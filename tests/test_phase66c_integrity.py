from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pytest

import eduecon_radar.services.migration_integrity as migration_integrity
from eduecon_radar.config import Settings
from eduecon_radar.db import Database
from eduecon_radar.resources import read_text
from eduecon_radar.schemas import JournalConfig, PaperMetadata
from eduecon_radar.services.ai_budget import AIBudgetManager, BudgetBlocked, BudgetLimits
from eduecon_radar.services.doi import canonicalize_doi, doi_url, validate_doi_syntax
from eduecon_radar.services.metadata_provenance import audit_metadata_provenance
from eduecon_radar.services.migration_integrity import (
    MigrationIntegrityError,
    baseline_migration_ledger,
    migrate_database,
    verify_migrations,
)
from eduecon_radar.services.runtime_doctor import run_doctor
from eduecon_radar.services.schema_manifest import (
    SchemaManifestError,
    generate_schema_manifest,
    verify_schema_manifest,
)
from eduecon_radar.services.task_lock import TaskLeaseManager
from eduecon_radar.services.translation import TranslationParameters, TranslationService


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" DOI: 10.1000/ABC. ", "10.1000/abc."),
        ("https://doi.org/10.1000/a%3Fb%3Dc", "10.1000/a?b=c"),
        ("https://dx.doi.org/10.1000/abc(def)", "10.1000/abc(def)"),
        ("10.1000/a#fragment", "10.1000/a#fragment"),
        ("\u00a010.1000/中文测试\u00a0", "10.1000/中文测试"),
        ("not-a-doi", None),
        (None, None),
    ],
)
def test_canonical_doi_is_lossless_and_validated(raw, expected):
    assert canonicalize_doi(raw) == expected
    assert validate_doi_syntax(expected) is (expected is not None)


def test_doi_url_encodes_reserved_characters():
    assert doi_url("10.1000/a?b=c") == "https://doi.org/10.1000/a%3Fb%3Dc"


@pytest.mark.parametrize("profile", ["production", "validation", "gold"])
def test_fresh_profile_migration_has_ordered_ledger(tmp_path, profile):
    database = tmp_path / f"{profile}.db"
    result = migrate_database(database, profile, dry_run=False)
    assert result.ledger_present and not result.pending and not result.errors
    assert verify_migrations(database, profile).applied == result.applied
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()


def test_repeated_migration_does_not_repeat_rows(tmp_path):
    database = tmp_path / "production.db"
    first = migrate_database(database, "production", dry_run=False)
    second = migrate_database(database, "production", dry_run=False)
    assert first.applied == second.applied
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == len(
            first.applied
        )


def test_ledger_hash_change_fails_closed(tmp_path):
    database = tmp_path / "production.db"
    migrate_database(database, "production", dry_run=False)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE schema_migrations SET migration_sha256='tampered' WHERE migration_id=1"
        )
        connection.commit()
    with pytest.raises(MigrationIntegrityError, match="migration_hash_mismatch"):
        verify_migrations(database, "production")


def test_profile_mismatch_fails_closed(tmp_path):
    database = tmp_path / "validation.db"
    migrate_database(database, "validation", dry_run=False)
    with pytest.raises(MigrationIntegrityError):
        verify_migrations(database, "production")


def test_ambiguous_unknown_schema_cannot_be_baselined(tmp_path):
    database = tmp_path / "unknown.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE unrelated(id INTEGER PRIMARY KEY)")
    with pytest.raises(MigrationIntegrityError, match="uniquely"):
        baseline_migration_ledger(database, "production", confirmed=True)
    with sqlite3.connect(database) as connection:
        assert not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='schema_migrations'"
        ).fetchone()


def test_exact_historical_schema_can_be_baselined(tmp_path):
    database = Database(tmp_path / "historical.db")
    database.init_schema()
    with database.connect() as connection:
        for filename in migration_integrity.PROFILE_MIGRATIONS["production"][:4]:
            connection.executescript(read_text(f"migrations/{filename}"))
    result = baseline_migration_ledger(database.path, "production", confirmed=True)
    assert result.applied[-1] == "0004_add_domain_semantic_runs.sql"
    assert result.pending[0] == "0007_add_runtime_leases_and_task_reviews.sql"


def test_failed_migration_rolls_back_schema_and_records_failure(tmp_path, monkeypatch):
    database = tmp_path / "failure.db"
    monkeypatch.setitem(
        migration_integrity.PROFILE_MIGRATIONS,
        "validation",
        ("0999_failure.sql",),
    )
    monkeypatch.setattr(
        migration_integrity,
        "migration_sha256",
        lambda filename: "a" * 64,
    )
    monkeypatch.setattr(
        migration_integrity,
        "read_text",
        lambda path: (
            "BEGIN IMMEDIATE; CREATE TABLE partial_write(id INTEGER); "
            "THIS IS INVALID; COMMIT;"
        ),
    )
    with pytest.raises(sqlite3.DatabaseError):
        migrate_database(database, "validation", dry_run=False)
    with sqlite3.connect(database) as connection:
        assert not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='partial_write'"
        ).fetchone()
        row = connection.execute(
            "SELECT success,error_type FROM schema_migrations WHERE migration_id=999"
        ).fetchone()
    assert row and row[0] == 0 and row[1]


def test_manifest_hash_is_stable_and_detects_index_difference(tmp_path):
    first = generate_schema_manifest("production")
    second = generate_schema_manifest("production")
    assert first["manifest_sha256"] == second["manifest_sha256"]
    database = tmp_path / "production.db"
    manifest = tmp_path / "manifest.json"
    migrate_database(database, "production", dry_run=False)
    generate_schema_manifest("production", output=manifest)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE INDEX injected_index ON papers(title)")
    with pytest.raises(SchemaManifestError, match="schema_manifest_mismatch"):
        verify_schema_manifest(database, "production", manifest)


def test_manifest_rejects_wrong_profile(tmp_path):
    database = tmp_path / "production.db"
    manifest = tmp_path / "validation.json"
    migrate_database(database, "production", dry_run=False)
    generate_schema_manifest("validation", output=manifest)
    with pytest.raises(SchemaManifestError, match="database_profile_mismatch"):
        verify_schema_manifest(database, "production", manifest)


def _canonical_production_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "production-manifest.json"
    generate_schema_manifest("production", output=manifest)
    return manifest


def _completed_baseline_production_database(path: Path) -> Path:
    database = Database(path)
    database.init_schema()
    with database.connect() as connection:
        for filename in migration_integrity.PROFILE_MIGRATIONS["production"][:4]:
            connection.executescript(read_text(f"migrations/{filename}"))
    baseline_migration_ledger(path, "production", confirmed=True)
    migrate_database(path, "production", dry_run=False)
    return path


def test_manifest_accepts_fresh_and_baseline_ledgers_with_same_schema(tmp_path):
    manifest = _canonical_production_manifest(tmp_path)
    fresh = tmp_path / "fresh.db"
    migrate_database(fresh, "production", dry_run=False)
    assert verify_schema_manifest(fresh, "production", manifest)["status"] == "ok"

    baseline = _completed_baseline_production_database(tmp_path / "baseline.db")
    assert verify_schema_manifest(baseline, "production", manifest)["status"] == "ok"


def test_manifest_ignores_migration_execution_history_and_provenance(tmp_path):
    manifest = _canonical_production_manifest(tmp_path)
    database = tmp_path / "history.db"
    migrate_database(database, "production", dry_run=False)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """UPDATE schema_migrations
            SET inferred=1,inference_method='exact_schema_fingerprint',
                source_manifest='different-history',applied_at='2000-01-01T00:00:00Z',
                execution_ms=999,application_version='historical',
                schema_fingerprint='historical-step-fingerprint'"""
        )
        connection.commit()
    assert verify_schema_manifest(database, "production", manifest)["status"] == "ok"


def test_manifest_still_rejects_migration_hash_and_missing_entry(tmp_path):
    manifest = _canonical_production_manifest(tmp_path)
    tampered = tmp_path / "tampered.db"
    migrate_database(tampered, "production", dry_run=False)
    with sqlite3.connect(tampered) as connection:
        connection.execute(
            "UPDATE schema_migrations SET migration_sha256='tampered' WHERE migration_id=1"
        )
        connection.commit()
    with pytest.raises(MigrationIntegrityError, match="migration_hash_mismatch"):
        verify_schema_manifest(tampered, "production", manifest)

    missing = tmp_path / "missing.db"
    migrate_database(missing, "production", dry_run=False)
    with sqlite3.connect(missing) as connection:
        # Drop the newest entry, whatever the production profile currently ends
        # with.  The remaining set stays a valid prefix, so the rejection must
        # come from the manifest comparison rather than from ordering — a
        # hard-coded migration id here silently became a *hole* in the middle of
        # the sequence as soon as a later migration was added.
        newest = connection.execute(
            "SELECT MAX(migration_id) FROM schema_migrations"
        ).fetchone()[0]
        connection.execute(
            "DELETE FROM schema_migrations WHERE migration_id=?", (newest,)
        )
        connection.commit()
    with pytest.raises(SchemaManifestError, match="schema_manifest_mismatch:migrations"):
        verify_schema_manifest(missing, "production", manifest)


# --------------------------------------------------------------------------- #
# Manifest source: a fresh database vs a named database
#
# `generate-schema-manifest` without `--database` describes a database built by
# running the migrations; with `--database` it describes the given database's
# real schema *and* its real ledger.  Only the second can carry a reconciled
# ledger's provenance, and the two must not be substitutable.
# --------------------------------------------------------------------------- #


def test_manifest_from_named_database_keeps_ledger_provenance_and_is_read_only(tmp_path):
    reconciled = _completed_baseline_production_database(tmp_path / "reconciled.db")

    def ledger_rows(path: Path) -> list[tuple[object, ...]]:
        with sqlite3.connect(path) as connection:
            return [
                tuple(row)
                for row in connection.execute(
                    """SELECT migration_id,filename,migration_sha256,inferred,
                              inference_method,schema_version_before,
                              schema_version_after
                       FROM schema_migrations ORDER BY migration_id"""
                )
            ]

    before_rows = ledger_rows(reconciled)
    before_bytes = hashlib.sha256(reconciled.read_bytes()).hexdigest()
    assert any(row[3] == 1 for row in before_rows), "fixture must exercise inferred rows"
    assert any(row[4] for row in before_rows), "fixture must record an inference method"

    manifest = tmp_path / "reconciled.json"
    payload = generate_schema_manifest("production", output=manifest, database=reconciled)

    # The manifest reports the ledger as it actually is, not as a clean run would
    # have written it.  (`inference_method` is deliberately not part of the
    # manifest; `inferred` is, and is the flag that distinguishes the sources.)
    assert [
        (
            m["migration_id"],
            m["filename"],
            m["migration_sha256"],
            m["inferred"],
            m["schema_version_before"],
            m["schema_version_after"],
        )
        for m in payload["migrations"]
    ] == [row[:3] + (row[3], row[5], row[6]) for row in before_rows]
    assert all(m["database_profile"] == "production" for m in payload["migrations"])

    # Naming a database must not write to it.
    assert ledger_rows(reconciled) == before_rows
    assert hashlib.sha256(reconciled.read_bytes()).hexdigest() == before_bytes

    # And the result is a valid manifest for that database.
    assert verify_schema_manifest(reconciled, "production", manifest)["status"] == "ok"


def test_a_fresh_built_manifest_cannot_substitute_for_production_provenance(tmp_path):
    """A scratch-database manifest must not be able to stand in for production.

    Schemas agree exactly (same fingerprint, same tables), so the only thing
    keeping the two sources apart is the recorded migration history.  This test
    pins both halves of that guarantee:

    * the manifest generated from a database carries that database's real
      provenance, so regenerating from a scratch database produces a *different*
      manifest for the same schema;
    * verification compares the recorded migration chain, so a manifest built
      from the wrong database is refused rather than accepted.

    Provenance flags themselves (``inferred``) are deliberately outside the
    compared identity — they record *how* a migration landed, not *what* the
    schema is — so the refusal below is driven by the recorded version chain,
    which is what actually differs in the production database.
    """
    fresh = tmp_path / "fresh.db"
    migrate_database(fresh, "production", dry_run=False)

    # Reproduce the shape the production ledger actually has: migrations that
    # were reconciled from schema evidence record a synthetic predecessor
    # (`migration_id:N`) instead of the previous filename, and are flagged.
    reconciled_path = tmp_path / "reconciled.db"
    migrate_database(reconciled_path, "production", dry_run=False)
    with sqlite3.connect(reconciled_path) as connection:
        connection.execute(
            """UPDATE schema_migrations
               SET schema_version_before=('migration_id:' || (migration_id - 1)),
                   inferred=1, inference_method='schema_object_match'
               WHERE migration_id IN (10, 11, 12)"""
        )
        connection.commit()

    from_db = generate_schema_manifest(
        "production", output=tmp_path / "from-db.json", database=reconciled_path
    )
    from_fresh = generate_schema_manifest("production", output=tmp_path / "from-fresh.json")

    # Same schema, different recorded provenance.
    assert from_db["schema_fingerprint"] == from_fresh["schema_fingerprint"]
    assert from_db["tables"] == from_fresh["tables"]
    assert from_db["manifest_sha256"] != from_fresh["manifest_sha256"]
    assert from_db["migrations"] != from_fresh["migrations"]

    # The database-derived manifest reports the reconciled flags verbatim,
    # while a fresh build would have claimed a clean, never-reconciled history.
    assert {m["migration_id"]: m["inferred"] for m in from_db["migrations"]}[10] == 1
    assert {m["inferred"] for m in from_fresh["migrations"]} == {0}

    # (a) a fresh-built manifest must not validate the reconciled database
    with pytest.raises(SchemaManifestError, match="schema_manifest_mismatch"):
        verify_schema_manifest(reconciled_path, "production", tmp_path / "from-fresh.json")

    # (b) and the reverse: a database-derived manifest must not validate a fresh build
    with pytest.raises(SchemaManifestError, match="schema_manifest_mismatch"):
        verify_schema_manifest(fresh, "production", tmp_path / "from-db.json")

    # Each one does validate its own source, so the refusal above is provenance,
    # not a malformed manifest.
    assert verify_schema_manifest(
        reconciled_path, "production", tmp_path / "from-db.json"
    )["status"] == "ok"
    assert verify_schema_manifest(fresh, "production", tmp_path / "from-fresh.json")[
        "status"
    ] == "ok"


def test_generate_schema_manifest_cli_accepts_database_and_stays_backward_compatible(
    tmp_path,
):
    """The CLI gains an explicit real-database mode without changing the default.

    Asserted from the written manifest rather than captured stdout: the command
    also logs, so parsing the console would test the logging stream instead of
    the manifest.  The default (fresh-database) source is exercised at the
    service level elsewhere in this module; here only the argparse contract and
    the new ``--database`` behaviour are pinned.
    """
    from eduecon_radar import cli

    project_root = Path(__file__).resolve().parents[1]
    database = tmp_path / "cli.db"
    migrate_database(database, "production", dry_run=False)
    before = hashlib.sha256(database.read_bytes()).hexdigest()

    # The option is optional: old invocations still parse.
    legacy = cli.build_parser().parse_args(
        ["generate-schema-manifest", "--profile", "production"]
    )
    assert legacy.database is None

    output = tmp_path / "cli-manifest.json"
    args = cli.build_parser().parse_args(
        [
            "generate-schema-manifest",
            "--profile",
            "production",
            "--database",
            str(database),
            "--output",
            str(output),
        ]
    )
    assert args.database == database

    original_cwd = Path.cwd()
    try:
        __import__("os").chdir(project_root)
        assert cli.run(args, Settings.load(project_root)) == 0
    finally:
        __import__("os").chdir(original_cwd)

    assert output.is_file()
    assert verify_schema_manifest(database, "production", output)["status"] == "ok"
    # Naming the database must not modify it.
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before

    payload = json.loads(output.read_text("utf-8"))
    assert len(payload["migrations"]) == len(
        [
            row
            for row in sqlite3.connect(database).execute(
                "SELECT 1 FROM schema_migrations WHERE success=1"
            )
        ]
    )


def test_manifest_still_rejects_migration_order_difference(tmp_path):
    manifest = _canonical_production_manifest(tmp_path)
    database = tmp_path / "wrong-order.db"
    migrate_database(database, "production", dry_run=False)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE schema_migrations SET migration_id=70 WHERE migration_id=7")
        connection.execute("UPDATE schema_migrations SET migration_id=7 WHERE migration_id=8")
        connection.execute("UPDATE schema_migrations SET migration_id=8 WHERE migration_id=70")
        connection.commit()
    with pytest.raises(MigrationIntegrityError, match="migration_order_mismatch"):
        verify_schema_manifest(database, "production", manifest)


def _paper(abstract: str | None) -> PaperMetadata:
    return PaperMetadata(
        doi="10.1000/TEST",
        title="A complete education title",
        abstract=abstract,
        authors=["Jane Doe"],
        publisher="Publisher",
        online_date=date(2026, 1, 1),
        source_metadata={"DOI": "10.1000/TEST", "_source_name": "crossref"},
    )


def _production_database(tmp_path: Path) -> tuple[Database, int]:
    database = Database(tmp_path / "production.db")
    migrate_database(database.path, "production", dry_run=False)
    journal_id = database.upsert_journal(
        JournalConfig(
            name="Journal",
            short_name="J",
            issn="0000-0000",
            rank=1,
            rank_source="test",
            priority="high",
            enabled=True,
        )
    )
    return database, journal_id


def test_resync_prevents_empty_and_placeholder_abstract_degradation(tmp_path):
    database, journal_id = _production_database(tmp_path)
    paper_id, _ = database.upsert_paper(_paper("A complete abstract " * 20), journal_id)
    database.upsert_paper(_paper("No abstract"), journal_id)
    with database.connect() as connection:
        row = connection.execute("SELECT abstract FROM papers WHERE id=?", (paper_id,)).fetchone()
    assert row["abstract"] == "A complete abstract " * 20
    audit = audit_metadata_provenance(database.path)
    assert audit["prevented_degradations"] >= 1


def test_resync_is_idempotent_and_preserves_local_metadata(tmp_path):
    database, journal_id = _production_database(tmp_path)
    paper = _paper("A complete abstract " * 20)
    paper_id, _ = database.upsert_paper(paper, journal_id)
    with database.connect() as connection:
        metadata = json.loads(
            connection.execute(
                "SELECT source_metadata_json FROM papers WHERE id=?", (paper_id,)
            ).fetchone()[0]
        )
        metadata["abstract_provenance"] = {"source": "manual"}
        connection.execute(
            "UPDATE papers SET source_metadata_json=? WHERE id=?",
            (json.dumps(metadata), paper_id),
        )
    database.upsert_paper(paper, journal_id)
    with database.connect() as connection:
        stored = json.loads(
            connection.execute(
                "SELECT source_metadata_json FROM papers WHERE id=?", (paper_id,)
            ).fetchone()[0]
        )
    assert stored["abstract_provenance"] == {"source": "manual"}


def _budget_task(tmp_path: Path) -> tuple[Database, int, TaskLeaseManager]:
    database, _ = _production_database(tmp_path)
    with database.connect() as connection:
        cursor = connection.execute(
            """INSERT INTO workflow_batches
            (name,source_type,status,source_config_json,screening_policy,created_at,updated_at)
            VALUES ('test','sync','running','{}','{}','2026-01-01','2026-01-01')"""
        )
        task_id = int(cursor.lastrowid)
    manager = TaskLeaseManager(database)
    manager.acquire(task_id, owner_id="test-owner", stage="paid")
    return database, task_id, manager


def test_budget_reserves_before_call_and_blocks_at_limit(tmp_path):
    database, task_id, _ = _budget_task(tmp_path)
    manager = AIBudgetManager(database)
    manager.ensure_default_budget(task_id)
    manager.set_budget(
        task_id,
        BudgetLimits(1, 1000, 1000, 100000, 1, 1),
        changed_by="test",
        reason="concurrency gate",
    )
    manager.reserve(
        task_id=task_id,
        stage="fast_semantic",
        provider="fake",
        model_name="fake",
        idempotency_key="one",
        estimated_input_tokens=10,
        estimated_output_tokens=10,
    )
    with pytest.raises(BudgetBlocked, match="max_calls"):
        manager.reserve(
            task_id=task_id,
            stage="fast_semantic",
            provider="fake",
            model_name="fake",
            idempotency_key="two",
            estimated_input_tokens=10,
            estimated_output_tokens=10,
        )


def test_two_connections_cannot_reserve_last_budget_twice(tmp_path):
    database, task_id, _ = _budget_task(tmp_path)
    manager = AIBudgetManager(database)
    manager.ensure_default_budget(task_id)
    manager.set_budget(
        task_id,
        BudgetLimits(1, 1000, 1000, 100000, 1, 1),
        changed_by="test",
        reason="last reservation",
    )

    def reserve(identity: str) -> str:
        try:
            AIBudgetManager(database).reserve(
                task_id=task_id,
                stage="fast_semantic",
                provider="fake",
                model_name="fake",
                idempotency_key=identity,
                estimated_input_tokens=10,
                estimated_output_tokens=10,
            )
            return "reserved"
        except BudgetBlocked:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(reserve, ("a", "b")))
    assert sorted(results) == ["blocked", "reserved"]


def test_missing_usage_is_conservatively_settled(tmp_path):
    database, task_id, _ = _budget_task(tmp_path)
    manager = AIBudgetManager(database)
    reservation = manager.reserve(
        task_id=task_id,
        stage="fast_semantic",
        provider="fake",
        model_name="fake",
        idempotency_key="missing-usage",
        estimated_input_tokens=10,
        estimated_output_tokens=10,
    )
    manager.settle(reservation, input_tokens=0, output_tokens=0)
    with database.connect() as connection:
        row = connection.execute(
            "SELECT usage_missing,actual_cost_microunits FROM ai_usage_reservations"
        ).fetchone()
    assert row["usage_missing"] == 1
    assert row["actual_cost_microunits"] > 0


def test_stale_fencing_token_cannot_settle(tmp_path):
    database, task_id, lease_manager = _budget_task(tmp_path)
    budget = AIBudgetManager(database)
    reservation = budget.reserve(
        task_id=task_id,
        stage="fast_semantic",
        provider="fake",
        model_name="fake",
        idempotency_key="stale",
        estimated_input_tokens=10,
        estimated_output_tokens=10,
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE workflow_task_leases SET status='stale' WHERE task_id=?", (task_id,)
        )
    lease_manager.acquire(task_id, owner_id="new-owner", stage="paid")
    with pytest.raises(BudgetBlocked, match="stale_fencing_token"):
        budget.settle(reservation, input_tokens=5, output_tokens=5)


def test_budget_blocked_translation_never_calls_client(tmp_path):
    database, journal_id = _production_database(tmp_path)
    paper_id, _ = database.upsert_paper(_paper("Original abstract " * 20), journal_id)
    with database.connect() as connection:
        task_id = int(
            connection.execute(
                """INSERT INTO workflow_batches
                (name,source_type,status,source_config_json,screening_policy,created_at,updated_at)
                VALUES ('budget','sync','running','{}','{}','2026-01-01','2026-01-01')
                RETURNING id"""
            ).fetchone()[0]
        )
    TaskLeaseManager(database).acquire(task_id, owner_id="owner", stage="title_translation")
    budget = AIBudgetManager(database)
    budget.ensure_default_budget(task_id)
    budget.set_budget(
        task_id,
        BudgetLimits(1, 100000, 100000, 1000000, 1, 1),
        changed_by="test",
        reason="block next call",
    )
    budget.reserve(
        task_id=task_id,
        stage="title_translation",
        provider="fake",
        model_name="fake",
        idempotency_key="consume-only-call",
        estimated_input_tokens=10,
        estimated_output_tokens=10,
    )

    class FakeClient:
        calls = 0

        def request_json(self, **kwargs):
            self.calls += 1
            raise AssertionError("budget block must happen before client call")

    client = FakeClient()
    service = TranslationService(
        database,
        Path(__file__).resolve().parents[1] / "prompts",
        client,
        "fake",
        budget_manager=budget,
    )
    with pytest.raises(BudgetBlocked):
        service.run(
            TranslationParameters(
                review_status=None,
                paper_ids=(paper_id,),
                translate_titles=True,
                translate_abstracts=False,
                task_id=task_id,
            )
        )
    assert client.calls == 0


def _doctor_settings(tmp_path: Path, database: Path, *, api_key: str | None) -> Settings:
    (tmp_path / "exports").mkdir(exist_ok=True)
    (tmp_path / "logs").mkdir(exist_ok=True)
    return Settings(
        project_root=tmp_path,
        database_url=f"sqlite:///{database.as_posix()}",
        journals_path=Path(__file__).resolve().parents[1] / "journals.yaml",
        prompts_dir=Path(__file__).resolve().parents[1] / "prompts",
        deepseek_api_key=api_key,
        deepseek_base_url="https://invalid.test",
        deepseek_model_fast="fake",
        deepseek_model_strong="fake",
        deepseek_reasoning_enabled=False,
        deepseek_timeout_seconds=1,
        deepseek_max_retries=1,
        openalex_api_key=None,
        unpaywall_email=None,
        log_level="INFO",
        app_timezone="Asia/Shanghai",
        ui_version="v05",
        domain_config_dir=Path(__file__).resolve().parents[1]
        / "config/domains/education_economics",
    )


def test_doctor_is_offline_read_only_and_never_prints_secret(tmp_path, monkeypatch):
    database = tmp_path / "doctor.db"
    migrate_database(database, "production", dry_run=False)
    before = hashlib.sha256(database.read_bytes()).hexdigest()

    def forbidden_network(*args, **kwargs):
        raise AssertionError("doctor must not call an external service")

    import httpx

    monkeypatch.setattr(httpx.Client, "send", forbidden_network)
    result = run_doctor(_doctor_settings(tmp_path, database, api_key="DO_NOT_PRINT_ME"))
    after = hashlib.sha256(database.read_bytes()).hexdigest()
    assert before == after
    encoded = json.dumps(result, sort_keys=True)
    assert "DO_NOT_PRINT_ME" not in encoded
    assert '"DEEPSEEK_API_KEY": "present"' in encoded
    assert result["status"] in {"blocked", "warning", "ok"}


def test_doctor_reports_missing_runtime_resource_without_crashing(tmp_path, monkeypatch):
    database = tmp_path / "doctor.db"
    migrate_database(database, "production", dry_run=False)

    def missing():
        raise FileNotFoundError("missing packaged prompt")

    monkeypatch.setattr(
        "eduecon_radar.services.runtime_doctor.verify_runtime_resources", missing
    )
    result = run_doctor(_doctor_settings(tmp_path, database, api_key=None))
    resource_check = next(
        item for item in result["checks"] if item["name"] == "runtime_resources"
    )
    assert resource_check["status"] == "error"
