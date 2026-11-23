"""Consolidated invariants of the product architecture.

These tests encode what "one trustworthy source" means after the core
consolidation: one storage authority, one journal source, one prompt source, one
provider list, and a packaged runtime that does not depend on the checkout.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tomllib
from pathlib import Path

import pytest

from eduecon_radar.config import Settings
from eduecon_radar.db import Database
from eduecon_radar.resources import runtime_path
from eduecon_radar.services import local_pipeline as pipeline
from eduecon_radar.services import profile_store
from eduecon_radar.services.migration_integrity import migrate_database

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "config" / "profiles"
CATALOG = ROOT / "config" / "journals" / "catalog.json"


# --- canonical storage -------------------------------------------------------


def _database(tmp_path: Path) -> Database:
    path = tmp_path / "consolidation.db"
    migrate_database(path, "production", dry_run=False)
    return Database(path)


def _seed(database: Database, *, doi: str = "10.1/consolidate") -> tuple[int, int]:
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO journals (name,short_name,issn,priority,enabled,created_at,updated_at)
            VALUES ('Journal','J','0000-0001','high',1,'t','t')
            ON CONFLICT(short_name) DO NOTHING"""
        )
        journal_id = connection.execute(
            "SELECT id FROM journals WHERE short_name='J'"
        ).fetchone()["id"]
        cursor = connection.execute(
            """INSERT INTO papers
            (doi,fingerprint,title,abstract,authors_json,journal_id,source_metadata_json,
             first_seen_at,last_seen_at,created_at,updated_at)
            VALUES (?,?,?,?,'[]',?,'{}','t','t','t','t')""",
            (doi, f"fp-{doi}", "T", "A", journal_id),
        )
        paper_id = int(cursor.lastrowid)
        run = connection.execute(
            """INSERT INTO sync_runs
            (started_at,status,requested_limit,matched_paper_ids,profile_id)
            VALUES ('t','success',10,?, 'education_economics')""",
            (json.dumps([paper_id]),),
        )
    run_id = int(run.lastrowid)
    pipeline.ensure_paper_statuses_for_paper_ids(database, run_id, (paper_id,))
    return run_id, paper_id


def _assess(database: Database, run_id: int, paper_id: int, decision: str) -> None:
    pipeline._record_profile_assessment(
        database,
        profiles_ready=True,
        profile_id="education_economics",
        run_id=run_id,
        paper_id=paper_id,
        classification={"decision": decision, "screening_basis": "title_and_abstract"},
        model_name="m",
        prompt_version="v1",
        title="T",
        abstract="A",
    )


def test_only_the_profile_table_records_a_judgement(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, paper_id = _seed(database)
    _assess(database, run_id, paper_id, "relevant")
    with database.connect() as connection:
        snapshot = connection.execute(
            "SELECT classification_json,model_name,prompt_version,classified_at "
            "FROM paper_processing_status WHERE sync_run_id=? AND paper_id=?",
            (run_id, paper_id),
        ).fetchone()
        stored = connection.execute(
            "SELECT COUNT(*) FROM paper_profile_assessments WHERE paper_id=?",
            (paper_id,),
        ).fetchone()[0]
    assert stored == 1
    assert snapshot["classification_json"] == "{}"
    assert snapshot["model_name"] is None
    assert snapshot["prompt_version"] is None
    assert snapshot["classified_at"] is None


def test_assessment_columns_cannot_be_written_through_the_run_api(
    tmp_path: Path,
) -> None:
    """The run-state writer refuses judgement fields, so they cannot be doubled."""
    database = _database(tmp_path)
    run_id, paper_id = _seed(database)
    pipeline._update_status(
        database,
        run_id,
        paper_id,
        status="classified",
        stage="complete",
        classification_json='{"decision": "relevant"}',
    )
    with database.connect() as connection:
        row = connection.execute(
            "SELECT classification_json FROM paper_processing_status WHERE paper_id=?",
            (paper_id,),
        ).fetchone()
    assert row["classification_json"] == "{}"


def test_historical_run_snapshot_stays_readable(tmp_path: Path) -> None:
    """A row written before the cutover is still shown for its own run."""
    database = _database(tmp_path)
    run_id, paper_id = _seed(database)
    with database.connect() as connection:
        connection.execute(
            "UPDATE paper_processing_status SET classification_json=?, model_name=? "
            "WHERE sync_run_id=? AND paper_id=?",
            (
                json.dumps({"decision": "irrelevant", "screening_basis": "title_only"}),
                "deepseek-v4-pro",
                run_id,
                paper_id,
            ),
        )
    item = pipeline.sync_run_papers(database, run_id)[0]
    assert item["assessment_source"] == "run_snapshot"
    assert item["processing"]["classification"]["decision"] == "irrelevant"


def test_run_outcome_is_never_presented_as_a_judgement(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, paper_id = _seed(database)
    with database.connect() as connection:
        connection.execute(
            "UPDATE paper_processing_status SET status='needs_manual_abstract' "
            "WHERE sync_run_id=? AND paper_id=?",
            (run_id, paper_id),
        )
    item = pipeline.sync_run_papers(database, run_id)[0]
    assert item["assessment_source"] == "run_state"
    assert item["processing"]["classification"] == {"decision": "needs_abstract"}


def test_profile_assessment_wins_over_the_run_snapshot(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, paper_id = _seed(database)
    with database.connect() as connection:
        connection.execute(
            "UPDATE paper_processing_status SET classification_json=? "
            "WHERE sync_run_id=? AND paper_id=?",
            (json.dumps({"decision": "irrelevant"}), run_id, paper_id),
        )
    _assess(database, run_id, paper_id, "relevant")
    item = pipeline.sync_run_papers(database, run_id)[0]
    assert item["assessment_source"] == "profile"
    assert item["processing"]["classification"]["decision"] == "relevant"


def test_no_drift_between_snapshot_and_canonical_judgement(tmp_path: Path) -> None:
    """The drift check reports rows where the two stores disagree."""
    database = _database(tmp_path)
    run_id, paper_id = _seed(database)
    _assess(database, run_id, paper_id, "relevant")
    assert profile_store.assessment_drift(database) == ()

    with database.connect() as connection:
        connection.execute(
            "UPDATE paper_processing_status SET classification_json=? "
            "WHERE sync_run_id=? AND paper_id=?",
            (json.dumps({"decision": "irrelevant"}), run_id, paper_id),
        )
    drift = profile_store.assessment_drift(database)
    assert len(drift) == 1
    assert drift[0]["kind"] == "snapshot_contradicts_assessment"


# --- one source per configuration kind ---------------------------------------


def test_one_journal_source() -> None:
    legacy = ROOT / "config" / "journals" / "education_economics.json"
    assert not legacy.exists(), "the per-domain journal registry must be gone"
    assert CATALOG.is_file()
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert data["catalog_version"]
    # The catalog is the one place journals live and it only grows: assert the
    # required entries rather than a frozen count.
    keys = {item["journal_key"] for item in data["journals"]}
    assert {
        "economics_of_education_review",
        "journal_of_labor_economics",
        "general_qje",
        "journal_of_economic_behavior_and_organization",
    } <= keys
    assert len(keys) == len(data["journals"])


def test_one_screening_prompt_source() -> None:
    for name in (
        "screening_primary_system.txt",
        "screening_primary_user.txt",
        "screening_review_system.txt",
        "screening_review_user.txt",
    ):
        assert not (ROOT / "prompts" / name).exists(), f"duplicate prompt {name}"
        for profile in ("education_economics", "labor_economics"):
            assert (PROFILES / profile / "prompts" / name).is_file()
    settings = Settings.load(ROOT)
    assert settings.screening_prompts_dir.is_dir()
    assert len(list(settings.screening_prompts_dir.glob("*.txt"))) == 4


def test_one_provider_list_without_retired_entries() -> None:
    import yaml

    payload = yaml.safe_load(
        (ROOT / "config" / "abstract_sources.yaml").read_text(encoding="utf-8")
    )
    names = list(payload["sources"])
    assert names == [
        "manual",
        "crossref",
        "repec",
        "openalex",
        "semantic_scholar",
        "doaj",
        "openaire",
    ]
    assert all(item["enabled"] for item in payload["sources"].values())
    assert not (ROOT / "config" / "education_ontology.yaml").exists()


# --- packaging ---------------------------------------------------------------


def test_every_runtime_file_is_packaged() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_packaging import main  # noqa: PLC0415

    assert main([]) == 0


def test_force_include_entries_all_exist() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entries = (
        data["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    )
    missing = [source for source in entries if not (ROOT / source).exists()]
    assert missing == []


@pytest.mark.parametrize(
    "relative",
    [
        "config/repec_series.json",
        "config/journals/catalog.json",
        "config/profiles/education_economics/profile.yaml",
        "config/profiles/labor_economics/prompts/screening_review_user.txt",
        "prompts/translation_abstract_system.txt",
        "migrations/0013_add_profile_scoped_assessment.sql",
    ],
)
def test_runtime_resource_resolves(relative: str) -> None:
    assert runtime_path(ROOT, relative).exists()


def test_resource_resolution_survives_a_missing_checkout_copy(tmp_path: Path) -> None:
    """A wheel install has no project copy; resolution must not need one."""
    resolved = runtime_path(tmp_path, "config/journals/catalog.json")
    assert resolved.exists()
    assert resolved != tmp_path / "config" / "journals" / "catalog.json"


# --- fresh install -----------------------------------------------------------


def test_fresh_database_supports_the_whole_product_path(tmp_path: Path) -> None:
    """A database built from zero can store and read a judgement."""
    database = _database(tmp_path)
    run_id, paper_id = _seed(database)
    _assess(database, run_id, paper_id, "relevant")
    item = pipeline.sync_run_papers(database, run_id)[0]
    assert item["assessment_source"] == "profile"
    assert pipeline.save_human_decisions(
        database, run_id, {paper_id: ("related", "note")}
    ) == (paper_id,)
    review = profile_store.reviews_for(database, "education_economics", (paper_id,))[
        paper_id
    ]
    assert (review.human_decision, review.human_note) == ("related", "note")
    with database.connect() as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_profile_isolation_survives_the_consolidation(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, paper_id = _seed(database)
    _assess(database, run_id, paper_id, "relevant")
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO paper_profile_assessments
            (profile_id,paper_id,decision,screening_basis,classification_json,
             model_name,prompt_version,updated_at)
            VALUES ('labor_economics',?,?,?,?,'m','lab-v1','t')""",
            (
                paper_id,
                "irrelevant",
                "title_and_abstract",
                json.dumps({"decision": "irrelevant"}),
            ),
        )
    education = pipeline.sync_run_papers(database, run_id, profile_id="education_economics")[0]
    labor = pipeline.sync_run_papers(database, run_id, profile_id="labor_economics")[0]
    assert education["processing"]["classification"]["decision"] == "relevant"
    assert labor["processing"]["classification"]["decision"] == "irrelevant"
    with sqlite3.connect(database.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
