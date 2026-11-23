from __future__ import annotations

import json
import multiprocessing
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from eduecon_radar.db import Database
from eduecon_radar.domain.loader import DomainConfigLoader
from eduecon_radar.domain.semantic_prompting import canonical_sha256, file_sha256
from eduecon_radar.schemas import JournalConfig, PaperMetadata
from eduecon_radar.services.exporting import mentor_records, personal_research_records
from eduecon_radar.services.migrations import (
    apply_p0_hardening_migration,
    apply_translation_migration,
    apply_workflow_migration,
)
from eduecon_radar.services.prototype_workflow import (
    LiteratureSearchScope,
    V05LiteratureOrchestrator,
    _save_state,
    create_prototype_literature_task,
    list_task_papers,
)
from eduecon_radar.services.runtime_bundle import (
    RuntimeBundleResolver,
    RuntimeBundleTrustError,
    RuntimeModelIdentity,
)
from eduecon_radar.services.task_lock import (
    TaskLeaseBusy,
    TaskLeaseError,
    TaskLeaseLost,
    TaskLeaseManager,
)
from eduecon_radar.services.task_review import (
    TaskReviewInput,
    get_task_review,
    mark_task_recovery,
    save_task_review,
    task_review_counts,
)
from eduecon_radar.services.workflow import add_batch_items, create_workflow_batch

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "migrations/0007_add_runtime_leases_and_task_reviews.sql"


def _database(tmp_path: Path, *, hardened: bool = True) -> tuple[Database, int]:
    database = Database(tmp_path / "phase66b.db")
    database.init_schema()
    apply_translation_migration(database, ROOT / "migrations/0001_add_paper_translations.sql")
    apply_workflow_migration(database, ROOT / "migrations/0002_add_workflow_batches.sql")
    journal_id = database.upsert_journal(
        JournalConfig(name="Test Journal", short_name="TJ", issn="0000-0000")
    )
    if hardened:
        apply_p0_hardening_migration(database, MIGRATION)
    return database, journal_id


def _paper(database: Database, journal_id: int, suffix: str = "shared") -> int:
    return database.upsert_paper(
        PaperMetadata(
            doi=f"10.1000/{suffix}",
            title=f"Paper {suffix}",
            abstract="Original abstract.",
            authors=["Researcher"],
            issue_date=date(2026, 7, 1),
        ),
        journal_id,
    )[0]


def _task(database: Database, paper_id: int, name: str) -> int:
    task_id = create_prototype_literature_task(
        database,
        name=name,
        scope=LiteratureSearchScope(("TJ",), date(2026, 7, 1), date(2026, 7, 2), 10),
    )
    add_batch_items(database, task_id, (paper_id,))
    return task_id


def _set_yaml(path: Path, section: str, **values: object) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload[section].update(values)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _released_runtime(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "runtime"
    shutil.copytree(ROOT / "config", project / "config")
    shutil.copytree(ROOT / "prompts", project / "prompts")
    domain = project / "config/domains/education_economics"
    version = "1.0.0-test-release"
    _set_yaml(domain / "metadata.yaml", "metadata", status="released", bundle_version=version)
    _set_yaml(domain / "profile.yaml", "profile", status="released")
    for filename, section in (
        ("ontology.yaml", "ontology"),
        ("exclusion_contexts.yaml", "exclusion_contexts"),
        ("subject_codes.yaml", "subject_codes"),
        ("journal_priors.yaml", "journal_priors"),
        ("scoring.yaml", "scoring"),
    ):
        _set_yaml(domain / filename, section, status="released")
    semantic_path = domain / "semantic_models.yaml"
    semantic = yaml.safe_load(semantic_path.read_text(encoding="utf-8"))
    engine = semantic["semantic_engine"]
    engine.update(status="released", live_ai_allowed=True)
    engine["fast"].update(
        status="released", enabled=True, model="fast-release", provider="deepseek-compatible"
    )
    engine["strong_review"].update(
        status="released",
        enabled=True,
        model="strong-release",
        provider="deepseek-compatible",
    )
    semantic_path.write_text(yaml.safe_dump(semantic, sort_keys=False), encoding="utf-8")
    prompt_manifest_path = domain / "prompts/manifest.yaml"
    prompts = yaml.safe_load(prompt_manifest_path.read_text(encoding="utf-8"))
    prompts["prompt_manifest"]["status"] = "released"
    for entry in prompts["prompt_manifest"]["prompts"]:
        entry["status"] = "released"
    prompt_manifest_path.write_text(
        yaml.safe_dump(prompts, sort_keys=False), encoding="utf-8"
    )
    resolver = RuntimeBundleResolver(project)
    resources = {
        relative: file_sha256(project / relative)
        for relative in resolver.REQUIRED_RESOURCE_PATHS
    }
    domain_bundle = DomainConfigLoader(domain).load()
    prompt_hash = canonical_sha256(
        {
            key: resources[key]
            for key in sorted(resources)
            if "/prompts/" in key.replace("\\", "/")
        }
    )
    body = {
        "schema_version": 1,
        "status": "released",
        "bundle_version": version,
        "domain_config_sha256": domain_bundle.config_sha256,
        "prompt_bundle_sha256": prompt_hash,
        "models": {
            "provider": "deepseek-compatible",
            "fast": "fast-release",
            "strong": "strong-release",
            "translation": "fast-release",
        },
        "resources": resources,
        "created_at": "2026-07-24T00:00:00+00:00",
        "released_at": "2026-07-24T00:00:00+00:00",
    }
    manifest = {**body, "bundle_sha256": canonical_sha256(body)}
    path = project / resolver.MANIFEST_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return project, path


def _lease_worker(database_path: str, ready, release, output) -> None:
    manager = TaskLeaseManager(Database(database_path), ttl_seconds=30)
    try:
        lease = manager.acquire(1, owner_id="worker-one", stage="fast_semantic")
        output.put(("acquired", lease.fencing_token))
        ready.set()
        release.wait(10)
        manager.release(lease)
    except Exception as exc:  # pragma: no cover - relayed to parent for assertion
        output.put(("error", type(exc).__name__, str(exc)))
        ready.set()


def test_released_bundle_loads_and_newer_candidate_or_test_does_not_replace_it(tmp_path):
    project, manifest = _released_runtime(tmp_path)
    candidate_dir = project / "config/runtime_bundles/candidate"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "newer.json").write_text(
        json.dumps({"status": "candidate", "created_at": "2099-01-01"}), encoding="utf-8"
    )
    resolution = RuntimeBundleResolver(project).resolve()
    assert resolution.status == "released"
    assert resolution.manifest_path == manifest
    assert resolution.models.fast == "fast-release"


@pytest.mark.parametrize(
    ("mutation", "failure"),
    (
        ("bundle_hash", "bundle_hash_mismatch"),
        ("missing_resource", "bundle_resource_missing"),
        ("prompt", "prompt_hash_mismatch"),
        ("prompt_bundle", "prompt_bundle_hash_mismatch"),
        ("ontology", "resource_hash_mismatch"),
        ("test_status", "invalid_bundle_manifest"),
    ),
)
def test_released_bundle_tampering_fails_closed(tmp_path, mutation, failure):
    project, manifest = _released_runtime(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if mutation == "bundle_hash":
        payload["bundle_sha256"] = "0" * 64
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "missing_resource":
        (project / "prompts/translation_title_system.txt").unlink()
    elif mutation == "prompt":
        path = project / "prompts/translation_title_system.txt"
        path.write_text(path.read_text(encoding="utf-8") + "\nmodified", encoding="utf-8")
    elif mutation == "prompt_bundle":
        payload["prompt_bundle_sha256"] = "0" * 64
        body = dict(payload)
        body.pop("bundle_sha256")
        payload["bundle_sha256"] = canonical_sha256(body)
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "test_status":
        payload["status"] = "test"
        body = dict(payload)
        body.pop("bundle_sha256")
        payload["bundle_sha256"] = canonical_sha256(body)
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path = project / "config/domains/education_economics/ontology.yaml"
        path.write_text(path.read_text(encoding="utf-8") + "\n# modified", encoding="utf-8")
    with pytest.raises(RuntimeBundleTrustError) as raised:
        RuntimeBundleResolver(project).resolve()
    assert raised.value.failure_type == failure


def test_missing_release_blocks_before_ai_and_developer_override_is_explicit(tmp_path):
    project = tmp_path / "runtime"
    shutil.copytree(ROOT / "config", project / "config")
    shutil.copytree(ROOT / "prompts", project / "prompts")
    resolver = RuntimeBundleResolver(project)
    with pytest.raises(RuntimeBundleTrustError, match="released_bundle_missing"):
        resolver.resolve()
    with pytest.raises(RuntimeBundleTrustError, match="explicit_override_required"):
        resolver.resolve(mode="test")
    development = resolver.resolve(
        mode="test",
        allow_unreleased_bundle=True,
        development_models=RuntimeModelIdentity(
            provider="fake", fast="fake-fast", strong="fake-strong", translation="fake-fast"
        ),
    )
    assert development.unreleased_override
    assert development.status == "test_fixture"
    assert yaml.safe_load(
        (project / "config/domains/education_economics/semantic_models.yaml").read_text(
            encoding="utf-8"
        )
    )["semantic_engine"]["status"] == "test_fixture"


def test_released_version_collision_is_rejected(tmp_path):
    project, manifest = _released_runtime(tmp_path)
    collision = json.loads(manifest.read_text(encoding="utf-8"))
    collision["bundle_sha256"] = "F" * 64
    (manifest.parent / "collision.json").write_text(json.dumps(collision), encoding="utf-8")
    with pytest.raises(RuntimeBundleTrustError, match="released_version_collision"):
        RuntimeBundleResolver(project).resolve()


def test_task_lease_acquire_heartbeat_release_and_independent_tasks(tmp_path):
    database, _ = _database(tmp_path)
    task_one = create_workflow_batch(database, name="one", source_type="sync")
    task_two = create_workflow_batch(database, name="two", source_type="sync")
    manager = TaskLeaseManager(database, ttl_seconds=30)
    now = datetime(2026, 7, 24, tzinfo=UTC)
    first = manager.acquire(task_one, owner_id="owner-one", stage="sync", now=now)
    reentered = manager.acquire(
        task_one, owner_id="owner-one", stage="title_translation", now=now
    )
    assert reentered.fencing_token == first.fencing_token
    with pytest.raises(TaskLeaseBusy):
        TaskLeaseManager(Database(database.path), ttl_seconds=30).acquire(
            task_one, owner_id="owner-two", stage="sync", now=now
        )
    other = manager.acquire(task_two, owner_id="owner-two", stage="sync", now=now)
    renewed = manager.heartbeat(
        reentered, stage="fast_semantic", now=now + timedelta(seconds=10)
    )
    assert renewed.expires_at > first.expires_at
    with pytest.raises(TaskLeaseLost):
        manager.release(
            first.__class__(
                first.task_id,
                "not-owner",
                first.fencing_token,
                first.acquired_at,
                first.heartbeat_at,
                first.expires_at,
                first.current_stage,
            )
        )
    manager.release(renewed, now=now + timedelta(seconds=11))
    manager.release(other, now=now + timedelta(seconds=11))
    assert manager.inspect(task_one)["status"] == "released"


def test_expired_lease_recovery_and_fencing_prevent_stale_commit(tmp_path):
    database, _ = _database(tmp_path)
    task_id = create_workflow_batch(database, name="stale", source_type="sync")
    manager = TaskLeaseManager(database, ttl_seconds=5)
    start = datetime(2026, 7, 24, tzinfo=UTC)
    old = manager.acquire(task_id, owner_id="old", stage="fast_semantic", now=start)
    with pytest.raises(TaskLeaseBusy):
        manager.recover_stale(task_id, now=start + timedelta(seconds=4))
    recovered = manager.recover_stale(task_id, now=start + timedelta(seconds=6))
    assert recovered["recovered"]
    new = manager.acquire(
        task_id, owner_id="new", stage="fast_semantic", now=start + timedelta(seconds=6)
    )
    assert new.fencing_token == old.fencing_token + 1
    with pytest.raises(TaskLeaseLost):
        manager.assert_current(old, now=start + timedelta(seconds=6))
    with pytest.raises(TaskLeaseLost):
        _save_state(
            database,
            task_id,
            {"current_step": "export"},
            lease_manager=manager,
            lease=old,
        )
    manager.release(new, now=start + timedelta(seconds=7))


def test_two_processes_cannot_hold_the_same_task_lease(tmp_path):
    database, _ = _database(tmp_path)
    create_workflow_batch(database, name="process race", source_type="sync")
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    output = context.Queue()
    process = context.Process(
        target=_lease_worker,
        args=(str(database.path), ready, release, output),
    )
    process.start()
    # Windows spawn imports the full application in the child process; allow slower CI hosts.
    assert ready.wait(30)
    assert output.get(timeout=5)[0] == "acquired"
    with pytest.raises(TaskLeaseBusy):
        TaskLeaseManager(Database(database.path)).acquire(
            1, owner_id="parent-two", stage="fast_semantic"
        )
    release.set()
    process.join(10)
    assert process.exitcode == 0


def test_second_executor_never_enters_paid_or_sync_stub(tmp_path):
    database, journal_id = _database(tmp_path)
    paper_id = _paper(database, journal_id)
    task_id = _task(database, paper_id, "Locked task")
    manager = TaskLeaseManager(database)
    held = manager.acquire(
        task_id, owner_id="first-ui-session", stage="automatic_processing"
    )
    calls: list[str] = []

    class ForbiddenSync:
        def sync(self, *args, **kwargs):
            calls.append("sync")
            raise AssertionError("locked executor entered workflow")

    orchestrator = V05LiteratureOrchestrator(
        database=database,
        journal_file=SimpleNamespace(journals=()),
        sync_service=ForbiddenSync(),
        translation_service=SimpleNamespace(),
        evidence_service=SimpleNamespace(),
        semantic_service=SimpleNamespace(),
        lease_manager=TaskLeaseManager(Database(database.path)),
    )
    with pytest.raises(TaskLeaseBusy):
        orchestrator.run(task_id)
    assert calls == []
    manager.release(held)


def test_sqlite_busy_is_clear_and_does_not_acquire(tmp_path, monkeypatch):
    database, _ = _database(tmp_path)
    create_workflow_batch(database, name="busy", source_type="sync")

    @contextmanager
    def locked():
        raise sqlite3.OperationalError("database is locked")
        yield  # pragma: no cover

    monkeypatch.setattr(database, "connect", locked)
    with pytest.raises(TaskLeaseError, match="no workflow work was started"):
        TaskLeaseManager(database).acquire(1, owner_id="owner", stage="sync")


def test_task_reviews_translations_notes_counts_and_exports_are_isolated(tmp_path):
    database, journal_id = _database(tmp_path)
    paper_id = _paper(database, journal_id)
    task_a = _task(database, paper_id, "Task A")
    task_b = _task(database, paper_id, "Task B")
    save_task_review(
        database,
        task_a,
        paper_id,
        TaskReviewInput(
            review_status="approved",
            final_decision="include",
            reviewer_notes="A note",
            title_override="A title",
            abstract_override="A abstract",
        ),
    )
    save_task_review(
        database,
        task_b,
        paper_id,
        TaskReviewInput(
            review_status="rejected",
            final_decision="exclude",
            reviewer_notes="B note",
            title_override="B title",
            abstract_override="B abstract",
        ),
    )
    assert get_task_review(database, task_a, paper_id)["reviewer_notes"] == "A note"
    assert get_task_review(database, task_b, paper_id)["reviewer_notes"] == "B note"
    assert task_review_counts(database, task_a) == {"approved": 1}
    assert task_review_counts(database, task_b) == {"rejected": 1}
    assert list_task_papers(database, task_a)[0]["chinese_title"] == "A title"
    assert list_task_papers(database, task_b)[0]["chinese_title"] == "B title"
    assert mentor_records(database, task_id=task_a)[0].chinese_abstract == "A abstract"
    assert mentor_records(database, task_id=task_b) == []
    assert personal_research_records(database, task_id=task_a)[0].human_personal_note == "A note"


def test_missed_recovery_and_wrong_task_are_isolated(tmp_path):
    database, journal_id = _database(tmp_path)
    paper_id = _paper(database, journal_id)
    task_a = _task(database, paper_id, "Task A")
    task_b = create_workflow_batch(database, name="Task B", source_type="sync")
    mark_task_recovery(database, task_a, paper_id, recovered=True)
    assert get_task_review(database, task_a, paper_id)["reincluded"] == 1
    assert get_task_review(database, task_b, paper_id) is None
    with pytest.raises(ValueError, match="does not belong"):
        save_task_review(
            database,
            task_b,
            paper_id,
            TaskReviewInput(review_status="approved", final_decision="include"),
        )


def test_v05_never_falls_back_to_legacy_global_review(tmp_path):
    database, journal_id = _database(tmp_path)
    paper_id = _paper(database, journal_id)
    task_id = _task(database, paper_id, "Task")
    with database.connect() as connection:
        connection.execute(
            "UPDATE reviews SET status='approved',reviewer_note='legacy' WHERE paper_id=?",
            (paper_id,),
        )
    assert list_task_papers(database, task_id)[0]["review_status"] == "pending"
    assert mentor_records(database, task_id=task_id) == []
    assert mentor_records(database)[0].human_note == "legacy"


def test_legacy_backfill_only_for_single_task_and_ambiguous_is_reported(tmp_path):
    database, journal_id = _database(tmp_path, hardened=False)
    single = _paper(database, journal_id, "single")
    ambiguous = _paper(database, journal_id, "ambiguous")
    task_one = create_workflow_batch(
        database, name="one", source_type="selected_papers", paper_ids=[single, ambiguous]
    )
    create_workflow_batch(
        database, name="two", source_type="selected_papers", paper_ids=[ambiguous]
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE reviews SET status='approved',reviewer_note='legacy single' WHERE paper_id=?",
            (single,),
        )
        connection.execute(
            "UPDATE reviews SET status='rejected',reviewer_note='legacy ambiguous' "
            "WHERE paper_id=?",
            (ambiguous,),
        )
    apply_p0_hardening_migration(database, MIGRATION)
    apply_p0_hardening_migration(database, MIGRATION)
    assert get_task_review(database, task_one, single)["source"] == "legacy_backfill"
    assert get_task_review(database, task_one, ambiguous) is None
    with database.connect() as connection:
        conflict = connection.execute(
            "SELECT paper_id,task_ids_json FROM workflow_task_review_migration_conflicts"
        ).fetchone()
        assert int(conflict["paper_id"]) == ambiguous
        assert set(json.loads(conflict["task_ids_json"])) == {task_one, task_one + 1}
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_review_upsert_respects_unique_task_paper_constraint(tmp_path):
    database, journal_id = _database(tmp_path)
    paper_id = _paper(database, journal_id)
    task_id = _task(database, paper_id, "Task")
    for note in ("first", "second"):
        save_task_review(
            database,
            task_id,
            paper_id,
            TaskReviewInput(
                review_status="approved", final_decision="include", reviewer_notes=note
            ),
        )
    with database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM workflow_task_reviews WHERE task_id=? AND paper_id=?",
            (task_id, paper_id),
        ).fetchone()[0] == 1
    assert get_task_review(database, task_id, paper_id)["reviewer_notes"] == "second"


def test_migration_failure_rolls_back_all_phase66b_tables(tmp_path):
    database, _ = _database(tmp_path, hardened=False)
    broken = tmp_path / "broken.sql"
    sql = MIGRATION.read_text(encoding="utf-8").replace(
        "\nCOMMIT;", "\nTHIS IS INVALID SQL;\nCOMMIT;"
    )
    broken.write_text(sql, encoding="utf-8")
    with pytest.raises(sqlite3.OperationalError):
        apply_p0_hardening_migration(database, broken)
    with database.connect() as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "workflow_task_leases" not in tables
    assert "workflow_task_reviews" not in tables


def test_production_builder_fails_before_constructing_ai_without_release(
    tmp_path, monkeypatch
):
    from eduecon_radar.services import prototype_workflow

    database, _ = _database(tmp_path)
    settings = SimpleNamespace(
        project_root=ROOT,
        deepseek_api_key="not-a-real-key",
        deepseek_base_url="https://example.invalid",
        deepseek_timeout_seconds=1,
        deepseek_max_retries=1,
        deepseek_model_fast="fast",
        deepseek_model_strong="strong",
        openalex_api_key=None,
    )
    called = False

    class ForbiddenClient:
        def __init__(self, **kwargs):
            nonlocal called
            called = True

    monkeypatch.setattr(prototype_workflow, "DeepSeekClient", ForbiddenClient)
    with pytest.raises(RuntimeBundleTrustError, match="released_bundle_missing"):
        prototype_workflow.build_live_v05_orchestrator(
            database,
            settings,
            SimpleNamespace(journals=()),
        )
    assert called is False
