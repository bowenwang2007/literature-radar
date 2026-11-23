from __future__ import annotations

import json
from pathlib import Path

import pytest

from eduecon_radar.db import Database
from eduecon_radar.services.migrations import (
    apply_translation_migration,
    apply_workflow_migration,
)
from eduecon_radar.services.prototype_workflow import mark_export_completed
from eduecon_radar.services.workflow import create_workflow_batch
from eduecon_radar.services.workflow_resume import (
    STAGES,
    StageCheckpoint,
    WorkflowCheckpointStore,
    WorkflowResumeService,
    fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def resumable(tmp_path):
    database = Database(tmp_path / "resume.db")
    database.init_schema()
    apply_translation_migration(database, ROOT / "migrations" / "0001_add_paper_translations.sql")
    apply_workflow_migration(database, ROOT / "migrations" / "0002_add_workflow_batches.sql")
    batch_id = create_workflow_batch(
        database,
        name="resume",
        source_type="sync",
        source_config={"journals": ["JEE"], "date_from": "2026-01-01", "date_to": "2026-01-02"},
    )
    store = WorkflowCheckpointStore(database)
    store.initialize_all(batch_id)
    return database, batch_id, store


def _complete(store, batch_id, stage, ids=(1, 2)):
    item = store.start(batch_id, stage, total=len(ids), config_fp="c", input_fp="i", item_ids=ids)
    return store.finish(batch_id, stage, item, succeeded=len(ids))


def test_fingerprint_is_stable_and_checkpoint_shape_is_complete():
    assert fingerprint({"b": 2, "a": 1}) == fingerprint({"a": 1, "b": 2})
    item = StageCheckpoint(config_fingerprint="c", input_fingerprint="i")
    assert item.status == "pending" and item.error_summary is None
    assert item.pending_item_ids == [] and item.failed_item_ids == [] and item.retryable


def test_new_task_starts_from_sync_and_all_eleven_stages_exist(resumable):
    database, batch_id, store = resumable
    state = WorkflowResumeService(database).get_resume_state(batch_id)
    assert state["current_stage"] == "sync"
    assert tuple(state["checkpoints"]) == STAGES


@pytest.mark.parametrize(
    ("completed", "expected"),
    [
        ("abstract_enrichment", "title_translation"),
        ("title_translation", "evidence"),
        ("evidence", "fast_semantic"),
        ("composite", "missed_recovery"),
        ("final_review", "export"),
    ],
)
def test_resume_after_stage_interruption(resumable, completed, expected):
    database, batch_id, store = resumable
    for stage in STAGES[: STAGES.index(completed) + 1]:
        _complete(store, batch_id, stage)
    assert WorkflowResumeService(database).get_resume_state(batch_id)["current_stage"] == expected


@pytest.mark.parametrize("stage", ["fast_semantic", "strong_semantic"])
def test_partial_failed_retries_only_failed_items(resumable, stage):
    database, batch_id, store = resumable
    item = store.start(batch_id, stage, total=3, config_fp="c", input_fp="i", item_ids=(1, 2, 3))
    store.finish(batch_id, stage, item, succeeded=2, failed=1, failed_item_ids=(3,))
    assert WorkflowResumeService(database).retry_stage_failures(batch_id, stage) == (3,)


def test_running_is_reported_as_interrupted_and_resumable(resumable):
    database, batch_id, store = resumable
    store.start(batch_id, "composite", total=2, config_fp="c", input_fp="i", item_ids=(1, 2))
    state = WorkflowResumeService(database).get_resume_state(batch_id)
    assert state["interrupted_stages"] == ["composite"]
    assert state["current_stage"] == "sync"


def test_completed_and_matching_fingerprint_skip(resumable):
    _, batch_id, store = resumable
    _complete(store, batch_id, "sync")
    assert not store.should_run(batch_id, "sync", "c", "i")
    assert store.should_run(batch_id, "sync", "changed", "i")


def test_fingerprint_change_invalidates_only_downstream(resumable):
    _, batch_id, store = resumable
    for stage in STAGES[:5]:
        _complete(store, batch_id, stage)
    store.invalidate_downstream(batch_id, "title_translation")
    assert store.get(batch_id, "title_translation").status == "completed"
    assert store.get(batch_id, "evidence").status == "pending"
    assert store.get(batch_id, "fast_semantic").status == "pending"


def test_result_summary_fields_are_preserved_by_atomic_merge(resumable):
    database, batch_id, store = resumable
    with database.connect() as connection:
        connection.execute(
            "UPDATE workflow_batches SET result_summary_json=? WHERE id=?",
            (json.dumps({"unrelated": {"keep": True}}), batch_id),
        )
    _complete(store, batch_id, "sync")
    with database.connect() as connection:
        value = json.loads(
            connection.execute(
                "SELECT result_summary_json FROM workflow_batches WHERE id=?", (batch_id,)
            ).fetchone()[0]
        )
    assert value["unrelated"] == {"keep": True}


def test_invalid_checkpoint_json_safely_degrades(resumable):
    database, batch_id, store = resumable
    with database.connect() as connection:
        connection.execute(
            "UPDATE workflow_batches SET result_summary_json='not json' WHERE id=?", (batch_id,)
        )
    assert store.load(batch_id) == {}
    _complete(store, batch_id, "sync")
    assert store.get(batch_id, "sync").status == "completed"


def test_abandon_marks_batch_without_deleting_items(resumable):
    database, batch_id, _ = resumable
    WorkflowResumeService(database).abandon_batch(batch_id)
    with database.connect() as connection:
        row = connection.execute(
            "SELECT status,last_error FROM workflow_batches WHERE id=?", (batch_id,)
        ).fetchone()
    assert tuple(row) == ("paused", "abandoned_by_user")


def test_resume_does_not_create_another_batch(resumable):
    database, batch_id, _ = resumable
    WorkflowResumeService(database).resume_batch(batch_id)
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM workflow_batches").fetchone()[0] == 1


def test_multiple_continue_clicks_are_claimed_once(resumable):
    database, batch_id, _ = resumable

    class RecursiveRunner:
        def run_stage(self, current_id, stage, item_ids=None):
            with pytest.raises(RuntimeError, match="already running"):
                WorkflowResumeService(database, self).resume_batch(current_id)
            return stage

    assert WorkflowResumeService(database, RecursiveRunner()).resume_batch(batch_id) == "sync"


def test_resume_retry_composite_and_export_preserve_all_human_data(resumable):
    database, batch_id, store = resumable
    now = "2026-07-22T00:00:00+00:00"
    metadata = {
        "abstract_source": "manual",
        "abstract_manual_url": "https://publisher.example/abstract",
    }
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO journals(id,name,short_name,issn,priority,created_at,updated_at) "
            "VALUES(1,'Journal','J','1234','high',?,?)",
            (now, now),
        )
        connection.execute(
            "INSERT INTO papers(id,fingerprint,title,abstract,authors_json,journal_id,"
            "source_metadata_json,first_seen_at,last_seen_at,created_at,updated_at) "
            "VALUES(1,'human-fp','Human paper','Human English abstract','[]',1,?,?,?,?,?)",
            (json.dumps(metadata), now, now, now, now),
        )
        connection.execute(
            "INSERT INTO workflow_batch_items(batch_id,paper_id,position,added_at) VALUES(?,1,1,?)",
            (batch_id, now),
        )
        connection.execute(
            "INSERT INTO paper_translations(paper_id,chinese_title,chinese_abstract,"
            "title_translation_status,abstract_translation_status,title_translation_source,"
            "abstract_translation_source,model_name,prompt_version,translated_at,updated_at) "
            "VALUES(1,'人工中文标题','人工中文摘要','human_edited','human_edited',"
            "'human_edited','human_edited','human','human',?,?)",
            (now, now),
        )
        connection.execute(
            "INSERT INTO reviews(paper_id,status,reviewer_note,edited_chinese_title,"
            "edited_chinese_summary,reviewed_at) VALUES(1,'approved','exported review',"
            "'人工最终标题','人工最终摘要',?)",
            (now,),
        )
        row = connection.execute(
            "SELECT result_summary_json FROM workflow_batches WHERE id=?", (batch_id,)
        ).fetchone()
        summary = json.loads(row[0])
        summary["v05_literature"] = {"recovered_ids": [1], "final_reviewed_ids": [1]}
        connection.execute(
            "UPDATE workflow_batches SET result_summary_json=? WHERE id=?",
            (json.dumps(summary, ensure_ascii=False), batch_id),
        )
    with database.connect() as connection:
        before = {
            "paper": tuple(
                connection.execute(
                    "SELECT abstract,source_metadata_json FROM papers WHERE id=1"
                ).fetchone()
            ),
            "translation": tuple(
                connection.execute(
                    "SELECT chinese_title,chinese_abstract,title_translation_source,"
                    "abstract_translation_source FROM paper_translations WHERE paper_id=1"
                ).fetchone()
            ),
            "review": tuple(
                connection.execute(
                    "SELECT status,reviewer_note,edited_chinese_title,edited_chinese_summary "
                    "FROM reviews WHERE paper_id=1"
                ).fetchone()
            ),
        }
    for stage in ("abstract_enrichment", "title_translation", "fast_semantic", "composite"):
        item = store.start(
            batch_id, stage, total=1, config_fp="retry", input_fp="same", item_ids=(1,)
        )
        store.finish(batch_id, stage, item, succeeded=1)
    mark_export_completed(database, batch_id)
    WorkflowResumeService(database).abandon_batch(batch_id)
    with database.connect() as connection:
        after = {
            "paper": tuple(
                connection.execute(
                    "SELECT abstract,source_metadata_json FROM papers WHERE id=1"
                ).fetchone()
            ),
            "translation": tuple(
                connection.execute(
                    "SELECT chinese_title,chinese_abstract,title_translation_source,"
                    "abstract_translation_source FROM paper_translations WHERE paper_id=1"
                ).fetchone()
            ),
            "review": tuple(
                connection.execute(
                    "SELECT status,reviewer_note,edited_chinese_title,edited_chinese_summary "
                    "FROM reviews WHERE paper_id=1"
                ).fetchone()
            ),
            "summary": json.loads(
                connection.execute(
                    "SELECT result_summary_json FROM workflow_batches WHERE id=?", (batch_id,)
                ).fetchone()[0]
            ),
        }
    assert after["paper"] == before["paper"]
    assert after["translation"] == before["translation"]
    assert after["review"] == before["review"]
    assert after["summary"]["v05_literature"]["recovered_ids"] == [1]
    assert after["summary"]["v05_literature"]["final_reviewed_ids"] == [1]
