from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest

from eduecon_radar.db import Database
from eduecon_radar.services.migrations import apply_workflow_migration
from eduecon_radar.services.prototype_workflow import (
    LiteratureSearchScope,
    create_prototype_literature_task,
    mark_export_completed,
    prototype_task_state,
    set_prototype_task_step,
)

ROOT = Path(__file__).resolve().parents[1]


def _database(tmp_path):
    database = Database(tmp_path / "prototype.db")
    database.init_schema()
    apply_workflow_migration(
        database, ROOT / "migrations" / "0002_add_workflow_batches.sql"
    )
    return database


def test_search_scope_validation_rejects_missing_journals_and_reverse_dates():
    with pytest.raises(ValueError, match="至少选择"):
        LiteratureSearchScope((), date(2026, 1, 1), date(2026, 1, 2), 10)
    with pytest.raises(ValueError, match="起始日期"):
        LiteratureSearchScope(("AER",), date(2026, 1, 2), date(2026, 1, 1), 10)


def test_prototype_task_scope_and_step_survive_new_database_instance(tmp_path):
    database = _database(tmp_path)
    scope = LiteratureSearchScope(
        ("AER", "QJE"), date(2026, 1, 1), date(2026, 2, 1), 25
    )
    task_id = create_prototype_literature_task(database, name="本次检索", scope=scope)

    restored = prototype_task_state(Database(database.path), task_id)
    assert restored["current_step"] == "automatic_processing"
    assert restored["scope"]["journals"] == ["AER", "QJE"]
    assert restored["scope"]["date_from"] == "2026-01-01"
    assert restored["scope"]["date_to"] == "2026-02-01"
    assert restored["scope"]["limit"] == 25
    assert restored["result"]["total"] == 0


def test_prototype_final_review_only_moves_forward_to_export(tmp_path):
    database = _database(tmp_path)
    task_id = create_prototype_literature_task(
        database,
        name="one way",
        scope=LiteratureSearchScope(
            ("AER",), date(2026, 1, 1), date(2026, 1, 2), 10
        ),
    )

    set_prototype_task_step(database, task_id, "missed_recovery")
    set_prototype_task_step(database, task_id, "final_review")
    with pytest.raises(ValueError, match="cannot move backwards"):
        set_prototype_task_step(database, task_id, "missed_recovery")
    set_prototype_task_step(database, task_id, "export")

    assert prototype_task_state(database, task_id)["current_step"] == "export"
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM screenings").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 0


def test_repeated_mark_export_completed_is_strict_file_level_noop(tmp_path):
    database = _database(tmp_path)
    task_id = create_prototype_literature_task(
        database,
        name="idempotent export",
        scope=LiteratureSearchScope(
            ("AER",), date(2026, 1, 1), date(2026, 1, 2), 10
        ),
    )
    set_prototype_task_step(database, task_id, "missed_recovery")
    set_prototype_task_step(database, task_id, "final_review")
    set_prototype_task_step(database, task_id, "export")
    mark_export_completed(database, task_id)
    with database.connect() as connection:
        before_row = tuple(
            connection.execute(
                "SELECT * FROM workflow_batches WHERE id=?", (task_id,)
            ).fetchone()
        )
    before_sha256 = hashlib.sha256(database.path.read_bytes()).hexdigest()

    mark_export_completed(database, task_id)

    with database.connect() as connection:
        after_row = tuple(
            connection.execute(
                "SELECT * FROM workflow_batches WHERE id=?", (task_id,)
            ).fetchone()
        )
    after_sha256 = hashlib.sha256(database.path.read_bytes()).hexdigest()
    assert after_row == before_row
    assert after_sha256 == before_sha256
