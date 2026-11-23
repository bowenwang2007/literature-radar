"""Migration ledger must describe the schema that actually exists.

The production ledger stopped at 0008 while migrations 0010-0012 had in fact
been applied through guarded Python helpers.  These tests pin the reconciliation
rules: evidence from real schema objects, no guessing, idempotent recording, and
a fresh database that ends up complete.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from eduecon_radar.db import Database
from eduecon_radar.services import migrations as migration_helpers
from eduecon_radar.services.migration_integrity import (
    PROFILE_MIGRATIONS,
    MigrationIntegrityError,
    migrate_database,
    verify_migrations,
)
from eduecon_radar.services.migration_ledger import (
    MigrationLedgerError,
    audit_migrations,
    migration_objects,
    reconcile_migration_ledger,
)

PRODUCTION = PROFILE_MIGRATIONS["production"]
#: The product-pipeline migrations whose ledger entries were missing when the
#: reconciler was written.  Named explicitly rather than sliced off the end of the
#: profile: the slice moved every time a later migration was legitimately added,
#: which silently changed what these tests were exercising.
RECONCILED = (
    "0010_add_overnight_pipeline.sql",
    "0011_add_product_safety_and_verification.sql",
    "0012_add_verification_intake.sql",
    "0013_add_profile_scoped_assessment.sql",
)
RECONCILED_IDS = tuple(int(name.split("_", 1)[0]) for name in RECONCILED)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fresh_migrated(tmp_path: Path) -> Path:
    database = tmp_path / "fresh.db"
    migrate_database(database, "production", dry_run=False)
    return database


def test_production_profile_covers_every_product_pipeline_migration() -> None:
    for name in RECONCILED:
        assert name in PRODUCTION
    # Ordering matters: a migration may not appear before its dependencies.
    assert list(PRODUCTION) == sorted(PRODUCTION)
    # The reconciler's set is a contiguous run at the end of the pipeline group.
    assert PRODUCTION.index(RECONCILED[0]) + len(RECONCILED) == len(PRODUCTION) - 1
    assert PRODUCTION[-1] == "0014_add_jel_observations.sql"


def test_fresh_database_reaches_a_complete_product_schema(tmp_path: Path) -> None:
    database = _fresh_migrated(tmp_path)
    handle = Database(database)
    assert migration_helpers.overnight_schema_ready(handle)
    assert migration_helpers.product_safety_schema_ready(handle)
    assert migration_helpers.verification_intake_schema_ready(handle)
    assert migration_helpers.translation_schema_ready(handle)
    assert verify_migrations(database, "production").pending == ()


def test_guarded_columns_are_part_of_the_migration_evidence() -> None:
    objects = {obj.label() for obj in migration_objects("0010_add_overnight_pipeline.sql")}
    # Columns SQLite cannot add conditionally are applied by Python; they must
    # still count as required objects when deciding whether 0010 was applied.
    assert "column:sync_runs.matched_paper_ids" in objects
    assert "column:paper_translations.source_abstract_hash" in objects
    tombstone = {
        obj.label() for obj in migration_objects("0012_add_verification_intake.sql")
    }
    assert tombstone == {"column:paper_tombstones.restored_at"}


def test_audit_is_read_only(tmp_path: Path) -> None:
    ids = ",".join(str(item) for item in RECONCILED_IDS)
    database = _fresh_migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute(
            f"DELETE FROM schema_migrations WHERE migration_id IN ({ids})"
        )
        connection.commit()
    before = _sha256(database)
    report = reconcile_migration_ledger(database, "production", PRODUCTION, confirmed=False)
    assert report["status"] == "audit"
    assert report["to_record"] == list(RECONCILED)
    assert _sha256(database) == before


def test_reconciler_records_complete_matches_once(tmp_path: Path) -> None:
    ids = ",".join(str(item) for item in RECONCILED_IDS)
    database = _fresh_migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute(
            f"DELETE FROM schema_migrations WHERE migration_id IN ({ids})"
        )
        connection.commit()

    first = reconcile_migration_ledger(
        database, "production", PRODUCTION, confirmed=True
    )
    assert first["recorded"] == list(RECONCILED)
    assert verify_migrations(database, "production").pending == ()

    second = reconcile_migration_ledger(
        database, "production", PRODUCTION, confirmed=True
    )
    assert second["recorded"] == []
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT inferred,inference_method FROM schema_migrations WHERE migration_id=10"
        ).fetchone()
    assert rows == (1, "schema_object_match:108/108")
    # The reconciler touched only the migrations it was asked about; a migration
    # applied normally keeps inferred=0 and no inference method.
    with sqlite3.connect(database) as connection:
        applied_row = connection.execute(
            "SELECT inferred,inference_method FROM schema_migrations WHERE migration_id=14"
        ).fetchone()
    assert applied_row == (0, None)
    with sqlite3.connect(database) as connection:
        profile_row = connection.execute(
            "SELECT inferred,inference_method FROM schema_migrations WHERE migration_id=13"
        ).fetchone()
    assert profile_row is not None and profile_row[0] == 1
    assert str(profile_row[1]).startswith("schema_object_match:")


def test_partial_schema_is_reported_and_never_recorded(tmp_path: Path) -> None:
    ids = ",".join(str(item) for item in RECONCILED_IDS)
    database = _fresh_migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute(
            f"DELETE FROM schema_migrations WHERE migration_id IN ({ids})"
        )
        connection.execute("DROP INDEX idx_provider_call_log_run")
        connection.commit()

    audit = {item.filename: item for item in audit_migrations(database, PRODUCTION)}
    assert audit["0010_add_overnight_pipeline.sql"].state == "partially_applied"

    report = reconcile_migration_ledger(database, "production", PRODUCTION, confirmed=False)
    assert report["status"] == "blocked"
    assert report["blocking"] == ["0010_add_overnight_pipeline.sql"]

    with pytest.raises(MigrationLedgerError, match="Refusing to reconcile"):
        reconcile_migration_ledger(database, "production", PRODUCTION, confirmed=True)
    with sqlite3.connect(database) as connection:
        # A blocker anywhere in the run stops the whole reconciliation, so none of
        # the deleted entries may have been written back.  Migrations outside the
        # run (for example a later one applied normally) are untouched by design.
        ids = ",".join(str(item) for item in RECONCILED_IDS)
        count = connection.execute(
            f"SELECT COUNT(*) FROM schema_migrations WHERE migration_id IN ({ids})"
        ).fetchone()[0]
    assert count == 0


def test_reconciler_still_records_when_a_later_migration_is_already_ledgered(
    tmp_path: Path,
) -> None:
    """A later *recorded* migration must not block an earlier unrecorded run.

    The reconciler chose the row it was continuing from as the newest ledger
    entry overall.  Once a later migration (here 0014) had been applied normally,
    every migration in the earlier run failed the ordering test and was silently
    skipped: the command reported success while recording nothing.
    """
    ids = ",".join(str(item) for item in RECONCILED_IDS)
    database = _fresh_migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute(
            f"DELETE FROM schema_migrations WHERE migration_id IN ({ids})"
        )
        # 0014 stays, exactly as it would after a normal `migrate-database`.
        remaining = connection.execute(
            "SELECT filename FROM schema_migrations WHERE migration_id=14"
        ).fetchone()
        connection.commit()
    assert remaining == ("0014_add_jel_observations.sql",)

    report = reconcile_migration_ledger(
        database, "production", PRODUCTION, confirmed=True
    )
    assert report["recorded"] == list(RECONCILED)
    assert report["status"] == "reconciled"
    assert verify_migrations(database, "production").pending == ()
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT migration_id FROM schema_migrations ORDER BY migration_id"
        ).fetchall()
    assert [row[0] for row in rows] == sorted(row[0] for row in rows)
    assert {row[0] for row in rows} >= set(RECONCILED_IDS) | {14}


def test_ledger_claim_without_schema_is_detected(tmp_path: Path) -> None:
    database = _fresh_migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("DROP INDEX idx_paper_processing_status_run")
        connection.commit()
    with pytest.raises(MigrationIntegrityError, match="ledger_object_missing"):
        verify_migrations(database, "production")
