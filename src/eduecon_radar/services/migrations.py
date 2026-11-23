from __future__ import annotations

import sqlite3
from pathlib import Path

from eduecon_radar.db import Database

#: The research profile that existing single-domain data migrates into.
DEFAULT_PROFILE_ID = "education_economics"


class TranslationMigrationRequired(RuntimeError):
    """Raised when the optional translation layer has not been installed."""


class WorkflowMigrationRequired(RuntimeError):
    """Raised when the optional workflow layer has not been installed."""


class P0HardeningMigrationRequired(RuntimeError):
    """Raised when task leases and task-scoped reviews are not installed."""


class DataIntegrityMigrationRequired(RuntimeError):
    """Raised when Phase 6.6C provenance and budget tables are unavailable."""


class OvernightPipelineMigrationRequired(RuntimeError):
    """Raised when the overnight output-first pipeline schema is unavailable."""


class ProductSafetyMigrationRequired(RuntimeError):
    """Raised when removal tombstones and retrieval verification are unavailable."""


class VerificationIntakeMigrationRequired(RuntimeError):
    """Raised when tombstone restore provenance (restored_at) is unavailable."""


_OVERNIGHT_SYNC_RUN_COLUMNS = (
    ("selected_journal_keys", "TEXT"),
    ("query_start_date", "TEXT"),
    ("query_end_date", "TEXT"),
    ("requested_limit", "INTEGER"),
    ("matched_paper_ids", "TEXT"),
    ("inserted_paper_ids", "TEXT"),
    ("updated_paper_ids", "TEXT"),
    ("unchanged_paper_ids", "TEXT"),
    ("duplicate_paper_ids", "TEXT"),
    ("rejected_candidates", "TEXT"),
    ("query_errors", "TEXT"),
    ("overall_status", "TEXT"),
    ("provider_returned_count", "INTEGER"),
    ("pages_requested", "INTEGER"),
    ("raw_candidate_count", "INTEGER"),
    ("normalized_count", "INTEGER"),
    ("stopped_reason", "TEXT"),
)

_OVERNIGHT_TRANSLATION_COLUMNS = (
    ("source_title_hash", "TEXT"),
    ("validation_status", "TEXT"),
    ("error_category", "TEXT"),
    ("error_message", "TEXT"),
    ("source_abstract_hash", "TEXT"),
)


def _existing_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }


def apply_overnight_columns(connection: sqlite3.Connection) -> None:
    """Additive columns that ``migrations/0010`` cannot express in SQL.

    SQLite has no conditional ``ADD COLUMN``, so the migration file documents
    the change and this guarded helper performs it.  Keeping it connection
    scoped lets the migration runner apply the *complete* effect of 0010 in one
    transaction instead of only the part written in SQL.
    """
    sync_columns = _existing_columns(connection, "sync_runs")
    for name, definition in _OVERNIGHT_SYNC_RUN_COLUMNS:
        if name not in sync_columns:
            connection.execute(f"ALTER TABLE sync_runs ADD COLUMN {name} {definition}")
    translation_columns = _existing_columns(connection, "paper_translations")
    for name, definition in _OVERNIGHT_TRANSLATION_COLUMNS:
        if name not in translation_columns:
            connection.execute(
                f"ALTER TABLE paper_translations ADD COLUMN {name} {definition}"
            )


def apply_verification_intake_column(connection: sqlite3.Connection) -> None:
    """Additive ``paper_tombstones.restored_at`` from ``migrations/0012``."""
    if "paper_tombstones" not in {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }:
        return
    if "restored_at" not in _existing_columns(connection, "paper_tombstones"):
        connection.execute("ALTER TABLE paper_tombstones ADD COLUMN restored_at TEXT")


def apply_profile_scoped_columns(connection: sqlite3.Connection) -> None:
    """Add ``sync_runs.profile_id`` (migration 0013) and backfill the default.

    Existing runs were made before profiles existed; they belong to the default
    profile, and saying so explicitly is better than leaving the column NULL and
    making every reader guess.
    """
    if "sync_runs" not in {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }:
        return
    if "profile_id" not in _existing_columns(connection, "sync_runs"):
        connection.execute("ALTER TABLE sync_runs ADD COLUMN profile_id TEXT")
    connection.execute(
        "UPDATE sync_runs SET profile_id=? WHERE profile_id IS NULL OR profile_id=''",
        (DEFAULT_PROFILE_ID,),
    )


#: Migration file -> guarded DDL applied by Python after the SQL body.
GUARDED_DDL = {
    "0010_add_overnight_pipeline.sql": apply_overnight_columns,
    "0012_add_verification_intake.sql": apply_verification_intake_column,
    "0013_add_profile_scoped_assessment.sql": apply_profile_scoped_columns,
}


def guarded_columns(filename: str) -> tuple[tuple[str, str, str], ...]:
    """Return ``(table, column, definition)`` rows added by the guarded helper.

    Used by the migration ledger to decide, from the real schema alone, whether
    a migration whose effect is partly expressed in Python has been applied.
    """
    if filename == "0010_add_overnight_pipeline.sql":
        return tuple(
            ("sync_runs", name, definition)
            for name, definition in _OVERNIGHT_SYNC_RUN_COLUMNS
        ) + tuple(
            ("paper_translations", name, definition)
            for name, definition in _OVERNIGHT_TRANSLATION_COLUMNS
        )
    if filename == "0012_add_verification_intake.sql":
        return (("paper_tombstones", "restored_at", "TEXT"),)
    if filename == "0013_add_profile_scoped_assessment.sql":
        return (("sync_runs", "profile_id", "TEXT"),)
    return ()


def overnight_schema_ready(database: Database) -> bool:
    if not database.path.is_file():
        return False
    uri = database.path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        names = {
            row[0]
            for row in connection.execute(
                """SELECT name FROM sqlite_master WHERE type='table' AND name IN
                ('paper_screenings','provider_call_log','paper_processing_status')"""
            ).fetchall()
        }
        if names != {
            "paper_screenings",
            "provider_call_log",
            "paper_processing_status",
        }:
            return False
        sync_columns = _existing_columns(connection, "sync_runs")
        translation_columns = _existing_columns(connection, "paper_translations")
    required_sync = {"selected_journal_keys", "matched_paper_ids", "overall_status"}
    required_translation = {
        "source_title_hash",
        "validation_status",
        "source_abstract_hash",
    }
    return required_sync <= sync_columns and required_translation <= translation_columns


def require_overnight_schema(database: Database) -> None:
    if not overnight_schema_ready(database):
        raise OvernightPipelineMigrationRequired(
            "Overnight output-first pipeline schema is not installed. Apply "
            "migrations/0010_add_overnight_pipeline.sql to an explicitly selected, "
            "archived database before starting the personal overnight workflow."
        )


def apply_overnight_migration(database: Database, migration_path: Path) -> None:
    """Apply the additive overnight pipeline migration (tables + guarded columns)."""
    if not database.path.is_file():
        raise FileNotFoundError(f"Database does not exist: {database.path}")
    sql = migration_path.read_text(encoding="utf-8")
    with database.connect() as connection:
        connection.executescript(sql)
        apply_overnight_columns(connection)


def apply_profile_migration(database: Database, migration_path: Path) -> None:
    """Install the profile-aware tables (0013) on an existing database.

    Same shape as the other additive appliers: explicit, idempotent, and safe to
    call on a database that already has the profile schema.
    """
    if not database.path.is_file():
        raise FileNotFoundError(f"Database does not exist: {database.path}")
    sql = migration_path.read_text(encoding="utf-8")
    with database.connect() as connection:
        connection.executescript(sql)
        apply_profile_scoped_columns(connection)


def product_safety_schema_ready(database: Database) -> bool:
    if not database.path.is_file():
        return False
    uri = database.path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            """SELECT name FROM sqlite_master WHERE type='table' AND name IN
            ('paper_tombstones','retrieval_verifications')"""
        ).fetchall()
    return {row[0] for row in rows} == {
        "paper_tombstones",
        "retrieval_verifications",
    }


def require_product_safety_schema(database: Database) -> None:
    if not product_safety_schema_ready(database):
        raise ProductSafetyMigrationRequired(
            "Phase 7C2 数据迁移尚未安装。请在备份后应用 "
            "migrations/0011_add_product_safety_and_verification.sql。"
        )


def apply_product_safety_migration(database: Database, migration_path: Path) -> None:
    """Install the additive Phase 7C2 safety and verification tables."""
    if not database.path.is_file():
        raise FileNotFoundError(f"Database does not exist: {database.path}")
    sql = migration_path.read_text(encoding="utf-8")
    with database.connect() as connection:
        connection.executescript(sql)


def verification_intake_schema_ready(database: Database) -> bool:
    """Return whether tombstones can record an explicit restore."""
    if not product_safety_schema_ready(database):
        return False
    uri = database.path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(paper_tombstones)").fetchall()
        }
    return "restored_at" in columns


def require_verification_intake_schema(database: Database) -> None:
    if not verification_intake_schema_ready(database):
        raise VerificationIntakeMigrationRequired(
            "Phase 7C4 数据迁移尚未安装。请在备份后应用 "
            "migrations/0012_add_verification_intake.sql。"
        )


def apply_verification_intake_migration(database: Database, migration_path: Path) -> None:
    """Additive restore-provenance column; safe to run once per database."""
    if not database.path.is_file():
        raise FileNotFoundError(f"数据库不存在：{database.path}")
    if verification_intake_schema_ready(database):
        return
    with database.connect() as connection:
        apply_verification_intake_column(connection)


def translation_schema_ready(database: Database) -> bool:
    if not database.path.is_file():
        return False
    uri = database.path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_translations'"
        ).fetchone()
    return row is not None


def require_translation_schema(database: Database) -> None:
    if not translation_schema_ready(database):
        raise TranslationMigrationRequired(
            "Translation Layer 尚未安装。请先备份数据库，再显式执行翻译层迁移。"
        )


def apply_translation_migration(database: Database, migration_path: Path) -> None:
    """Apply the additive translation migration to an explicitly selected database."""
    if not database.path.is_file():
        raise FileNotFoundError(f"数据库不存在：{database.path}")
    sql = migration_path.read_text(encoding="utf-8")
    with database.connect() as connection:
        connection.executescript(sql)


def workflow_schema_ready(database: Database) -> bool:
    if not database.path.is_file():
        return False
    uri = database.path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            """SELECT name FROM sqlite_master WHERE type='table'
            AND name IN ('workflow_batches','workflow_batch_items')"""
        ).fetchall()
    return {row[0] for row in rows} == {"workflow_batches", "workflow_batch_items"}


def require_workflow_schema(database: Database) -> None:
    if not workflow_schema_ready(database):
        raise WorkflowMigrationRequired(
            "Workflow Layer 尚未安装。请先备份数据库，再显式执行工作流迁移。"
        )


def apply_workflow_migration(database: Database, migration_path: Path) -> None:
    """Apply the additive workflow migration to an explicitly selected database."""
    if not database.path.is_file():
        raise FileNotFoundError(f"数据库不存在：{database.path}")
    sql = migration_path.read_text(encoding="utf-8")
    with database.connect() as connection:
        connection.executescript(sql)


def p0_hardening_schema_ready(database: Database) -> bool:
    if not database.path.is_file():
        return False
    uri = database.path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            """SELECT name FROM sqlite_master WHERE type='table' AND name IN
            ('workflow_task_leases','workflow_task_reviews',
             'workflow_task_review_migration_conflicts')"""
        ).fetchall()
    return {row[0] for row in rows} == {
        "workflow_task_leases",
        "workflow_task_reviews",
        "workflow_task_review_migration_conflicts",
    }


def require_p0_hardening_schema(database: Database) -> None:
    if not p0_hardening_schema_ready(database):
        raise P0HardeningMigrationRequired(
            "Phase 6.6B schema is not installed. Apply migration "
            "migrations/0007_add_runtime_leases_and_task_reviews.sql to an explicitly "
            "selected, archived database before starting the v0.5 workflow."
        )


def apply_p0_hardening_migration(database: Database, migration_path: Path) -> None:
    """Apply the additive Phase 6.6B migration to an explicitly selected database."""
    if not database.path.is_file():
        raise FileNotFoundError(f"Database does not exist: {database.path}")
    sql = migration_path.read_text(encoding="utf-8")
    with database.connect() as connection:
        connection.executescript(sql)


def data_integrity_schema_ready(database: Database) -> bool:
    if not database.path.is_file():
        return False
    uri = database.path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        names = {
            row[0]
            for row in connection.execute(
                """SELECT name FROM sqlite_master WHERE type='table' AND name IN
                ('paper_metadata_provenance','ai_task_budgets','ai_stage_budgets',
                 'ai_usage_reservations','ai_budget_changes')"""
            ).fetchall()
        }
    required = {
        "paper_metadata_provenance",
        "ai_task_budgets",
        "ai_stage_budgets",
        "ai_usage_reservations",
        "ai_budget_changes",
    }
    return names == required


def require_data_integrity_schema(database: Database) -> None:
    if not data_integrity_schema_ready(database):
        raise DataIntegrityMigrationRequired(
            "Phase 6.6C schema is not installed. Dry-run and verify migrations first, "
            "then explicitly apply production migration 0008 after backup."
        )


def apply_data_integrity_migration(database: Database, migration_path: Path) -> None:
    """Apply additive Phase 6.6C tables to an explicitly selected database."""
    if not database.path.is_file():
        raise FileNotFoundError(f"Database does not exist: {database.path}")
    sql = migration_path.read_text(encoding="utf-8")
    with database.connect() as connection:
        connection.executescript(sql)
