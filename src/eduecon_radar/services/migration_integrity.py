from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from eduecon_radar import __version__
from eduecon_radar.db import SCHEMA, utcnow
from eduecon_radar.resources import read_text, resource_sha256
from eduecon_radar.services.migration_ledger import audit_migrations
from eduecon_radar.services.migrations import GUARDED_DDL

PROFILE_MIGRATIONS: dict[str, tuple[str, ...]] = {
    "production": (
        "0001_add_paper_translations.sql",
        "0002_add_workflow_batches.sql",
        "0003_add_domain_evidence_runs.sql",
        "0004_add_domain_semantic_runs.sql",
        "0007_add_runtime_leases_and_task_reviews.sql",
        "0008_add_metadata_provenance_and_ai_budgets.sql",
        # The product pipeline migrations were applied through the guarded
        # Python helpers but were never listed in any profile, so the ledger
        # could not describe the schema that actually existed.  They belong to
        # the production profile and are recorded by the reconciler rather than
        # re-executed.
        "0010_add_overnight_pipeline.sql",
        "0011_add_product_safety_and_verification.sql",
        "0012_add_verification_intake.sql",
        "0013_add_profile_scoped_assessment.sql",
        # JEL codes are a global paper fact, so the observation ledger that
        # backs them belongs to the production profile.  No guarded Python
        # helper is needed: migration 0014 is expressed entirely in SQL.
        "0014_add_jel_observations.sql",
    ),
    "gold": (
        "0005_add_gold_set_and_calibration.sql",
        "0009_add_gold_governance.sql",
    ),
    "validation": ("0006_add_validation_batches.sql",),
}

LEDGER_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_id INTEGER NOT NULL,
    filename TEXT NOT NULL UNIQUE,
    migration_sha256 TEXT NOT NULL,
    applied_at TEXT NOT NULL,
    application_version TEXT NOT NULL,
    schema_version_before TEXT NOT NULL,
    schema_version_after TEXT NOT NULL,
    execution_ms INTEGER NOT NULL,
    success INTEGER NOT NULL CHECK (success IN (0,1)),
    error_type TEXT,
    error_message TEXT,
    inferred INTEGER NOT NULL DEFAULT 0 CHECK (inferred IN (0,1)),
    inference_method TEXT,
    schema_fingerprint TEXT,
    source_manifest TEXT,
    database_profile TEXT NOT NULL,
    PRIMARY KEY(database_profile,migration_id)
);
"""


class MigrationIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class MigrationStatus:
    database: str
    database_profile: str
    ledger_present: bool
    inferred_version: str | None
    applied: tuple[str, ...]
    pending: tuple[str, ...]
    hashes: dict[str, str]
    errors: tuple[str, ...]


def migration_sha256(filename: str) -> str:
    return resource_sha256(f"migrations/{filename}")


def migration_status(database: Path, profile: str) -> MigrationStatus:
    expected = _expected(profile)
    hashes = {name: migration_sha256(name) for name in expected}
    if not database.is_file():
        return MigrationStatus(
            str(database.resolve()), profile, False, "empty", (), expected, hashes, ()
        )
    with _read_only(database) as connection:
        ledger_present = _table_exists(connection, "schema_migrations")
        if not ledger_present:
            matches = infer_historical_versions(connection, profile)
            errors = () if len(matches) == 1 else (f"ambiguous_schema:{','.join(matches)}",)
            version = matches[0] if len(matches) == 1 else None
            count = _version_count(version, expected)
            return MigrationStatus(
                str(database.resolve()),
                profile,
                False,
                version,
                expected[:count],
                expected[count:],
                hashes,
                errors,
            )
        rows = connection.execute(
            """SELECT filename,migration_sha256,success,database_profile
            FROM schema_migrations ORDER BY migration_id"""
        ).fetchall()
    errors: list[str] = []
    if any(row["database_profile"] != profile for row in rows):
        errors.append("database_profile_mismatch")
    successful = tuple(row["filename"] for row in rows if row["success"])
    if successful != expected[: len(successful)]:
        errors.append("migration_order_mismatch")
    for row in rows:
        if row["filename"] not in hashes:
            errors.append(f"unexpected_migration:{row['filename']}")
        elif row["success"] and row["migration_sha256"] != hashes[row["filename"]]:
            errors.append(f"migration_hash_mismatch:{row['filename']}")
    return MigrationStatus(
        str(database.resolve()),
        profile,
        True,
        successful[-1] if successful else "empty",
        successful,
        expected[len(successful) :],
        hashes,
        tuple(dict.fromkeys(errors)),
    )


def verify_migrations(database: Path, profile: str) -> MigrationStatus:
    status = migration_status(database, profile)
    if status.errors:
        raise MigrationIntegrityError("; ".join(status.errors))
    if database.is_file() and not status.ledger_present:
        raise MigrationIntegrityError("migration_ledger_missing")
    if database.is_file() and status.applied:
        # A ledger row is a claim; the schema is the evidence.  Report a
        # recorded migration whose objects are absent instead of trusting it.
        drift = [
            f"ledger_object_missing:{audit.filename}:{','.join(audit.missing)}"
            for audit in audit_migrations(database, status.applied)
            if audit.missing
        ]
        if drift:
            raise MigrationIntegrityError("; ".join(drift))
    return status


def baseline_migration_ledger(database: Path, profile: str, *, confirmed: bool) -> MigrationStatus:
    if not confirmed:
        raise MigrationIntegrityError("baseline requires explicit confirmation")
    if not database.is_file():
        raise FileNotFoundError(f"Database does not exist: {database}")
    status = migration_status(database, profile)
    if status.ledger_present:
        return verify_migrations(database, profile)
    if status.errors or status.inferred_version is None:
        raise MigrationIntegrityError(
            "Schema cannot be uniquely baselined: " + "; ".join(status.errors)
        )
    expected = _expected(profile)
    count = _version_count(status.inferred_version, expected)
    with closing(sqlite3.connect(database)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(LEDGER_DDL)
            fingerprint = schema_fingerprint(connection, exclude_ledger=True)
            previous = "base" if profile == "production" else "empty"
            for filename in expected[:count]:
                migration_id = _migration_id(filename)
                connection.execute(
                    """INSERT INTO schema_migrations
                    (migration_id,filename,migration_sha256,applied_at,application_version,
                     schema_version_before,schema_version_after,execution_ms,success,
                     inferred,inference_method,schema_fingerprint,source_manifest,database_profile)
                    VALUES (?,?,?,?,?,?,?,?,1,1,?,?,?,?)""",
                    (
                        migration_id,
                        filename,
                        migration_sha256(filename),
                        utcnow(),
                        __version__,
                        previous,
                        filename,
                        0,
                        "exact_schema_fingerprint",
                        fingerprint,
                        f"builtin:{profile}:{status.inferred_version}",
                        profile,
                    ),
                )
                previous = filename
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return verify_migrations(database, profile)


def create_personal_database(database: Path) -> None:
    """Publish a completely initialized new database without replacing another file."""
    database.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=database.parent, prefix=".radar-init-") as work:
        ready = Path(work) / "ready.sqlite"
        migrate_database(ready, "production", dry_run=False)
        # Same-volume link is atomic on NTFS and POSIX, including concurrent first opens.
        os.link(ready, database)


def migrate_database(
    database: Path,
    profile: str,
    *,
    dry_run: bool = True,
    baseline_confirmed: bool = False,
) -> MigrationStatus:
    _expected(profile)
    if not database.exists():
        if dry_run:
            return migration_status(database, profile)
        database.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            if profile == "production":
                connection.executescript(SCHEMA)
            connection.execute(LEDGER_DDL)
            connection.commit()
    status = migration_status(database, profile)
    if not status.ledger_present:
        if dry_run:
            return status
        # A database whose schema is exactly the untouched base (nothing has
        # ever been applied) is unambiguous: there is no migration history to
        # get wrong, so a fresh install does not need the confirmation phrase.
        # Anything later than that still requires it, because baselining would
        # then be a statement about data that already exists.
        nothing_applied = status.inferred_version in {"base", "empty"}
        status = baseline_migration_ledger(
            database,
            profile,
            confirmed=baseline_confirmed or nothing_applied,
        )
    if status.errors:
        raise MigrationIntegrityError("; ".join(status.errors))
    if dry_run:
        return status
    for filename in status.pending:
        _apply_one(database, profile, filename)
    return verify_migrations(database, profile)


def infer_historical_versions(
    connection: sqlite3.Connection, profile: str
) -> tuple[str, ...]:
    actual = schema_fingerprint(connection, exclude_ledger=True)
    matches: list[str] = []
    expected = _expected(profile)
    for count in range(0, len(expected) + 1):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.db"
            with closing(sqlite3.connect(candidate)) as generated:
                generated.row_factory = sqlite3.Row
                generated.execute("PRAGMA foreign_keys=ON")
                if profile == "production":
                    generated.executescript(SCHEMA)
                for filename in expected[:count]:
                    generated.executescript(read_text(f"migrations/{filename}"))
                    guarded = GUARDED_DDL.get(filename)
                    if guarded is not None:
                        # The candidate schema must reproduce the *complete*
                        # effect of a migration, guarded columns included, or
                        # the fingerprint comparison would never match a real
                        # database.
                        guarded(generated)
                if schema_fingerprint(generated, exclude_ledger=True) == actual:
                    version = (
                        expected[count - 1]
                        if count
                        else "base"
                        if profile == "production"
                        else "empty"
                    )
                    matches.append(version)
    return tuple(matches)


def schema_fingerprint(
    connection: sqlite3.Connection, *, exclude_ledger: bool = False
) -> str:
    rows = connection.execute(
        """SELECT type,name,tbl_name,sql FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"""
    ).fetchall()
    payload = []
    for row in rows:
        name = row["name"] if isinstance(row, sqlite3.Row) else row[1]
        if exclude_ledger and name == "schema_migrations":
            continue
        values = dict(row) if isinstance(row, sqlite3.Row) else {
            "type": row[0], "name": row[1], "tbl_name": row[2], "sql": row[3]
        }
        values["sql"] = _normalize_sql(values["sql"])
        payload.append(values)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _apply_one(database: Path, profile: str, filename: str) -> None:
    status = verify_migrations(database, profile)
    if filename not in status.pending or filename != status.pending[0]:
        raise MigrationIntegrityError(f"out_of_order_migration:{filename}")
    sql = _transaction_body(read_text(f"migrations/{filename}"))
    started = time.monotonic()
    with closing(sqlite3.connect(database, timeout=30)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            connection.executescript("BEGIN IMMEDIATE;\n" + sql)
            # Some migrations cannot express their full effect in SQL (SQLite
            # has no conditional ADD COLUMN).  Applying only the SQL body was
            # how a freshly migrated database ended up without the guarded
            # columns, so the complete effect is applied here, in the same
            # transaction, before the fingerprint is taken.
            guarded = GUARDED_DDL.get(filename)
            if guarded is not None:
                guarded(connection)
            fingerprint_before = schema_fingerprint(connection)
            connection.execute(
                """INSERT INTO schema_migrations
                (migration_id,filename,migration_sha256,applied_at,application_version,
                 schema_version_before,schema_version_after,execution_ms,success,
                 inferred,inference_method,schema_fingerprint,source_manifest,database_profile)
                VALUES (?,?,?,?,?,?,?,?,1,0,NULL,?,NULL,?)
                ON CONFLICT(filename) DO UPDATE SET
                  migration_sha256=excluded.migration_sha256,
                  applied_at=excluded.applied_at,
                  application_version=excluded.application_version,
                  schema_version_before=excluded.schema_version_before,
                  schema_version_after=excluded.schema_version_after,
                  execution_ms=excluded.execution_ms,
                  success=1,error_type=NULL,error_message=NULL,
                  inferred=0,inference_method=NULL,
                  schema_fingerprint=excluded.schema_fingerprint,
                  source_manifest=NULL,database_profile=excluded.database_profile""",
                (
                    _migration_id(filename),
                    filename,
                    migration_sha256(filename),
                    utcnow(),
                    __version__,
                    (
                        status.applied[-1]
                        if status.applied
                        else "base"
                        if profile == "production"
                        else "empty"
                    ),
                    filename,
                    int((time.monotonic() - started) * 1000),
                    fingerprint_before,
                    profile,
                ),
            )
            connection.commit()
        except Exception as exc:
            connection.rollback()
            with closing(sqlite3.connect(database)) as failure_connection:
                failure_connection.execute(LEDGER_DDL)
                failure_connection.execute(
                    """INSERT INTO schema_migrations
                    (migration_id,filename,migration_sha256,applied_at,application_version,
                     schema_version_before,schema_version_after,execution_ms,success,
                     error_type,error_message,inferred,database_profile)
                    VALUES (?,?,?,?,?,?,?,?,0,?,?,0,?)
                    ON CONFLICT(filename) DO UPDATE SET
                      migration_sha256=excluded.migration_sha256,
                      applied_at=excluded.applied_at,
                      application_version=excluded.application_version,
                      execution_ms=excluded.execution_ms,
                      success=0,error_type=excluded.error_type,
                      error_message=excluded.error_message""",
                    (
                        _migration_id(filename),
                        filename,
                        migration_sha256(filename),
                        utcnow(),
                        __version__,
                        (
                            status.applied[-1]
                            if status.applied
                            else "base"
                            if profile == "production"
                            else "empty"
                        ),
                        filename,
                        int((time.monotonic() - started) * 1000),
                        type(exc).__name__,
                        str(exc)[:500],
                        profile,
                    ),
                )
                failure_connection.commit()
            raise


def _transaction_body(sql: str) -> str:
    lines = []
    for line in sql.splitlines():
        normalized = line.strip().upper()
        if normalized in {"BEGIN;", "BEGIN IMMEDIATE;", "COMMIT;"}:
            continue
        if normalized.startswith("PRAGMA FOREIGN_KEYS"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _expected(profile: str) -> tuple[str, ...]:
    try:
        expected = PROFILE_MIGRATIONS[profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported database profile: {profile}") from exc
    for filename in expected:
        try:
            migration_sha256(filename)
        except FileNotFoundError as exc:
            raise MigrationIntegrityError(f"missing_migration:{filename}") from exc
    return expected


def _migration_id(filename: str) -> int:
    match = re.match(r"^(\d{4})_", filename)
    if not match:
        raise MigrationIntegrityError(f"Invalid migration filename: {filename}")
    return int(match.group(1))


def _version_count(version: str | None, expected: tuple[str, ...]) -> int:
    if version in {None, "empty", "base"}:
        return 0
    try:
        return expected.index(version) + 1
    except ValueError as exc:
        raise MigrationIntegrityError(f"Unknown inferred version: {version}") from exc


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


@contextmanager
def _read_only(database: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def _normalize_sql(sql: str | None) -> str | None:
    return " ".join((sql or "").split()) or None


def status_payload(status: MigrationStatus) -> dict[str, object]:
    return asdict(status)
