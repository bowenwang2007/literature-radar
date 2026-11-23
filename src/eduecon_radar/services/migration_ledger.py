"""Evidence-based reconciliation between ``schema_migrations`` and the real schema.

The ledger and the database had drifted apart: migrations 0010-0012 were applied
to production through guarded Python helpers, but no database profile listed
them, so ``schema_migrations`` could not describe the schema it was supposed to
record.  Re-running them blindly would have been wrong (0012 is a non-idempotent
``ALTER TABLE``), and guessing was not acceptable either.

This module answers the question from the other direction: it derives the exact
set of schema objects each migration file creates (tables, columns, indexes,
triggers, plus the guarded columns that SQLite cannot express conditionally) and
compares that set with ``sqlite_master`` / ``PRAGMA table_info``.  Only complete
matches are recorded, always flagged ``inferred=1`` with the evidence method and
the object counts; anything ambiguous is reported and refused.

Nothing here executes a migration body.  Recording is metadata only.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from eduecon_radar import __version__
from eduecon_radar.db import utcnow
from eduecon_radar.resources import read_text, resource_sha256
from eduecon_radar.services.migrations import guarded_columns

_CONSTRAINT_KEYWORDS = ("CONSTRAINT", "PRIMARY", "UNIQUE", "CHECK", "FOREIGN")

_CREATE_TABLE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"'\[]?([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
_CREATE_INDEX = re.compile(
    r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"'\[]?([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
_CREATE_TRIGGER = re.compile(
    r"CREATE\s+TRIGGER\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"'\[]?([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
_ALTER_ADD_COLUMN = re.compile(
    r"ALTER\s+TABLE\s+[\"'\[]?([A-Za-z_][A-Za-z0-9_]*)[\"'\]]?\s+"
    r"ADD\s+COLUMN\s+[\"'\[]?([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
_LINE_COMMENT = re.compile(r"--[^\n]*")


class MigrationLedgerError(RuntimeError):
    pass


@dataclass(frozen=True)
class SchemaObject:
    """One object a migration is expected to bring into existence."""

    kind: str  # table | column | index | trigger
    name: str
    table: str | None = None

    def label(self) -> str:
        return f"{self.kind}:{self.table}.{self.name}" if self.table else f"{self.kind}:{self.name}"


@dataclass(frozen=True)
class LedgerRow:
    filename: str
    migration_id: int
    inferred: bool
    success: bool
    schema_version_after: str


@dataclass(frozen=True)
class MigrationAudit:
    filename: str
    recorded: bool
    inferred: bool
    total: int
    present: int
    missing: tuple[str, ...]

    @property
    def state(self) -> str:
        if self.recorded:
            return "recorded_verified" if not self.missing else "recorded_object_missing"
        if self.total and self.present == self.total:
            return "applied_unrecorded"
        if self.present == 0:
            return "not_applied"
        return "partially_applied"

    def payload(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "state": self.state,
            "recorded": self.recorded,
            "inferred": self.inferred,
            "objects_present": self.present,
            "objects_total": self.total,
            "missing": list(self.missing),
        }


def _strip_comments(sql: str) -> str:
    return _LINE_COMMENT.sub("", sql)


def _table_body(sql: str, start: int) -> str:
    """Return the text of the first balanced parenthesis group after ``start``."""
    open_index = sql.find("(", start)
    if open_index < 0:
        return ""
    depth = 0
    quote: str | None = None
    for index in range(open_index, len(sql)):
        char = sql[index]
        if quote:
            if char == quote:
                quote = None
            continue
        if char in "'\"`":
            quote = char
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return sql[open_index + 1 : index]
    return ""


def _split_top_level(body: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    current: list[str] = []
    for char in body:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in "'\"`":
            quote = char
            current.append(char)
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current))
    return parts


def _column_name(definition: str) -> str | None:
    stripped = definition.strip()
    if not stripped:
        return None
    upper = stripped.upper()
    if any(upper.startswith(keyword) for keyword in _CONSTRAINT_KEYWORDS):
        return None
    token = stripped.split(None, 1)[0]
    token = token.strip('"\'`[]')
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
        return None
    return token


def parse_migration_objects(sql: str, *, filename: str = "") -> tuple[SchemaObject, ...]:
    """Derive every schema object a migration file is expected to create."""
    cleaned = _strip_comments(sql)
    objects: list[SchemaObject] = []
    for match in _CREATE_TABLE.finditer(cleaned):
        table = match.group(1)
        objects.append(SchemaObject("table", table))
        for definition in _split_top_level(_table_body(cleaned, match.end())):
            column = _column_name(definition)
            if column:
                objects.append(SchemaObject("column", column, table))
    for match in _CREATE_INDEX.finditer(cleaned):
        objects.append(SchemaObject("index", match.group(1)))
    for match in _CREATE_TRIGGER.finditer(cleaned):
        objects.append(SchemaObject("trigger", match.group(1)))
    for match in _ALTER_ADD_COLUMN.finditer(cleaned):
        objects.append(SchemaObject("column", match.group(2), match.group(1)))
    for table, column, _definition in guarded_columns(filename):
        obj = SchemaObject("column", column, table)
        if obj not in objects:
            objects.append(obj)
    return tuple(dict.fromkeys(objects))


def migration_objects(filename: str) -> tuple[SchemaObject, ...]:
    return parse_migration_objects(read_text(f"migrations/{filename}"), filename=filename)


def _object_present(connection: sqlite3.Connection, obj: SchemaObject) -> bool:
    if obj.kind == "column":
        assert obj.table is not None
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (obj.table,)
        ).fetchone()
        if row is None:
            return False
        columns = {
            str(item[1]) for item in connection.execute(f"PRAGMA table_info({obj.table})")
        }
        return obj.name in columns
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type=? AND name=?", (obj.kind, obj.name)
    ).fetchone()
    return row is not None


def read_ledger(database: Path) -> dict[str, LedgerRow]:
    if not database.is_file():
        return {}
    uri = database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        present = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
        ).fetchone()
        if present is None:
            return {}
        rows = connection.execute(
            """SELECT filename,migration_id,inferred,success,schema_version_after
            FROM schema_migrations"""
        ).fetchall()
    return {
        str(row["filename"]): LedgerRow(
            filename=str(row["filename"]),
            migration_id=int(row["migration_id"]),
            inferred=bool(row["inferred"]),
            success=bool(row["success"]),
            schema_version_after=str(row["schema_version_after"]),
        )
        for row in rows
    }


def audit_migrations(
    database: Path,
    filenames: Sequence[str],
    *,
    ledger: dict[str, LedgerRow] | None = None,
) -> tuple[MigrationAudit, ...]:
    """Compare every expected migration with the objects that actually exist."""
    rows = read_ledger(database) if ledger is None else ledger
    audits: list[MigrationAudit] = []
    uri = f"{database.resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        for filename in filenames:
            objects = migration_objects(filename)
            missing = tuple(
                obj.label() for obj in objects if not _object_present(connection, obj)
            )
            row = rows.get(filename)
            audits.append(
                MigrationAudit(
                    filename=filename,
                    recorded=bool(row and row.success),
                    inferred=bool(row and row.inferred),
                    total=len(objects),
                    present=len(objects) - len(missing),
                    missing=missing,
                )
            )
    return tuple(audits)


def reconcile_migration_ledger(
    database: Path,
    profile: str,
    filenames: Sequence[str],
    *,
    confirmed: bool = False,
) -> dict[str, object]:
    """Record migrations whose full object set already exists in the schema.

    ``confirmed=False`` is a pure read-only audit.  Refuses to act when a
    migration is only partially present, or when the ledger claims a migration
    whose objects are missing.
    """
    if not database.is_file():
        raise MigrationLedgerError(f"Database does not exist: {database}")
    audits = audit_migrations(database, filenames)
    report: dict[str, object] = {
        "database": str(database.resolve()),
        "database_profile": profile,
        "confirmed": confirmed,
        "migrations": [audit.payload() for audit in audits],
    }
    blocking = [
        audit.filename
        for audit in audits
        if audit.state in {"partially_applied", "recorded_object_missing"}
    ]
    to_record = [audit for audit in audits if audit.state == "applied_unrecorded"]
    report["to_record"] = [audit.filename for audit in to_record]
    report["blocking"] = blocking
    if blocking:
        details = {
            audit.filename: list(audit.missing)
            for audit in audits
            if audit.filename in blocking
        }
        report["status"] = "blocked"
        report["blocking_detail"] = details
        if not confirmed:
            return report
        raise MigrationLedgerError(
            "Refusing to reconcile: schema does not match the migration record "
            f"({json.dumps(details, ensure_ascii=False)})"
        )
    if not confirmed:
        report["status"] = "audit"
        return report
    recorded = _record(database, profile, to_record, ledger=read_ledger(database))
    report["recorded"] = recorded
    report["status"] = "reconciled"
    report["after"] = [
        audit.payload() for audit in audit_migrations(database, filenames)
    ]
    return report


def _record(
    database: Path,
    profile: str,
    audits: Iterable[MigrationAudit],
    *,
    ledger: dict[str, LedgerRow],
) -> list[str]:
    recorded: list[str] = []
    with closing(sqlite3.connect(database, timeout=30)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        ordered = sorted(audits, key=lambda audit: _migration_id(audit.filename))
        if not ordered:
            return recorded
        # The "previous" row is the migration this run continues from, so it must
        # be the newest ledger entry *below* the run being recorded — not the
        # newest entry overall.  Using the global maximum silently skipped the
        # whole run whenever a later migration had already been applied normally,
        # which is exactly the hole this reconciler exists to close.
        first_id = _migration_id(ordered[0].filename)
        previous = max(
            (
                row.migration_id
                for row in ledger.values()
                if row.success and row.migration_id < first_id
            ),
            default=0,
        )
        for audit in ordered:
            migration_id = _migration_id(audit.filename)
            if migration_id <= previous:
                continue
            before = _version_for(previous, ledger, profile)
            connection.execute(
                """INSERT INTO schema_migrations
                (migration_id,filename,migration_sha256,applied_at,application_version,
                 schema_version_before,schema_version_after,execution_ms,success,
                 error_type,error_message,inferred,inference_method,schema_fingerprint,
                 source_manifest,database_profile)
                VALUES (?,?,?,?,?,?,?,?,1,NULL,NULL,1,?,?,?,?)
                ON CONFLICT(filename) DO NOTHING""",
                (
                    migration_id,
                    audit.filename,
                    resource_sha256(f"migrations/{audit.filename}"),
                    utcnow(),
                    __version__,
                    before,
                    audit.filename,
                    0,
                    f"schema_object_match:{audit.present}/{audit.total}",
                    _schema_fingerprint(database),
                    f"reconciled:{profile}",
                    profile,
                ),
            )
            recorded.append(audit.filename)
            previous = migration_id
        connection.commit()
    return recorded


def _version_for(
    migration_id: int, ledger: dict[str, LedgerRow], profile: str
) -> str:
    if migration_id == 0:
        return "base" if profile == "production" else "empty"
    for row in ledger.values():
        if row.migration_id == migration_id and row.success:
            return row.schema_version_after
    return f"migration_id:{migration_id}"


def _schema_fingerprint(database: Path) -> str:
    uri = f"{database.resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        rows = connection.execute(
            """SELECT type,name,tbl_name,sql FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"""
        ).fetchall()
    payload = [
        {"type": row[0], "name": row[1], "tbl_name": row[2], "sql": row[3]}
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _migration_id(filename: str) -> int:
    prefix = filename.split("_", 1)[0]
    try:
        return int(prefix)
    except ValueError as exc:  # pragma: no cover - guarded by file naming
        raise MigrationLedgerError(f"Unparseable migration filename: {filename}") from exc
