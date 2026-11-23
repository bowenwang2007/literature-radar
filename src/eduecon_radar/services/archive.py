from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

ARCHIVE_TABLES = (
    "papers",
    "screenings",
    "paper_translations",
    "reviews",
    "workflow_batches",
    "workflow_batch_items",
)


@dataclass(frozen=True)
class DatabaseArchiveManifest:
    source_path: str
    archive_path: str
    manifest_path: str
    created_at: str
    file_size: int
    sha256: str
    integrity_result: str
    table_counts: dict[str, int | None]
    current_git_commit: str | None
    logical_application_version: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def inspect_database_read_only(path: Path) -> tuple[str, dict[str, int | None]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Database does not exist: {resolved}")
    uri = resolved.as_uri() + "?mode=ro"
    counts: dict[str, int | None] = {}
    with sqlite3.connect(uri, uri=True) as connection:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity_check failed: {integrity}")
        available = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for table in ARCHIVE_TABLES:
            counts[table] = (
                int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                if table in available
                else None
            )
    return integrity, counts


def compare_archived_protected_data(source_path: Path, archive_path: Path) -> tuple[str, ...]:
    """Compare protected business tables without writing either SQLite database."""
    differences: list[str] = []
    source_uri = source_path.resolve().as_uri() + "?mode=ro"
    archive_uri = archive_path.resolve().as_uri() + "?mode=ro"
    with (
        sqlite3.connect(source_uri, uri=True) as source,
        sqlite3.connect(archive_uri, uri=True) as archive,
    ):
        for table in ARCHIVE_TABLES:
            source_columns = _table_columns(source, table)
            archive_columns = _table_columns(archive, table)
            if source_columns != archive_columns:
                differences.append(f"protected_table_schema_changed:{table}")
                continue
            if not source_columns:
                continue
            source_rows = source.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            archive_rows = archive.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            if source_rows != archive_rows:
                differences.append(f"protected_table_data_changed:{table}")
    return tuple(differences)


def _table_columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})"))


def archive_database(
    *,
    source_path: Path,
    archive_dir: Path,
    label: str,
    confirm_read_only_source: bool,
    current_git_commit: str | None,
    logical_application_version: str,
    now: datetime | None = None,
) -> DatabaseArchiveManifest:
    if not confirm_read_only_source:
        raise ValueError("Use --confirm-read-only-source to confirm the source is read-only")
    source = source_path.resolve()
    destination_dir = archive_dir.resolve()
    integrity, counts = inspect_database_read_only(source)
    source_hash = sha256_file(source)
    source_size = source.stat().st_size
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip()).strip("-.")
    if not safe_label:
        raise ValueError("Archive label must contain at least one safe character")
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive_path = destination_dir / f"{source.stem}.{safe_label}.{timestamp}{source.suffix}"
    manifest_path = archive_path.with_suffix(archive_path.suffix + ".manifest.json")
    if archive_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Archive target already exists: {archive_path}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    created_at = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
    created_archive = False
    created_manifest = False
    try:
        with source.open("rb") as source_stream, archive_path.open("xb") as target_stream:
            created_archive = True
            shutil.copyfileobj(source_stream, target_stream, length=1024 * 1024)
        archive_hash = sha256_file(archive_path)
        if archive_hash != source_hash or archive_path.stat().st_size != source_size:
            raise RuntimeError("Archive verification failed: source and archive differ")
        manifest = DatabaseArchiveManifest(
            source_path=str(source),
            archive_path=str(archive_path),
            manifest_path=str(manifest_path),
            created_at=created_at,
            file_size=source_size,
            sha256=source_hash,
            integrity_result=integrity,
            table_counts=counts,
            current_git_commit=current_git_commit,
            logical_application_version=logical_application_version,
        )
        with manifest_path.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(asdict(manifest), output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
        created_manifest = True
        return manifest
    except Exception:
        if created_manifest and manifest_path.exists():
            manifest_path.unlink()
        if created_archive and archive_path.exists():
            archive_path.unlink()
        raise
