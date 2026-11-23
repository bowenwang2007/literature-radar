from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Any

from eduecon_radar.services.migration_integrity import (
    migrate_database,
    schema_fingerprint,
    verify_migrations,
)

MANIFEST_VERSION = "1"


class SchemaManifestError(RuntimeError):
    pass


def generate_schema_manifest(
    profile: str,
    *,
    output: Path | None = None,
    database: Path | None = None,
) -> dict[str, Any]:
    if database is None:
        with tempfile.TemporaryDirectory() as directory:
            fresh = Path(directory) / f"{profile}.db"
            migrate_database(fresh, profile, dry_run=False)
            payload = _manifest_for_database(fresh, profile)
    else:
        verify_migrations(database, profile)
        payload = _manifest_for_database(database, profile)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def verify_schema_manifest(
    database: Path, profile: str, manifest_path: Path
) -> dict[str, Any]:
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected.get("database_profile") != profile:
        raise SchemaManifestError("database_profile_mismatch")
    expected_hash = expected.get("manifest_sha256")
    if expected_hash != _payload_sha256(expected):
        raise SchemaManifestError("manifest_hash_mismatch")
    actual = generate_schema_manifest(profile, database=database)
    differences = _diff_schema(expected, actual)
    if differences:
        raise SchemaManifestError("schema_manifest_mismatch:" + ",".join(differences[:20]))
    return {
        "status": "ok",
        "database": str(database.resolve()),
        "database_profile": profile,
        "manifest_sha256": expected_hash,
        "integrity_check": actual["integrity_check"],
        "foreign_key_violations": len(actual["foreign_key_check"]),
    }


def _manifest_for_database(database: Path, profile: str) -> dict[str, Any]:
    uri = database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        tables: list[dict[str, Any]] = []
        table_rows = connection.execute(
            """SELECT name,sql FROM sqlite_master WHERE type='table'
            AND name NOT LIKE 'sqlite_%' ORDER BY name"""
        ).fetchall()
        for table in table_rows:
            name = table["name"]
            columns = [
                {
                    "cid": int(row["cid"]),
                    "name": row["name"],
                    "type": row["type"],
                    "not_null": bool(row["notnull"]),
                    "default": row["dflt_value"],
                    "primary_key_position": int(row["pk"]),
                }
                for row in connection.execute(
                    f"PRAGMA table_xinfo({_quote_identifier(name)})"
                ).fetchall()
            ]
            foreign_keys = [
                dict(row)
                for row in connection.execute(
                    f"PRAGMA foreign_key_list({_quote_identifier(name)})"
                ).fetchall()
            ]
            indexes = []
            for row in connection.execute(
                f"PRAGMA index_list({_quote_identifier(name)})"
            ).fetchall():
                index_name = row["name"]
                indexes.append(
                    {
                        "name": index_name,
                        "unique": bool(row["unique"]),
                        "origin": row["origin"],
                        "partial": bool(row["partial"]),
                        "columns": [
                            dict(item)
                            for item in connection.execute(
                                f"PRAGMA index_xinfo({_quote_identifier(index_name)})"
                            ).fetchall()
                        ],
                        "sql": _sqlite_sql(connection, "index", index_name),
                    }
                )
            tables.append(
                {
                    "name": name,
                    "sql": _normalize_sql(table["sql"]),
                    "columns": columns,
                    "primary_key": [
                        item["name"]
                        for item in sorted(
                            columns, key=lambda value: value["primary_key_position"]
                        )
                        if item["primary_key_position"]
                    ],
                    "foreign_keys": foreign_keys,
                    "indexes": sorted(indexes, key=lambda item: item["name"]),
                    "check_constraints": _check_constraints(table["sql"]),
                }
            )
        migrations = [
            dict(row)
            for row in connection.execute(
                """SELECT migration_id,filename,migration_sha256,schema_version_before,
                schema_version_after,inferred,database_profile
                FROM schema_migrations WHERE success=1 ORDER BY migration_id"""
            ).fetchall()
        ]
        payload: dict[str, Any] = {
            "manifest_version": MANIFEST_VERSION,
            "database_profile": profile,
            "migrations": migrations,
            "tables": tables,
            "triggers": _objects(connection, "trigger"),
            "views": _objects(connection, "view"),
            "sqlite_version": sqlite3.sqlite_version,
            "application_git_commit": _git_commit(),
            "schema_fingerprint": schema_fingerprint(connection),
            "integrity_check": connection.execute("PRAGMA integrity_check").fetchone()[0],
            "foreign_key_check": [
                dict(row) for row in connection.execute("PRAGMA foreign_key_check").fetchall()
            ],
        }
    payload["manifest_sha256"] = _payload_sha256(payload)
    return payload


def _diff_schema(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    differences = []
    if _stable_migrations(expected.get("migrations")) != _stable_migrations(
        actual.get("migrations")
    ):
        differences.append("migrations")
    keys = (
        "database_profile",
        "tables",
        "triggers",
        "views",
        "schema_fingerprint",
        "integrity_check",
        "foreign_key_check",
    )
    differences.extend(key for key in keys if expected.get(key) != actual.get(key))
    return differences


def _stable_migrations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    stable_fields = (
        "database_profile",
        "migration_id",
        "filename",
        "migration_sha256",
        "schema_version_before",
        "schema_version_after",
    )
    return [
        {
            **{field: migration.get(field) for field in stable_fields},
            # Existing manifests contain only successful rows and predate this
            # explicit field. Absence therefore has the same meaning as success=1.
            "success": int(migration.get("success", 1)),
        }
        for migration in value
    ]


def _payload_sha256(payload: dict[str, Any]) -> str:
    canonical = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _objects(connection: sqlite3.Connection, object_type: str) -> list[dict[str, str]]:
    return [
        {"name": row["name"], "table": row["tbl_name"], "sql": _normalize_sql(row["sql"])}
        for row in connection.execute(
            """SELECT name,tbl_name,sql FROM sqlite_master
            WHERE type=? AND name NOT LIKE 'sqlite_%' ORDER BY name""",
            (object_type,),
        ).fetchall()
    ]


def _sqlite_sql(connection: sqlite3.Connection, object_type: str, name: str) -> str | None:
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type=? AND name=?", (object_type, name)
    ).fetchone()
    return _normalize_sql(row["sql"]) if row else None


def _check_constraints(sql: str | None) -> list[str]:
    if not sql:
        return []
    upper = sql.upper()
    result: list[str] = []
    position = 0
    while True:
        start = upper.find("CHECK", position)
        if start < 0:
            break
        opening = sql.find("(", start)
        if opening < 0:
            break
        depth = 0
        for index in range(opening, len(sql)):
            if sql[index] == "(":
                depth += 1
            elif sql[index] == ")":
                depth -= 1
                if depth == 0:
                    result.append(_normalize_sql(sql[start : index + 1]) or "")
                    position = index + 1
                    break
        else:
            break
    return result


def _quote_identifier(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _normalize_sql(sql: str | None) -> str | None:
    return " ".join((sql or "").split()) or None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        top = Path(result.stdout.strip()).resolve()
        package_root = Path(__file__).resolve().parents[3]
        if top != package_root:
            return None
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
            cwd=top,
        )
        return commit.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
