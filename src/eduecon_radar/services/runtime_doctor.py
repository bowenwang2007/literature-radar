from __future__ import annotations

import importlib
import os
import platform
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from eduecon_radar import __version__
from eduecon_radar.config import Settings
from eduecon_radar.resources import verify_runtime_resources
from eduecon_radar.services.doi import canonicalize_doi
from eduecon_radar.services.migration_integrity import migration_status
from eduecon_radar.services.runtime_bundle import RuntimeBundleResolver
from eduecon_radar.services.schema_manifest import (
    SchemaManifestError,
    verify_schema_manifest,
)


def run_doctor(settings: Settings) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, detail: Any) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    add("python_version", "ok", platform.python_version())
    add("package_version", "ok", __version__)
    git_commit = _git_commit(settings.project_root)
    add("git_commit", "ok" if git_commit else "warning", git_commit)
    add(
        "cwd_independence",
        "ok",
        {"cwd": str(Path.cwd()), "project_root": str(settings.project_root)},
    )
    try:
        resources = verify_runtime_resources()
        add("runtime_resources", resources["status"], {"count": resources["resource_count"]})
    except Exception as exc:
        add("runtime_resources", "error", _error(exc))

    database = settings.database_path.resolve()
    if not database.is_file():
        add("database", "blocked", {"path": str(database), "reason": "missing"})
    else:
        try:
            status = migration_status(database, "production")
            add(
                "migration_ledger",
                "ok" if status.ledger_present and not status.errors else "blocked",
                {
                    "database": status.database,
                    "ledger_present": status.ledger_present,
                    "applied": list(status.applied),
                    "pending": list(status.pending),
                    "errors": list(status.errors),
                },
            )
        except Exception as exc:
            add("migration_ledger", "error", _error(exc))
        try:
            integrity, foreign_keys = _database_integrity(database)
            add("integrity_check", "ok" if integrity == "ok" else "error", integrity)
            add(
                "foreign_key_check",
                "ok" if not foreign_keys else "error",
                {"violations": foreign_keys},
            )
        except Exception as exc:
            add("database_integrity", "error", _error(exc))
        manifest = settings.project_root / "manifests" / "production_schema_manifest.json"
        if manifest.is_file():
            try:
                result = verify_schema_manifest(database, "production", manifest)
                add("schema_manifest", "ok", result)
            except (SchemaManifestError, OSError, RuntimeError) as exc:
                add("schema_manifest", "blocked", _error(exc))
        else:
            add("schema_manifest", "blocked", "production manifest missing")
        _add_database_schema_checks(add, database)

    try:
        bundle = RuntimeBundleResolver(settings.project_root).inspect()
        add("released_bundle_trust", "ok" if bundle["trusted"] else "blocked", bundle)
        add(
            "production_ai_safety",
            "ok" if not bundle["trusted"] else "warning",
            (
                "blocked before client construction"
                if not bundle["trusted"]
                else "released bundle trusted"
            ),
        )
    except Exception as exc:
        add("released_bundle_trust", "blocked", _error(exc))

    add(
        "writable_directories",
        "ok" if _directories_writable(settings.project_root) else "warning",
        {
            "exports": str(settings.project_root / "exports"),
            "logs": str(settings.project_root / "logs"),
        },
    )
    add(
        "api_keys",
        "ok" if settings.deepseek_api_key else "warning",
        {
            "DEEPSEEK_API_KEY": "present" if settings.deepseek_api_key else "missing",
            "OPENALEX_API_KEY": "present" if settings.openalex_api_key else "missing",
        },
    )
    validation = settings.project_root / "data" / "validation.db"
    add(
        "validation_studio_isolation",
        "ok" if validation.resolve() != database else "error",
        {"validation_database": str(validation.resolve()), "production_database": str(database)},
    )
    try:
        importlib.import_module("eduecon_radar.ui.app")
        for module in (
            "eduecon_radar.ui.views.personal_mode",
            "eduecon_radar.ui.views.paper_library",
        ):
            importlib.import_module(module)
        add("streamlit_import", "ok", "all page modules imported; no render or network")
    except Exception as exc:
        add("streamlit_import", "error", _error(exc))

    order = {"ok": 0, "warning": 1, "blocked": 2, "error": 3}
    overall = max((item["status"] for item in checks), key=order.get)
    return {"status": overall, "checks": checks}


def _add_database_schema_checks(add, database: Path) -> None:
    uri = database.as_uri() + "?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            tables = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            add(
                "metadata_provenance_schema",
                "ok" if "paper_metadata_provenance" in tables else "blocked",
                "present" if "paper_metadata_provenance" in tables else "migration 0008 required",
            )
            if "workflow_task_leases" in tables:
                stale = int(
                    connection.execute(
                        """SELECT COUNT(*) FROM workflow_task_leases
                        WHERE status='active' AND expires_at<=datetime('now')"""
                    ).fetchone()[0]
                )
                add("task_leases", "warning" if stale else "ok", {"stale": stale})
            else:
                add("task_leases", "blocked", "migration 0007 required")
            if "ai_usage_reservations" in tables:
                unsettled = int(
                    connection.execute(
                        """SELECT COUNT(*) FROM ai_usage_reservations
                        WHERE status='reserved'"""
                    ).fetchone()[0]
                )
                add(
                    "unsettled_budget_reservations",
                    "warning" if unsettled else "ok",
                    {"count": unsettled},
                )
            else:
                add("unsettled_budget_reservations", "blocked", "migration 0008 required")
    except sqlite3.DatabaseError as exc:
        add("database_schema", "error", _error(exc))
    add(
        "doi_normalization",
        "ok" if canonicalize_doi("HTTPS://DOI.ORG/10.1000/ABC.") == "10.1000/abc." else "error",
        "canonicalizer self-test",
    )


def _database_integrity(database: Path) -> tuple[str, list[dict[str, Any]]]:
    uri = database.as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_keys = [dict(row) for row in connection.execute("PRAGMA foreign_key_check")]
    return integrity, foreign_keys


def _directories_writable(project_root: Path) -> bool:
    return all(
        path.exists() and os.access(path, os.W_OK)
        for path in (project_root / "exports", project_root / "logs")
    )


def _git_commit(project_root: Path) -> str | None:
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout.strip()
        if Path(top).resolve() != project_root.resolve():
            return None
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _error(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:500]}
