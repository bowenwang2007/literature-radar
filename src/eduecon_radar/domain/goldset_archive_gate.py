from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from eduecon_radar.services.archive import inspect_database_read_only, sha256_file


@dataclass(frozen=True)
class ArchiveGateResult:
    valid: bool
    manifest_path: str
    source_path: str | None
    archive_path: str | None
    source_sha256: str | None
    reasons: tuple[str, ...]


def validate_archive_manifest(
    manifest_path: Path,
    *,
    production_database: Path,
    active_database: Path | None = None,
    source_change_explanation: str | None = None,
    require_git_commit: bool = True,
) -> ArchiveGateResult:
    reasons: list[str] = []
    manifest_path = manifest_path.resolve()
    production_database = production_database.resolve()
    if not manifest_path.is_file():
        return ArchiveGateResult(
            False,
            str(manifest_path),
            None,
            None,
            None,
            ("archive_manifest_missing",),
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        return ArchiveGateResult(
            False,
            str(manifest_path),
            None,
            None,
            None,
            (f"invalid_archive_manifest:{exc}",),
        )
    source = Path(str(payload.get("source_path", ""))).resolve()
    archive = Path(str(payload.get("archive_path", ""))).resolve()
    expected_hash = str(payload.get("sha256", "")).upper()
    if source != production_database:
        reasons.append("source_path_does_not_match_production")
    if source == archive:
        reasons.append("archive_path_equals_source")
    if active_database and archive == active_database.resolve():
        reasons.append("archive_is_active_database")
    if not source.is_file():
        reasons.append("source_database_missing")
    if not archive.is_file():
        reasons.append("archive_database_missing")
    source_hash = sha256_file(source) if source.is_file() else None
    archive_hash = sha256_file(archive) if archive.is_file() else None
    if not expected_hash:
        reasons.append("manifest_sha256_missing")
    if (
        source_hash
        and source_hash != expected_hash
        and not (source_change_explanation or "").strip()
    ):
        reasons.append("current_source_sha256_mismatch")
    if archive_hash and archive_hash != expected_hash:
        reasons.append("archive_sha256_mismatch")
    if payload.get("integrity_result") != "ok":
        reasons.append("manifest_integrity_not_ok")
    if not payload.get("created_at"):
        reasons.append("manifest_created_at_missing")
    if require_git_commit and not payload.get("current_git_commit"):
        reasons.append("manifest_git_commit_missing")
    counts = payload.get("table_counts")
    if not isinstance(counts, dict) or not {"papers", "screenings", "reviews"} <= set(counts):
        reasons.append("manifest_table_counts_incomplete")
    if archive.is_file():
        try:
            integrity, archive_counts = inspect_database_read_only(archive)
            if integrity != "ok":
                reasons.append("archive_integrity_not_ok")
            if isinstance(counts, dict):
                for name, expected in counts.items():
                    if name in archive_counts and archive_counts[name] != expected:
                        reasons.append(f"archive_table_count_mismatch:{name}")
        except (OSError, RuntimeError) as exc:
            reasons.append(f"archive_read_only_validation_failed:{exc}")
    return ArchiveGateResult(
        valid=not reasons,
        manifest_path=str(manifest_path),
        source_path=str(source),
        archive_path=str(archive),
        source_sha256=source_hash,
        reasons=tuple(reasons),
    )
