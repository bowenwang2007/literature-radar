from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from eduecon_radar.resources import read_text

PROVENANCE_FIELDS = (
    "doi",
    "title",
    "abstract",
    "authors_json",
    "publisher",
    "article_type",
    "online_date",
    "issue_date",
    "volume",
    "issue",
    "pages",
    "source_url",
    "open_access_url",
    "keywords_json",
    "jel_codes_json",
)
PLACEHOLDER_ABSTRACT = re.compile(
    r"^(?:abstract\s+unavailable|no\s+abstract|not\s+available|n/?a|none|null)[.!]?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MergeDecision:
    field_name: str
    value: Any
    changed: bool
    decision: str | None
    reason: str | None
    quality_flags: tuple[str, ...] = ()


def load_source_priorities(path: Path | None = None) -> tuple[str, dict[str, int]]:
    payload = yaml.safe_load(
        path.read_text(encoding="utf-8")
        if path is not None
        else read_text("config/metadata_sources.yaml")
    )
    return str(payload["version"]), {
        str(name): int(details["priority"])
        for name, details in payload["sources"].items()
    }


def source_identity(source_metadata: dict[str, Any]) -> tuple[str, str | None]:
    source = str(
        source_metadata.get("_source_name")
        or source_metadata.get("source")
        or ("crossref" if source_metadata.get("DOI") else "legacy_unknown")
    ).casefold()
    record_id = source_metadata.get("DOI") or source_metadata.get("id")
    return source, str(record_id) if record_id else None


def merge_field(
    field_name: str,
    current: Any,
    incoming: Any,
    *,
    current_priority: int,
    incoming_priority: int,
) -> MergeDecision:
    if _empty(incoming):
        return MergeDecision(
            field_name,
            current,
            False,
            "degraded_prevention" if not _empty(current) else None,
            "empty_value_cannot_replace_non_empty" if not _empty(current) else None,
            ("incoming_empty",),
        )
    if _empty(current):
        return MergeDecision(
            field_name, incoming, True, "supplemented", "filled_missing_value"
        )
    if _equivalent(current, incoming):
        return MergeDecision(field_name, current, False, None, None)
    if field_name == "abstract":
        current_flags = abstract_quality_flags(str(current))
        incoming_flags = abstract_quality_flags(str(incoming))
        if incoming_flags and not current_flags:
            return MergeDecision(
                field_name,
                current,
                False,
                "degraded_prevention",
                "lower_quality_abstract_rejected",
                incoming_flags,
            )
        if len(str(incoming).strip()) < len(str(current).strip()) * 0.7:
            return MergeDecision(
                field_name,
                current,
                False,
                "degraded_prevention",
                "likely_truncated_abstract_rejected",
                ("likely_truncated",),
            )
    if field_name == "title" and _looks_truncated(str(incoming), str(current)):
        return MergeDecision(
            field_name,
            current,
            False,
            "degraded_prevention",
            "likely_truncated_title_rejected",
            ("likely_truncated",),
        )
    if field_name == "doi":
        return MergeDecision(
            field_name,
            current,
            False,
            "conflict",
            "canonical_doi_conflict_requires_review",
        )
    if incoming_priority < current_priority:
        return MergeDecision(
            field_name,
            current,
            False,
            "ignored",
            "lower_priority_source",
        )
    if field_name in {"online_date", "issue_date"}:
        return MergeDecision(
            field_name,
            incoming,
            True,
            "accepted",
            "same_date_semantics_new_observation",
        )
    return MergeDecision(
        field_name, incoming, True, "accepted", "equal_or_higher_priority_source"
    )


def abstract_quality_flags(value: str | None) -> tuple[str, ...]:
    text = " ".join((value or "").split())
    flags: list[str] = []
    if PLACEHOLDER_ABSTRACT.fullmatch(text):
        flags.append("placeholder")
    if text and len(text) < 80:
        flags.append("very_short")
    if text.endswith(("…", "...")):
        flags.append("likely_truncated")
    return tuple(flags)


def deep_merge_metadata(current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    result = dict(current)
    for key, value in incoming.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge_metadata(result[key], value)
        else:
            result[key] = value
    return result


def provenance_schema_ready(connection: sqlite3.Connection) -> bool:
    return (
        connection.execute(
            """SELECT 1 FROM sqlite_master WHERE type='table'
            AND name='paper_metadata_provenance'"""
        ).fetchone()
        is not None
    )


def current_source_priority(
    connection: sqlite3.Connection, paper_id: int, field_name: str
) -> int:
    row = connection.execute(
        """SELECT source_priority FROM paper_metadata_provenance
        WHERE paper_id=? AND field_name=? AND is_current=1 ORDER BY id DESC LIMIT 1""",
        (paper_id, field_name),
    ).fetchone()
    return int(row[0]) if row else 0


def record_provenance(
    connection: sqlite3.Connection,
    *,
    paper_id: int,
    field_name: str,
    value: Any,
    source_name: str,
    source_record_id: str | None,
    source_priority: int,
    decision: str,
    reason: str | None,
    quality_flags: tuple[str, ...] = (),
    is_current: bool,
    observed_at: str | None = None,
) -> None:
    value_hash = value_sha256(value) if not _empty(value) else None
    previous = connection.execute(
        """SELECT id FROM paper_metadata_provenance
        WHERE paper_id=? AND field_name=? AND is_current=1 ORDER BY id DESC LIMIT 1""",
        (paper_id, field_name),
    ).fetchone()
    if is_current:
        connection.execute(
            """UPDATE paper_metadata_provenance SET is_current=0
            WHERE paper_id=? AND field_name=? AND is_current=1""",
            (paper_id, field_name),
        )
    now = datetime.now(UTC).isoformat()
    connection.execute(
        """INSERT OR IGNORE INTO paper_metadata_provenance
        (paper_id,field_name,active_value_hash,source_name,source_record_id,
         source_priority,source_observed_at,fetched_at,is_current,quality_flags_json,
         decision,replacement_reason,previous_provenance_id,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            paper_id,
            field_name,
            value_hash,
            source_name,
            source_record_id,
            source_priority,
            observed_at,
            now,
            int(is_current),
            json.dumps(quality_flags),
            decision,
            reason,
            int(previous[0]) if previous else None,
            now,
        ),
    )


def inspect_paper_provenance(database: Path, paper_id: int) -> dict[str, Any]:
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [
            dict(row)
            for row in connection.execute(
                """SELECT * FROM paper_metadata_provenance WHERE paper_id=?
                ORDER BY field_name,id""",
                (paper_id,),
            ).fetchall()
        ]
    return {"paper_id": paper_id, "records": rows}


def audit_metadata_provenance(database: Path) -> dict[str, Any]:
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        if not provenance_schema_ready(connection):
            return {"status": "blocked", "reason": "provenance_schema_missing"}
        total = int(connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
        covered = int(
            connection.execute(
                "SELECT COUNT(DISTINCT paper_id) FROM paper_metadata_provenance"
            ).fetchone()[0]
        )
        decisions = {
            row["decision"]: int(row["count"])
            for row in connection.execute(
                """SELECT decision,COUNT(*) count FROM paper_metadata_provenance
                GROUP BY decision"""
            ).fetchall()
        }
        abstract_total = int(
            connection.execute(
                "SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL AND trim(abstract)!=''"
            ).fetchone()[0]
        )
    return {
        "status": "ok",
        "total_papers": total,
        "papers_with_provenance": covered,
        "missing_provenance": total - covered,
        "abstract_coverage": {"available": abstract_total, "missing": total - abstract_total},
        "decisions": decisions,
        "prevented_degradations": decisions.get("degraded_prevention", 0),
    }


def metadata_quality_report(database: Path) -> dict[str, Any]:
    report = audit_metadata_provenance(database)
    if report.get("status") != "ok":
        return report
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        abstracts = connection.execute(
            "SELECT id,abstract FROM papers WHERE abstract IS NOT NULL"
        ).fetchall()
        report["placeholder_abstracts"] = [
            int(row["id"])
            for row in abstracts
            if "placeholder" in abstract_quality_flags(row["abstract"])
        ]
        report["truncated_abstracts"] = [
            int(row["id"])
            for row in abstracts
            if "likely_truncated" in abstract_quality_flags(row["abstract"])
        ]
        report["source_distribution"] = {
            row["source_name"]: int(row["count"])
            for row in connection.execute(
                """SELECT source_name,COUNT(*) count FROM paper_metadata_provenance
                WHERE is_current=1 GROUP BY source_name"""
            ).fetchall()
        }
        report["conflicts"] = int(
            connection.execute(
                """SELECT COUNT(*) FROM paper_metadata_provenance
                WHERE decision='conflict'"""
            ).fetchone()[0]
        )
        report["online_first_transitions"] = int(
            connection.execute(
                """SELECT COUNT(*) FROM papers WHERE online_date IS NOT NULL
                AND issue_date IS NOT NULL AND (volume IS NOT NULL OR issue IS NOT NULL)"""
            ).fetchone()[0]
        )
        report["duplicate_candidates"] = [
            {
                "title": row["title"],
                "journal_id": int(row["journal_id"]),
                "publication_date": row["publication_date"],
                "paper_ids": json.loads(row["paper_ids"]),
            }
            for row in connection.execute(
                """SELECT title,journal_id,COALESCE(issue_date,online_date) publication_date,
                json_group_array(id) paper_ids FROM papers
                GROUP BY title,journal_id,COALESCE(issue_date,online_date)
                HAVING COUNT(*)>1 AND COUNT(DISTINCT COALESCE(doi,''))>1
                LIMIT 20"""
            ).fetchall()
        ]
    from eduecon_radar.services.doi_audit import audit_doi_normalization

    report["doi_anomalies"] = audit_doi_normalization(database)
    return report


def value_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _looks_truncated(incoming: str, current: str) -> bool:
    left = " ".join(incoming.split())
    right = " ".join(current.split())
    return len(left) < len(right) * 0.7 or (
        len(left) < len(right) and right.casefold().startswith(left.casefold())
    )


def _equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return " ".join(left.split()) == " ".join(right.split())
    return left == right


def value_is_empty(value: Any) -> bool:
    """True when a value carries no fact.

    The JSON containers ``[]`` and ``{}`` count as empty: a source that returned
    no keywords or no JEL codes has not asserted anything, and treating ``[]`` as
    a real observation would let it overwrite a populated list and would write a
    provenance record claiming a source supplied "no codes".
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in {"", "[]", "{}"}
    return False


#: Internal alias kept for the call sites that predate the public name.
_empty = value_is_empty
