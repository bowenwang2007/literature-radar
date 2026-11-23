from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import date

from eduecon_radar.db import Database, utcnow
from eduecon_radar.models import ReviewStatus
from eduecon_radar.services.migrations import (
    translation_schema_ready,
    workflow_schema_ready,
)


@dataclass(frozen=True)
class CandidateFilters:
    journal: str | None = None
    decision: str | None = None
    status: str | None = None
    minimum_confidence: float | None = None
    date_from: date | None = None
    date_to: date | None = None
    has_abstract: bool | None = None
    has_open_access: bool | None = None


@dataclass(frozen=True)
class PaperFilters:
    journals: tuple[str, ...] = ()
    ai_statuses: tuple[str, ...] = ()
    review_statuses: tuple[str, ...] = ()
    date_from: date | None = None
    date_to: date | None = None
    search: str | None = None
    has_abstract: bool | None = None
    has_title_translation: bool | None = None
    has_abstract_translation: bool | None = None
    batch_id: int | None = None
    sort_by: str = "publication_desc"


@dataclass(frozen=True)
class PaperPage:
    items: list[dict]
    total: int
    page: int
    page_size: int

    @property
    def total_pages(self) -> int:
        return max(1, (self.total + self.page_size - 1) // self.page_size)


LATEST_SCREENING_JOIN = """
LEFT JOIN screenings s ON s.id=(
    SELECT s2.id FROM screenings s2 WHERE s2.paper_id=p.id
    ORDER BY s2.screened_at DESC,s2.id DESC LIMIT 1
)
"""

AI_STATUS_SQL = """CASE
    WHEN s.id IS NULL THEN 'awaiting_screening'
    WHEN s.ai_decision='unprocessed' THEN 'screening_failed'
    WHEN s.ai_decision='exclude' THEN 'ai_excluded'
    WHEN s.ai_decision='include' THEN 'ai_included'
    WHEN s.ai_decision='borderline' THEN 'ai_borderline'
    ELSE 'unknown'
END"""

AI_STATUSES = {
    "awaiting_screening",
    "screening_failed",
    "ai_excluded",
    "ai_included",
    "ai_borderline",
}

SORT_ORDERS = {
    "publication_desc": """COALESCE(p.issue_date,p.online_date) IS NULL,
        COALESCE(p.issue_date,p.online_date) DESC,p.id DESC""",
    "screening_desc": "s.screened_at IS NULL,s.screened_at DESC,p.id DESC",
    "journal": "j.name ASC,COALESCE(p.issue_date,p.online_date) DESC,p.id DESC",
}


def dashboard_stats(database: Database) -> dict[str, object]:
    with database.connect() as connection:
        row = connection.execute(
            f"""SELECT
            SUM(CASE WHEN s.id IS NULL THEN 1 ELSE 0 END) awaiting_screening,
            SUM(CASE WHEN s.ai_decision='unprocessed' THEN 1 ELSE 0 END) screening_failed,
            SUM(CASE WHEN s.ai_decision='exclude' THEN 1 ELSE 0 END) ai_excluded,
            SUM(CASE WHEN s.ai_decision='include' THEN 1 ELSE 0 END) ai_included,
            SUM(CASE WHEN s.ai_decision='borderline' THEN 1 ELSE 0 END) ai_borderline,
            SUM(CASE WHEN s.ai_decision IN ('include','borderline')
                AND r.status='pending' THEN 1 ELSE 0 END) review_pending,
            SUM(CASE WHEN r.status='approved' THEN 1 ELSE 0 END) approved,
            SUM(CASE WHEN r.status='rejected' THEN 1 ELSE 0 END) rejected,
            SUM(CASE WHEN r.status='needs_revision' THEN 1 ELSE 0 END) needs_revision,
            COUNT(*) total
            FROM papers p {LATEST_SCREENING_JOIN} LEFT JOIN reviews r ON r.paper_id=p.id"""
        ).fetchone()
        recent = connection.execute(
            """SELECT finished_at,records_inserted,records_failed,ai_calls,input_tokens,
            output_tokens,estimated_cost FROM sync_runs ORDER BY id DESC LIMIT 1"""
        ).fetchone()
        failure_rows = connection.execute(
            f"""SELECT s.ai_reason FROM papers p {LATEST_SCREENING_JOIN}
            WHERE s.ai_decision='unprocessed'"""
        ).fetchall()
    # sqlite3.Row iteration yields values, so keys() is required here.
    stats = {key: int(row[key] or 0) for key in row.keys()} if row else {}  # noqa: SIM118
    failure_counts = Counter(classify_screening_failure(item["ai_reason"]) for item in failure_rows)
    stats["screening_failure_reasons"] = dict(sorted(failure_counts.items()))
    stats.update(dict(recent) if recent else {})
    return stats


def classify_screening_failure(reason: str | None) -> str:
    if not reason:
        return "unknown"
    normalized = reason.casefold()
    if any(
        marker in normalized
        for marker in (
            "invalid structured ai response",
            "validation error",
            "invalid json",
            "jsondecodeerror",
        )
    ):
        return "validation_error"
    if any(
        marker in normalized
        for marker in ("missing abstract", "abstract missing", "no abstract available")
    ):
        return "missing_abstract"
    if "ai disabled" in normalized:
        return "ai_disabled"
    if any(
        marker in normalized
        for marker in (
            "api error",
            "http error",
            "status code",
            "bad request",
            "rate limit",
            "timed out",
            "timeout",
            "connection error",
            "connection failed",
        )
    ):
        return "api_error"
    return "unknown"


def list_all_papers(
    database: Database,
    filters: PaperFilters | None = None,
    *,
    page: int = 1,
    page_size: int = 20,
) -> PaperPage:
    filters = filters or PaperFilters()
    if page < 1:
        raise ValueError("page must be at least 1")
    if page_size < 1:
        raise ValueError("page_size must be at least 1")
    if filters.sort_by not in SORT_ORDERS:
        raise ValueError(f"Invalid sort order: {filters.sort_by}")
    unknown_ai_statuses = set(filters.ai_statuses) - AI_STATUSES
    if unknown_ai_statuses:
        raise ValueError(f"Invalid AI statuses: {sorted(unknown_ai_statuses)}")
    valid_review_statuses = {item.value for item in ReviewStatus}
    unknown_review_statuses = set(filters.review_statuses) - valid_review_statuses
    if unknown_review_statuses:
        raise ValueError(f"Invalid review statuses: {sorted(unknown_review_statuses)}")

    has_translation_schema = translation_schema_ready(database)
    has_workflow_schema = workflow_schema_ready(database)
    conditions = ["1=1"]
    params: list[object] = []
    if filters.journals:
        placeholders = ",".join("?" for _ in filters.journals)
        conditions.append(f"j.short_name IN ({placeholders})")
        params.extend(filters.journals)
    if filters.ai_statuses:
        placeholders = ",".join("?" for _ in filters.ai_statuses)
        conditions.append(f"{AI_STATUS_SQL} IN ({placeholders})")
        params.extend(filters.ai_statuses)
    if filters.review_statuses:
        placeholders = ",".join("?" for _ in filters.review_statuses)
        conditions.append(f"r.status IN ({placeholders})")
        params.extend(filters.review_statuses)
    if filters.date_from:
        conditions.append("COALESCE(p.issue_date,p.online_date)>=?")
        params.append(filters.date_from.isoformat())
    if filters.date_to:
        conditions.append("COALESCE(p.issue_date,p.online_date)<=?")
        params.append(filters.date_to.isoformat())
    if filters.search and filters.search.strip():
        search = f"%{filters.search.strip()}%"
        searchable = [
            "p.title LIKE ?",
            "COALESCE(p.abstract,'') LIKE ?",
            "COALESCE(p.authors_json,'') LIKE ?",
            "COALESCE(p.doi,'') LIKE ?",
        ]
        if has_translation_schema:
            searchable.append("COALESCE(pt.chinese_title,'') LIKE ?")
        conditions.append(f"({' OR '.join(searchable)})")
        params.extend(search for _ in searchable)
    if filters.has_abstract is not None:
        conditions.append(
            "p.abstract IS NOT NULL" if filters.has_abstract else "p.abstract IS NULL"
        )
    translation_filters = (
        (filters.has_title_translation, "pt.chinese_title"),
        (filters.has_abstract_translation, "pt.chinese_abstract"),
    )
    for requested, field in translation_filters:
        if requested is None:
            continue
        if not has_translation_schema:
            conditions.append("0=1" if requested else "1=1")
        elif requested:
            conditions.append(f"NULLIF(TRIM({field}),'') IS NOT NULL")
        else:
            conditions.append(f"NULLIF(TRIM({field}),'') IS NULL")
    if filters.batch_id is not None:
        if not has_workflow_schema:
            conditions.append("0=1")
        else:
            conditions.append(
                "EXISTS (SELECT 1 FROM workflow_batch_items wbi "
                "WHERE wbi.paper_id=p.id AND wbi.batch_id=?)"
            )
            params.append(filters.batch_id)

    if has_translation_schema:
        translation_join = "LEFT JOIN paper_translations pt ON pt.paper_id=p.id"
        translation_fields = """pt.id formal_translation_id,
            pt.chinese_title formal_chinese_title,
            pt.chinese_abstract formal_chinese_abstract,
            pt.title_translation_status,pt.abstract_translation_status,
            pt.title_translation_source,pt.abstract_translation_source,
            pt.model_name translation_model_name,
            pt.prompt_version translation_prompt_version,
            pt.translated_at translation_translated_at,
            pt.updated_at translation_updated_at"""
    else:
        translation_join = ""
        translation_fields = """NULL formal_translation_id,
            NULL formal_chinese_title,NULL formal_chinese_abstract,
            NULL title_translation_status,NULL abstract_translation_status,
            NULL title_translation_source,NULL abstract_translation_source,
            NULL translation_model_name,NULL translation_prompt_version,
            NULL translation_translated_at,NULL translation_updated_at"""
    if has_workflow_schema:
        task_names_field = """(SELECT GROUP_CONCAT(wb.name, '；')
            FROM workflow_batch_items wbi
            JOIN workflow_batches wb ON wb.id=wbi.batch_id
            WHERE wbi.paper_id=p.id) task_names"""
    else:
        task_names_field = "NULL task_names"
    from_sql = f"""FROM papers p JOIN journals j ON j.id=p.journal_id
        {LATEST_SCREENING_JOIN} LEFT JOIN reviews r ON r.paper_id=p.id {translation_join}
        WHERE {" AND ".join(conditions)}"""
    with database.connect() as connection:
        total = int(connection.execute(f"SELECT COUNT(*) {from_sql}", params).fetchone()[0])
        total_pages = max(1, (total + page_size - 1) // page_size)
        effective_page = min(page, total_pages)
        offset = (effective_page - 1) * page_size
        rows = connection.execute(
            f"""SELECT p.id,p.title,p.abstract,p.doi,p.authors_json,p.source_url,
            p.source_metadata_json,
            p.open_access_url,
            j.name journal_name,j.short_name journal,
            COALESCE(p.issue_date,p.online_date) publication_date,
            {AI_STATUS_SQL} ai_status,s.id screening_id,s.ai_decision,s.ai_confidence,s.ai_reason,
            s.rule_score,s.chinese_title ai_chinese_title,
            s.research_question ai_research_question,s.data_and_sample ai_data_and_sample,
            s.method ai_method,s.identification_strategy ai_identification_strategy,
            s.main_result ai_main_result,s.chinese_summary ai_chinese_summary,
            s.topics_json,s.missing_information_json,s.model_name,s.screened_at,
            r.status review_status,r.reviewer_note,r.reviewed_at,
            r.edited_chinese_title,r.edited_chinese_summary,{translation_fields},
            {task_names_field}
            {from_sql} ORDER BY {SORT_ORDERS[filters.sort_by]} LIMIT ? OFFSET ?""",
            (*params, page_size, offset),
        ).fetchall()
    return PaperPage(
        items=[_decode_all_paper(row) for row in rows],
        total=total,
        page=effective_page,
        page_size=page_size,
    )


def matching_paper_ids(database: Database, filters: PaperFilters | None = None) -> list[int]:
    result = list_all_papers(database, filters, page=1, page_size=1_000_000)
    return [int(item["id"]) for item in result.items]


def _decode_all_paper(row: sqlite3.Row) -> dict:
    item = dict(row)
    item["authors"] = json.loads(item.pop("authors_json") or "[]")
    item["missing_information"] = json.loads(item.pop("missing_information_json") or "[]")
    item["ai_topics"] = json.loads(item.pop("topics_json") or "[]")
    metadata = json.loads(item.pop("source_metadata_json") or "{}")
    item["abstract_provenance"] = metadata.get("abstract_provenance") or {
        "source": metadata.get("abstract_source")
    }
    publication_date = item.get("publication_date")
    item["publication_year"] = publication_date[:4] if publication_date else None
    return item


def set_review_status(
    database: Database, paper_id: int, status: str, *, note: str | None = None
) -> None:
    allowed = {
        ReviewStatus.APPROVED.value,
        ReviewStatus.REJECTED.value,
        ReviewStatus.NEEDS_REVISION.value,
    }
    if status not in allowed:
        raise ValueError(f"Invalid review action: {status}")
    with database.connect() as connection:
        if note is None:
            cursor = connection.execute(
                "UPDATE reviews SET status=?,reviewed_at=? WHERE paper_id=?",
                (status, utcnow(), paper_id),
            )
        else:
            cursor = connection.execute(
                """UPDATE reviews SET status=?,reviewer_note=?,reviewed_at=?
                WHERE paper_id=?""",
                (status, note.strip() or None, utcnow(), paper_id),
            )
        if cursor.rowcount != 1:
            raise ValueError(f"Paper not found: {paper_id}")


def list_candidates(database: Database, filters: CandidateFilters) -> list[dict]:
    conditions = ["1=1"]
    params: list[object] = []
    mapping = {
        "journal": ("j.short_name=?", filters.journal),
        "decision": ("s.ai_decision=?", filters.decision),
        "status": ("r.status=?", filters.status),
    }
    for condition, value in mapping.values():
        if value:
            conditions.append(condition)
            params.append(value)
    if filters.minimum_confidence is not None:
        conditions.append("s.ai_confidence>=?")
        params.append(filters.minimum_confidence)
    if filters.date_from:
        conditions.append("COALESCE(p.issue_date,p.online_date)>=?")
        params.append(filters.date_from.isoformat())
    if filters.date_to:
        conditions.append("COALESCE(p.issue_date,p.online_date)<=?")
        params.append(filters.date_to.isoformat())
    if filters.has_abstract is not None:
        conditions.append(
            "p.abstract IS NOT NULL" if filters.has_abstract else "p.abstract IS NULL"
        )
    if filters.has_open_access is not None:
        conditions.append(
            "p.open_access_url IS NOT NULL"
            if filters.has_open_access
            else "p.open_access_url IS NULL"
        )
    sql = f"""SELECT p.id,j.short_name journal,p.title,p.authors_json,
        COALESCE(p.issue_date,p.online_date) paper_date,s.ai_decision,s.ai_confidence,
        s.ai_reason,s.topics_json,r.status
        FROM papers p JOIN journals j ON j.id=p.journal_id
        {LATEST_SCREENING_JOIN} JOIN reviews r ON r.paper_id=p.id
        WHERE {" AND ".join(conditions)} ORDER BY p.first_seen_at DESC"""
    with database.connect() as connection:
        rows = connection.execute(sql, params).fetchall()
    return [_decode_candidate(row) for row in rows]


def _decode_candidate(row: sqlite3.Row) -> dict:
    item = dict(row)
    item["authors"] = json.loads(item.pop("authors_json"))
    item["topics"] = json.loads(item.pop("topics_json") or "[]")
    return item


def get_paper_detail(database: Database, paper_id: int) -> dict | None:
    with database.connect() as connection:
        row = connection.execute(
            f"""SELECT p.*,j.name journal_name,j.short_name journal_short_name,
            s.rule_score,s.rule_hits_json,s.ai_decision,s.ai_confidence,s.ai_reason,
            s.topics_json,s.education_level,s.country_or_region,s.method,
            s.research_question,s.data_and_sample,s.identification_strategy,s.main_result,
            s.heterogeneity,s.policy_implication,s.chinese_title,s.chinese_summary,
            s.missing_information_json,s.model_name,s.prompt_version,r.status,r.reviewer_note,
            r.edited_chinese_title,r.edited_chinese_summary,r.verified_numbers,
            r.verified_causal_language,r.reviewed_at
            FROM papers p JOIN journals j ON j.id=p.journal_id
            {LATEST_SCREENING_JOIN} JOIN reviews r ON r.paper_id=p.id WHERE p.id=?""",
            (paper_id,),
        ).fetchone()
    if not row:
        return None
    item = dict(row)
    for field in (
        "authors_json",
        "keywords_json",
        "jel_codes_json",
        "source_metadata_json",
        "rule_hits_json",
        "topics_json",
        "missing_information_json",
    ):
        item[field.removesuffix("_json")] = json.loads(item.pop(field) or "[]")
    return item


def save_review(
    database: Database,
    paper_id: int,
    *,
    status: str,
    note: str,
    chinese_title: str,
    chinese_summary: str,
    verified_numbers: bool,
    verified_causal_language: bool,
) -> None:
    if status not in {item.value for item in ReviewStatus}:
        raise ValueError(f"Invalid review status: {status}")
    if status == ReviewStatus.APPROVED and not (verified_numbers and verified_causal_language):
        raise ValueError("批准前必须完成关键数字和因果措辞核验")
    with database.connect() as connection:
        cursor = connection.execute(
            """UPDATE reviews SET status=?,reviewer_note=?,edited_chinese_title=?,
            edited_chinese_summary=?,verified_numbers=?,verified_causal_language=?,reviewed_at=?
            WHERE paper_id=?""",
            (
                status,
                note or None,
                chinese_title or None,
                chinese_summary or None,
                int(verified_numbers),
                int(verified_causal_language),
                utcnow(),
                paper_id,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"Paper not found: {paper_id}")
