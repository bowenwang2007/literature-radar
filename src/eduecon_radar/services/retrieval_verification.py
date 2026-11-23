from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Protocol

from eduecon_radar.db import Database, utcnow
from eduecon_radar.services.deduplication import normalize_title
from eduecon_radar.services.doi import canonicalize_doi
from eduecon_radar.services.local_pipeline import sync_run_papers
from eduecon_radar.services.migrations import require_product_safety_schema
from eduecon_radar.services.resilient_http import ResilientHttpClient


@dataclass(frozen=True)
class IndependentWork:
    source_id: str
    doi: str | None
    title: str
    publication_date: str | None
    work_type: str | None
    is_paratext: bool


@dataclass(frozen=True)
class VerificationResult:
    sync_run_id: int
    source_name: str
    status: str
    local_count: int
    independent_count: int | None
    comparable_independent_count: int | None
    exact_doi_matches: int
    possible_missing_count: int
    source_only_count: int
    local_only_count: int
    query_count: int
    duration_ms: int
    details: dict[str, object]
    error_message: str | None = None
    verified_at: str = ""


class OpenAlexVerificationSource:
    """Small, bounded OpenAlex query used only as independent evidence."""

    source_name = "OpenAlex"

    def __init__(
        self,
        api_key: str | None = None,
        client: ResilientHttpClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.http = client or ResilientHttpClient(
            timeout=20,
            max_retries=1,
            rate_limit_per_second=2,
            max_inline_sleep_seconds=1,
        )
        self.query_count = 0

    def fetch(
        self,
        journal_issns: tuple[tuple[str, ...], ...],
        since: date,
        until: date,
    ) -> tuple[IndependentWork, ...]:
        records: dict[str, IndependentWork] = {}
        for issns in journal_issns:
            usable = tuple(dict.fromkeys(value for value in issns if value))
            if not usable:
                continue
            cursor = "*"
            for _page in range(5):
                params: dict[str, object] = {
                    "filter": (
                        "primary_location.source.issn:"
                        + "|".join(usable)
                        + f",from_publication_date:{since.isoformat()}"
                        + f",to_publication_date:{until.isoformat()}"
                    ),
                    "select": (
                        "id,doi,display_name,publication_date,type,is_paratext,"
                        "primary_location"
                    ),
                    "per_page": 100,
                    "cursor": cursor,
                }
                if self.api_key:
                    params["api_key"] = self.api_key
                self.query_count += 1
                payload = self.http.get_json(
                    "https://api.openalex.org/works", params=params
                )
                items = payload.get("results") or []
                if not isinstance(items, list):
                    raise ValueError("OpenAlex results is not a list")
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    work = IndependentWork(
                        source_id=str(item.get("id") or ""),
                        doi=canonicalize_doi(item.get("doi")),
                        title=str(item.get("display_name") or "").strip(),
                        publication_date=str(item.get("publication_date") or "") or None,
                        work_type=str(item.get("type") or "") or None,
                        is_paratext=bool(item.get("is_paratext")),
                    )
                    key = work.doi or work.source_id or f"title:{normalize_title(work.title)}"
                    records[key] = work
                meta = payload.get("meta") or {}
                next_cursor = meta.get("next_cursor") if isinstance(meta, dict) else None
                if not items or len(items) < 100 or not next_cursor or next_cursor == cursor:
                    break
                cursor = str(next_cursor)
        return tuple(records.values())


class CrossrefWorkSource(Protocol):
    def fetch_work(self, doi: str) -> dict[str, Any] | None: ...


def _crossref_date(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    parts = value.get("date-parts")
    if not isinstance(parts, list) or not parts or not isinstance(parts[0], list):
        return None
    usable = [str(part) for part in parts[0][:3] if part is not None]
    if not usable:
        return None
    return "-".join(
        part.zfill(2) if index else part
        for index, part in enumerate(usable)
    )


def _within(value: str | None, since: date, until: date) -> bool:
    if not value:
        return False
    try:
        parts = [int(part) for part in value.split("-")]
        padded = [*parts, 1, 1]
        candidate = date(padded[0], padded[1], padded[2])
    except (TypeError, ValueError):
        return False
    return since <= candidate <= until


def _audit_crossref_gap(
    item: dict[str, object],
    *,
    since: date,
    until: date,
    crossref: CrossrefWorkSource,
) -> dict[str, object]:
    doi = canonicalize_doi(item.get("doi"))
    if not doi:
        return item
    try:
        record = crossref.fetch_work(doi)
    except Exception as exc:
        return {**item, "crossref_audit_error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    if not record:
        return {**item, "gap_classification": "crossref_metadata_unavailable"}
    dates = {
        "published_online": _crossref_date(record.get("published-online")),
        "published_print": _crossref_date(record.get("published-print")),
        "issued": _crossref_date(record.get("issued")),
        "created": _crossref_date(record.get("created")),
    }
    published_in_scope = any(
        _within(dates[key], since, until)
        for key in ("published_online", "published_print", "issued")
    )
    created_in_scope = _within(dates["created"], since, until)
    if published_in_scope:
        classification = "true_primary_retrieval_gap"
        reason = "Crossref 出版日期位于检索窗口内，主检索本应覆盖，需检查分页或筛选。"
    elif created_in_scope:
        classification = "date_basis_difference"
        reason = (
            "Crossref 在检索窗口内创建了记录，但出版/刊期日期落在窗口外；"
            "旧版仅按出版日期检索会漏掉，现已用创建日期补充检索修复。"
        )
    elif dates["created"] and dates["created"] > until.isoformat():
        classification = "crossref_metadata_lag"
        reason = "Crossref 在检索窗口结束后才创建记录，属于元数据时滞。"
    else:
        classification = "other"
        reason = "现有日期字段不能完全解释差异，需要人工核对来源记录。"
    return {
        **item,
        "reason": reason,
        "gap_classification": classification,
        "crossref_dates": dates,
        "crossref_title": str((record.get("title") or [""])[0]),
    }


def _is_comparable(work: IndependentWork) -> bool:
    return not work.is_paratext and (work.work_type or "").casefold() not in {
        "paratext",
        "editorial",
        "erratum",
        "correction",
    }


def _explain_source_only(
    work: IndependentWork, local_titles: set[str]
) -> tuple[str, bool]:
    if normalize_title(work.title) in local_titles:
        return "题名已匹配，标识符口径不同", False
    if not _is_comparable(work):
        return "独立来源标记为前后页或非研究内容", False
    return "独立来源存在可比记录，主检索中未找到相同 DOI", True


def _persist(database: Database, result: VerificationResult) -> None:
    require_product_safety_schema(database)
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO retrieval_verifications
            (sync_run_id,source_name,status,local_count,independent_count,
             comparable_independent_count,exact_doi_matches,possible_missing_count,
             source_only_count,local_only_count,query_count,duration_ms,detail_json,
             error_message,verified_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(sync_run_id) DO UPDATE SET
             source_name=excluded.source_name,status=excluded.status,
             local_count=excluded.local_count,independent_count=excluded.independent_count,
             comparable_independent_count=excluded.comparable_independent_count,
             exact_doi_matches=excluded.exact_doi_matches,
             possible_missing_count=excluded.possible_missing_count,
             source_only_count=excluded.source_only_count,
             local_only_count=excluded.local_only_count,
             query_count=excluded.query_count,duration_ms=excluded.duration_ms,
             detail_json=excluded.detail_json,error_message=excluded.error_message,
             verified_at=excluded.verified_at""",
            (
                result.sync_run_id,
                result.source_name,
                result.status,
                result.local_count,
                result.independent_count,
                result.comparable_independent_count,
                result.exact_doi_matches,
                result.possible_missing_count,
                result.source_only_count,
                result.local_only_count,
                result.query_count,
                result.duration_ms,
                json.dumps(result.details, ensure_ascii=False),
                result.error_message,
                result.verified_at,
            ),
        )


def verify_sync_run(
    database: Database,
    sync_run_id: int,
    *,
    journal_issns: tuple[tuple[str, ...], ...],
    source: OpenAlexVerificationSource,
    crossref: CrossrefWorkSource | None = None,
) -> VerificationResult:
    """Compare the persisted primary result with an on-demand independent index."""
    require_product_safety_schema(database)
    with database.connect() as connection:
        run = connection.execute(
            "SELECT query_start_date,query_end_date FROM sync_runs WHERE id=?",
            (sync_run_id,),
        ).fetchone()
    if run is None or not run["query_start_date"] or not run["query_end_date"]:
        raise ValueError("同步记录缺少可核验的日期范围")
    local = sync_run_papers(database, sync_run_id)
    started = time.perf_counter()
    verified_at = utcnow()
    try:
        independent = source.fetch(
            journal_issns,
            date.fromisoformat(str(run["query_start_date"])),
            date.fromisoformat(str(run["query_end_date"])),
        )
    except Exception as exc:
        result = VerificationResult(
            sync_run_id=sync_run_id,
            source_name=source.source_name,
            status="unavailable",
            local_count=len(local),
            independent_count=None,
            comparable_independent_count=None,
            exact_doi_matches=0,
            possible_missing_count=0,
            source_only_count=0,
            local_only_count=0,
            query_count=source.query_count,
            duration_ms=round((time.perf_counter() - started) * 1000),
            details={"date_basis": "OpenAlex publication_date；主检索为 Crossref pub-date"},
            error_message=f"{type(exc).__name__}: {str(exc)[:300]}",
            verified_at=verified_at,
        )
        _persist(database, result)
        return result

    local_by_doi = {
        doi: paper for paper in local if (doi := canonicalize_doi(paper.get("doi")))
    }
    independent_by_doi = {work.doi: work for work in independent if work.doi}
    local_titles = {normalize_title(str(paper.get("title") or "")) for paper in local}
    exact = set(local_by_doi) & set(independent_by_doi)
    unmatched_local = [
        paper
        for paper in local
        if canonicalize_doi(paper.get("doi")) not in exact
    ]
    unmatched_independent = [work for work in independent if work.doi not in exact]

    # DOI is the primary identity.  A unique record can still be compared when
    # one source omits or disagrees on DOI, so pair the remaining records by a
    # normalized title before declaring either side unmatched.
    local_by_title: dict[str, list[dict[str, object]]] = {}
    for paper in unmatched_local:
        title_key = normalize_title(str(paper.get("title") or ""))
        if title_key:
            local_by_title.setdefault(title_key, []).append(paper)
    title_matches: list[dict[str, object]] = []
    source_only: list[IndependentWork] = []
    title_matched_local_ids: set[int] = set()
    for work in unmatched_independent:
        title_key = normalize_title(work.title)
        candidates = local_by_title.get(title_key) if title_key else None
        if candidates:
            paper = candidates.pop(0)
            title_matched_local_ids.add(id(paper))
            title_matches.append(
                {
                    "title": work.title,
                    "local_doi": canonicalize_doi(paper.get("doi")),
                    "independent_doi": work.doi,
                    "is_comparable": _is_comparable(work),
                    "reason": "题名一致，DOI 缺失或标识符口径不同",
                }
            )
        else:
            source_only.append(work)
    local_only = [
        paper for paper in unmatched_local if id(paper) not in title_matched_local_ids
    ]
    explained = []
    possible: list[dict[str, object]] = []
    for work in source_only:
        reason, is_possible = _explain_source_only(work, local_titles)
        item = {**asdict(work), "reason": reason}
        (possible if is_possible else explained).append(item)
    date_audit_queries = 0
    if crossref is not None and possible:
        audited = []
        for item in possible:
            if canonicalize_doi(item.get("doi")):
                date_audit_queries += 1
            audited.append(
                _audit_crossref_gap(
                    item,
                    since=date.fromisoformat(str(run["query_start_date"])),
                    until=date.fromisoformat(str(run["query_end_date"])),
                    crossref=crossref,
                )
            )
        possible = audited
    comparable_count = sum(1 for work in independent if _is_comparable(work))
    # Every comparable independent record is either confirmed by DOI/title (or
    # explained as duplicate metadata) or remains a possible gap.  Persist the
    # user-facing partition explicitly so its numbers always add up.
    confirmed_overlap_count = max(0, comparable_count - len(possible))
    if local and not independent:
        # An empty secondary index cannot substantiate primary completeness.
        status = "unavailable"
    elif possible:
        status = "possible_gaps"
    elif (
        not source_only
        and not local_only
        and len(exact) == len(local) == len(independent)
    ):
        status = "consistent"
    else:
        status = "broadly_consistent"
    result = VerificationResult(
        sync_run_id=sync_run_id,
        source_name=source.source_name,
        status=status,
        local_count=len(local),
        independent_count=len(independent),
        comparable_independent_count=comparable_count,
        exact_doi_matches=len(exact),
        possible_missing_count=len(possible),
        source_only_count=len(source_only),
        local_only_count=len(local_only),
        query_count=source.query_count + date_audit_queries,
        duration_ms=round((time.perf_counter() - started) * 1000),
        details={
            "date_basis": (
                "OpenAlex publication_date；当前主检索联合 Crossref pub-date "
                "与 created-date（旧运行可能仅使用 pub-date）"
            ),
            "possible_missing": possible,
            "explained_source_only": explained,
            "title_matches": title_matches,
            "confirmed_overlap_count": confirmed_overlap_count,
            "insufficient_independent_evidence": bool(local and not independent),
            "local_only": [
                {
                    "doi": canonicalize_doi(paper.get("doi")),
                    "title": paper.get("title"),
                    "publication_date": paper.get("publication_date"),
                    "reason": "OpenAlex 未收录、日期口径不同或记录尚未更新",
                }
                for paper in local_only
            ],
        },
        verified_at=verified_at,
    )
    _persist(database, result)
    return result


def cached_verification(
    database: Database, sync_run_id: int
) -> VerificationResult | None:
    require_product_safety_schema(database)
    with database.connect() as connection:
        row = connection.execute(
            "SELECT * FROM retrieval_verifications WHERE sync_run_id=?",
            (sync_run_id,),
        ).fetchone()
    if row is None:
        return None
    return VerificationResult(
        sync_run_id=int(row["sync_run_id"]),
        source_name=str(row["source_name"]),
        status=str(row["status"]),
        local_count=int(row["local_count"]),
        independent_count=(
            int(row["independent_count"])
            if row["independent_count"] is not None
            else None
        ),
        comparable_independent_count=(
            int(row["comparable_independent_count"])
            if row["comparable_independent_count"] is not None
            else None
        ),
        exact_doi_matches=int(row["exact_doi_matches"]),
        possible_missing_count=int(row["possible_missing_count"]),
        source_only_count=int(row["source_only_count"]),
        local_only_count=int(row["local_only_count"]),
        query_count=int(row["query_count"]),
        duration_ms=int(row["duration_ms"]),
        details=json.loads(row["detail_json"] or "{}"),
        error_message=str(row["error_message"]) if row["error_message"] else None,
        verified_at=str(row["verified_at"]),
    )
