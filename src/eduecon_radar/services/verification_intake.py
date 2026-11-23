"""Phase 7C4 — verification-candidate intake into the formal product chain.

A cached retrieval-verification result may name papers that the historical
primary sync missed. This module turns such a candidate into a real library
paper on explicit user confirmation, then attaches it to the same per-paper
processing chain used by normal retrieval (title translation, abstract
enrichment, abstract translation, automatic screening, human review).

Invariants:
* DOI deduplication always re-reads the current database; a stale verification
  snapshot is only a display cache, never an authority.
* Historical sync facts (``sync_runs.matched_paper_ids`` and the inserted /
  updated id lists) are never rewritten.
* A previously removed paper is only re-admitted after the user explicitly
  confirms; the tombstone archive is preserved and marked ``restored_at``.
* Partial failures never roll back the already-persisted paper; each step is
  recorded on ``paper_processing_status`` for a later targeted retry.
* Ordinary page renders only call the read-only state helper; every provider
  and AI call happens inside an explicit user-triggered intake action.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from eduecon_radar.db import Database, utcnow
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.doi import canonicalize_doi
from eduecon_radar.services.journal_registry import JournalRegistry
from eduecon_radar.services.local_pipeline import (
    LocalPipelineServices,
    ensure_paper_statuses_for_paper_ids,
    process_batch,
    translate_titles_for_paper_ids,
)
from eduecon_radar.services.paper_lifecycle import (
    restore_paper_tombstone,
)

LOGGER = logging.getLogger(__name__)

INTAKE_SOURCE = "retrieval_verification"
DISCOVERED_BY = "OpenAlex"


class CandidateIntakeError(RuntimeError):
    """Raised when a candidate cannot enter the product chain."""


class CandidateMissingIdentity(CandidateIntakeError):
    """No resolvable DOI after Crossref/OpenAlex metadata lookup."""


class CandidatePreviouslyRemoved(CandidateIntakeError):
    """The paper was removed before; explicit user confirmation is required."""

    def __init__(
        self,
        message: str,
        *,
        tombstone_id: int,
        has_human_data: bool,
        removed_at: str | None,
    ) -> None:
        super().__init__(message)
        self.tombstone_id = tombstone_id
        self.has_human_data = has_human_data
        self.removed_at = removed_at


class JournalResolutionError(CandidateIntakeError):
    """The candidate could not be attributed to one tracked journal."""


@dataclass(frozen=True)
class CandidateIntakeState:
    """Read-only, DB-only state used by the verification cards."""

    doi: str | None
    title: str | None
    paper_id: int | None
    in_library: bool
    attached_to_run: bool
    processing_status: str | None
    stage: str | None
    human_decision: str | None
    classification_decision: str | None
    tombstoned: bool
    tombstone_has_human_data: bool = False
    tombstone_removed_at: str | None = None

    @property
    def review_ready(self) -> bool:
        return bool(
            self.attached_to_run
            and self.processing_status == "classified"
            and (self.human_decision or "undecided") == "undecided"
        )


@dataclass(frozen=True)
class IntakeStep:
    step: str
    status: str  # done | skipped | pending | failed
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationIntakeResult:
    outcome: str  # created | restored | attached_existing | already_in_library
    paper_id: int
    doi: str | None
    title: str | None
    steps: tuple[IntakeStep, ...] = field(default_factory=tuple)
    review_ready: bool = False
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def entered_chain(self) -> bool:
        return self.outcome in {"created", "restored", "attached_existing"}


class CrossrefWorkClient(Protocol):
    def fetch_work(self, doi: str) -> dict[str, Any] | None: ...


class OpenAlexWorkClient(Protocol):
    def fetch_by_doi(self, doi: str) -> PaperMetadata | None: ...

    def fetch_by_id(self, source_id: str) -> PaperMetadata | None: ...


def _candidate_text(candidate: dict[str, Any], key: str) -> str | None:
    value = candidate.get(key)
    if value is None:
        return None
    return str(value).strip() or None


def _crossref_issns(record: dict[str, Any] | None) -> tuple[str, ...]:
    if not record:
        return ()
    raw = record.get("ISSN") or []
    if isinstance(raw, str):
        raw = [raw]
    return tuple(dict.fromkeys(str(item) for item in raw if item))


def _openalex_issns(work: dict[str, Any] | None) -> tuple[str, ...]:
    if not work:
        return ()
    source = (work.get("primary_location") or {}).get("source") or {}
    value = source.get("issn") or source.get("issn_l")
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(dict.fromkeys(str(item) for item in value if item))


def _safe_provider_failure(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc).replace('Authorization', '[redacted]')[:300]}"


def candidate_intake_state(
    database: Database,
    run_id: int,
    candidate: dict[str, Any],
) -> CandidateIntakeState:
    """Compute the current card state from the database only (no providers)."""
    return candidate_intake_states(database, run_id, (candidate,))[0]


def candidate_intake_states(
    database: Database,
    run_id: int,
    candidates: Sequence[dict[str, Any]],
) -> tuple[CandidateIntakeState, ...]:
    """Read-only batch card state using a single database connection.

    The verification page may render several candidates at once; this avoids
    one connection per candidate while keeping the exact same semantics as the
    per-candidate helper.  Malformed cache entries degrade to a neutral state
    instead of taking down the whole verification section.
    """
    normalized = [dict(item) for item in candidates]
    dois = tuple(
        dict.fromkeys(
            doi for item in normalized if (doi := canonicalize_doi(item.get("doi")))
        )
    )
    with database.connect() as connection:
        papers_by_doi: dict[str, Any] = {}
        tombstones_by_doi: dict[str, Any] = {}
        status_by_paper: dict[int, Any] = {}
        if dois:
            placeholders = ",".join("?" for _ in dois)
            papers_by_doi = {
                str(row["doi"]): row
                for row in connection.execute(
                    f"SELECT id,doi,title FROM papers WHERE doi IN ({placeholders})",
                    dois,
                ).fetchall()
            }
            try:
                tombstone_rows = connection.execute(
                    f"""SELECT id,doi,has_human_data,removed_at,restored_at
                    FROM paper_tombstones WHERE doi IN ({placeholders})""",
                    dois,
                ).fetchall()
            except sqlite3.OperationalError:
                # Pre-0012 databases have no restored_at column; treat every
                # tombstone as active and disable the restore affordance.
                tombstone_rows = connection.execute(
                    f"""SELECT id,doi,has_human_data,removed_at,NULL restored_at
                    FROM paper_tombstones WHERE doi IN ({placeholders})""",
                    dois,
                ).fetchall()
            tombstones_by_doi = {str(row["doi"]): row for row in tombstone_rows}
        paper_ids = tuple(int(row["id"]) for row in papers_by_doi.values())
        if paper_ids:
            placeholders = ",".join("?" for _ in paper_ids)
            status_by_paper = {
                int(row["paper_id"]): row
                for row in connection.execute(
                    f"""SELECT paper_id,status,stage,human_decision,classification_json
                    FROM paper_processing_status
                    WHERE sync_run_id=? AND paper_id IN ({placeholders})""",
                    (run_id, *paper_ids),
                ).fetchall()
            }
    results: list[CandidateIntakeState] = []
    for candidate in normalized:
        try:
            doi = canonicalize_doi(candidate.get("doi"))
            paper = papers_by_doi.get(doi) if doi else None
            tombstone = tombstones_by_doi.get(doi) if doi else None
            paper_id = int(paper["id"]) if paper else None
            attached = (
                status_by_paper.get(paper_id) if paper_id is not None else None
            )
            tombstone_active = bool(tombstone) and not tombstone["restored_at"]
            classification = (
                json.loads(attached["classification_json"] or "{}")
                if attached
                else {}
            )
            classification = (
                classification if isinstance(classification, dict) else {}
            )
            results.append(
                CandidateIntakeState(
                    doi=doi,
                    title=_candidate_text(candidate, "title"),
                    paper_id=paper_id,
                    in_library=paper_id is not None,
                    attached_to_run=attached is not None,
                    processing_status=str(attached["status"]) if attached else None,
                    stage=str(attached["stage"]) if attached else None,
                    human_decision=str(attached["human_decision"])
                    if attached
                    else None,
                    classification_decision=(
                        str(classification.get("decision") or "") if attached else None
                    ),
                    tombstoned=tombstone_active,
                    tombstone_has_human_data=bool(
                        tombstone and tombstone["has_human_data"]
                    ),
                    tombstone_removed_at=(
                        str(tombstone["removed_at"]) if tombstone else None
                    ),
                )
            )
        except Exception as exc:
            LOGGER.warning(
                "verification intake card state skipped malformed candidate: %s",
                exc,
            )
            results.append(
                CandidateIntakeState(
                    doi=canonicalize_doi(candidate.get("doi")),
                    title=_candidate_text(candidate, "title"),
                    paper_id=None,
                    in_library=False,
                    attached_to_run=False,
                    processing_status=None,
                    stage=None,
                    human_decision=None,
                    classification_decision=None,
                    tombstoned=False,
                )
            )
    return tuple(results)


def _resolve_journal(
    database: Database,
    registry: JournalRegistry,
    run: dict[str, Any],
    *,
    crossref_record: dict[str, Any] | None,
    openalex_work: dict[str, Any] | None,
    doi: str,
) -> tuple[int, str]:
    selected_keys = run.get("selected_journal_keys") or []
    if isinstance(selected_keys, str):
        try:
            selected_keys = json.loads(selected_keys or "[]")
        except (TypeError, ValueError):
            selected_keys = []
    selected = {
        str(key)
        for key in (selected_keys if isinstance(selected_keys, Sequence) else [])
    }
    candidates: list[tuple[int, Any]] = []
    issns = tuple(
        dict.fromkeys((*_crossref_issns(crossref_record), *_openalex_issns(openalex_work)))
    )
    for issn in issns:
        try:
            journal = registry.resolve(issn)
        except Exception:
            journal = None
        if journal is not None:
            candidates.append((1 if journal.journal_key in selected else 0, journal))
    if candidates:
        candidates.sort(key=lambda item: (-item[0], item[1].canonical_name))
        journal = candidates[0][1]
        journal_id = database.upsert_journal(journal.as_journal_config())
        return journal_id, journal.canonical_name
    # Fall back to an unambiguous run-level selection: candidates from an
    # independent index normally belong to the journal that was verified.
    if len(selected) == 1:
        try:
            journal = registry.by_key(next(iter(selected)))
        except KeyError as exc:
            raise JournalResolutionError(
                f"无法确认 DOI {doi} 的期刊归属：核验记录中的期刊 "
                f"({next(iter(selected))}) 不在当前期刊配置中。"
            ) from exc
        journal_id = database.upsert_journal(journal.as_journal_config())
        return journal_id, journal.canonical_name
    raise JournalResolutionError(
        f"无法确认 DOI {doi} 的期刊归属：Crossref/OpenAlex 均未返回 ISSN，"
        "且本次核验涉及多本期刊。"
    )


def _merge_dates(
    crossref_paper: PaperMetadata | None,
    openalex_paper: PaperMetadata | None,
    publication_date: str | None,
) -> tuple[date | None, date | None]:
    online = crossref_paper.online_date if crossref_paper else None
    issue = crossref_paper.issue_date if crossref_paper else None
    if openalex_paper is not None and openalex_paper.issue_date is not None:
        if issue is None:
            issue = openalex_paper.issue_date
        elif online is None:
            online = issue
            issue = openalex_paper.issue_date
    if issue is None and online is None and publication_date:
        try:
            issue = date.fromisoformat(publication_date[:10])
        except ValueError:
            issue = None
    return online, issue


def _build_paper_metadata(
    candidate: dict[str, Any],
    *,
    doi: str,
    crossref_record: dict[str, Any] | None,
    openalex_work: dict[str, Any] | None,
    run_id: int,
    journal_name: str,
    restored: bool,
    metadata_warnings: tuple[str, ...],
) -> PaperMetadata:
    from eduecon_radar.services.crossref import parse_crossref_item
    from eduecon_radar.services.openalex import parse_openalex_work

    crossref_paper = (
        parse_crossref_item(crossref_record) if crossref_record else None
    )
    openalex_paper = parse_openalex_work(openalex_work) if openalex_work else None
    candidate_title = _candidate_text(candidate, "title") or ""
    title = (
        crossref_paper.title
        if crossref_paper and crossref_paper.title.strip()
        else openalex_paper.title
        if openalex_paper and openalex_paper.title.strip()
        else candidate_title
    )
    authors = (
        crossref_paper.authors
        if crossref_paper and crossref_paper.authors
        else openalex_paper.authors
        if openalex_paper and openalex_paper.authors
        else []
    )
    abstract = (
        crossref_paper.abstract
        if crossref_paper and crossref_paper.abstract
        else openalex_paper.abstract
        if openalex_paper
        else None
    )
    online_date, issue_date = _merge_dates(
        crossref_paper,
        openalex_paper,
        _candidate_text(candidate, "publication_date"),
    )
    source_metadata = dict(crossref_record or {})
    if openalex_work:
        source_metadata["openalex_record"] = openalex_work
    source_metadata["eduecon_verification_intake"] = {
        "source": INTAKE_SOURCE,
        "discovered_by": DISCOVERED_BY,
        "verification_sync_run_id": run_id,
        "accepted_at": utcnow(),
        "candidate": {
            key: candidate.get(key)
            for key in (
                "doi",
                "title",
                "publication_date",
                "work_type",
                "reason",
                "gap_classification",
            )
        },
        "metadata_sources": [
            name
            for name, value in (
                ("crossref", crossref_record),
                ("openalex", openalex_work),
            )
            if value is not None
        ],
        "metadata_warnings": list(metadata_warnings),
        "journal": journal_name,
        "restored_from_removal": bool(restored),
    }
    paper = PaperMetadata(
        doi=doi,
        title=title,
        abstract=abstract,
        authors=authors,
        publisher=(
            crossref_paper.publisher
            if crossref_paper and crossref_paper.publisher
            else None
        ),
        article_type=(
            crossref_paper.article_type
            if crossref_paper and crossref_paper.article_type
            else openalex_paper.article_type
            if openalex_paper
            else None
        ),
        online_date=online_date,
        issue_date=issue_date,
        volume=(
            crossref_paper.volume
            if crossref_paper and crossref_paper.volume
            else openalex_paper.volume
            if openalex_paper
            else None
        ),
        issue=(
            crossref_paper.issue
            if crossref_paper and crossref_paper.issue
            else openalex_paper.issue
            if openalex_paper
            else None
        ),
        pages=(
            crossref_paper.pages
            if crossref_paper and crossref_paper.pages
            else openalex_paper.pages
            if openalex_paper
            else None
        ),
        source_url=(
            crossref_paper.source_url
            if crossref_paper and crossref_paper.source_url
            else openalex_paper.source_url
            if openalex_paper
            else None
        ),
        open_access_url=(
            openalex_paper.open_access_url
            if openalex_paper and openalex_paper.open_access_url
            else None
        ),
        keywords=(
            crossref_paper.keywords
            if crossref_paper and crossref_paper.keywords
            else openalex_paper.keywords
            if openalex_paper
            else []
        ),
        source_metadata=source_metadata,
    )
    return paper


def _record_intake_error(
    database: Database,
    run_id: int,
    paper_id: int,
    *,
    error_type: str,
    error_message: str,
    stage: str,
) -> None:
    with database.connect() as connection:
        connection.execute(
            """UPDATE paper_processing_status
            SET status='needs_retry',stage=?,error_type=?,error_message=?,
            updated_at=?
            WHERE sync_run_id=? AND paper_id=?""",
            (stage, error_type, error_message[:500], utcnow(), run_id, paper_id),
        )


def _processing_summary(
    database: Database, run_id: int, paper_id: int
) -> tuple[str | None, str | None]:
    with database.connect() as connection:
        row = connection.execute(
            """SELECT status,human_decision FROM paper_processing_status
            WHERE sync_run_id=? AND paper_id=?""",
            (run_id, paper_id),
        ).fetchone()
    if not row:
        return None, None
    return str(row["status"]), str(row["human_decision"] or "undecided")


def intake_verification_candidate(
    database: Database,
    run_id: int,
    candidate: dict[str, Any],
    *,
    registry: JournalRegistry,
    crossref: CrossrefWorkClient | None = None,
    openalex: OpenAlexWorkClient | None = None,
    services: LocalPipelineServices,
    allow_restore: bool = False,
    progress: Callable[[str], None] | None = None,
) -> VerificationIntakeResult:
    """Persist one confirmed candidate and run the standard processing chain."""
    steps: list[IntakeStep] = []
    errors: list[str] = []
    title = _candidate_text(candidate, "title")
    doi = canonicalize_doi(candidate.get("doi"))

    if progress:
        progress("正在核对论文身份与当前论文库状态…")
    state = candidate_intake_state(database, run_id, candidate)
    steps.append(
        IntakeStep(
            "dedup",
            "done",
            {
                "doi": doi,
                "in_library": state.in_library,
                "attached_to_run": state.attached_to_run,
                "tombstoned": state.tombstoned,
            },
        )
    )
    if doi and state.tombstoned:
        if not allow_restore:
            raise CandidatePreviouslyRemoved(
                "这篇论文此前被你主动移除；重新加入需要你明确确认。",
                tombstone_id=state.paper_id or 0,
                has_human_data=state.tombstone_has_human_data,
                removed_at=state.tombstone_removed_at,
            )
        restore_paper_tombstone(database, doi=doi)
        steps.append(
            IntakeStep(
                "restore",
                "done",
                {"doi": doi, "removed_at": state.tombstone_removed_at},
            )
        )

    if state.in_library and state.paper_id is not None:
        paper_id = state.paper_id
        if state.attached_to_run:
            steps.append(
                IntakeStep("metadata", "skipped", {"reason": "already_in_library"})
            )
            steps.append(
                IntakeStep("insert", "skipped", {"reason": "already_in_library"})
            )
            if state.processing_status == "classified":
                steps.append(
                    IntakeStep("chain", "skipped", {"reason": "already_classified"})
                )
                return VerificationIntakeResult(
                    outcome="already_in_library",
                    paper_id=paper_id,
                    doi=doi,
                    title=title,
                    steps=tuple(steps),
                    review_ready=state.review_ready,
                    errors=tuple(errors),
                )
            # Idempotent retry path: the paper is attached but not finished.
        else:
            steps.append(
                IntakeStep(
                    "metadata", "skipped", {"reason": "in_library_unattached"}
                )
            )
            steps.append(
                IntakeStep("insert", "skipped", {"reason": "already_exists"})
            )
        outcome = "attached_existing"
    else:
        if progress:
            progress("正在从 Crossref / OpenAlex 获取完整元数据…")
        crossref_record: dict[str, Any] | None = None
        openalex_work: dict[str, Any] | None = None
        metadata_warnings: list[str] = []
        if doi is None:
            if openalex is not None:
                try:
                    source_id = _candidate_text(candidate, "source_id")
                    if source_id:
                        work = openalex.fetch_by_id(source_id)
                        if work is not None:
                            doi = canonicalize_doi(work.doi)
                            openalex_work = dict(work.source_metadata)
                except Exception as exc:
                    metadata_warnings.append(
                        f"openalex_id_lookup: {_safe_provider_failure(exc)}"
                    )
            if doi is None:
                raise CandidateMissingIdentity(
                    "该候选缺少 DOI，且无法从 OpenAlex 记录解析出 DOI；"
                    "暂时不能可靠加入论文库。"
                )
        if crossref is not None:
            try:
                crossref_record = crossref.fetch_work(doi)
            except Exception as exc:
                metadata_warnings.append(
                    f"crossref_metadata: {_safe_provider_failure(exc)}"
                )
        if openalex is not None and openalex_work is None:
            try:
                openalex_paper = openalex.fetch_by_doi(doi)
                if openalex_paper is not None:
                    openalex_work = dict(openalex_paper.source_metadata)
            except Exception as exc:
                metadata_warnings.append(
                    f"openalex_metadata: {_safe_provider_failure(exc)}"
                )
        if metadata_warnings:
            LOGGER.warning(
                "verification intake metadata warnings: %s", metadata_warnings
            )
        if progress:
            progress("正在确认期刊归属并写入论文库…")
        with database.connect() as connection:
            try:
                run = connection.execute(
                    "SELECT selected_journal_keys FROM sync_runs WHERE id=?",
                    (run_id,),
                ).fetchone()
            except sqlite3.OperationalError:
                # Legacy sync_runs predate the overnight columns; journal
                # resolution then has to rely on provider ISSNs alone.
                run = None
        run = dict(run) if run else {}
        journal_id, journal_name = _resolve_journal(
            database,
            registry,
            run,
            crossref_record=crossref_record,
            openalex_work=openalex_work,
            doi=doi,
        )
        steps.append(
            IntakeStep("journal", "done", {"journal": journal_name, "doi": doi})
        )
        restored = bool(
            doi
            and any(
                step.step == "restore" and step.status == "done" for step in steps
            )
        )
        try:
            paper = _build_paper_metadata(
                candidate,
                doi=doi,
                crossref_record=crossref_record,
                openalex_work=openalex_work,
                run_id=run_id,
                journal_name=journal_name,
                restored=restored,
                metadata_warnings=tuple(metadata_warnings),
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise CandidateIntakeError(
                "元数据无法解析，论文未加入：" + _safe_provider_failure(exc)
            ) from exc
        if progress:
            progress("正在保存论文元数据（自动去重）…")
        paper_id, inserted = database.upsert_paper(paper, journal_id)
        if not inserted:
            outcome = "attached_existing"
            steps.append(
                IntakeStep(
                    "insert",
                    "done",
                    {
                        "paper_id": paper_id,
                        "inserted": False,
                        "reason": "doi_or_title_match",
                    },
                )
            )
        else:
            outcome = "restored" if restored else "created"
            steps.append(
                IntakeStep(
                    "insert",
                    "done",
                    {"paper_id": paper_id, "inserted": True, "restored": restored},
                )
            )

    if progress:
        progress("正在接入本次核验对应的处理链…")
    ensure_paper_statuses_for_paper_ids(database, run_id, (paper_id,))
    steps.append(IntakeStep("attach", "done", {"paper_id": paper_id}))

    if progress:
        progress("正在补译中文标题…")
    try:
        title_result = translate_titles_for_paper_ids(
            database, run_id, (paper_id,), services
        )
        title_status = (
            "done"
            if title_result.get("translated")
            else "pending"
            if title_result.get("pending")
            else "failed"
            if title_result.get("failed")
            else "done"
        )
        steps.append(
            IntakeStep(
                "title_translation",
                title_status,
                {
                    "translated": title_result.get("translated", 0),
                    "pending": title_result.get("pending", 0),
                    "failed": title_result.get("failed", 0),
                },
            )
        )
        if title_result.get("failed"):
            errors.append(
                f"中文标题翻译失败 {title_result['failed']} 次尝试；"
                "英文论文已保留，可在处理区重试。"
            )
    except Exception as exc:
        _record_intake_error(
            database,
            run_id,
            paper_id,
            error_type="intake_title_translation",
            error_message=_safe_provider_failure(exc),
            stage="title_translation",
        )
        errors.append("中文标题翻译失败：" + _safe_provider_failure(exc))
        steps.append(
            IntakeStep("title_translation", "failed", {"error": errors[-1]})
        )

    if progress:
        progress("正在获取英文摘要并生成中文摘要、自动初筛…")
    chain_step = IntakeStep("chain", "pending", {})
    try:
        partial = process_batch(database, run_id, (paper_id,), services)
        completed = partial.get("completed") or []
        needs_abstract = partial.get("needs_abstract") or []
        needs_retry = partial.get("needs_retry") or []
        if paper_id in completed:
            chain_step = IntakeStep(
                "chain", "done", {"ai_calls": partial.get("ai_calls")}
            )
        elif paper_id in needs_abstract:
            chain_step = IntakeStep(
                "chain",
                "pending",
                {"reason": "needs_manual_abstract"},
            )
            errors.append("英文摘要暂时未获取到；论文已保留，等待补摘要。")
        elif paper_id in needs_retry:
            chain_step = IntakeStep(
                "chain",
                "failed",
                {"reason": "needs_retry"},
            )
            errors.append("处理链部分步骤需要重试；已完成数据已保留。")
    except Exception as exc:
        _record_intake_error(
            database,
            run_id,
            paper_id,
            error_type="intake_chain",
            error_message=_safe_provider_failure(exc),
            stage="full_chain_processing",
        )
        errors.append("自动处理失败：" + _safe_provider_failure(exc))
        chain_step = IntakeStep("chain", "failed", {"error": errors[-1]})
    steps.append(chain_step)

    status, human_decision = _processing_summary(database, run_id, paper_id)
    review_ready = status == "classified" and human_decision == "undecided"
    return VerificationIntakeResult(
        outcome=outcome,
        paper_id=paper_id,
        doi=doi,
        title=title,
        steps=tuple(steps),
        review_ready=review_ready,
        errors=tuple(dict.fromkeys(errors)),
    )


def intake_verification_candidates(
    database: Database,
    run_id: int,
    candidates: Sequence[dict[str, Any]],
    *,
    registry: JournalRegistry,
    crossref: CrossrefWorkClient | None = None,
    openalex: OpenAlexWorkClient | None = None,
    services: LocalPipelineServices,
    allow_restore: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[VerificationIntakeResult, ...]:
    """Intake multiple candidates with per-paper failure isolation."""
    results: list[VerificationIntakeResult] = []
    for index, candidate in enumerate(candidates, 1):
        title = _candidate_text(candidate, "title") or "未知论文"
        if progress:
            progress(f"正在加入 {index}/{len(candidates)}：{title[:64]}")
        try:
            results.append(
                intake_verification_candidate(
                    database,
                    run_id,
                    candidate,
                    registry=registry,
                    crossref=crossref,
                    openalex=openalex,
                    services=services,
                    allow_restore=allow_restore,
                    progress=progress,
                )
            )
        except Exception as exc:
            errors = [str(exc)]
            LOGGER.warning(
                "verification intake skipped candidate %s: %s",
                title,
                exc,
            )
            results.append(
                VerificationIntakeResult(
                    outcome="skipped",
                    paper_id=-1,
                    doi=canonicalize_doi(candidate.get("doi")),
                    title=title,
                    steps=(),
                    review_ready=False,
                    errors=errors,
                )
            )
    return tuple(results)
