"""Unified, read-only paper lifecycle state.

Phase 7D: the English abstract, the Chinese abstract and the automatic
judgment are three separate facts.  This module is the single source of truth
for those facts so that UI statistics, the library reprocessing action and the
review workspace stop guessing from a single status enum.

Stabilisation pass: the screening *basis* vocabulary is also defined here.
Workspace, library export, library filters and run statistics previously each
re-derived it with their own ad-hoc comparisons, which produced three
different sets of categories for one stored value.  ``classify_screening_basis``
is now the only place that decides what a stored basis means.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from eduecon_radar.db import Database


class ScreeningBasisKind(StrEnum):
    """How much evidence an automatic judgment actually had."""

    NONE = "none"
    TITLE_ONLY = "title_only"
    TITLE_AND_ABSTRACT = "title_and_abstract"
    DETERMINISTIC_RULE = "deterministic_metadata_rule"
    #: A judgment exists but its basis was never persisted by the writing code.
    #: We must not infer the basis from today's abstract: a paper whose
    #: abstract arrived later may still have been judged on the title alone.
    HISTORICAL_UNKNOWN = "historical_unknown"


#: Stored basis values written by the current code, mapped to their meaning.
_RECORDED_BASES: dict[str, ScreeningBasisKind] = {
    "title_only": ScreeningBasisKind.TITLE_ONLY,
    "title_and_abstract": ScreeningBasisKind.TITLE_AND_ABSTRACT,
    "deterministic_metadata_rule": ScreeningBasisKind.DETERMINISTIC_RULE,
}


def classify_screening_basis(basis: object) -> ScreeningBasisKind:
    """Map a stored ``screening_basis`` value onto its meaning.

    An absent value returns :attr:`ScreeningBasisKind.NONE`; any unrecognised
    value is treated as :attr:`ScreeningBasisKind.HISTORICAL_UNKNOWN` so that
    legacy rows are reported honestly instead of being silently upgraded to a
    complete judgment.
    """
    value = str(basis or "").strip()
    if not value:
        return ScreeningBasisKind.NONE
    return _RECORDED_BASES.get(value, ScreeningBasisKind.HISTORICAL_UNKNOWN)


#: Decisions that count as an actual automatic judgment.
JUDGMENT_DECISIONS = frozenset({"relevant", "irrelevant", "uncertain"})


@dataclass(frozen=True)
class AbstractChainState:
    has_en_abstract: bool
    has_cn_abstract: bool
    cn_translation_status: str | None
    cn_translation_source: str | None
    abstract_source: str | None
    retrieval_status: str | None
    retrieval_retryable: bool
    retrieval_execution_state: str | None


@dataclass(frozen=True)
class ScreeningState:
    has_judgment: bool
    decision: str | None
    screening_basis: str | None
    model_name: str | None

    @property
    def basis_kind(self) -> ScreeningBasisKind:
        return classify_screening_basis(self.screening_basis)

    @property
    def is_full_judgment(self) -> bool:
        """Judged on title *and* abstract — the only complete-judgment basis."""
        return (
            self.has_judgment
            and self.basis_kind is ScreeningBasisKind.TITLE_AND_ABSTRACT
        )

    @property
    def is_historical_unknown_basis(self) -> bool:
        """A judgment exists, but the code that wrote it recorded no basis."""
        return (
            self.has_judgment
            and self.basis_kind is ScreeningBasisKind.HISTORICAL_UNKNOWN
        )


def screening_state_from_classification(
    classification: object,
    *,
    model_name: str | None = None,
) -> ScreeningState:
    """Build a :class:`ScreeningState` from a serialised classification blob.

    Single parser for the ``classification_json`` payload.  Callers that
    already hold the raw JSON (library rows, run snapshots) must use this
    instead of re-implementing the "is there a judgment?" test, so that
    question has exactly one definition.
    """
    payload = classification if isinstance(classification, dict) else {}
    decision = str(payload.get("decision") or "").strip() or None
    basis = str(payload.get("screening_basis") or "").strip() or None
    return ScreeningState(
        has_judgment=decision in JUDGMENT_DECISIONS,
        decision=decision,
        screening_basis=basis,
        model_name=model_name,
    )


@dataclass(frozen=True)
class PaperLifecycleState:
    paper_id: int
    abstract: AbstractChainState
    screening: ScreeningState
    human_decision: str
    processing_status: str | None
    stage: str | None

    @property
    def human_decided(self) -> bool:
        return self.human_decision != "undecided"


def _has_text(value: object) -> bool:
    return bool(str(value or "").strip())


def translation_ready(text: object, status: object, source: object) -> bool:
    return _has_text(text) and (
        source == "human_edited" or status not in {"stale", "pending", "failed"}
    )


def usable_translation_sql(part: str) -> str:
    """Project current translations without altering archived or human-edited text."""
    if part not in {"abstract", "title"}:
        raise ValueError("unknown translation field")
    return (f"CASE WHEN pt.{part}_translation_source='human_edited' OR "
            f"COALESCE(pt.{part}_translation_status,'') NOT IN ('stale','pending','failed') "
            f"THEN pt.chinese_{part} END")


def reprocessing_label(
    has_en: bool, has_cn: bool, decision: object, basis: object
) -> str | None:
    """What a paper still needs, or ``None`` when nothing is left to do.

    A non-research judgment that already reached a decision is complete: there
    is no abstract to fetch for a journal front-matter item, so offering to
    "complete and re-analyse" it would be a false action.
    """
    basis_kind = classify_screening_basis(basis)
    decided = decision in {"relevant", "irrelevant", "uncertain"}
    if basis_kind is ScreeningBasisKind.DETERMINISTIC_RULE and decided:
        return None
    if not has_en:
        return "补全并重新分析"
    if not has_cn:
        return "生成中文摘要并重新分析"
    if not decided or basis_kind is not ScreeningBasisKind.TITLE_AND_ABSTRACT:
        return "重新分析"
    return None


def paper_lifecycle_state(
    database: Database, paper_id: int, *, profile_id: str | None = None
) -> PaperLifecycleState | None:
    """Compute the current state from the database only (no providers/AI).

    The judgement is read from ``paper_profile_assessments`` — the single
    authority — for the run's profile (or ``profile_id`` when given).  A row
    written before migration 0013 has no profile assessment, so its run-scoped
    ``classification_json`` is still read; that fallback exists for historical
    traceability only, and a run outcome such as "needs an abstract" is never
    presented as a judgement.
    """
    with database.connect() as connection:
        row = connection.execute(
            """SELECT p.id,p.abstract,p.source_metadata_json,
            pt.chinese_abstract,pt.abstract_translation_status,
            pt.abstract_translation_source,
            ps.status,ps.stage,ps.human_decision,ps.classification_json,
            ps.model_name,ps.sync_run_id,
            sr.profile_id run_profile_id
            FROM papers p
            LEFT JOIN paper_translations pt ON pt.paper_id=p.id
            LEFT JOIN paper_processing_status ps
              ON ps.paper_id=p.id
             AND ps.sync_run_id=(SELECT MAX(sync_run_id)
                                 FROM paper_processing_status
                                 WHERE paper_id=p.id)
            LEFT JOIN sync_runs sr ON sr.id=ps.sync_run_id
            WHERE p.id=?""",
            (paper_id,),
        ).fetchone()
        if row is None:
            return None
        assessment = None
        resolved_profile: str | None = None
        for candidate in (profile_id, row["run_profile_id"]):
            if not candidate:
                continue
            assessment = _assessment_row(connection, paper_id, str(candidate))
            if assessment is not None:
                resolved_profile = str(candidate)
                break
        if resolved_profile is None and row["run_profile_id"]:
            resolved_profile = str(row["run_profile_id"])
        human = _review_row(connection, paper_id, resolved_profile)
        if human is None:
            human = str(row["human_decision"] or "undecided")
    metadata = json.loads(row["source_metadata_json"] or "{}")
    metadata = metadata if isinstance(metadata, dict) else {}
    enrichment = metadata.get("abstract_enrichment")
    enrichment = enrichment if isinstance(enrichment, dict) else {}
    if assessment is not None:
        classification = assessment
        model_name = assessment.get("_model_name")
    else:
        classification = json.loads(row["classification_json"] or "{}")
        classification = classification if isinstance(classification, dict) else {}
        model_name = row["model_name"]
        if not classification.get("decision"):
            derived = _RUN_STATE_DECISIONS.get(str(row["status"] or ""))
            classification = {"decision": derived} if derived else {}
    abstract = AbstractChainState(
        has_en_abstract=_has_text(row["abstract"]),
        has_cn_abstract=translation_ready(
            row["chinese_abstract"], row["abstract_translation_status"],
            row["abstract_translation_source"],
        ),
        cn_translation_status=row["abstract_translation_status"],
        cn_translation_source=row["abstract_translation_source"],
        abstract_source=metadata.get("abstract_source"),
        retrieval_status=str(enrichment.get("status") or "") or None,
        retrieval_retryable=bool(enrichment.get("retryable")),
        retrieval_execution_state=str(
            enrichment.get("execution_state") or ""
        )
        or None,
    )
    screening = screening_state_from_classification(
        classification, model_name=model_name
    )
    return PaperLifecycleState(
        paper_id=int(row["id"]),
        abstract=abstract,
        screening=screening,
        human_decision=human,
        processing_status=row["status"],
        stage=row["stage"],
    )


#: Run outcomes that must not be mistaken for a judgement.
_RUN_STATE_DECISIONS = {
    "needs_manual_abstract": "needs_abstract",
    "needs_retry": "needs_retry",
    "failed": "needs_retry",
}


def _assessment_row(
    connection: object, paper_id: int, profile_id: str
) -> dict[str, object] | None:
    """The profile's stored assessment payload, if any, with its model name."""
    getter = getattr(connection, "execute", None)
    if getter is None:
        return None
    try:
        row = getter(
            "SELECT classification_json,model_name FROM paper_profile_assessments "
            "WHERE profile_id=? AND paper_id=?",
            (profile_id, paper_id),
        ).fetchone()
    except Exception:
        # The profile tables do not exist yet (pre-0013 database).
        return None
    if row is None:
        return None
    payload = json.loads(row["classification_json"] or "{}")
    if not isinstance(payload, dict) or not payload.get("decision"):
        return None
    return {**payload, "_model_name": row["model_name"]}


def _review_row(
    connection: object, paper_id: int, profile_id: str | None
) -> str | None:
    """The profile's stored human decision, if one was saved."""
    if not profile_id:
        return None
    getter = getattr(connection, "execute", None)
    if getter is None:
        return None
    try:
        row = getter(
            "SELECT human_decision FROM paper_profile_reviews "
            "WHERE profile_id=? AND paper_id=?",
            (profile_id, paper_id),
        ).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    return str(row["human_decision"] or "undecided")


def abstract_state_label(state: AbstractChainState) -> str:
    """Compact Chinese label for the abstract chain (UI helper)."""
    if not state.has_en_abstract:
        if state.retrieval_retryable:
            return "英文摘要可重试"
        if state.retrieval_execution_state == "not_applicable":
            return "无需摘要"
        if state.retrieval_execution_state is None:
            # No enrichment record exists at all: retrieval was never run for
            # this paper, or it ran before the enrichment audit was recorded.
            # Reporting plain "missing" would hide a recoverable gap and would
            # wrongly imply every source had already been tried.
            return "缺英文摘要（未尝试获取）"
        return "缺英文摘要"
    if not state.has_cn_abstract:
        if state.cn_translation_status == "failed":
            return "中文摘要待重试"
        if state.cn_translation_status == "pending":
            return "中文摘要待生成"
        return "缺中文摘要"
    return "摘要完整"


def screening_state_label(state: ScreeningState) -> str:
    """Compact Chinese label for the automatic judgment (UI helper).

    This is the single user-visible vocabulary for a judgment.  Exports,
    filters, review and statistics must derive from here rather than
    re-comparing raw basis strings.
    """
    if not state.has_judgment:
        return "无自动判断"
    match state.basis_kind:
        case ScreeningBasisKind.TITLE_ONLY:
            return "标题初判"
        case ScreeningBasisKind.TITLE_AND_ABSTRACT:
            return "完整判断"
        case ScreeningBasisKind.DETERMINISTIC_RULE:
            return "非研究内容判断"
        case _:
            return "历史判断（依据未记录）"


def screening_basis_count_key(state: ScreeningState) -> str | None:
    """Which run-statistic bucket a judgment belongs to.

    Returns ``None`` when there is no judgment at all.  Every judgment lands in
    exactly one bucket, so the buckets sum to the automatic-judgment count.
    """
    if not state.has_judgment:
        return None
    match state.basis_kind:
        case ScreeningBasisKind.TITLE_ONLY:
            return "title_only_judgments"
        case ScreeningBasisKind.TITLE_AND_ABSTRACT:
            return "full_judgments"
        case ScreeningBasisKind.DETERMINISTIC_RULE:
            return "non_research_judgments"
        case _:
            return "unrecorded_basis_judgments"


def library_reprocess_label(state: PaperLifecycleState) -> str | None:
    """What a library paper still needs, or None when it is complete."""
    return reprocessing_label(
        state.abstract.has_en_abstract, state.abstract.has_cn_abstract,
        state.screening.decision, state.screening.screening_basis,
    )



#: Processing statuses that mean "the chain stopped short of a judgment".
#: Kept next to the state vocabulary so the invariant check and the
#: continuation selector cannot disagree about what "unfinished" means.
INCOMPLETE_PROCESSING_STATUSES = frozenset(
    {"pending", "processing", "abstract_ready", "classification_ready", "needs_retry"}
)


@dataclass(frozen=True)
class LifecycleViolation:
    """A persisted row whose state contradicts what the pipeline promises."""

    paper_id: int
    kind: str
    detail: str


def lifecycle_violations(
    rows: list[dict[str, object]],
) -> tuple[LifecycleViolation, ...]:
    """Report rows that break the documented lifecycle contract.

    Two contracts are checked, both of which were violated silently in
    production before this check existed:

    ``abstract_without_judgment``
        The chain retrieved an abstract but never produced an automatic
        judgment.  Nothing in the UI says such a paper still needs screening.

    ``finishable_status_stalled``
        The status sits in an intermediate stage while the paper already has
        everything needed to finish.  Before the fix these papers were never
        selected for continuation, so they were stranded permanently.

    This is deliberately an *observation*, not a repair: it never guesses why a
    row looks odd and never invents data.  ``attempts`` is NOT used as a signal
    -- production shows it is not maintained consistently, so it cannot
    distinguish "never attempted" from "attempted".
    """
    violations: list[LifecycleViolation] = []
    for row in rows:
        state = row.get("state")
        if not isinstance(state, PaperLifecycleState):
            continue
        status = str(row.get("processing_status") or "pending")
        if status not in INCOMPLETE_PROCESSING_STATUSES:
            continue
        if state.screening.has_judgment:
            continue
        if state.abstract.has_en_abstract:
            violations.append(
                LifecycleViolation(
                    paper_id=state.paper_id,
                    kind="abstract_without_judgment",
                    detail=(
                        f"abstract present, status={status}, "
                        f"stage={state.stage!r}, no automatic judgment"
                    ),
                )
            )
        if state.abstract.has_cn_abstract:
            violations.append(
                LifecycleViolation(
                    paper_id=state.paper_id,
                    kind="finishable_status_stalled",
                    detail=(
                        "abstract and Chinese abstract both present but no "
                        f"judgment; status={status} means nothing will resume it"
                    ),
                )
            )
    return tuple(violations)


def abstract_state_payload(state: PaperLifecycleState) -> dict[str, Any]:
    return {
        "abstract_label": abstract_state_label(state.abstract),
        "screening_label": screening_state_label(state.screening),
        "has_en_abstract": state.abstract.has_en_abstract,
        "has_cn_abstract": state.abstract.has_cn_abstract,
        "has_judgment": state.screening.has_judgment,
        "screening_basis": state.screening.screening_basis,
        "human_decided": state.human_decided,
    }
