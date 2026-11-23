"""Product-path pipeline keyed by sync_run instead of workflow tasks.

Phase 7B5: the local app never creates a personal task. One sync run stores the
precise matched_paper_ids and every paper has its own processing-status row.
The workflow_batches tables remain only as historical data; the product path
does not query, create, or write them.

Profile-aware refactor: a retrieval run is a *discovery snapshot* for one
research profile, and the long-term judgement (automatic assessment plus human
review) lives in the profile-scoped tables of ``services/profile_store.py``.
The run-scoped columns in ``paper_processing_status`` keep recording what
happened during that run and stay readable for historical rows.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Protocol

from eduecon_radar.db import Database, utcnow
from eduecon_radar.services import profile_store
from eduecon_radar.services.deterministic_screening import (
    FRONT_MATTER_PROMPT_VERSION,
)
from eduecon_radar.services.paper_lifecycle_state import (
    screening_basis_count_key,
    screening_state_from_classification,
    translation_ready,
    usable_translation_sql,
)
from eduecon_radar.services.ui_labels import zh_status

LOGGER = logging.getLogger(__name__)


@dataclass
class LocalPipelineServices:
    enrichment: object | None = None
    translation: object | None = None
    screening: object | None = None
    batch_titles: object | None = None
    #: Research profile the services were assembled for.
    profile_id: str | None = None


class Translator(Protocol):
    def run_batch_titles(
        self, paper_ids: tuple[int, ...], *, task_id: int | None = None
    ) -> object: ...


#: Processing statuses that a continuation run must pick up.
#:
#: ``abstract_ready`` and ``classification_ready`` are *intermediate* stages,
#: not terminal outcomes: the chain sets them before translation / screening
#: respectively.  If the process stops in between (interrupt, crash, closed
#: browser) the paper is left in one of them, and a continuation that only
#: looked at pending/processing/needs_retry would never touch it again --
#: stranding a paper whose abstract is already retrieved.  They belong here so
#: "can this paper still make progress?" has one answer for the workbench and
#: the CLI alike.
RESUMABLE_PROCESSING_STATUSES = frozenset(
    {
        "pending",
        "processing",
        "abstract_ready",
        "classification_ready",
        "needs_retry",
        "failed",
    }
)


def resumable_paper_ids(
    database: Database, sync_run_id: int
) -> tuple[int, ...]:
    """Prefer earlier attempts so repeated failures cannot starve untouched papers."""
    return tuple(
        int(paper["id"])
        for paper in sorted(sync_run_papers(database, sync_run_id),
                            key=lambda p: int(p["processing"].get("attempts") or 0))
        if str((paper.get("processing") or {}).get("status") or "pending")
        in RESUMABLE_PROCESSING_STATUSES
    )


def run_profile_id(database: Database, run_id: int) -> str:
    """The profile a retrieval run was made for.

    Databases that predate migration 0013 have no column; they belong to the
    default profile, which is exactly what the backfill establishes.
    """
    with database.connect() as connection:
        try:
            row = connection.execute(
                "SELECT profile_id FROM sync_runs WHERE id=?", (run_id,)
            ).fetchone()
        except sqlite3.OperationalError:
            return profile_store.DEFAULT_PROFILE_ID
    if row is None or not row["profile_id"]:
        return profile_store.DEFAULT_PROFILE_ID
    return str(row["profile_id"])


def _record_profile_assessment(
    database: Database,
    *,
    profiles_ready: bool,
    profile_id: str,
    run_id: int,
    paper_id: int,
    classification: dict[str, object],
    model_name: str | None,
    prompt_version: str | None,
    title: str | None = None,
    abstract: str | None = None,
) -> None:
    """Store the judgement in its single canonical place.

    ``paper_profile_assessments`` is the authority.  Only a database that
    predates migration 0013 still falls back to the run-scoped column, because
    it has nowhere else to put the judgement; once 0013 is applied the same code
    path never writes an assessment twice.
    """
    if not profiles_ready:
        _write_legacy_run_judgment(
            database,
            run_id=run_id,
            paper_id=paper_id,
            classification=classification,
            model_name=model_name,
            prompt_version=prompt_version,
        )
        return
    fingerprint = (
        profile_store.assessment_fingerprint(
            profile_id, prompt_version or "", title or "", abstract
        )
        if title is not None
        else None
    )
    profile_store.record_assessment(
        database,
        profile_id=profile_id,
        paper_id=paper_id,
        classification=classification,
        model_name=model_name,
        prompt_version=prompt_version,
        source_run_id=run_id,
        input_fingerprint=fingerprint,
    )


def _write_legacy_run_judgment(
    database: Database,
    *,
    run_id: int,
    paper_id: int,
    classification: dict[str, object],
    model_name: str | None,
    prompt_version: str | None,
) -> None:
    """Store a judgement in the run snapshot — **only** for a pre-0013 database.

    Kept as its own function so the one remaining place that writes a judgement
    next to a run is visible in review, and so the ordinary run-state writer can
    refuse those columns outright.  A database with the profile schema never
    reaches this path (see ``_record_profile_assessment``).
    """
    now = utcnow()
    with database.connect() as connection:
        connection.execute(
            """UPDATE paper_processing_status
            SET classification_json=?,model_name=?,prompt_version=?,classified_at=?,
                updated_at=?
            WHERE sync_run_id=? AND paper_id=?""",
            (
                json.dumps(classification, ensure_ascii=False),
                model_name,
                prompt_version,
                now,
                now,
                run_id,
                paper_id,
            ),
        )


def _screening_prompt_version(services: LocalPipelineServices) -> str:
    """The profile's declared screening prompt version, with a safe default."""
    return (
        getattr(services.screening, "prompt_version", None) or "2026-07-31-screening-v1"
    )


def services_profile_id(
    database: Database, run_id: int, services: LocalPipelineServices | None = None
) -> str:
    """Prefer the profile the services were built for, else the run's profile."""
    if services is not None and services.profile_id:
        return str(services.profile_id)
    return run_profile_id(database, run_id)


def lifecycle_invariant_report(
    database: Database, sync_run_id: int | None = None
) -> dict[str, object]:
    """Read-only scan for papers whose persisted state breaks the lifecycle contract.

    Exists because production contained 6 papers that were "abstract retrieved,
    translated, but never judged" and nothing surfaced them.  Reports what is
    observable; never repairs and never guesses.
    """
    from eduecon_radar.services.paper_lifecycle_state import (
        lifecycle_violations,
        paper_lifecycle_state,
    )

    with database.connect() as connection:
        sql = "SELECT DISTINCT paper_id FROM paper_processing_status"
        params: tuple[object, ...] = ()
        if sync_run_id is not None:
            sql = (
                "SELECT DISTINCT paper_id FROM paper_processing_status WHERE sync_run_id=?"
            )
            params = (sync_run_id,)
        paper_ids = [int(row["paper_id"]) for row in connection.execute(sql, params)]

    rows: list[dict[str, object]] = []
    for paper_id in paper_ids:
        state = paper_lifecycle_state(database, paper_id)
        if state is None:
            continue
        with database.connect() as connection:
            row = connection.execute(
                "SELECT status FROM paper_processing_status "
                "WHERE paper_id=? ORDER BY sync_run_id DESC LIMIT 1",
                (paper_id,),
            ).fetchone()
        rows.append(
            {
                "state": state,
                "processing_status": row["status"] if row else "pending",
            }
        )

    violations = lifecycle_violations(rows)
    return {
        "scan_sync_run_id": sync_run_id,
        "papers_scanned": len(rows),
        "papers_with_judgment": sum(
            1
            for row in rows
            if isinstance(row["state"], object) and row["state"].screening.has_judgment
        ),
        "violations": [
            {"paper_id": v.paper_id, "kind": v.kind, "detail": v.detail}
            for v in violations
        ],
        "violation_count": len(violations),
    }


def ensure_paper_statuses(database: Database, sync_run_id: int) -> tuple[int, ...]:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT matched_paper_ids FROM sync_runs WHERE id=?", (sync_run_id,)
        ).fetchone()
    if not row:
        raise ValueError(f"sync run not found: {sync_run_id}")
    paper_ids = tuple(int(item) for item in json.loads(row[0] or "[]"))
    return ensure_paper_statuses_for_paper_ids(database, sync_run_id, paper_ids)


def ensure_paper_statuses_for_paper_ids(
    database: Database,
    sync_run_id: int,
    paper_ids: tuple[int, ...],
) -> tuple[int, ...]:
    """Attach explicit papers to a sync run without rewriting its matched set.

    Verification intake reuses the same per-paper processing chain as a normal
    retrieval while leaving the historical ``matched_paper_ids`` untouched, so
    the run still reports the original retrieval facts.
    """
    normalized = tuple(dict.fromkeys(int(item) for item in paper_ids))
    scoped = profile_store.profile_schema_ready(database)
    profile_id = run_profile_id(database, sync_run_id) if scoped else None
    ids_json = json.dumps(normalized)
    with database.connect() as connection:
        present = connection.execute(
            "SELECT p.id, ps.paper_id attached FROM json_each(?) ids "
            "JOIN papers p ON p.id=ids.value LEFT JOIN paper_processing_status ps "
            "ON ps.paper_id=p.id AND ps.sync_run_id=? ORDER BY ids.key",
            (ids_json, sync_run_id),
        ).fetchall()
        missing = [row["id"] for row in present if row["attached"] is None]
        if missing:
            now = utcnow()
            if scoped:
                prior = ("LEFT JOIN paper_profile_reviews old ON old.paper_id=p.id "
                         "AND old.profile_id=?")
                scope = [profile_id]
                decided_at = "old.decided_at"
            else:
                prior = """LEFT JOIN paper_processing_status old ON old.rowid=(
                    SELECT rowid FROM paper_processing_status prev WHERE prev.paper_id=p.id
                    ORDER BY human_decided_at DESC, sync_run_id DESC LIMIT 1)"""
                scope = []
                decided_at = "old.human_decided_at"
            connection.execute(
                f"""INSERT INTO paper_processing_status
                (sync_run_id,paper_id,status,stage,created_at,updated_at,human_decision,
                 human_note,human_decided_at)
                SELECT ?,p.id,'pending','pending',?,?,coalesce(old.human_decision,'undecided'),
                       old.human_note,{decided_at}
                FROM json_each(?) ids JOIN papers p ON p.id=ids.value {prior} WHERE true
                ON CONFLICT(sync_run_id,paper_id) DO NOTHING""",
                (sync_run_id, now, now, json.dumps(missing), *scope),
            )
        return tuple(row["id"] for row in present)


def list_sync_runs(database: Database) -> list[dict[str, object]]:
    try:
        with database.connect() as connection:
            rows = connection.execute(
                """SELECT id,started_at,finished_at,status,selected_journal_keys,
                query_start_date,query_end_date,requested_limit,matched_paper_ids,
                provider_returned_count,pages_requested,raw_candidate_count,
                normalized_count,stopped_reason
                FROM sync_runs WHERE requested_limit IS NOT NULL
                ORDER BY id DESC"""
            ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.OperationalError:
        return []


def latest_sync_run(database: Database) -> dict[str, object] | None:
    runs = list_sync_runs(database)
    return runs[0] if runs else None


def _status_row_values(
    database: Database, run_id: int, paper_id: int
) -> dict[str, object]:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT * FROM paper_processing_status "
            "WHERE sync_run_id=? AND paper_id=?",
            (run_id, paper_id),
        ).fetchone()
    return dict(row) if row else {}


def _update_status(
    database: Database,
    run_id: int,
    paper_id: int,
    **updates: object,
) -> None:
    """Write the *run audit snapshot* for one paper.

    ``paper_processing_status`` records what happened during one retrieval run:
    stage, status, attempts, error, abstract-retrieval and title-translation
    facts.  The long-term judgement lives in ``paper_profile_assessments`` and
    the human decision in ``paper_profile_reviews``; the assessment columns are
    deliberately **not** accepted here, so the product chain cannot write the
    same judgement twice.  Historical rows keep their values and stay readable
    (see ``sync_run_papers``).
    """
    now = utcnow()
    allowed = {
        "status",
        "stage",
        "attempts",
        "title_translation_json",
        "abstract_retrieval_json",
        "human_decision",
        "human_note",
        "human_decided_at",
        "error_type",
        "error_message",
    }
    sets = {key: value for key, value in updates.items() if key in allowed}
    sets["updated_at"] = now
    assignments = ", ".join(f"{key}=?" for key in sets)
    values = list(sets.values())
    with database.connect() as connection:
        if sets.get("status") == "classified" and "error_type" not in sets:
            # Only clear errors from stages completed by this chain. A failed
            # title translation from intake is a separate, still-open issue.
            connection.execute(
                """UPDATE paper_processing_status SET error_type=NULL,error_message=NULL
                WHERE sync_run_id=? AND paper_id=?
                  AND error_type IN ('title_screening_failed','translation_failed')""",
                (run_id, paper_id),
            )
        connection.execute(
            f"""UPDATE paper_processing_status SET {assignments}
            WHERE sync_run_id=? AND paper_id=?""",
            (*values, run_id, paper_id),
        )


def sync_run_papers(
    database: Database, run_id: int, *, profile_id: str | None = None
) -> list[dict[str, object]]:
    """Papers discovered by one retrieval run, judged under one profile.

    The run supplies the discovery facts (which papers, which processing state);
    the profile supplies the long-term judgement.  Where a profile row exists it
    wins over the historical run-scoped column, so switching profiles really
    switches the answers instead of showing the first profile's decision twice.
    """
    profile = profile_id or run_profile_id(database, run_id)
    profiles_ready = profile_store.profile_schema_ready(database)
    with database.connect() as connection:
        rows = connection.execute(
            f"""SELECT p.id,p.title,p.abstract,p.authors_json,p.doi,p.source_url,
            p.open_access_url,p.volume,p.issue,p.publisher,p.source_metadata_json,
            COALESCE(p.issue_date,p.online_date) publication_date,j.name journal,
            {usable_translation_sql("title")} chinese_title,
            {usable_translation_sql("abstract")} chinese_abstract,pt.title_translation_status,
            pt.title_translation_source,pt.abstract_translation_status,
            pt.abstract_translation_source,pt.model_name translation_model_name,
            pt.prompt_version translation_prompt_version,
            ps.status processing_status,ps.stage,ps.attempts,
            ps.classification_json,ps.title_translation_json,
            ps.abstract_retrieval_json,ps.human_decision,ps.human_note,
            ps.model_name,ps.prompt_version,ps.classified_at,
            ps.error_type,ps.error_message
            FROM paper_processing_status ps
            JOIN papers p ON p.id=ps.paper_id
            JOIN journals j ON j.id=p.journal_id
            LEFT JOIN paper_translations pt ON pt.paper_id=p.id
            WHERE ps.sync_run_id=?
            ORDER BY p.id""",
            (run_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["authors"] = json.loads(item.pop("authors_json") or "[]")
        metadata = json.loads(item.pop("source_metadata_json") or "{}")
        item["source_metadata"] = metadata
        item["abstract_source"] = metadata.get("abstract_source")
        item["abstract_enrichment"] = metadata.get("abstract_enrichment")
        classification = json.loads(item.pop("classification_json") or "{}")
        title_translation = json.loads(item.pop("title_translation_json") or "{}")
        abstract_retrieval = json.loads(item.pop("abstract_retrieval_json") or "{}")
        item["processing"] = {
            "status": item.pop("processing_status") or "pending",
            "stage": item.pop("stage") or "pending",
            "attempts": item.pop("attempts") or 0,
            "classification": classification,
            "title_translation": title_translation,
            "abstract_retrieval": abstract_retrieval,
            "model_name": item.pop("model_name"),
            "prompt_version": item.pop("prompt_version"),
            "classified_at": item.pop("classified_at"),
            "error_type": item.pop("error_type"),
            "error_message": item.pop("error_message"),
        }
        item["human_decision"] = item.pop("human_decision") or "undecided"
        item["reviewer_notes"] = item.pop("human_note")
        item["profile_id"] = profile
        item["assessment_source"] = "run_snapshot"
        result.append(item)
    if result:
        paper_ids = tuple(int(item["id"]) for item in result)
        assessments = (
            profile_store.assessments_for(database, profile, paper_ids)
            if profiles_ready
            else {}
        )
        reviews = (
            profile_store.reviews_for(database, profile, paper_ids)
            if profiles_ready
            else {}
        )
        for item in result:
            paper_id = int(item["id"])
            assessment = assessments.get(paper_id)
            if assessment is not None:
                # Canonical: the profile owns the judgement.
                item["processing"]["classification"] = assessment.classification
                item["processing"]["model_name"] = assessment.model_name
                item["processing"]["prompt_version"] = assessment.prompt_version
                item["processing"]["classified_at"] = assessment.assessed_at
                item["assessment_source"] = "profile"
                profile_store.reflect_saved_assessment(item, assessment.input_fingerprint)
            elif item["processing"]["classification"]:
                # Historical row written before the single-authority cutover.
                # It stays readable for exactly this run, never re-used as the
                # profile's current judgement.
                item["assessment_source"] = "run_snapshot"
            else:
                # No judgement exists for this profile: what the run could or
                # could not do is a run outcome, so say that instead of
                # inventing an assessment.
                derived = _decision_from_run_state(
                    str(item["processing"].get("status") or "")
                )
                if derived:
                    item["processing"]["classification"] = {"decision": derived}
                    item["assessment_source"] = "run_state"
            review = reviews.get(paper_id)
            if review is not None:
                item["human_decision"] = review.human_decision
                item["reviewer_notes"] = review.human_note
    return result


#: Run outcomes that are not judgements but must still be visible to the reader.
_RUN_STATE_DECISIONS = {
    "needs_manual_abstract": "needs_abstract",
    "needs_retry": "needs_retry",
    "failed": "needs_retry",
}


def _decision_from_run_state(status: str) -> str | None:
    return _RUN_STATE_DECISIONS.get(status)


def snapshot_counts(papers: list[dict[str, object]]) -> dict[str, int]:
    counts = {
        "retrieved": len(papers),
        "pending_processing": 0,
        "missing_abstract": 0,
        "missing_cn_abstract": 0,
        "needs_retry": 0,
        "review_pending": 0,
        "completed": 0,
        "english_abstracts": 0,
        "chinese_abstracts": 0,
        "automatic_judgments": 0,
        "title_only_judgments": 0,
        "full_judgments": 0,
        "non_research_judgments": 0,
        "unrecorded_basis_judgments": 0,
        "human_decisions": 0,
        "title_translated": 0,
    }
    for paper in papers:
        processing = dict(paper.get("processing") or {})
        status = str(processing.get("status") or "pending")
        human = str(paper.get("human_decision") or "undecided")
        classification = dict(processing.get("classification") or {})
        if status in RESUMABLE_PROCESSING_STATUSES - {"needs_retry", "failed"}:
            counts["pending_processing"] += 1
        has_en_abstract = bool(str(paper.get("abstract") or "").strip())
        has_cn_abstract = translation_ready(
            paper.get("chinese_abstract"), paper.get("abstract_translation_status"),
            paper.get("abstract_translation_source"),
        )
        if not has_en_abstract:
            counts["missing_abstract"] += 1
        elif not has_cn_abstract:
            counts["missing_cn_abstract"] += 1
        if status in {"needs_retry", "failed"}:
            counts["needs_retry"] += 1
        if human == "undecided":
            counts["review_pending"] += 1
        if human != "undecided":
            counts["completed"] += 1
            counts["human_decisions"] += 1
        if has_en_abstract:
            counts["english_abstracts"] += 1
        if has_cn_abstract:
            counts["chinese_abstracts"] += 1
        if status == "classified":
            # One shared classifier decides both "is there a judgment?" and
            # which basis bucket it belongs to, so the buckets always sum to
            # the automatic-judgment count.
            screening = screening_state_from_classification(classification)
            if screening.has_judgment:
                counts["automatic_judgments"] += 1
                key = screening_basis_count_key(screening)
                if key is not None:
                    counts[key] += 1
        if str(paper.get("chinese_title") or "").strip():
            counts["title_translated"] += 1
    return counts


def sync_run_snapshot(
    database: Database,
    run_id: int,
    *,
    run: dict[str, object] | None = None,
) -> dict[str, object]:
    if run is not None and int(run["id"]) != run_id:
        raise ValueError("provided sync run does not match run_id")
    if run is None:
        for candidate in list_sync_runs(database):
            if int(candidate["id"]) == run_id:
                run = candidate
                break
    if run is None:
        raise ValueError(f"sync run not found: {run_id}")
    papers = sync_run_papers(database, run_id)
    return {
        "run": run,
        "papers": papers,
        "counts": snapshot_counts(papers),
    }


def translate_titles(
    database: Database,
    run_id: int,
    services: LocalPipelineServices,
) -> dict[str, int]:
    paper_ids = ensure_paper_statuses(database, run_id)
    return translate_titles_for_paper_ids(database, run_id, paper_ids, services)


def translate_titles_for_paper_ids(
    database: Database,
    run_id: int,
    paper_ids: tuple[int, ...],
    services: LocalPipelineServices,
) -> dict[str, int]:
    """Translate missing titles for an explicit paper set on one sync run."""
    paper_ids = tuple(dict.fromkeys(int(item) for item in paper_ids))
    translator = services.batch_titles
    if translator is None:
        now = utcnow()
        with database.connect() as connection:
            for paper_id in paper_ids:
                connection.execute(
                    """UPDATE paper_processing_status
                    SET title_translation_json=?,updated_at=?
                    WHERE sync_run_id=? AND paper_id=?""",
                    (
                        json.dumps(
                            {
                                "status": "title_translation_pending",
                                "reason": "deepseek_api_key_missing",
                                "updated_at": now,
                            }
                        ),
                        now,
                        run_id,
                        paper_id,
                    ),
                )
        return {"translated": 0, "pending": len(paper_ids), "failed": 0, "ai_calls": 0}
    result = translator.run_batch_titles(paper_ids, task_id=None)
    now = utcnow()
    placeholders = ",".join("?" for _ in paper_ids)
    with database.connect() as connection:
        rows = connection.execute(
            f"""SELECT pt.paper_id,pt.chinese_title,pt.title_translation_status,
            pt.title_translation_source FROM paper_translations pt
            WHERE pt.paper_id IN ({placeholders})""",
            list(paper_ids),
        ).fetchall()
    translated = failed = pending = 0
    by_id = {int(row["paper_id"]): row for row in rows}
    with database.connect() as connection:
        for paper_id in paper_ids:
            row = by_id.get(paper_id)
            if row and row["chinese_title"]:
                status = (
                    "manual_title_preserved"
                    if row["title_translation_source"] == "human_edited"
                    else "translated"
                )
                translated += 1
            elif row and row["title_translation_status"] == "failed":
                status = "title_translation_retry"
                failed += 1
            else:
                status = "title_translation_pending"
                pending += 1
            connection.execute(
                """UPDATE paper_processing_status
                SET title_translation_json=?,updated_at=?
                WHERE sync_run_id=? AND paper_id=?""",
                (
                    json.dumps(
                        {
                            "status": status,
                            "source": row["title_translation_source"] if row else None,
                            "updated_at": now,
                        }
                    ),
                    now,
                    run_id,
                    paper_id,
                ),
            )
    record_run_ai_usage(
        database, run_id, ai_calls=int(getattr(result, "ai_calls", 0) or 0),
        input_tokens=int(getattr(result, "input_tokens", 0) or 0),
        output_tokens=int(getattr(result, "output_tokens", 0) or 0),
    )
    return {
        "translated": translated,
        "pending": pending,
        "failed": failed,
        "ai_calls": int(getattr(result, "ai_calls", 0) or 0),
    }


def _paper_row(database: Database, paper_id: int) -> dict[str, object]:
    with database.connect() as connection:
        row = connection.execute(
            """SELECT p.*,j.name journal FROM papers p
            JOIN journals j ON j.id=p.journal_id WHERE p.id=?""",
            (paper_id,),
        ).fetchone()
    return dict(row) if row else {}


def record_run_ai_usage(
    database: Database,
    run_id: int,
    *,
    ai_calls: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Add a processing batch's real model usage to its retrieval run.

    ``sync_runs.ai_calls`` / ``input_tokens`` / ``output_tokens`` are the audit
    record of what a run cost.  The screening and translation services count
    those numbers already, but nothing wrote them back, so every processed run
    claimed zero.

    Two properties make this safe:

    * **Additive, never assigning.**  ``process_batch`` is resumable, so one run
      can be processed by several invocations.  Writing a total would erase the
      earlier batches; adding accumulates them.
    * **Scoped to the run that did the work.**  The legacy screening path
      attributes usage to ``ORDER BY id DESC LIMIT 1`` — the newest run in the
      database, which is not necessarily the one being processed.  A run id is
      required here precisely so that cannot happen.
    """
    if ai_calls == 0 and input_tokens == 0 and output_tokens == 0:
        return
    with database.connect() as connection:
        connection.execute(
            """UPDATE sync_runs SET ai_calls=COALESCE(ai_calls,0)+?,
            input_tokens=COALESCE(input_tokens,0)+?,
            output_tokens=COALESCE(output_tokens,0)+? WHERE id=?""",
            (int(ai_calls), int(input_tokens), int(output_tokens), int(run_id)),
        )


def process_batch(
    database: Database,
    run_id: int,
    paper_ids: tuple[int, ...],
    services: LocalPipelineServices,
    *,
    force_abstract_refresh: bool = False,
    profile_id: str | None = None,
) -> dict[str, object]:
    # Translation pulls in the AI SDK.  Keep it off the ordinary read-only UI
    # import path; this operation only runs after an explicit processing action.
    from eduecon_radar.services.translation import TranslationParameters

    profile = profile_id or services_profile_id(database, run_id, services)
    profiles_ready = profile_store.profile_schema_ready(database)
    result: dict[str, object] = {
        "completed": [],
        "needs_abstract": [],
        "needs_retry": [],
        "ai_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    for paper_id in paper_ids:
        row = _paper_row(database, paper_id)
        if not row:
            result["needs_retry"].append(paper_id)
            continue
        with database.connect() as connection:
            status_row = connection.execute(
                "SELECT status,attempts FROM paper_processing_status "
                "WHERE sync_run_id=? AND paper_id=?",
                (run_id, paper_id),
            ).fetchone()
        prior_processing_status = status_row["status"] if status_row else "pending"
        _update_status(
            database, run_id, paper_id, status="processing", stage="abstract_retrieval",
            attempts=int(status_row["attempts"] or 0) + 1 if status_row else 1,
        )
        has_abstract = bool((row.get("abstract") or "").strip())
        if not has_abstract and services.enrichment is not None:
            services.enrichment.enrich_papers(
                (paper_id,),
                workflow_batch_id=None,
                force_refresh=force_abstract_refresh,
            )
            row = _paper_row(database, paper_id)
            has_abstract = bool((row.get("abstract") or "").strip())
        if not has_abstract:
            metadata = json.loads(row.get("source_metadata_json") or "{}")
            audit = dict(metadata.get("abstract_enrichment") or {})
            if audit.get("execution_state") == "not_applicable":
                classification = {
                    "decision": "irrelevant",
                    "screening_basis": "deterministic_metadata_rule",
                    "education_centrality": "none",
                    "economic_relevance": "none",
                    "confidence": "high",
                    "evidence": [row.get("title") or ""],
                    "reason": (
                        "Journal front matter is not a research article; "
                        "an English abstract is not expected."
                    ),
                }
                _update_status(
                    database,
                    run_id,
                    paper_id,
                    status="classified",
                    stage="complete",
                    abstract_retrieval_json=json.dumps(
                        {
                            "status": "not_applicable",
                            "final_reason": audit.get("final_reason"),
                            "attempts": [],
                            "retryable": False,
                            "updated_at": utcnow(),
                        }
                    ),
                )
                _record_profile_assessment(
                    database,
                    profiles_ready=profiles_ready,
                    profile_id=profile,
                    run_id=run_id,
                    paper_id=paper_id,
                    classification=classification,
                    model_name="deterministic_metadata_rule",
                    prompt_version=FRONT_MATTER_PROMPT_VERSION,
                    title=row.get("title"),
                    abstract=None,
                )
                result["completed"].append(paper_id)
                continue
            if services.screening is not None:
                title_outcome = None
                try:
                    title_outcome = services.screening.screen_paper(
                        paper_id,
                        task_id=None,
                        force=prior_processing_status == "needs_retry",
                    )
                except Exception:
                    LOGGER.exception(
                        "title-only screening failed run_id=%s paper_id=%s",
                        run_id,
                        paper_id,
                    )
                if title_outcome is not None:
                    result["ai_calls"] = int(result["ai_calls"]) + int(title_outcome.ai_calls or 0)
                    for token_field in ("input_tokens", "output_tokens"):
                        result[token_field] = int(result[token_field]) + int(
                            getattr(title_outcome, token_field, 0) or 0
                        )
                if title_outcome is None or title_outcome.status == "needs_retry":
                    _update_status(
                        database, run_id, paper_id, status="needs_retry",
                        stage="title_only_screening", error_type="title_screening_failed",
                        error_message="标题初判暂未完成，可重试；英文摘要仍可继续获取。",
                    )
                    result["needs_retry"].append(paper_id)
                    continue
                if title_outcome.status == "classified":
                    classification = dict(title_outcome.classification)
                    classification.setdefault("screening_basis", "title_only")
                    _update_status(
                        database,
                        run_id,
                        paper_id,
                        status="classified",
                        stage="title_only_screening",
                    )
                    _record_profile_assessment(
                        database,
                        profiles_ready=profiles_ready,
                        profile_id=profile,
                        run_id=run_id,
                        paper_id=paper_id,
                        classification=classification,
                        model_name=getattr(services.screening, "model", None),
                        prompt_version=_screening_prompt_version(services),
                        title=row.get("title"),
                        abstract=None,
                    )
                    result["needs_abstract"].append(paper_id)
                    continue
            retryable = bool(audit.get("retryable")) or audit.get("execution_state") == "pending"
            status = "needs_retry" if retryable else "needs_manual_abstract"
            _update_status(
                database,
                run_id,
                paper_id,
                status=status,
                stage="awaiting_abstract_retrieval",
                abstract_retrieval_json=json.dumps(
                    {
                        "status": "needs_retry" if retryable else "needs_manual_abstract",
                        "final_reason": audit.get("final_reason"),
                        "attempts": list(audit.get("attempts") or []),
                        "retryable": retryable,
                        "updated_at": utcnow(),
                    }
                ),
            )
            result["needs_abstract"].append(paper_id)
            continue
        _update_status(
            database,
            run_id,
            paper_id,
            status="abstract_ready",
            stage="abstract_translation",
            abstract_retrieval_json=json.dumps(
                _successful_abstract_retrieval(row), ensure_ascii=False
            ),
        )
        if services.translation is not None:
            stats = services.translation.run(
                TranslationParameters(
                    review_status=None,
                    limit=1,
                    missing_only=True,
                    paper_ids=(paper_id,),
                    translate_titles=False,
                    translate_abstracts=True,
                    task_id=None,
                )
            )
            result["ai_calls"] = int(result["ai_calls"]) + int(stats.ai_calls)
            result["input_tokens"] = int(result["input_tokens"]) + int(stats.input_tokens)
            result["output_tokens"] = int(result["output_tokens"]) + int(stats.output_tokens)
            if stats.failed:
                _update_status(
                    database,
                    run_id,
                    paper_id,
                    status="needs_retry",
                    stage="abstract_translation",
                    error_type="translation_failed",
                    error_message="; ".join(stats.failure_reasons)[:500],
                )
                result["needs_retry"].append(paper_id)
                continue
        if services.screening is None:
            _update_status(
                database,
                run_id,
                paper_id,
                status="needs_retry",
                stage="screening_unavailable",
            )
            result["needs_retry"].append(paper_id)
            continue
        outcome = services.screening.screen_paper(
            paper_id,
            task_id=None,
            force=prior_processing_status == "needs_retry",
        )
        result["ai_calls"] = int(result["ai_calls"]) + int(outcome.ai_calls)
        # Token usage travels with the call count: recording that a run made N
        # calls but consumed no tokens would be a half-truth in the same row.
        result["input_tokens"] = int(result["input_tokens"]) + int(
            getattr(outcome, "input_tokens", 0) or 0
        )
        result["output_tokens"] = int(result["output_tokens"]) + int(
            getattr(outcome, "output_tokens", 0) or 0
        )
        status = (
            "classified"
            if outcome.status == "classified"
            else "needs_retry"
            if outcome.status == "needs_retry"
            else "needs_manual_abstract"
        )
        _update_status(
            database,
            run_id,
            paper_id,
            status=status,
            stage="complete" if outcome.status == "classified" else "strong_classification",
        )
        if outcome.status == "classified":
            _record_profile_assessment(
                database,
                profiles_ready=profiles_ready,
                profile_id=profile,
                run_id=run_id,
                paper_id=paper_id,
                classification=dict(outcome.classification),
                model_name=getattr(services.screening, "model", None),
                prompt_version=_screening_prompt_version(services),
                title=row.get("title"),
                abstract=row.get("abstract"),
            )
            result["completed"].append(paper_id)
        elif outcome.status == "needs_retry":
            result["needs_retry"].append(paper_id)
        else:
            result["needs_abstract"].append(paper_id)
    # Persist what this batch actually cost.  The counters above are the real
    # per-call counts reported by the screening and translation services; without
    # this write the run's audit row would keep claiming zero model calls.
    record_run_ai_usage(
        database,
        run_id,
        ai_calls=int(result["ai_calls"]),
        input_tokens=int(result["input_tokens"]),
        output_tokens=int(result["output_tokens"]),
    )
    return result


def _successful_abstract_retrieval(row: dict[str, object]) -> dict[str, object]:
    metadata = json.loads(row.get("source_metadata_json") or "{}")
    audit = metadata.get("abstract_enrichment")
    audit = audit if isinstance(audit, dict) else {}
    return {
        "status": "abstract_ready",
        "source": metadata.get("abstract_source"),
        "successful_source": audit.get("successful_source"),
        "final_reason": audit.get("final_reason"),
        "attempts": list(audit.get("attempts") or []),
        "retryable": False,
        "updated_at": utcnow(),
    }


def save_human_decisions(
    database: Database,
    run_id: int,
    drafts: dict[int, tuple[str, str | None]],
    *,
    profile_id: str | None = None,
) -> tuple[int, ...]:
    """Save human decisions for the run's profile.

    Human review belongs to the *paper under a profile*: the decision is stored
    once in ``paper_profile_reviews`` and mirrored onto the paper's processing
    rows **of the same profile**, so a freshly created run agrees with the
    library while another profile's decisions stay untouched.
    """
    normalized = {}
    for paper_id, (decision, note) in drafts.items():
        decision = str(decision or "").strip().casefold() or "undecided"
        if decision not in {"related", "irrelevant", "uncertain", "undecided"}:
            raise ValueError("Unsupported human decision")
        normalized[paper_id] = (decision, (note or "").strip() or None)
    profile = profile_id or run_profile_id(database, run_id)
    if profile_store.profile_schema_ready(database):
        with database.connect() as connection:
            run_paper_ids = tuple(
                int(row["paper_id"])
                for row in connection.execute(
                    "SELECT paper_id FROM paper_processing_status WHERE sync_run_id=?",
                    (run_id,),
                ).fetchall()
            )
        saved = list(
            profile_store.record_human_reviews(
                database,
                profile_id=profile,
                drafts=normalized,
                paper_ids=run_paper_ids,
            )
        )
        now = utcnow()
        with database.connect() as connection:
            for paper_id in saved:
                draft = normalized[paper_id]
                connection.execute(
                    """UPDATE paper_processing_status
                    SET human_decision=?,human_note=?,human_decided_at=?,updated_at=?
                    WHERE paper_id=? AND sync_run_id IN
                        (SELECT id FROM sync_runs WHERE profile_id=?)""",
                    (*draft, now, now, paper_id, profile),
                )
        return tuple(saved)

    saved = []
    now = utcnow()
    with database.connect() as connection:
        rows = connection.execute(
            """SELECT paper_id,human_decision,human_note
            FROM paper_processing_status WHERE sync_run_id=?""",
            (run_id,),
        ).fetchall()
    current = {int(row["paper_id"]): (row["human_decision"], row["human_note"]) for row in rows}
    with database.connect() as connection:
        for paper_id, draft in normalized.items():
            if draft[0] == "undecided":
                continue
            if paper_id not in current or current.get(paper_id) == draft:
                continue
            # Human review belongs to the *paper*, not to the retrieval batch:
            # every processing row for this paper is updated so the library and
            # a freshly created run agree (see
            # ensure_paper_statuses_for_paper_ids, which copies review data into
            # a new run). Keying this on paper_id alone is deliberate.
            connection.execute(
                """UPDATE paper_processing_status
                SET human_decision=?,human_note=?,human_decided_at=?,updated_at=?
                WHERE paper_id=?""",
                (*draft, now, now, paper_id),
            )
            saved.append(paper_id)
    return tuple(saved)


def retranslate_abstract(
    database: Database,
    run_id: int,
    paper_id: int,
    services: LocalPipelineServices,
) -> dict[str, object]:
    # See process_batch: loading the AI stack belongs to the user-triggered
    # operation, not to Streamlit's initial render.
    from eduecon_radar.services.translation import TranslationParameters

    if services.translation is None:
        raise ValueError("Translation service is not configured")
    stats = services.translation.run(
        TranslationParameters(
            review_status=None,
            limit=1,
            missing_only=False,
            paper_ids=(paper_id,),
            translate_titles=False,
            translate_abstracts=True,
            task_id=None,
        )
    )
    if stats.failed:
        raise RuntimeError(
            "Abstract translation failed: " + "; ".join(stats.failure_reasons)[:500]
        )
    if services.screening is not None:
        outcome = services.screening.screen_paper(
            paper_id,
            task_id=None,
            force=True,
        )
        status = (
            "classified"
            if outcome.status == "classified"
            else "needs_retry"
            if outcome.status == "needs_retry"
            else "needs_manual_abstract"
        )
        _update_status(
            database,
            run_id,
            paper_id,
            status=status,
            stage="complete" if status == "classified" else "strong_classification",
        )
        if status == "classified":
            _record_profile_assessment(
                database,
                profiles_ready=profile_store.profile_schema_ready(database),
                profile_id=services_profile_id(database, run_id, services),
                run_id=run_id,
                paper_id=paper_id,
                classification=dict(outcome.classification),
                model_name=getattr(services.screening, "model", None),
                prompt_version=_screening_prompt_version(services),
            )
        return {"ai_calls": stats.ai_calls + int(outcome.ai_calls or 0)}
    _update_status(
        database,
        run_id,
        paper_id,
        status="abstract_ready",
        stage="abstract_translation",
    )
    return {"ai_calls": stats.ai_calls}


def library_processing_run(database: Database, paper_id: int, profile_id: str | None) -> int:
    """Reuse this paper's run within the chosen direction, including library-only work."""
    scoped = profile_id is not None and profile_store.profile_schema_ready(database)
    with database.connect() as connection:
        if scoped:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT MAX(ps.sync_run_id) run_id FROM paper_processing_status ps "
                "JOIN sync_runs sr ON sr.id=ps.sync_run_id WHERE ps.paper_id=? AND sr.profile_id=?",
                (paper_id, profile_id),
            ).fetchone()
            if row["run_id"] is None:
                now = utcnow()
                cursor = connection.execute(
                    "INSERT INTO sync_runs (profile_id,started_at,finished_at,status) "
                    "VALUES (?,?,?,'success')", (profile_id, now, now),
                )
                run_id = int(cursor.lastrowid)
                connection.execute(
                    "INSERT INTO paper_processing_status "
                    "(sync_run_id,paper_id,status,stage,created_at,updated_at) "
                    "VALUES (?,?,'pending','pending',?,?)", (run_id, paper_id, now, now),
                )
                return run_id
        else:
            row = connection.execute(
                """SELECT MAX(sync_run_id) run_id
                FROM paper_processing_status WHERE paper_id=?""",
                (paper_id,),
            ).fetchone()
    if row is None or row["run_id"] is None:
        raise ValueError("论文没有可用的处理记录，无法补全")
    run_id = int(row["run_id"])
    return run_id


def reprocess_library_paper(
    database: Database,
    paper_id: int,
    services: LocalPipelineServices,
    *,
    force_abstract_refresh: bool = True,
) -> dict[str, object]:
    """Complete and re-analyze one library paper through the main chain.

    The paper keeps its latest sync run; missing titles, the English abstract,
    the Chinese abstract and the automatic judgment are all re-derived by the
    same processing chain used by normal retrieval and verification intake.
    """
    run_id = library_processing_run(database, paper_id, services.profile_id)
    translate_titles_for_paper_ids(database, run_id, (paper_id,), services)
    partial = process_batch(
        database,
        run_id,
        (paper_id,),
        services,
        force_abstract_refresh=force_abstract_refresh,
    )
    with database.connect() as connection:
        status_row = connection.execute(
            """SELECT status,stage,error_type,error_message
            FROM paper_processing_status
            WHERE sync_run_id=? AND paper_id=?""",
            (run_id, paper_id),
        ).fetchone()
    return {
        "paper_id": paper_id,
        "run_id": run_id,
        "partial": partial,
        "status": dict(status_row) if status_row else None,
    }


def rescreen_paper(
    database: Database,
    run_id: int,
    paper_id: int,
    services: LocalPipelineServices,
) -> dict[str, object]:
    if services.screening is None:
        raise ValueError("Screening service is not configured")
    outcome = services.screening.screen_paper(paper_id, task_id=None, force=True)
    status = (
        "classified"
        if outcome.status == "classified"
        else "needs_retry"
        if outcome.status == "needs_retry"
        else "needs_manual_abstract"
    )
    _update_status(
        database,
        run_id,
        paper_id,
        status=status,
        stage="complete" if outcome.status == "classified" else "strong_classification",
    )
    if status == "classified":
        _record_profile_assessment(
            database,
            profiles_ready=profile_store.profile_schema_ready(database),
            profile_id=run_profile_id(database, run_id),
            run_id=run_id,
            paper_id=paper_id,
            classification=dict(outcome.classification),
            model_name=getattr(services.screening, "model", None),
            prompt_version=_screening_prompt_version(services),
        )
    return outcome.classification


def status_label(status: str) -> str:
    return zh_status(status)


def export_mentor_csv(papers: list[dict[str, object]]) -> str:
    fields = [
        "paper_id", "doi", "original_title", "chinese_title", "authors",
        "journal", "publication_date", "original_abstract", "abstract_source",
        "chinese_abstract", "abstract_translation_source", "source_url", "human_note",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for paper in papers:
        writer.writerow(
            {
                "paper_id": paper["id"],
                "doi": paper.get("doi") or "",
                "original_title": paper["title"],
                "chinese_title": paper.get("chinese_title") or "",
                "authors": "; ".join(paper.get("authors") or []),
                "journal": paper.get("journal") or "",
                "publication_date": paper.get("publication_date") or "",
                "original_abstract": paper.get("abstract") or "",
                "abstract_source": paper.get("abstract_source") or "",
                "chinese_abstract": paper.get("chinese_abstract") or "",
                "abstract_translation_source": paper.get(
                    "abstract_translation_source"
                )
                or "",
                "source_url": paper.get("source_url") or "",
                "human_note": paper.get("reviewer_notes") or "",
            }
        )
    return "\ufeff" + output.getvalue()


def export_research_csv(papers: list[dict[str, object]]) -> str:
    fields = [
        "paper_id", "doi", "original_title", "chinese_title", "authors",
        "journal", "publication_date", "original_abstract", "abstract_source",
        "chinese_abstract", "abstract_translation_source", "source_url",
        "automatic_decision", "education_centrality", "economic_relevance",
        "confidence", "direct_evidence", "ai_reason", "human_decision", "human_note",
        "profile_id", "screening_basis", "model_name", "prompt_version",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for paper in papers:
        classification = dict(paper.get("processing", {}).get("classification") or {})
        writer.writerow(
            {
                "paper_id": paper["id"],
                "profile_id": paper.get("profile_id") or "",
                "screening_basis": classification.get("screening_basis") or "",
                "model_name": paper.get("processing", {}).get("model_name") or "",
                "prompt_version": paper.get("processing", {}).get("prompt_version") or "",
                "doi": paper.get("doi") or "",
                "original_title": paper["title"],
                "chinese_title": paper.get("chinese_title") or "",
                "authors": "; ".join(paper.get("authors") or []),
                "journal": paper.get("journal") or "",
                "publication_date": paper.get("publication_date") or "",
                "original_abstract": paper.get("abstract") or "",
                "abstract_source": paper.get("abstract_source") or "",
                "chinese_abstract": paper.get("chinese_abstract") or "",
                "abstract_translation_source": paper.get(
                    "abstract_translation_source"
                )
                or "",
                "source_url": paper.get("source_url") or "",
                "automatic_decision": classification.get("decision") or "",
                "education_centrality": classification.get("education_centrality") or "",
                "economic_relevance": classification.get("economic_relevance") or "",
                "confidence": classification.get("confidence") or "",
                "direct_evidence": " | ".join(classification.get("evidence") or []),
                "ai_reason": classification.get("reason") or "",
                "human_decision": paper.get("human_decision") or "undecided",
                "human_note": paper.get("reviewer_notes") or "",
            }
        )
    return "\ufeff" + output.getvalue()


def export_mentor_markdown(papers: list[dict[str, object]]) -> str:
    lines = ["# Literature Radar 文献清单", ""]
    for paper in papers:
        if paper.get("human_decision") != "related":
            continue
        lines.extend(
            [
                f"## {paper['title']}",
                "",
                f"**中文标题：** {paper.get('chinese_title') or 'Unavailable'}",
                f"**作者：** {', '.join(paper.get('authors') or []) or 'Unknown'}",
                f"**期刊：** {paper.get('journal') or ''}",
                f"**发表日期：** {paper.get('publication_date') or 'Unavailable'}",
                f"**DOI：** {paper.get('doi') or 'Unavailable'}",
                f"**Source Link：** {paper.get('source_url') or 'Unavailable'}",
                "",
                "### Original Abstract / 论文原版摘要",
                "",
                str(paper.get("abstract") or "Original abstract unavailable."),
                "",
                "### Chinese Abstract Translation / 中文摘要翻译",
                "",
                str(paper.get("chinese_abstract") or "Chinese translation unavailable."),
                "",
                "### Human Note / 人工备注",
                "",
                str(paper.get("reviewer_notes") or ""),
                "",
            ]
        )
    return "\n".join(lines)


def export_research_markdown(papers: list[dict[str, object]]) -> str:
    lines = ["# Literature Radar 文献研究记录", ""]
    for paper in papers:
        processing = dict(paper.get("processing") or {})
        classification = dict(processing.get("classification") or {})
        lines.extend(
            [
                f"## {paper['title']}",
                "",
                f"- Chinese title: {paper.get('chinese_title') or ''}",
                f"- Authors: {'; '.join(paper.get('authors') or [])}",
                f"- Journal: {paper.get('journal') or ''}",
                f"- Publication date: {paper.get('publication_date') or ''}",
                f"- DOI: {paper.get('doi') or ''}",
                f"- Abstract source: {paper.get('abstract_source') or ''}",
                f"- Automatic decision: {classification.get('decision') or ''}",
                f"- Education centrality: {classification.get('education_centrality') or ''}",
                f"- Economic relevance: {classification.get('economic_relevance') or ''}",
                f"- Confidence: {classification.get('confidence') or ''}",
                f"- Direct evidence: {' | '.join(classification.get('evidence') or [])}",
                f"- AI reason: {classification.get('reason') or ''}",
                f"- Human decision: {paper.get('human_decision') or 'undecided'}",
                f"- Human note: {paper.get('reviewer_notes') or ''}",
                "",
                "### Original Abstract",
                "",
                str(paper.get("abstract") or "Original abstract unavailable."),
                "",
                "### Chinese Abstract Translation",
                "",
                str(paper.get("chinese_abstract") or "Chinese translation unavailable."),
                "",
            ]
        )
    return "\n".join(lines)
