# --- not part of the product core: see docs/PRODUCT_CORE.md ---
# LEGACY (workflow batch tables). Not reachable from the product core.

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import date

from eduecon_radar.db import Database, utcnow
from eduecon_radar.journals import JournalFile
from eduecon_radar.services.ai_screening import ScreeningService, ScreeningStats
from eduecon_radar.services.ingestion import SyncService
from eduecon_radar.services.migrations import (
    require_translation_schema,
    require_workflow_schema,
)
from eduecon_radar.services.translation import (
    TranslationParameters,
    TranslationService,
    TranslationStats,
)
from eduecon_radar.services.web_sync import SyncParameters, run_sync

WORKFLOW_STATUSES = {
    "draft",
    "ready",
    "running",
    "paused",
    "ready_for_review",
    "completed",
    "failed",
}
INCOMPLETE_STATUSES = ("draft", "ready", "running", "paused", "ready_for_review", "failed")
FINAL_REVIEW_STATUSES = ("approved", "rejected", "needs_revision")
SUCCESSFUL_SCREENING_DECISIONS = ("include", "borderline", "exclude")


@dataclass(frozen=True)
class WorkflowPolicy:
    limit: int = 10
    retry_failed: bool = False
    force_rescreen: bool = False
    allow_strong: bool = True
    translate_borderline: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.limit <= 100:
            raise ValueError("Workflow limit must be between 1 and 100")

    @property
    def translation_decisions(self) -> tuple[str, ...]:
        return ("include", "borderline") if self.translate_borderline else ("include",)

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            self.limit,
            self.retry_failed,
            self.force_rescreen,
            self.allow_strong,
            self.translate_borderline,
        )


@dataclass(frozen=True)
class WorkflowPreview:
    batch_id: int
    batch_name: str
    source_type: str
    batch_total: int
    sync_journals: int
    sync_fetched: int
    screening_planned: int
    screening_skipped: int
    translation_decisions: tuple[str, ...]
    title_skipped: int
    title_pending: int
    title_calls_max: int
    abstract_calls_max: int
    abstract_unavailable: int
    translation_skipped: int
    human_protected: int
    model_name: str
    max_ai_calls: int
    signature: tuple[object, ...]


@dataclass(frozen=True)
class WorkflowRunResult:
    batch_id: int
    status: str
    summary: dict[str, object]


def create_workflow_batch(
    database: Database,
    *,
    name: str,
    source_type: str,
    paper_ids: Iterable[int] = (),
    source_config: dict[str, object] | None = None,
    policy: WorkflowPolicy | None = None,
) -> int:
    require_workflow_schema(database)
    if source_type not in {"selected_papers", "sync"}:
        raise ValueError("Unsupported workflow source")
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("Batch name is required")
    now = utcnow()
    selected_policy = policy or WorkflowPolicy()
    screening_policy, translation_policy = _policy_json(selected_policy)
    with database.connect() as connection:
        cursor = connection.execute(
            """INSERT INTO workflow_batches
            (name,source_type,status,source_config_json,screening_policy,translation_policy,
             created_at,updated_at,result_summary_json)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                normalized_name,
                source_type,
                "draft",
                json.dumps(source_config or {}, ensure_ascii=False, sort_keys=True),
                screening_policy,
                translation_policy,
                now,
                now,
                "{}",
            ),
        )
        batch_id = int(cursor.lastrowid)
    add_batch_items(database, batch_id, paper_ids)
    return batch_id


def add_batch_items(database: Database, batch_id: int, paper_ids: Iterable[int]) -> int:
    require_workflow_schema(database)
    unique_ids = list(dict.fromkeys(int(paper_id) for paper_id in paper_ids))
    if not unique_ids:
        return 0
    now = utcnow()
    added = 0
    with database.connect() as connection:
        if not connection.execute(
            "SELECT 1 FROM workflow_batches WHERE id=?", (batch_id,)
        ).fetchone():
            raise ValueError(f"Workflow batch not found: {batch_id}")
        existing_papers = {
            int(row[0])
            for row in connection.execute(
                f"SELECT id FROM papers WHERE id IN ({','.join('?' for _ in unique_ids)})",
                unique_ids,
            )
        }
        missing = set(unique_ids) - existing_papers
        if missing:
            raise ValueError(f"Papers not found: {sorted(missing)}")
        position = int(
            connection.execute(
                "SELECT COALESCE(MAX(position),0) FROM workflow_batch_items WHERE batch_id=?",
                (batch_id,),
            ).fetchone()[0]
        )
        for paper_id in unique_ids:
            position += 1
            cursor = connection.execute(
                """INSERT OR IGNORE INTO workflow_batch_items
                (batch_id,paper_id,position,added_at) VALUES (?,?,?,?)""",
                (batch_id, paper_id, position, now),
            )
            added += max(cursor.rowcount, 0)
        connection.execute("UPDATE workflow_batches SET updated_at=? WHERE id=?", (now, batch_id))
    return added


def remove_batch_items(database: Database, batch_id: int, paper_ids: Iterable[int]) -> int:
    require_workflow_schema(database)
    unique_ids = list(dict.fromkeys(int(paper_id) for paper_id in paper_ids))
    if not unique_ids:
        return 0
    with database.connect() as connection:
        cursor = connection.execute(
            f"""DELETE FROM workflow_batch_items WHERE batch_id=?
            AND paper_id IN ({",".join("?" for _ in unique_ids)})""",
            (batch_id, *unique_ids),
        )
        connection.execute(
            "UPDATE workflow_batches SET updated_at=? WHERE id=?", (utcnow(), batch_id)
        )
    return max(cursor.rowcount, 0)


def batch_paper_ids(database: Database, batch_id: int) -> list[int]:
    require_workflow_schema(database)
    with database.connect() as connection:
        rows = connection.execute(
            """SELECT paper_id FROM workflow_batch_items WHERE batch_id=?
            ORDER BY position,paper_id""",
            (batch_id,),
        ).fetchall()
    return [int(row[0]) for row in rows]


def batch_item_count(database: Database, batch_id: int) -> int:
    require_workflow_schema(database)
    with database.connect() as connection:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM workflow_batch_items WHERE batch_id=?", (batch_id,)
            ).fetchone()[0]
        )


def get_workflow_batch(database: Database, batch_id: int) -> dict[str, object]:
    require_workflow_schema(database)
    with database.connect() as connection:
        row = connection.execute(
            """SELECT wb.*,(SELECT COUNT(*) FROM workflow_batch_items wbi
            WHERE wbi.batch_id=wb.id) item_count FROM workflow_batches wb WHERE wb.id=?""",
            (batch_id,),
        ).fetchone()
    if not row:
        raise ValueError(f"Workflow batch not found: {batch_id}")
    return _decode_batch(row)


def list_workflow_batches(
    database: Database, *, incomplete_only: bool = False
) -> list[dict[str, object]]:
    require_workflow_schema(database)
    where = ""
    params: tuple[object, ...] = ()
    if incomplete_only:
        where = f"WHERE wb.status IN ({','.join('?' for _ in INCOMPLETE_STATUSES)})"
        params = INCOMPLETE_STATUSES
    with database.connect() as connection:
        rows = connection.execute(
            f"""SELECT wb.*,(SELECT COUNT(*) FROM workflow_batch_items wbi
            WHERE wbi.batch_id=wb.id) item_count FROM workflow_batches wb {where}
            ORDER BY wb.updated_at DESC,wb.id DESC""",
            params,
        ).fetchall()
    return [_decode_batch(row) for row in rows]


def update_workflow_policy(database: Database, batch_id: int, policy: WorkflowPolicy) -> None:
    require_workflow_schema(database)
    screening_policy, translation_policy = _policy_json(policy)
    with database.connect() as connection:
        cursor = connection.execute(
            """UPDATE workflow_batches SET screening_policy=?,translation_policy=?,
            status=CASE WHEN status='draft' THEN 'ready' ELSE status END,updated_at=?
            WHERE id=?""",
            (screening_policy, translation_policy, utcnow(), batch_id),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"Workflow batch not found: {batch_id}")


def workflow_policy(batch: dict[str, object]) -> WorkflowPolicy:
    screening = dict(batch.get("screening_policy") or {})
    translation = dict(batch.get("translation_policy") or {})
    decisions = tuple(translation.get("decisions") or ("include", "borderline"))
    return WorkflowPolicy(
        limit=int(screening.get("limit", 10)),
        retry_failed=bool(screening.get("retry_failed", False)),
        force_rescreen=bool(screening.get("force_rescreen", False)),
        allow_strong=bool(screening.get("allow_strong", True)),
        translate_borderline="borderline" in decisions,
    )


def batch_review_progress(database: Database, batch_id: int) -> dict[str, int]:
    require_workflow_schema(database)
    with database.connect() as connection:
        row = connection.execute(
            """SELECT COUNT(*) total,
            SUM(CASE WHEN r.status='pending' THEN 1 ELSE 0 END) remaining
            FROM workflow_batch_items wbi
            JOIN reviews r ON r.paper_id=wbi.paper_id
            JOIN screenings s ON s.id=(SELECT s2.id FROM screenings s2
                WHERE s2.paper_id=wbi.paper_id ORDER BY s2.screened_at DESC,s2.id DESC LIMIT 1)
            WHERE wbi.batch_id=?
            AND s.ai_decision IN ('include','borderline','unprocessed')""",
            (batch_id,),
        ).fetchone()
    total = int(row["total"] or 0)
    remaining = int(row["remaining"] or 0)
    return {"total": total, "reviewed": total - remaining, "remaining": remaining}


def complete_batch_if_reviewed(database: Database, batch_id: int) -> bool:
    progress = batch_review_progress(database, batch_id)
    batch = get_workflow_batch(database, batch_id)
    if progress["remaining"] or batch["status"] not in {"ready_for_review", "completed"}:
        return False
    now = utcnow()
    with database.connect() as connection:
        connection.execute(
            """UPDATE workflow_batches SET status='completed',completed_at=?,updated_at=?
            WHERE id=?""",
            (now, now, batch_id),
        )
    return True


class WorkflowOrchestrator:
    def __init__(
        self,
        database: Database,
        *,
        screening_service: ScreeningService,
        translation_service: TranslationService,
        model_name: str,
        sync_service: SyncService | None = None,
        journal_file: JournalFile | None = None,
    ) -> None:
        self.database = database
        self.screening_service = screening_service
        self.translation_service = translation_service
        self.model_name = model_name
        self.sync_service = sync_service
        self.journal_file = journal_file

    def preview(self, batch_id: int, policy: WorkflowPolicy) -> WorkflowPreview:
        require_workflow_schema(self.database)
        require_translation_schema(self.database)
        batch = get_workflow_batch(self.database, batch_id)
        sync_result: dict[str, object] = {}
        if batch["source_type"] == "sync" and not batch.get("sync_completed_at"):
            sync_result = self._sync(dry_run=True, batch=batch)
        facts = self._facts(batch_id)
        screen_ids = self._screening_candidate_ids(facts, policy)[: policy.limit]
        eligible_existing = [
            fact
            for fact in facts
            if fact["review_status"] == "pending"
            and fact["ai_decision"] in policy.translation_decisions
        ]
        batch_total = len(facts) or int(sync_result.get("records_fetched", 0) or 0)
        screening_planned = len(screen_ids) if facts else min(batch_total, policy.limit)
        predicted_new = screening_planned
        possible = [*eligible_existing]
        title_calls = sum(not fact["chinese_title"] for fact in facts)
        title_skipped = sum(bool(fact["chinese_title"]) for fact in facts)
        if sync_result:
            title_calls = batch_total
            title_skipped = 0
        abstract_calls = sum(
            bool(fact["abstract"]) and not fact["chinese_abstract"] for fact in possible
        )
        abstract_missing = sum(not fact["abstract"] for fact in possible)
        if sync_result:
            missing = int(sync_result.get("records_without_abstract", 0) or 0)
            abstract_missing += missing
            abstract_calls += max(predicted_new - missing, 0)
        translation_skipped = sum(_translation_complete(fact) for fact in eligible_existing)
        human_protected = sum(
            fact["title_translation_source"] == "human_edited"
            or fact["abstract_translation_source"] == "human_edited"
            for fact in facts
        )
        screening_skipped = sum(
            fact["review_status"] in FINAL_REVIEW_STATUSES
            or fact["ai_decision"] in SUCCESSFUL_SCREENING_DECISIONS
            for fact in facts
        )
        signature = self.signature(batch, policy)
        return WorkflowPreview(
            batch_id=batch_id,
            batch_name=str(batch["name"]),
            source_type=str(batch["source_type"]),
            batch_total=batch_total,
            sync_journals=int(sync_result.get("journals_checked", 0) or 0),
            sync_fetched=int(sync_result.get("records_fetched", 0) or 0),
            screening_planned=screening_planned,
            screening_skipped=screening_skipped,
            translation_decisions=policy.translation_decisions,
            title_skipped=title_skipped,
            title_pending=title_calls,
            title_calls_max=min(title_calls, policy.limit),
            abstract_calls_max=min(abstract_calls, policy.limit),
            abstract_unavailable=abstract_missing,
            translation_skipped=translation_skipped,
            human_protected=human_protected,
            model_name=self.model_name,
            max_ai_calls=min(title_calls, policy.limit)
            + screening_planned * 3
            + min(abstract_calls, policy.limit),
            signature=signature,
        )

    def signature(self, batch: dict[str, object], policy: WorkflowPolicy) -> tuple[object, ...]:
        return (
            int(batch["id"]),
            str(batch["source_type"]),
            json.dumps(batch.get("source_config") or {}, ensure_ascii=False, sort_keys=True),
            policy.signature,
            tuple(batch_paper_ids(self.database, int(batch["id"]))),
        )

    def run(
        self,
        batch_id: int,
        policy: WorkflowPolicy,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> WorkflowRunResult:
        require_workflow_schema(self.database)
        require_translation_schema(self.database)
        batch = get_workflow_batch(self.database, batch_id)
        update_workflow_policy(self.database, batch_id, policy)
        self._set_status(batch_id, "running", error=None)
        summary: dict[str, object] = {
            "sync": {},
            "screening": {},
            "title_translation": {},
            "abstract_translation": {},
        }
        try:
            if batch["source_type"] == "sync" and not batch.get("sync_completed_at"):
                _notify(progress, "正在同步")
                sync_stats = self._sync(dry_run=False, batch=batch)
                summary["sync"] = sync_stats
                ids = list(sync_stats.get("inserted_paper_ids", []))
                config = dict(batch.get("source_config") or {})
                if config.get("include_updated"):
                    ids.extend(self._awaiting_ids(sync_stats.get("updated_paper_ids", [])))
                add_batch_items(self.database, batch_id, ids)
                if int(sync_stats.get("records_failed", 0) or 0):
                    return self._pause(
                        batch_id,
                        summary,
                        "同步部分失败；继续时将重试未完成同步。",
                    )
                self._mark_stage(batch_id, "sync_completed_at")

            _notify(progress, "正在准备中文标题")
            title_ids = self._title_candidate_ids(batch_id)[: policy.limit]
            title_stats = TranslationStats()
            if title_ids:
                title_stats = self.translation_service.run(
                    TranslationParameters(
                        review_status=None,
                        limit=policy.limit,
                        missing_only=True,
                        overwrite_human=False,
                        paper_ids=tuple(title_ids),
                        translate_titles=True,
                        translate_abstracts=False,
                    )
                )
            summary["title_translation"] = asdict(title_stats)
            missing_titles = self._missing_title_count(batch_id)
            failed_titles = self._failed_title_count(batch_id)
            if missing_titles > failed_titles:
                return self._pause(
                    batch_id,
                    summary,
                    "仍有标题受本次数量上限影响；继续工作流后再进行筛选。",
                )

            _notify(progress, "正在进行AI筛选")
            facts = self._facts(batch_id)
            screen_ids = self._screening_candidate_ids(facts, policy)[: policy.limit]
            screening_stats = ScreeningStats()
            if screen_ids:
                screening_stats = self.screening_service.run(
                    paper_ids=screen_ids,
                    allow_strong=policy.allow_strong,
                    force_rescreen=policy.force_rescreen,
                )
            summary["screening"] = asdict(screening_stats)

            _notify(progress, "正在翻译相关论文摘要")
            translate_ids = self._translation_candidate_ids(batch_id, policy.translation_decisions)[
                : policy.limit
            ]
            translation_stats = TranslationStats()
            if translate_ids:
                translation_stats = self.translation_service.run(
                    TranslationParameters(
                        review_status="pending",
                        limit=policy.limit,
                        missing_only=True,
                        overwrite_human=False,
                        paper_ids=tuple(translate_ids),
                        translate_titles=False,
                        translate_abstracts=True,
                    )
                )
            summary["abstract_translation"] = asdict(translation_stats)

            _notify(progress, "正在准备审核队列")
            decision_counts = self._decision_counts(batch_id)
            progress_counts = batch_review_progress(self.database, batch_id)
            summary.update(decision_counts)
            summary["ready_for_review"] = progress_counts["remaining"]
            summary["screening_skipped"] = self._successful_screening_count(batch_id)
            summary["translation_skipped"] = self._complete_translation_count(batch_id)
            summary["human_protected"] = self._human_protected_count(batch_id)

            remaining_screen = self._unscreened_count(batch_id)
            failed_screen = self._failed_screening_count(batch_id)
            remaining_translation = self._translation_candidate_ids(
                batch_id, policy.translation_decisions
            )
            failed_abstract = self._failed_abstract_count(batch_id)
            if not remaining_screen:
                self._mark_stage(batch_id, "screening_completed_at")
            if len(remaining_translation) <= failed_abstract:
                self._mark_stage(batch_id, "translation_completed_at")
            if remaining_screen or len(remaining_translation) > failed_abstract:
                return self._pause(
                    batch_id,
                    summary,
                    "仍有筛选或摘要翻译受本次数量上限影响。",
                )
            warnings = []
            if failed_titles:
                warnings.append(f"标题翻译失败 {failed_titles} 篇")
            if failed_screen:
                warnings.append(f"筛选失败 {failed_screen} 篇")
            if failed_abstract:
                warnings.append(f"摘要翻译失败 {failed_abstract} 篇")
            self._save_summary(
                batch_id,
                summary,
                status="ready_for_review",
                error="；".join(warnings) or None,
            )
            return WorkflowRunResult(batch_id, "ready_for_review", summary)
        except Exception as exc:
            self._save_summary(batch_id, summary, status="failed", error=str(exc)[:1000])
            raise

    def _sync(self, *, dry_run: bool, batch: dict[str, object]) -> dict[str, object]:
        if self.sync_service is None or self.journal_file is None:
            raise ValueError("Sync workflow dependencies are not configured")
        config = dict(batch.get("source_config") or {})
        parameters = SyncParameters(
            tuple(str(item) for item in config.get("journals", [])),
            date.fromisoformat(str(config["since"])),
            int(config["limit"]),
        )
        return run_sync(
            self.sync_service,
            self.journal_file,
            parameters,
            dry_run=dry_run,
        )

    def _facts(self, batch_id: int) -> list[dict[str, object]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT p.id,p.abstract,r.status review_status,s.ai_decision,
                pt.chinese_title,pt.chinese_abstract,pt.title_translation_source,
                pt.abstract_translation_source
                FROM workflow_batch_items wbi JOIN papers p ON p.id=wbi.paper_id
                JOIN reviews r ON r.paper_id=p.id
                LEFT JOIN screenings s ON s.id=(SELECT s2.id FROM screenings s2
                    WHERE s2.paper_id=p.id ORDER BY s2.screened_at DESC,s2.id DESC LIMIT 1)
                LEFT JOIN paper_translations pt ON pt.paper_id=p.id
                WHERE wbi.batch_id=? ORDER BY wbi.position,p.id""",
                (batch_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _screening_candidate_ids(
        facts: list[dict[str, object]], policy: WorkflowPolicy
    ) -> list[int]:
        selected = []
        for fact in facts:
            if fact["review_status"] in FINAL_REVIEW_STATUSES:
                continue
            decision = fact["ai_decision"]
            if (
                policy.force_rescreen
                or decision is None
                or (decision == "unprocessed" and policy.retry_failed)
            ):
                selected.append(int(fact["id"]))
        return selected

    def _translation_candidate_ids(self, batch_id: int, decisions: tuple[str, ...]) -> list[int]:
        placeholders = ",".join("?" for _ in decisions)
        with self.database.connect() as connection:
            rows = connection.execute(
                f"""SELECT p.id FROM workflow_batch_items wbi
                JOIN papers p ON p.id=wbi.paper_id JOIN reviews r ON r.paper_id=p.id
                JOIN screenings s ON s.id=(SELECT s2.id FROM screenings s2
                    WHERE s2.paper_id=p.id ORDER BY s2.screened_at DESC,s2.id DESC LIMIT 1)
                LEFT JOIN paper_translations pt ON pt.paper_id=p.id
                WHERE wbi.batch_id=? AND r.status='pending'
                AND s.ai_decision IN ({placeholders})
                AND p.abstract IS NOT NULL AND pt.chinese_abstract IS NULL
                ORDER BY wbi.position,p.id""",
                (batch_id, *decisions),
            ).fetchall()
        return [int(row[0]) for row in rows]

    def _title_candidate_ids(self, batch_id: int) -> list[int]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT p.id FROM workflow_batch_items wbi
                JOIN papers p ON p.id=wbi.paper_id
                LEFT JOIN paper_translations pt ON pt.paper_id=p.id
                WHERE wbi.batch_id=? AND (
                    pt.chinese_title IS NULL OR
                    (p.abstract IS NULL AND
                     COALESCE(pt.abstract_translation_status,'pending')!='unavailable')
                ) ORDER BY wbi.position,p.id""",
                (batch_id,),
            ).fetchall()
        return [int(row[0]) for row in rows]

    def _awaiting_ids(self, paper_ids: object) -> list[int]:
        ids = [int(item) for item in paper_ids] if isinstance(paper_ids, list) else []
        if not ids:
            return []
        with self.database.connect() as connection:
            rows = connection.execute(
                f"""SELECT p.id FROM papers p WHERE p.id IN ({",".join("?" for _ in ids)})
                AND NOT EXISTS (SELECT 1 FROM screenings s WHERE s.paper_id=p.id)""",
                ids,
            ).fetchall()
        return [int(row[0]) for row in rows]

    def _decision_counts(self, batch_id: int) -> dict[str, int]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT s.ai_decision,COUNT(*) count FROM workflow_batch_items wbi
                JOIN screenings s ON s.id=(SELECT s2.id FROM screenings s2
                    WHERE s2.paper_id=wbi.paper_id ORDER BY s2.screened_at DESC,s2.id DESC LIMIT 1)
                WHERE wbi.batch_id=? GROUP BY s.ai_decision""",
                (batch_id,),
            ).fetchall()
        counts = {"include": 0, "borderline": 0, "exclude": 0, "unprocessed": 0}
        counts.update({str(row[0]): int(row[1]) for row in rows})
        return counts

    def _count_where(self, batch_id: int, joins: str, condition: str) -> int:
        with self.database.connect() as connection:
            return int(
                connection.execute(
                    f"""SELECT COUNT(*) FROM workflow_batch_items wbi {joins}
                    WHERE wbi.batch_id=? AND {condition}""",
                    (batch_id,),
                ).fetchone()[0]
            )

    def _successful_screening_count(self, batch_id: int) -> int:
        return self._count_where(
            batch_id,
            "JOIN screenings s ON s.id=(SELECT s2.id FROM screenings s2 "
            "WHERE s2.paper_id=wbi.paper_id ORDER BY s2.screened_at DESC,s2.id DESC LIMIT 1)",
            "s.ai_decision IN ('include','borderline','exclude')",
        )

    def _failed_screening_count(self, batch_id: int) -> int:
        return self._count_where(
            batch_id,
            "JOIN screenings s ON s.id=(SELECT s2.id FROM screenings s2 "
            "WHERE s2.paper_id=wbi.paper_id ORDER BY s2.screened_at DESC,s2.id DESC LIMIT 1)",
            "s.ai_decision='unprocessed'",
        )

    def _unscreened_count(self, batch_id: int) -> int:
        return self._count_where(
            batch_id,
            "",
            "NOT EXISTS (SELECT 1 FROM screenings s WHERE s.paper_id=wbi.paper_id)",
        )

    def _missing_title_count(self, batch_id: int) -> int:
        return self._count_where(
            batch_id,
            "LEFT JOIN paper_translations pt ON pt.paper_id=wbi.paper_id",
            "pt.chinese_title IS NULL",
        )

    def _failed_title_count(self, batch_id: int) -> int:
        return self._count_where(
            batch_id,
            "JOIN paper_translations pt ON pt.paper_id=wbi.paper_id",
            "pt.title_translation_status='failed'",
        )

    def _complete_translation_count(self, batch_id: int) -> int:
        return self._count_where(
            batch_id,
            "JOIN papers p ON p.id=wbi.paper_id JOIN paper_translations pt ON pt.paper_id=p.id",
            "pt.chinese_title IS NOT NULL AND "
            "(p.abstract IS NULL OR pt.chinese_abstract IS NOT NULL)",
        )

    def _failed_abstract_count(self, batch_id: int) -> int:
        return self._count_where(
            batch_id,
            "JOIN paper_translations pt ON pt.paper_id=wbi.paper_id",
            "pt.abstract_translation_status='failed'",
        )

    def _human_protected_count(self, batch_id: int) -> int:
        return self._count_where(
            batch_id,
            "JOIN paper_translations pt ON pt.paper_id=wbi.paper_id",
            "pt.title_translation_source='human_edited' "
            "OR pt.abstract_translation_source='human_edited'",
        )

    def _mark_stage(self, batch_id: int, field: str) -> None:
        if field not in {
            "sync_completed_at",
            "screening_completed_at",
            "translation_completed_at",
        }:
            raise ValueError("Unsupported workflow stage")
        now = utcnow()
        with self.database.connect() as connection:
            connection.execute(
                f"UPDATE workflow_batches SET {field}=?,updated_at=? WHERE id=?",
                (now, now, batch_id),
            )

    def _set_status(self, batch_id: int, status: str, *, error: str | None) -> None:
        if status not in WORKFLOW_STATUSES:
            raise ValueError("Unsupported workflow status")
        with self.database.connect() as connection:
            connection.execute(
                """UPDATE workflow_batches SET status=?,last_error=?,updated_at=? WHERE id=?""",
                (status, error, utcnow(), batch_id),
            )

    def _pause(self, batch_id: int, summary: dict[str, object], error: str) -> WorkflowRunResult:
        self._save_summary(batch_id, summary, status="paused", error=error)
        return WorkflowRunResult(batch_id, "paused", summary)

    def _save_summary(
        self,
        batch_id: int,
        summary: dict[str, object],
        *,
        status: str,
        error: str | None,
    ) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """UPDATE workflow_batches SET status=?,result_summary_json=?,last_error=?,
                updated_at=? WHERE id=?""",
                (
                    status,
                    json.dumps(summary, ensure_ascii=False, default=str),
                    error,
                    utcnow(),
                    batch_id,
                ),
            )


def _decode_batch(row: object) -> dict[str, object]:
    item = dict(row)  # type: ignore[arg-type]
    item["source_config"] = json.loads(item.pop("source_config_json") or "{}")
    item["screening_policy"] = json.loads(item["screening_policy"] or "{}")
    item["translation_policy"] = json.loads(item["translation_policy"] or "{}")
    item["result_summary"] = json.loads(item.pop("result_summary_json") or "{}")
    return item


def _policy_json(policy: WorkflowPolicy) -> tuple[str, str]:
    screening = {
        "limit": policy.limit,
        "retry_failed": policy.retry_failed,
        "force_rescreen": policy.force_rescreen,
        "allow_strong": policy.allow_strong,
    }
    translation = {
        "decisions": list(policy.translation_decisions),
        "missing_only": True,
        "overwrite_human": False,
    }
    return (
        json.dumps(screening, ensure_ascii=False, sort_keys=True),
        json.dumps(translation, ensure_ascii=False, sort_keys=True),
    )


def _translation_complete(fact: dict[str, object]) -> bool:
    return bool(fact["chinese_title"] and (not fact["abstract"] or fact["chinese_abstract"]))


def _notify(callback: Callable[[str], None] | None, message: str) -> None:
    if callback:
        callback(message)
