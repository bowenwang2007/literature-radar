# --- not part of the product core: see docs/PRODUCT_CORE.md ---
# LEGACY (workflow batch tables). Not reachable from the product core.

"""Atomic JSON-backed checkpoints and resumable workflow coordination."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from eduecon_radar.db import Database

STAGES = (
    "sync",
    "abstract_enrichment",
    "title_translation",
    "evidence",
    "fast_semantic",
    "strong_semantic",
    "composite",
    "missed_recovery",
    "abstract_translation",
    "final_review",
    "export",
)
STATUSES = {"pending", "running", "completed", "partial_failed", "failed", "budget_blocked"}
_ACTIVE_BATCHES: set[int] = set()
_ACTIVE_LOCK = threading.Lock()


class WorkflowAlreadyRunning(RuntimeError):
    """Raised for a duplicate in-process resume request."""


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class StageCheckpoint:
    status: str = "pending"
    started_at: str | None = None
    completed_at: str | None = None
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    pending_item_ids: list[int] = field(default_factory=list)
    failed_item_ids: list[int] = field(default_factory=list)
    config_fingerprint: str = ""
    input_fingerprint: str = ""
    retryable: bool = True
    error_summary: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unsupported checkpoint status: {self.status}")

    @classmethod
    def from_dict(cls, value: object) -> StageCheckpoint:
        if not isinstance(value, dict):
            return cls()
        known = cls.__dataclass_fields__
        try:
            return cls(**{key: item for key, item in value.items() if key in known})
        except (TypeError, ValueError):
            return cls()


class WorkflowCheckpointStore:
    """Preserves unrelated summary fields and writes a stage in one transaction."""

    summary_key = "phase6c_checkpoints"

    def __init__(self, database: Database) -> None:
        self.database = database

    def load(self, batch_id: int) -> dict[str, dict[str, Any]]:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT result_summary_json FROM workflow_batches WHERE id=?", (batch_id,)
            ).fetchone()
        summary = _safe_object(row[0] if row else None)
        stages = summary.get(self.summary_key)
        return dict(stages) if isinstance(stages, dict) else {}

    def get(self, batch_id: int, stage: str) -> StageCheckpoint:
        _require_stage(stage)
        return StageCheckpoint.from_dict(self.load(batch_id).get(stage))

    def should_run(self, batch_id: int, stage: str, config_fp: str, input_fp: str) -> bool:
        value = self.get(batch_id, stage)
        matches = (
            value.status == "completed"
            and value.config_fingerprint == config_fp
            and value.input_fingerprint == input_fp
        )
        if value.status == "completed" and not matches:
            self.invalidate_downstream(batch_id, stage)
        return not matches

    def save(self, batch_id: int, stage: str, checkpoint: StageCheckpoint) -> None:
        _require_stage(stage)
        now = _now()
        with self.database.connect() as connection:
            # IMMEDIATE serializes concurrent read/merge/write operations in SQLite.
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT result_summary_json FROM workflow_batches WHERE id=?", (batch_id,)
            ).fetchone()
            if not row:
                raise ValueError(f"workflow batch not found: {batch_id}")
            summary = _safe_object(row[0])
            stages = summary.get(self.summary_key)
            stages = dict(stages) if isinstance(stages, dict) else {}
            stages[stage] = asdict(checkpoint)
            summary[self.summary_key] = stages
            connection.execute(
                "UPDATE workflow_batches SET result_summary_json=?,updated_at=? WHERE id=?",
                (json.dumps(summary, ensure_ascii=False, sort_keys=True), now, batch_id),
            )

    def start(
        self,
        batch_id: int,
        stage: str,
        *,
        total: int,
        config_fp: str,
        input_fp: str,
        item_ids: tuple[int, ...] | list[int] = (),
    ) -> StageCheckpoint:
        ids = list(dict.fromkeys(int(item) for item in item_ids))
        item = StageCheckpoint(
            status="running",
            started_at=_now(),
            total=total,
            pending_item_ids=ids,
            config_fingerprint=config_fp,
            input_fingerprint=input_fp,
        )
        self.save(batch_id, stage, item)
        return item

    def finish(
        self,
        batch_id: int,
        stage: str,
        checkpoint: StageCheckpoint,
        *,
        succeeded: int,
        failed: int = 0,
        skipped: int = 0,
        failed_item_ids: tuple[int, ...] | list[int] = (),
        retryable: bool = True,
        errors: dict[str, str] | None = None,
    ) -> StageCheckpoint:
        failed_ids = list(dict.fromkeys(int(item) for item in failed_item_ids))
        checkpoint.status = (
            "completed" if not failed else ("partial_failed" if succeeded or skipped else "failed")
        )
        checkpoint.completed_at = _now()
        checkpoint.succeeded = succeeded
        checkpoint.failed = failed
        checkpoint.skipped = skipped
        checkpoint.failed_item_ids = failed_ids
        checkpoint.pending_item_ids = failed_ids
        checkpoint.retryable = retryable
        checkpoint.error_summary = dict(errors or {})
        self.save(batch_id, stage, checkpoint)
        return checkpoint

    def invalidate_downstream(self, batch_id: int, stage: str) -> None:
        """Invalidate only downstream stages; the changed stage is rerun by its caller."""
        _require_stage(stage)
        stages = self.load(batch_id)
        for downstream in STAGES[STAGES.index(stage) + 1 :]:
            if downstream in stages:
                old = StageCheckpoint.from_dict(stages[downstream])
                self.save(
                    batch_id,
                    downstream,
                    StageCheckpoint(
                        total=old.total,
                        pending_item_ids=list(old.pending_item_ids or old.failed_item_ids),
                    ),
                )

    def initialize_all(self, batch_id: int) -> None:
        existing = self.load(batch_id)
        for stage in STAGES:
            if stage not in existing:
                self.save(batch_id, stage, StageCheckpoint())


class StageRunner(Protocol):
    def run_stage(self, batch_id: int, stage: str, item_ids: tuple[int, ...] | None = None) -> Any:
        """Run a workflow stage and persist its own final checkpoint."""


class WorkflowResumeService:
    def __init__(self, database: Database, runner: StageRunner | None = None) -> None:
        self.database = database
        self.store = WorkflowCheckpointStore(database)
        self.runner = runner

    def get_resume_state(self, batch_id: int) -> dict[str, Any]:
        checkpoints = self.store.load(batch_id)
        normalized = {stage: StageCheckpoint.from_dict(checkpoints.get(stage)) for stage in STAGES}
        completed = [stage for stage in STAGES if normalized[stage].status == "completed"]
        current = next((stage for stage in STAGES if normalized[stage].status != "completed"), None)
        interrupted = [stage for stage in STAGES if normalized[stage].status == "running"]
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT status,source_config_json,updated_at FROM workflow_batches WHERE id=?",
                (batch_id,),
            ).fetchone()
        if not row:
            raise ValueError(f"workflow batch not found: {batch_id}")
        return {
            "batch_id": batch_id,
            "batch_status": row["status"],
            "scope": _safe_object(row["source_config_json"]),
            "completed_stages": completed,
            "current_stage": current,
            "interrupted_stages": interrupted,
            "updated_at": row["updated_at"],
            "checkpoints": {stage: asdict(item) for stage, item in normalized.items()},
        }

    def resume_batch(self, batch_id: int) -> Any:
        if not self.runner:
            return self.get_resume_state(batch_id)
        with _batch_claim(batch_id):
            state = self.get_resume_state(batch_id)
            stage = state["current_stage"]
            if stage is None:
                return state
            checkpoint = StageCheckpoint.from_dict(state["checkpoints"][stage])
            ids = checkpoint.failed_item_ids or checkpoint.pending_item_ids
            return self.runner.run_stage(batch_id, stage, tuple(ids) if ids else None)

    def retry_stage_failures(self, batch_id: int, stage_name: str) -> Any:
        _require_stage(stage_name)
        checkpoint = self.store.get(batch_id, stage_name)
        if checkpoint.status not in {"partial_failed", "failed", "running"}:
            return self.get_resume_state(batch_id)
        if not checkpoint.retryable:
            return self.get_resume_state(batch_id)
        if not self.runner:
            return tuple(checkpoint.failed_item_ids or checkpoint.pending_item_ids)
        with _batch_claim(batch_id):
            ids = tuple(checkpoint.failed_item_ids or checkpoint.pending_item_ids)
            return self.runner.run_stage(batch_id, stage_name, ids or None)

    def retry_abstract_failures(self, batch_id: int, temporary_only: bool) -> Any:
        checkpoint = self.store.get(batch_id, "abstract_enrichment")
        if temporary_only and not checkpoint.retryable:
            return self.get_resume_state(batch_id)
        specialized = getattr(self.runner, "retry_abstract_failures", None)
        if specialized:
            with _batch_claim(batch_id):
                return specialized(batch_id, temporary_only=temporary_only)
        return self.retry_stage_failures(batch_id, "abstract_enrichment")

    def retry_all_missing_abstracts(self, batch_id: int) -> Any:
        specialized = getattr(self.runner, "retry_abstract_failures", None)
        if not specialized:
            return self.retry_stage_failures(batch_id, "abstract_enrichment")
        with _batch_claim(batch_id):
            return specialized(batch_id, all_missing=True)

    def abandon_batch(self, batch_id: int) -> None:
        with self.database.connect() as connection:
            connection.execute(
                "UPDATE workflow_batches SET status='paused',last_error=?,updated_at=? WHERE id=? "
                "AND status!='completed'",
                ("abandoned_by_user", _now(), batch_id),
            )


class _batch_claim:
    def __init__(self, batch_id: int) -> None:
        self.batch_id = batch_id

    def __enter__(self) -> None:
        with _ACTIVE_LOCK:
            if self.batch_id in _ACTIVE_BATCHES:
                raise WorkflowAlreadyRunning(
                    "workflow batch is already running; duplicate resume was ignored"
                )
            _ACTIVE_BATCHES.add(self.batch_id)

    def __exit__(self, *args: object) -> None:
        with _ACTIVE_LOCK:
            _ACTIVE_BATCHES.discard(self.batch_id)


def _safe_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _require_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"unsupported workflow stage: {stage}")


def _now() -> str:
    return datetime.now(UTC).isoformat()
