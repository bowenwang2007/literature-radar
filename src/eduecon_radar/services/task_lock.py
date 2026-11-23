from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from eduecon_radar.db import Database


class TaskLeaseError(RuntimeError):
    """Base class for fail-closed workflow lease errors."""


class TaskLeaseBusy(TaskLeaseError):
    def __init__(self, summary: dict[str, object]) -> None:
        self.summary = summary
        super().__init__(
            "Task is already being processed by another executor "
            f"(owner={summary['owner']}, stage={summary['stage']}, "
            f"heartbeat={summary['heartbeat_at']}, expires={summary['expires_at']})"
        )


class TaskLeaseLost(TaskLeaseError):
    """Raised when a stale executor tries to continue or commit."""


@dataclass(frozen=True)
class TaskLease:
    task_id: int
    owner_id: str
    fencing_token: int
    acquired_at: str
    heartbeat_at: str
    expires_at: str
    current_stage: str

    @property
    def safe_summary(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "owner": _safe_owner(self.owner_id),
            "stage": self.current_stage,
            "heartbeat_at": self.heartbeat_at,
            "expires_at": self.expires_at,
            "fencing_token": self.fencing_token,
        }


class TaskLeaseManager:
    def __init__(self, database: Database, *, ttl_seconds: int = 300) -> None:
        if ttl_seconds < 5:
            raise ValueError("lease TTL must be at least five seconds")
        self.database = database
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def new_owner_id() -> str:
        return uuid.uuid4().hex

    def acquire(
        self,
        task_id: int,
        *,
        owner_id: str,
        stage: str,
        now: datetime | None = None,
    ) -> TaskLease:
        owner = owner_id.strip()
        if not owner:
            raise ValueError("owner_id is required")
        current = _as_utc(now)
        timestamp = current.isoformat()
        expires = (current + timedelta(seconds=self.ttl_seconds)).isoformat()
        try:
            with self.database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                if not connection.execute(
                    "SELECT 1 FROM workflow_batches WHERE id=?", (task_id,)
                ).fetchone():
                    raise ValueError(f"workflow task not found: {task_id}")
                row = connection.execute(
                    """SELECT * FROM workflow_task_leases
                    WHERE task_id=? AND status='active'""",
                    (task_id,),
                ).fetchone()
                if row and _parse(row["expires_at"]) > current:
                    if row["owner_id"] != owner:
                        raise TaskLeaseBusy(_row_summary(row))
                    connection.execute(
                        """UPDATE workflow_task_leases SET heartbeat_at=?,expires_at=?,
                        current_stage=?,updated_at=? WHERE id=?""",
                        (timestamp, expires, stage, timestamp, row["id"]),
                    )
                    return TaskLease(
                        task_id,
                        owner,
                        int(row["fencing_token"]),
                        row["acquired_at"],
                        timestamp,
                        expires,
                        stage,
                    )
                if row:
                    connection.execute(
                        """UPDATE workflow_task_leases SET status='stale',released_at=?,
                        updated_at=? WHERE id=? AND status='active'""",
                        (timestamp, timestamp, row["id"]),
                    )
                token = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(fencing_token),0)+1 "
                        "FROM workflow_task_leases WHERE task_id=?",
                        (task_id,),
                    ).fetchone()[0]
                )
                cursor = connection.execute(
                    """INSERT INTO workflow_task_leases
                    (task_id,owner_id,acquired_at,heartbeat_at,expires_at,released_at,status,
                     current_stage,fencing_token,created_at,updated_at)
                    VALUES (?,?,?,?,?,NULL,'active',?,?,?,?)""",
                    (
                        task_id,
                        owner,
                        timestamp,
                        timestamp,
                        expires,
                        stage,
                        token,
                        timestamp,
                        timestamp,
                    ),
                )
                if cursor.rowcount != 1:
                    raise TaskLeaseError("Task lease could not be acquired")
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).casefold() or "busy" in str(exc).casefold():
                raise TaskLeaseError(
                    "Task lease database is busy; no workflow work was started"
                ) from exc
            raise
        return TaskLease(task_id, owner, token, timestamp, timestamp, expires, stage)

    def heartbeat(
        self, lease: TaskLease, *, stage: str | None = None, now: datetime | None = None
    ) -> TaskLease:
        current = _as_utc(now)
        timestamp = current.isoformat()
        expires = (current + timedelta(seconds=self.ttl_seconds)).isoformat()
        current_stage = stage or lease.current_stage
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._assert_row(connection, lease, current)
            connection.execute(
                """UPDATE workflow_task_leases SET heartbeat_at=?,expires_at=?,
                current_stage=?,updated_at=? WHERE id=?""",
                (timestamp, expires, current_stage, timestamp, row["id"]),
            )
        return TaskLease(
            lease.task_id,
            lease.owner_id,
            lease.fencing_token,
            lease.acquired_at,
            timestamp,
            expires,
            current_stage,
        )

    def assert_current(
        self, lease: TaskLease, *, now: datetime | None = None, connection=None
    ) -> None:
        current = _as_utc(now)
        if connection is not None:
            self._assert_row(connection, lease, current)
            return
        with self.database.connect() as managed:
            self._assert_row(managed, lease, current)

    def release(self, lease: TaskLease, *, now: datetime | None = None) -> None:
        timestamp = _as_utc(now).isoformat()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT id,owner_id,fencing_token,status FROM workflow_task_leases
                WHERE task_id=? AND status='active'""",
                (lease.task_id,),
            ).fetchone()
            if not row:
                raise TaskLeaseLost("Task lease is no longer active")
            if (
                row["owner_id"] != lease.owner_id
                or int(row["fencing_token"]) != lease.fencing_token
            ):
                raise TaskLeaseLost("Only the current lease owner can release this task")
            connection.execute(
                """UPDATE workflow_task_leases SET status='released',released_at=?,
                updated_at=? WHERE id=?""",
                (timestamp, timestamp, row["id"]),
            )

    def inspect(self, task_id: int) -> dict[str, object] | None:
        with self.database.connect() as connection:
            row = connection.execute(
                """SELECT * FROM workflow_task_leases WHERE task_id=?
                ORDER BY fencing_token DESC LIMIT 1""",
                (task_id,),
            ).fetchone()
        return _row_summary(row) if row else None

    def recover_stale(self, task_id: int, *, now: datetime | None = None) -> dict[str, object]:
        current = _as_utc(now)
        timestamp = current.isoformat()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT * FROM workflow_task_leases WHERE task_id=? AND status='active'""",
                (task_id,),
            ).fetchone()
            if not row:
                return {"task_id": task_id, "recovered": False, "reason": "no_active_lease"}
            if _parse(row["expires_at"]) > current:
                raise TaskLeaseBusy(_row_summary(row))
            connection.execute(
                """UPDATE workflow_task_leases SET status='stale',released_at=?,
                updated_at=? WHERE id=?""",
                (timestamp, timestamp, row["id"]),
            )
        return {
            "task_id": task_id,
            "recovered": True,
            "fencing_token": int(row["fencing_token"]),
            "expired_at": row["expires_at"],
        }

    @staticmethod
    def _assert_row(connection, lease: TaskLease, now: datetime):
        row = connection.execute(
            """SELECT * FROM workflow_task_leases WHERE task_id=? AND status='active'""",
            (lease.task_id,),
        ).fetchone()
        if (
            not row
            or row["owner_id"] != lease.owner_id
            or int(row["fencing_token"]) != lease.fencing_token
            or _parse(row["expires_at"]) <= now
        ):
            raise TaskLeaseLost(
                "Workflow lease expired or was superseded; result was not committed"
            )
        return row


def _row_summary(row) -> dict[str, object]:
    expired = row["status"] == "active" and _parse(row["expires_at"]) <= datetime.now(UTC)
    return {
        "task_id": int(row["task_id"]),
        "owner": _safe_owner(row["owner_id"]),
        "stage": row["current_stage"],
        "heartbeat_at": row["heartbeat_at"],
        "expires_at": row["expires_at"],
        "status": row["status"],
        "fencing_token": int(row["fencing_token"]),
        "released_at": row["released_at"],
        "expired": expired,
    }


def classify_task_execution(
    task_status: str, lease: dict[str, object] | None
) -> str:
    """Classify task execution without mutating the lease or task."""
    if (
        lease
        and lease.get("status") == "active"
        and not bool(lease.get("expired"))
    ):
        return "running_with_live_lease"
    if task_status == "running":
        return "orphaned_running"
    return "idle"


def _safe_owner(owner: str) -> str:
    return f"{owner[:8]}…" if len(owner) > 8 else owner


def _as_utc(value: datetime | None) -> datetime:
    current = value or datetime.now(UTC)
    if current.tzinfo is None:
        return current.replace(tzinfo=UTC)
    return current.astimezone(UTC)


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
