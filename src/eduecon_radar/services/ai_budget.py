from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from eduecon_radar.db import Database, utcnow
from eduecon_radar.resources import read_text

if TYPE_CHECKING:
    from typing import Protocol

    class ReservationHook(Protocol):
        """A reservation guard supplied by an outer research layer.

        The product core knows only this shape: reserve before a paid call,
        release it when the call never happened.  Leases, runtime-bundle
        identity and authorization documents belong to the layer that
        implements the hook, not here.
        """

        def reserve(
            self,
            *,
            task_id: int,
            reservation_key: str,
            input_tokens: int,
            output_tokens: int,
            cost_microunits: int,
        ) -> object: ...

        def release(self, reservation_key: str) -> None: ...

        def inspect(self) -> dict[str, object]: ...

STAGES = (
    "title_translation",
    "fast_semantic",
    "strong_semantic",
    "abstract_translation",
)


class BudgetBlocked(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"AI budget blocked before request: {reason}")
        self.reason = reason


@dataclass(frozen=True)
class BudgetReservation:
    id: int
    reservation_uid: str
    task_id: int
    stage: str
    fencing_token: int
    already_settled: bool = False


@dataclass(frozen=True)
class BudgetLimits:
    max_calls: int
    max_input_tokens: int
    max_output_tokens: int
    max_estimated_cost_microunits: int
    max_strong_calls: int
    max_translation_calls: int


def load_budget_policy(path: Path | None = None) -> dict[str, Any]:
    payload = yaml.safe_load(
        path.read_text(encoding="utf-8")
        if path is not None
        else read_text("config/ai_budget.yaml")
    )
    defaults = payload.get("defaults")
    if not isinstance(defaults, dict):
        raise ValueError("AI budget defaults are required; missing is never unlimited")
    numeric = (
        "max_calls",
        "max_input_tokens",
        "max_output_tokens",
        "max_estimated_cost_microunits",
        "max_strong_calls",
        "max_translation_calls",
    )
    if any(not isinstance(defaults.get(key), int) or defaults[key] <= 0 for key in numeric):
        raise ValueError("AI budget defaults must be positive integers")
    if any(stage not in payload.get("stages", {}) for stage in STAGES):
        raise ValueError("Every paid stage requires a budget")
    return payload


class AIBudgetManager:
    def __init__(
        self,
        database: Database,
        *,
        require_active_lease: bool = True,
        reservation_hook: ReservationHook | None = None,
    ) -> None:
        self.database = database
        self.policy = load_budget_policy()
        self.require_active_lease = require_active_lease
        #: Optional outer guard (a governed pilot).  The product core never
        #: learns what is behind it.
        self.reservation_hook = reservation_hook

    def ensure_default_budget(self, task_id: int) -> None:
        defaults = self.policy["defaults"]
        now = utcnow()
        with self.database.connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO ai_task_budgets
                (task_id,currency,price_version,max_calls,max_input_tokens,max_output_tokens,
                 max_estimated_cost_microunits,max_strong_calls,max_translation_calls,
                 created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    task_id,
                    self.policy["currency"],
                    self.policy["version"],
                    defaults["max_calls"],
                    defaults["max_input_tokens"],
                    defaults["max_output_tokens"],
                    defaults["max_estimated_cost_microunits"],
                    defaults["max_strong_calls"],
                    defaults["max_translation_calls"],
                    now,
                    now,
                ),
            )
            for stage, limits in self.policy["stages"].items():
                connection.execute(
                    """INSERT OR IGNORE INTO ai_stage_budgets
                    (task_id,stage,max_calls,max_input_tokens,max_output_tokens,
                     max_estimated_cost_microunits,created_at,updated_at)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        task_id,
                        stage,
                        limits["max_calls"],
                        limits["max_input_tokens"],
                        limits["max_output_tokens"],
                        limits["max_estimated_cost_microunits"],
                        now,
                        now,
                    ),
                )

    def reserve(
        self,
        *,
        task_id: int,
        stage: str,
        provider: str,
        model_name: str,
        idempotency_key: str,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
    ) -> BudgetReservation:
        if stage not in STAGES:
            raise ValueError(f"Unsupported paid stage: {stage}")
        if estimated_input_tokens <= 0 or estimated_output_tokens <= 0:
            raise ValueError("Token reservation estimates must be positive")
        self.ensure_default_budget(task_id)
        reserved_cost = self._cost(estimated_input_tokens, estimated_output_tokens)
        connection = sqlite3.connect(self.database.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        pilot_key = f"{task_id}:{stage}:{idempotency_key}"
        pilot_reserved = False
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """SELECT * FROM ai_usage_reservations
                WHERE task_id=? AND stage=? AND idempotency_key=?""",
                (task_id, stage, idempotency_key),
            ).fetchone()
            if existing:
                if existing["status"] == "settled":
                    connection.commit()
                    return BudgetReservation(
                        int(existing["id"]),
                        existing["reservation_uid"],
                        task_id,
                        stage,
                        int(existing["fencing_token"]),
                        True,
                    )
                raise BudgetBlocked(f"idempotency_{existing['status']}")
            fencing_token = self._fencing_token(connection, task_id)
            task_budget = connection.execute(
                "SELECT * FROM ai_task_budgets WHERE task_id=?", (task_id,)
            ).fetchone()
            stage_budget = connection.execute(
                "SELECT * FROM ai_stage_budgets WHERE task_id=? AND stage=?",
                (task_id, stage),
            ).fetchone()
            if task_budget is None or stage_budget is None:
                raise BudgetBlocked("budget_configuration_missing")
            usage = self._usage(connection, task_id)
            stage_usage = self._usage(connection, task_id, stage=stage)
            projected = {
                "calls": usage["calls"] + 1,
                "input": usage["input"] + estimated_input_tokens,
                "output": usage["output"] + estimated_output_tokens,
                "cost": usage["cost"] + reserved_cost,
            }
            stage_projected = {
                "calls": stage_usage["calls"] + 1,
                "input": stage_usage["input"] + estimated_input_tokens,
                "output": stage_usage["output"] + estimated_output_tokens,
                "cost": stage_usage["cost"] + reserved_cost,
            }
            self._check(projected, task_budget, "task")
            self._check(stage_projected, stage_budget, stage)
            if stage == "strong_semantic" and projected["calls"] > 0:
                strong = self._usage(connection, task_id, stage="strong_semantic")["calls"] + 1
                if strong > int(task_budget["max_strong_calls"]):
                    raise BudgetBlocked("max_strong_calls")
            if stage in {"title_translation", "abstract_translation"}:
                translations = sum(
                    self._usage(connection, task_id, stage=name)["calls"]
                    for name in ("title_translation", "abstract_translation")
                ) + 1
                if translations > int(task_budget["max_translation_calls"]):
                    raise BudgetBlocked("max_translation_calls")
            if self.reservation_hook is not None:
                # The hook raises BudgetBlocked itself when it refuses; the
                # product core does not need to know why it refused.
                pilot_reserved = self.reservation_hook.reserve(
                    task_id=task_id,
                    reservation_key=pilot_key,
                    input_tokens=estimated_input_tokens,
                    output_tokens=estimated_output_tokens,
                    cost_microunits=reserved_cost,
                )
            uid = str(uuid.uuid4())
            cursor = connection.execute(
                """INSERT INTO ai_usage_reservations
                (reservation_uid,task_id,stage,provider,model_name,idempotency_key,
                 fencing_token,reserved_calls,reserved_input_tokens,reserved_output_tokens,
                 reserved_cost_microunits,status,reserved_at,updated_at)
                VALUES (?,?,?,?,?,?,?,1,?,?,?,'reserved',?,?)""",
                (
                    uid,
                    task_id,
                    stage,
                    provider,
                    model_name,
                    idempotency_key,
                    fencing_token,
                    estimated_input_tokens,
                    estimated_output_tokens,
                    reserved_cost,
                    utcnow(),
                    utcnow(),
                ),
            )
            reservation = BudgetReservation(
                int(cursor.lastrowid), uid, task_id, stage, fencing_token
            )
            connection.commit()
            return reservation
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            if pilot_reserved and self.reservation_hook is not None:
                self.reservation_hook.release(pilot_key)
            raise
        finally:
            connection.close()

    def settle(
        self,
        reservation: BudgetReservation,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        failed: bool = False,
        error_type: str | None = None,
    ) -> None:
        if reservation.already_settled:
            return
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM ai_usage_reservations WHERE id=?", (reservation.id,)
            ).fetchone()
            if row is None or row["status"] != "reserved":
                raise BudgetBlocked("reservation_not_active")
            self._assert_fencing(connection, reservation)
            usage_missing = not input_tokens or not output_tokens
            actual_input = (
                int(input_tokens)
                if input_tokens and input_tokens > 0
                else row["reserved_input_tokens"]
            )
            actual_output = (
                int(output_tokens)
                if output_tokens and output_tokens > 0
                else row["reserved_output_tokens"]
            )
            actual_cost = self._cost(actual_input, actual_output)
            connection.execute(
                """UPDATE ai_usage_reservations SET actual_input_tokens=?,
                actual_output_tokens=?,actual_cost_microunits=?,usage_missing=?,status=?,
                error_type=?,settled_at=?,updated_at=? WHERE id=? AND status='reserved'""",
                (
                    actual_input,
                    actual_output,
                    actual_cost,
                    int(usage_missing),
                    "failed" if failed else "settled",
                    error_type,
                    utcnow(),
                    utcnow(),
                    reservation.id,
                ),
            )

    def inspect(self, task_id: int) -> dict[str, Any]:
        uri = self.database.path.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                    "SELECT * FROM ai_task_budgets WHERE task_id=?", (task_id,)
                ).fetchone()
            if row is None:
                return {
                    "status": "blocked",
                    "task_id": task_id,
                    "reason": "task_budget_missing",
                }
            budget = dict(row)
            stages = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM ai_stage_budgets WHERE task_id=? ORDER BY stage",
                    (task_id,),
                ).fetchall()
            ]
            usage = self._usage(connection, task_id)
            unsettled = int(
                connection.execute(
                    """SELECT COUNT(*) FROM ai_usage_reservations
                    WHERE task_id=? AND status='reserved'""",
                    (task_id,),
                ).fetchone()[0]
            )
        result = {
            "status": "ok",
            "task_id": task_id,
            "budget": budget,
            "stage_budgets": stages,
            "usage": usage,
            "remaining": {
                "calls": budget["max_calls"] - usage["calls"],
                "input_tokens": budget["max_input_tokens"] - usage["input"],
                "output_tokens": budget["max_output_tokens"] - usage["output"],
                "cost_microunits": budget["max_estimated_cost_microunits"] - usage["cost"],
            },
            "unsettled_reservations": unsettled,
        }
        if self.reservation_hook is not None:
            result["reservation_guard"] = self.reservation_hook.inspect()
        return result

    def set_budget(
        self,
        task_id: int,
        limits: BudgetLimits,
        *,
        changed_by: str,
        reason: str,
    ) -> dict[str, Any]:
        if any(value <= 0 for value in asdict(limits).values()):
            raise ValueError("Unlimited, null and non-positive budgets are forbidden")
        self.ensure_default_budget(task_id)
        with self.database.connect() as connection:
            previous = dict(
                connection.execute(
                    "SELECT * FROM ai_task_budgets WHERE task_id=?", (task_id,)
                ).fetchone()
            )
            usage = self._usage(connection, task_id)
            if (
                limits.max_calls < usage["calls"]
                or limits.max_input_tokens < usage["input"]
                or limits.max_output_tokens < usage["output"]
                or limits.max_estimated_cost_microunits < usage["cost"]
            ):
                raise ValueError("Budget cannot be lowered below consumed or reserved usage")
            now = utcnow()
            connection.execute(
                """UPDATE ai_task_budgets SET max_calls=?,max_input_tokens=?,
                max_output_tokens=?,max_estimated_cost_microunits=?,max_strong_calls=?,
                max_translation_calls=?,updated_at=? WHERE task_id=?""",
                (*asdict(limits).values(), now, task_id),
            )
            current = dict(
                connection.execute(
                    "SELECT * FROM ai_task_budgets WHERE task_id=?", (task_id,)
                ).fetchone()
            )
            connection.execute(
                """INSERT INTO ai_budget_changes
                (task_id,changed_by,previous_budget_json,new_budget_json,reason,created_at)
                VALUES (?,?,?,?,?,?)""",
                (
                    task_id,
                    changed_by,
                    json.dumps(previous, sort_keys=True),
                    json.dumps(current, sort_keys=True),
                    reason,
                    now,
                ),
            )
        return self.inspect(task_id)

    def unsettled(self, task_id: int | None = None) -> list[dict[str, Any]]:
        uri = self.database.path.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            sql = "SELECT * FROM ai_usage_reservations WHERE status='reserved'"
            values: tuple[object, ...] = ()
            if task_id is not None:
                sql += " AND task_id=?"
                values = (task_id,)
            return [dict(row) for row in connection.execute(sql, values).fetchall()]

    def _cost(self, input_tokens: int, output_tokens: int) -> int:
        return max(
            1,
            input_tokens * int(self.policy["conservative_microunits_per_input_token"])
            + output_tokens * int(self.policy["conservative_microunits_per_output_token"]),
        )

    def _fencing_token(self, connection: sqlite3.Connection, task_id: int) -> int:
        row = connection.execute(
            """SELECT fencing_token,expires_at FROM workflow_task_leases
            WHERE task_id=? AND status='active' ORDER BY fencing_token DESC LIMIT 1""",
            (task_id,),
        ).fetchone()
        if not self.require_active_lease:
            return int(row["fencing_token"]) if row else 1
        if row is None:
            raise BudgetBlocked("active_task_lease_required")
        expires = datetime.fromisoformat(row["expires_at"])
        if expires <= datetime.now(UTC):
            raise BudgetBlocked("task_lease_expired")
        return int(row["fencing_token"])

    def _assert_fencing(
        self, connection: sqlite3.Connection, reservation: BudgetReservation
    ) -> None:
        if not self.require_active_lease:
            return
        row = connection.execute(
            """SELECT fencing_token FROM workflow_task_leases
            WHERE task_id=? AND status='active' ORDER BY fencing_token DESC LIMIT 1""",
            (reservation.task_id,),
        ).fetchone()
        if row is None or int(row["fencing_token"]) != reservation.fencing_token:
            raise BudgetBlocked("stale_fencing_token")

    @staticmethod
    def _usage(
        connection: sqlite3.Connection, task_id: int, *, stage: str | None = None
    ) -> dict[str, int]:
        condition = "task_id=? AND status IN ('reserved','settled','failed')"
        values: list[object] = [task_id]
        if stage:
            condition += " AND stage=?"
            values.append(stage)
        row = connection.execute(
            f"""SELECT COALESCE(SUM(reserved_calls),0) calls,
            COALESCE(SUM(COALESCE(actual_input_tokens,reserved_input_tokens)),0) input,
            COALESCE(SUM(COALESCE(actual_output_tokens,reserved_output_tokens)),0) output,
            COALESCE(SUM(COALESCE(actual_cost_microunits,reserved_cost_microunits)),0) cost
            FROM ai_usage_reservations WHERE {condition}""",  # noqa: S608
            values,
        ).fetchone()
        return {key: int(row[key]) for key in ("calls", "input", "output", "cost")}

    @staticmethod
    def _check(projected: dict[str, int], budget: sqlite3.Row, prefix: str) -> None:
        mapping = {
            "calls": "max_calls",
            "input": "max_input_tokens",
            "output": "max_output_tokens",
            "cost": "max_estimated_cost_microunits",
        }
        for usage_key, budget_key in mapping.items():
            if projected[usage_key] > int(budget[budget_key]):
                raise BudgetBlocked(f"{prefix}_{budget_key}")


def estimate_tokens(text: str, output_limit: int) -> tuple[int, int]:
    return max(1, (len(text) + 3) // 4), max(1, output_limit)
