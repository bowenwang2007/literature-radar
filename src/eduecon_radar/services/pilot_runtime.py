# --- not part of the product core: see docs/PRODUCT_CORE.md ---
# RESEARCH/GOVERNANCE. Reachable only from the governed Gold pilot branch.

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from eduecon_radar.db import Database
from eduecon_radar.domain.semantic_prompting import canonical_sha256, file_sha256
from eduecon_radar.services.runtime_bundle import (
    RuntimeBundleResolver,
    RuntimeModelIdentity,
)
from eduecon_radar.services.task_lock import TaskLeaseManager

PILOT_CONFIRMATION = "我确认这是受控试运行，结果不代表正式发布模型"
DEFAULT_AUTHORIZATION_PATH = Path("data/development/pilot_authorization.json")
DEFAULT_RESUME_AUTHORIZATION_DIR = Path(
    "data/development/pilot_resume_authorizations"
)
RESUME_EXISTING_TASK_SCOPE = "resume_existing_task"
_RESUME_STAGE_ORDER = (
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
_FILE_LOCK = threading.RLock()


class PilotAuthorizationError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"controlled_pilot_blocked:{reason}")


class PilotReservationGuard:
    """Adapts a pilot authorization to the product core's reservation hook.

    ``AIBudgetManager`` only asks a guard to ``reserve``/``release``/``inspect``;
    every pilot concept (authorization document, runtime-bundle identity, model
    snapshot, per-task limits) stays in this module.  A refusal is translated
    into the budget layer's own ``BudgetBlocked`` here, so the product core
    never imports anything about pilots.
    """

    def __init__(
        self,
        service: PilotAuthorizationService,
        authorization_id: str,
        models: RuntimeModelIdentity,
    ) -> None:
        self._service = service
        self._authorization_id = authorization_id
        self._models = models

    def reserve(
        self,
        *,
        task_id: int,
        reservation_key: str,
        input_tokens: int,
        output_tokens: int,
        cost_microunits: int,
    ) -> object:
        from eduecon_radar.services.ai_budget import BudgetBlocked

        try:
            return self._service.reserve_ai(
                authorization_id=self._authorization_id,
                models=self._models,
                task_id=task_id,
                reservation_key=reservation_key,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_microunits=cost_microunits,
            )
        except PilotAuthorizationError as exc:
            raise BudgetBlocked(exc.reason) from exc

    def release(self, reservation_key: str) -> None:
        self._service.release_ai(reservation_key)

    def inspect(self) -> dict[str, object]:
        return self._service.inspect(self._models)


@dataclass(frozen=True)
class PilotLimits:
    max_tasks: int = 1
    max_papers_per_task: int = 20
    max_ai_calls: int = 100
    max_input_tokens: int = 250_000
    max_output_tokens: int = 100_000
    max_cost_microunits: int = 2_000_000

    def validate(self, *, allow_zero_tasks: bool = False) -> None:
        values = asdict(self)
        if allow_zero_tasks:
            if values["max_tasks"] != 0 or any(
                value <= 0 for key, value in values.items() if key != "max_tasks"
            ):
                raise PilotAuthorizationError("resume_limits_invalid")
            return
        if any(value <= 0 for value in values.values()):
            raise PilotAuthorizationError("limits_must_be_positive")


@dataclass(frozen=True)
class ResumePilotLimits:
    max_ai_calls: int = 50
    max_input_tokens: int = 125_000
    max_output_tokens: int = 50_000
    max_cost_microunits: int = 1_000_000

    def validate(self) -> None:
        if any(value <= 0 for value in asdict(self).values()):
            raise PilotAuthorizationError("resume_limits_must_be_positive")


@dataclass(frozen=True)
class PilotAuthorization:
    authorization_id: str
    created_at: str
    expires_at: str
    status: str
    operator_note: str
    runtime_config_sha256: str
    prompt_bundle_sha256: str
    models: dict[str, str]
    limits: PilotLimits
    runtime_bundle_sha256: str | None = None
    scope: str = "create_tasks"
    existing_task_id: int | None = None
    original_authorization_id: str | None = None


class PilotAuthorizationService:
    def __init__(
        self, project_root: Path, authorization_path: Path | None = None
    ) -> None:
        self.project_root = project_root.resolve()
        self.path = (
            authorization_path.resolve()
            if authorization_path is not None
            else (self.project_root / DEFAULT_AUTHORIZATION_PATH).resolve()
        )

    def runtime_snapshot(self, models: RuntimeModelIdentity) -> dict[str, object]:
        resolver = RuntimeBundleResolver(self.project_root)
        runtime = resolver.resolve(
            mode="development",
            allow_unreleased_bundle=True,
            development_models=models,
        )
        prompt_resources = {
            relative: file_sha256(resolver.project_root / relative)
            for relative in resolver.REQUIRED_RESOURCE_PATHS
            if "prompts/" in relative.replace("\\", "/")
        }
        return {
            "runtime": runtime,
            "runtime_config_sha256": canonical_sha256(
                {
                    "domain": runtime.domain_bundle.config_sha256,
                    "semantic": runtime.semantic_config.model_dump(mode="json"),
                }
            ),
            "prompt_bundle_sha256": canonical_sha256(prompt_resources),
            "models": models.model_dump(),
        }

    def create(
        self,
        *,
        confirmation: str,
        operator_note: str,
        models: RuntimeModelIdentity,
        limits: PilotLimits | None = None,
        expires_in_hours: int = 8,
        now: datetime | None = None,
    ) -> PilotAuthorization:
        if confirmation.strip() != PILOT_CONFIRMATION:
            raise PilotAuthorizationError("confirmation_phrase_mismatch")
        note = operator_note.strip()
        if not note:
            raise PilotAuthorizationError("operator_note_required")
        if not 1 <= expires_in_hours <= 24:
            raise PilotAuthorizationError("expiry_must_be_between_1_and_24_hours")
        selected_limits = limits or PilotLimits()
        selected_limits.validate()
        current = now or datetime.now(UTC)
        snapshot = self.runtime_snapshot(models)
        authorization = PilotAuthorization(
            authorization_id=str(uuid.uuid4()),
            created_at=current.isoformat(),
            expires_at=(current + timedelta(hours=expires_in_hours)).isoformat(),
            status="active",
            operator_note=note,
            runtime_config_sha256=str(snapshot["runtime_config_sha256"]),
            prompt_bundle_sha256=str(snapshot["prompt_bundle_sha256"]),
            models=dict(snapshot["models"]),
            limits=selected_limits,
            runtime_bundle_sha256=str(snapshot["runtime"].bundle_sha256),
        )
        payload = {
            "schema_version": 1,
            **asdict(authorization),
            "authorization_type": "create_tasks",
            "usage": {
                "tasks": 0,
                "papers": 0,
                "ai_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_microunits": 0,
            },
            "task_claims": {},
            "ai_reservations": {},
        }
        self._write(payload)
        return authorization

    def inspect(
        self, models: RuntimeModelIdentity, *, now: datetime | None = None
    ) -> dict[str, object]:
        if not self.path.is_file():
            return {"valid": False, "status": "missing", "reasons": ["authorization_missing"]}
        try:
            payload = self._read()
            authorization = self._authorization(payload)
        except (OSError, ValueError, KeyError, TypeError, PilotAuthorizationError) as exc:
            return {
                "valid": False,
                "status": "invalid",
                "reasons": [f"authorization_invalid:{type(exc).__name__}"],
            }
        reasons: list[str] = []
        current = _as_utc(now)
        status = authorization.status
        if authorization.status == "revoked":
            reasons.append("authorization_revoked")
        if _parse_datetime(authorization.expires_at) <= current:
            status = "expired"
            reasons.append("authorization_expired")
        try:
            snapshot = self.runtime_snapshot(models)
        except Exception as exc:
            return {
                "valid": False,
                "status": "invalid",
                "reasons": [f"runtime_integrity_error:{type(exc).__name__}"],
                "authorization": authorization,
                "remaining": {},
                "usage": dict(payload.get("usage") or {}),
                "expires_at": authorization.expires_at,
            }
        if authorization.runtime_config_sha256 != snapshot["runtime_config_sha256"]:
            reasons.append("runtime_config_sha256_changed")
        if authorization.prompt_bundle_sha256 != snapshot["prompt_bundle_sha256"]:
            reasons.append("prompt_bundle_sha256_changed")
        if (
            authorization.runtime_bundle_sha256
            and authorization.runtime_bundle_sha256
            != snapshot["runtime"].bundle_sha256
        ):
            reasons.append("runtime_bundle_sha256_changed")
        if authorization.models != snapshot["models"]:
            reasons.append("model_identities_changed")
        usage = dict(payload.get("usage") or {})
        remaining = {
            "tasks": max(0, authorization.limits.max_tasks - int(usage.get("tasks", 0))),
            "papers": max(
                0,
                authorization.limits.max_tasks
                * authorization.limits.max_papers_per_task
                - int(usage.get("papers", 0)),
            ),
            "ai_calls": max(
                0, authorization.limits.max_ai_calls - int(usage.get("ai_calls", 0))
            ),
            "input_tokens": max(
                0,
                authorization.limits.max_input_tokens
                - int(usage.get("input_tokens", 0)),
            ),
            "output_tokens": max(
                0,
                authorization.limits.max_output_tokens
                - int(usage.get("output_tokens", 0)),
            ),
            "cost_microunits": max(
                0,
                authorization.limits.max_cost_microunits
                - int(usage.get("cost_microunits", 0)),
            ),
        }
        return {
            "valid": not reasons and status in {"active", "consumed"},
            "status": status,
            "reasons": reasons,
            "authorization": authorization,
            "remaining": remaining,
            "usage": usage,
            "expires_at": authorization.expires_at,
            "runtime": snapshot["runtime"],
        }

    def require(
        self,
        authorization_id: str,
        models: RuntimeModelIdentity,
        *,
        task_id: int | None = None,
    ) -> PilotAuthorization:
        inspected = self.inspect(models)
        if not inspected.get("valid"):
            raise PilotAuthorizationError(
                str((inspected.get("reasons") or [inspected.get("status")])[0])
            )
        authorization = inspected["authorization"]
        assert isinstance(authorization, PilotAuthorization)
        if authorization.authorization_id != authorization_id:
            raise PilotAuthorizationError("authorization_id_mismatch")
        if task_id is not None:
            payload = self._read()
            claims = dict(payload.get("task_claims") or {})
            if not any(int(item.get("task_id") or 0) == task_id for item in claims.values()):
                raise PilotAuthorizationError("task_not_bound_to_authorization")
        return authorization

    def claim_task(
        self,
        *,
        authorization_id: str,
        models: RuntimeModelIdentity,
        task_key: str,
        requested_papers: int,
    ) -> dict[str, object]:
        with _FILE_LOCK:
            if self._read().get("scope") == RESUME_EXISTING_TASK_SCOPE:
                raise PilotAuthorizationError(
                    "resume_authorization_cannot_create_task"
                )
            authorization = self.require(authorization_id, models)
            payload = self._read()
            claims = dict(payload.get("task_claims") or {})
            if task_key in claims:
                existing = dict(claims[task_key])
                if int(existing.get("requested_papers") or 0) != requested_papers:
                    raise PilotAuthorizationError("task_key_conflict")
                return existing
            if requested_papers > authorization.limits.max_papers_per_task:
                raise PilotAuthorizationError("max_papers_per_task")
            usage = dict(payload["usage"])
            pending = [
                dict(item)
                for item in claims.values()
                if dict(item).get("task_id") is None
            ]
            if int(usage["tasks"]) + len(pending) + 1 > authorization.limits.max_tasks:
                raise PilotAuthorizationError("max_tasks")
            pending_papers = sum(
                int(item.get("requested_papers") or 0) for item in pending
            )
            paper_capacity = (
                authorization.limits.max_tasks
                * authorization.limits.max_papers_per_task
            )
            if (
                int(usage["papers"]) + pending_papers + requested_papers
                > paper_capacity
            ):
                raise PilotAuthorizationError("max_papers")
            claim = {
                "claim_id": str(uuid.uuid4()),
                "task_key": task_key,
                "requested_papers": requested_papers,
                "task_id": None,
                "created_at": datetime.now(UTC).isoformat(),
            }
            claims[task_key] = claim
            payload["task_claims"] = claims
            self._write(payload)
            return claim

    def bind_task(self, task_key: str, task_id: int) -> None:
        with _FILE_LOCK:
            payload = self._read()
            if payload.get("scope") == RESUME_EXISTING_TASK_SCOPE:
                raise PilotAuthorizationError(
                    "resume_authorization_cannot_create_task"
                )
            claims = dict(payload.get("task_claims") or {})
            claim = dict(claims.get(task_key) or {})
            if not claim:
                raise PilotAuthorizationError("task_claim_missing")
            existing = claim.get("task_id")
            if existing is not None and int(existing) != task_id:
                raise PilotAuthorizationError("task_claim_already_bound")
            if existing is not None:
                return
            authorization = self._authorization(payload)
            usage = dict(payload["usage"])
            claim["task_id"] = task_id
            claims[task_key] = claim
            payload["task_claims"] = claims
            usage["tasks"] = int(usage["tasks"]) + 1
            usage["papers"] = int(usage["papers"]) + int(
                claim.get("requested_papers") or 0
            )
            payload["usage"] = usage
            if int(usage["tasks"]) >= authorization.limits.max_tasks:
                payload["status"] = "consumed"
            self._write(payload)

    def release_unbound_task(self, task_key: str) -> None:
        with _FILE_LOCK:
            payload = self._read()
            claims = dict(payload.get("task_claims") or {})
            claim = dict(claims.get(task_key) or {})
            if not claim or claim.get("task_id") is not None:
                return
            claims.pop(task_key, None)
            payload["task_claims"] = claims
            self._write(payload)

    def reserve_ai(
        self,
        *,
        authorization_id: str,
        models: RuntimeModelIdentity,
        task_id: int,
        reservation_key: str,
        input_tokens: int,
        output_tokens: int,
        cost_microunits: int,
    ) -> bool:
        with _FILE_LOCK:
            authorization = self.require(
                authorization_id, models, task_id=task_id
            )
            payload = self._read()
            reservations = dict(payload.get("ai_reservations") or {})
            if reservation_key in reservations:
                return False
            usage = dict(payload["usage"])
            projected = {
                "ai_calls": int(usage["ai_calls"]) + 1,
                "input_tokens": int(usage["input_tokens"]) + input_tokens,
                "output_tokens": int(usage["output_tokens"]) + output_tokens,
                "cost_microunits": int(usage["cost_microunits"]) + cost_microunits,
            }
            checks = {
                "ai_calls": authorization.limits.max_ai_calls,
                "input_tokens": authorization.limits.max_input_tokens,
                "output_tokens": authorization.limits.max_output_tokens,
                "cost_microunits": authorization.limits.max_cost_microunits,
            }
            for key, maximum in checks.items():
                if projected[key] > maximum:
                    raise PilotAuthorizationError(f"pilot_max_{key}")
            usage.update(projected)
            reservations[reservation_key] = {
                "task_id": task_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_microunits": cost_microunits,
                "reserved_at": datetime.now(UTC).isoformat(),
            }
            payload["usage"] = usage
            payload["ai_reservations"] = reservations
            self._write(payload)
            return True

    def release_ai(self, reservation_key: str) -> None:
        with _FILE_LOCK:
            payload = self._read()
            reservations = dict(payload.get("ai_reservations") or {})
            item = reservations.pop(reservation_key, None)
            if not item:
                return
            usage = dict(payload["usage"])
            usage["ai_calls"] = max(0, int(usage["ai_calls"]) - 1)
            for key in ("input_tokens", "output_tokens", "cost_microunits"):
                usage[key] = max(0, int(usage[key]) - int(item[key]))
            payload["usage"] = usage
            payload["ai_reservations"] = reservations
            self._write(payload)

    @property
    def resume_authorization_dir(self) -> Path:
        return (self.project_root / DEFAULT_RESUME_AUTHORIZATION_DIR).resolve()

    def resume_service(self, authorization_id: str) -> PilotAuthorizationService:
        try:
            normalized = str(uuid.UUID(authorization_id))
        except ValueError as exc:
            raise PilotAuthorizationError("resume_authorization_id_invalid") from exc
        return PilotAuthorizationService(
            self.project_root,
            authorization_path=self.resume_authorization_dir / f"{normalized}.json",
        )

    def inspect_resume_task(
        self,
        database: Database,
        task_id: int,
        models: RuntimeModelIdentity,
        *,
        now: datetime | None = None,
        _original_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        current = _as_utc(now)
        with database.connect() as connection:
            row = connection.execute(
                """SELECT id,status,last_error,source_config_json,result_summary_json
                FROM workflow_batches WHERE id=?""",
                (task_id,),
            ).fetchone()
            paper_ids = tuple(
                int(item[0])
                for item in connection.execute(
                    """SELECT paper_id FROM workflow_batch_items
                    WHERE batch_id=? ORDER BY position,paper_id""",
                    (task_id,),
                ).fetchall()
            )
        if row is None:
            return {
                "valid": False,
                "task_id": task_id,
                "reasons": ["task_not_found"],
            }
        source = _safe_object(row["source_config_json"])
        summary = _safe_object(row["result_summary_json"])
        v05 = _safe_object(summary.get("v05_literature"))
        checkpoints = _safe_object(summary.get("phase6c_checkpoints"))
        current_checkpoint = next(
            (
                stage
                for stage in _RESUME_STAGE_ORDER
                if _safe_object(checkpoints.get(stage)).get("status") != "completed"
            ),
            None,
        )
        checkpoint = (
            _safe_object(checkpoints.get(current_checkpoint))
            if current_checkpoint
            else {}
        )
        pending_item_ids = tuple(
            int(item)
            for item in (
                checkpoint.get("failed_item_ids")
                or checkpoint.get("pending_item_ids")
                or ()
            )
            if int(item) in set(paper_ids)
        )
        reasons: list[str] = []
        status = str(row["status"] or "")
        last_error = str(row["last_error"] or "")
        if status == "completed":
            reasons.append("task_completed")
        elif status == "paused" and last_error == "abandoned_by_user":
            reasons.append("task_abandoned_by_user")
        elif status not in {"paused", "failed", "running", "interrupted"}:
            reasons.append("task_status_not_resumable")
        if source.get("runtime_mode") != "controlled_pilot":
            reasons.append("task_runtime_mode_not_controlled_pilot")
        if v05.get("release_status") != "NOT_RELEASED":
            reasons.append("task_release_status_not_not_released")

        lease = TaskLeaseManager(database).inspect(task_id)
        live_lease = bool(
            lease
            and lease.get("status") == "active"
            and not bool(lease.get("expired"))
        )
        if live_lease:
            reasons.append("live_lease_present")

        original_id = str(source.get("pilot_authorization_id") or "")
        original_payload = _original_snapshot
        if original_payload is None and self.path.is_file():
            candidate = self._read()
            if str(candidate.get("authorization_id") or "") == original_id:
                original_payload = candidate
        original_status = "missing"
        if original_payload is None:
            reasons.append("original_authorization_missing")
        else:
            raw_status = str(original_payload.get("status") or "")
            original_status = raw_status
            expires_at = _parse_datetime(str(original_payload.get("expires_at") or ""))
            if raw_status == "revoked":
                reasons.append("original_authorization_revoked")
            elif expires_at <= current:
                original_status = "expired"
            elif raw_status == "active":
                reasons.append("original_authorization_still_active")
            elif raw_status != "consumed":
                reasons.append("original_authorization_status_invalid")
            claims = _safe_object(original_payload.get("task_claims"))
            bound_claim = next(
                (
                    _safe_object(item)
                    for item in claims.values()
                    if int(_safe_object(item).get("task_id") or 0) == task_id
                ),
                None,
            )
            if bound_claim is None:
                reasons.append("original_authorization_not_bound_to_task")
            elif len(paper_ids) > int(
                bound_claim.get("requested_papers") or 0
            ):
                reasons.append("task_paper_scope_exceeds_original_claim")

        snapshot: dict[str, object] | None = None
        try:
            snapshot = self.runtime_snapshot(models)
        except Exception as exc:
            reasons.append(f"runtime_integrity_error:{type(exc).__name__}")
        expected_models = (
            _safe_object(source.get("pilot_models_snapshot"))
            or _safe_object((original_payload or {}).get("models"))
        )
        hash_matches = {
            "runtime_config": False,
            "runtime_bundle": False,
            "prompt_bundle": False,
            "models": False,
        }
        if snapshot is not None:
            hash_matches = {
                "runtime_config": source.get("runtime_config_sha256")
                == snapshot["runtime_config_sha256"],
                "runtime_bundle": source.get("runtime_bundle_sha256")
                == snapshot["runtime"].bundle_sha256,
                "prompt_bundle": source.get("prompt_bundle_sha256")
                == snapshot["prompt_bundle_sha256"],
                "models": expected_models == snapshot["models"],
            }
            mapping = {
                "runtime_config": "runtime_config_sha256_mismatch",
                "runtime_bundle": "runtime_bundle_sha256_mismatch",
                "prompt_bundle": "prompt_bundle_sha256_mismatch",
                "models": "model_identity_mismatch",
            }
            reasons.extend(
                mapping[key] for key, matched in hash_matches.items() if not matched
            )
        return {
            "valid": not reasons,
            "task_id": task_id,
            "task_status": status,
            "last_error": row["last_error"],
            "runtime_mode": source.get("runtime_mode"),
            "release_status": v05.get("release_status"),
            "original_authorization_id": original_id,
            "original_authorization_status": original_status,
            "current_checkpoint": current_checkpoint,
            "pending_item_ids": pending_item_ids,
            "pending_count": len(pending_item_ids),
            "paper_ids": paper_ids,
            "lease": lease,
            "live_lease": live_lease,
            "hash_matches": hash_matches,
            "reasons": list(dict.fromkeys(reasons)),
            "source_config": source,
            "runtime_snapshot": snapshot,
            "original_authorization_snapshot": original_payload,
        }

    def issue_resume_authorization(
        self,
        database: Database,
        *,
        task_id: int,
        models: RuntimeModelIdentity,
        limits: ResumePilotLimits,
        expires_in_hours: int,
        confirmed: bool,
        issuer_mode: str,
        now: datetime | None = None,
    ) -> PilotAuthorization:
        if not confirmed:
            raise PilotAuthorizationError("resume_confirmation_required")
        if not 1 <= expires_in_hours <= 24:
            raise PilotAuthorizationError("expiry_must_be_between_1_and_24_hours")
        limits.validate()
        current = _as_utc(now)
        inspected = self.inspect_resume_task(
            database, task_id, models, now=current
        )
        if not inspected.get("valid"):
            raise PilotAuthorizationError(str(inspected["reasons"][0]))
        runtime_snapshot = inspected["runtime_snapshot"]
        assert isinstance(runtime_snapshot, dict)
        pilot_limits = PilotLimits(
            max_tasks=0,
            max_papers_per_task=max(1, len(inspected["paper_ids"])),
            **asdict(limits),
        )
        pilot_limits.validate(allow_zero_tasks=True)
        expires_at = (current + timedelta(hours=expires_in_hours)).isoformat()
        for path in self._resume_authorization_paths():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                existing = self._authorization(payload)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            if (
                existing.scope == RESUME_EXISTING_TASK_SCOPE
                and existing.existing_task_id == task_id
                and existing.status in {"active", "consumed"}
                and _parse_datetime(existing.expires_at) > current
                and existing.runtime_config_sha256
                == inspected["source_config"]["runtime_config_sha256"]
                and existing.runtime_bundle_sha256
                == inspected["source_config"]["runtime_bundle_sha256"]
                and existing.prompt_bundle_sha256
                == inspected["source_config"]["prompt_bundle_sha256"]
                and existing.models == models.model_dump()
                and existing.limits == pilot_limits
            ):
                if tuple(
                    int(item) for item in payload.get("task_paper_ids") or ()
                ) != tuple(inspected["paper_ids"]) or _safe_object(
                    payload.get("task_scope_snapshot")
                ) != _task_scope_snapshot(inspected["source_config"]):
                    raise PilotAuthorizationError(
                        "resume_task_scope_changed_after_authorization"
                    )
                return existing
        authorization = PilotAuthorization(
            authorization_id=str(uuid.uuid4()),
            created_at=current.isoformat(),
            expires_at=expires_at,
            status="active",
            operator_note="resume_after_interruption",
            runtime_config_sha256=str(
                inspected["source_config"]["runtime_config_sha256"]
            ),
            prompt_bundle_sha256=str(
                inspected["source_config"]["prompt_bundle_sha256"]
            ),
            models=models.model_dump(),
            limits=pilot_limits,
            runtime_bundle_sha256=str(
                inspected["source_config"]["runtime_bundle_sha256"]
            ),
            scope=RESUME_EXISTING_TASK_SCOPE,
            existing_task_id=task_id,
            original_authorization_id=str(
                inspected["original_authorization_id"]
            ),
        )
        payload = {
            "schema_version": 2,
            "authorization_type": RESUME_EXISTING_TASK_SCOPE,
            **asdict(authorization),
            "issued_at": current.isoformat(),
            "issuer_mode": issuer_mode,
            "reason": "resume_after_interruption",
            "release_status": "NOT_RELEASED",
            "runtime_mode": "controlled_pilot",
            "max_new_tasks": 0,
            "task_paper_ids": list(inspected["paper_ids"]),
            "remaining_pending_item_ids": list(inspected["pending_item_ids"]),
            "task_scope_snapshot": _task_scope_snapshot(
                inspected["source_config"]
            ),
            "original_authorization_snapshot": inspected[
                "original_authorization_snapshot"
            ],
            "usage": {
                "tasks": 0,
                "papers": 0,
                "ai_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_microunits": 0,
            },
            "task_claims": {
                f"resume:{task_id}": {
                    "claim_id": str(uuid.uuid4()),
                    "task_id": task_id,
                    "task_key": f"resume:{task_id}",
                    "requested_papers": 0,
                    "created_at": current.isoformat(),
                }
            },
            "ai_reservations": {},
            "resume_bindings": {},
            "use_count": 0,
        }
        resume_service = self.resume_service(authorization.authorization_id)
        resume_service._write(payload)
        try:
            self._append_task_resume_reference(
                database, task_id, authorization, issuer_mode=issuer_mode
            )
        except Exception:
            resume_service.path.unlink(missing_ok=True)
            raise
        return authorization

    def find_resume_authorization(
        self,
        database: Database,
        task_id: int,
        models: RuntimeModelIdentity,
        *,
        now: datetime | None = None,
    ) -> tuple[PilotAuthorizationService, PilotAuthorization] | None:
        current = _as_utc(now)
        for path in reversed(self._resume_authorization_paths()):
            resume_service = PilotAuthorizationService(
                self.project_root, authorization_path=path
            )
            try:
                payload = resume_service._read()
                authorization = resume_service.require_resume_task(
                    database,
                    str(payload.get("authorization_id") or ""),
                    models,
                    task_id,
                    now=current,
                )
            except (OSError, ValueError, KeyError, TypeError, PilotAuthorizationError):
                continue
            return resume_service, authorization
        return None

    def require_resume_task(
        self,
        database: Database,
        authorization_id: str,
        models: RuntimeModelIdentity,
        task_id: int,
        *,
        now: datetime | None = None,
    ) -> PilotAuthorization:
        payload = self._read()
        authorization = self.require(
            authorization_id, models, task_id=task_id
        )
        if (
            authorization.scope != RESUME_EXISTING_TASK_SCOPE
            or authorization.existing_task_id != task_id
        ):
            raise PilotAuthorizationError("resume_authorization_task_mismatch")
        inspector = PilotAuthorizationService(self.project_root)
        inspected = inspector.inspect_resume_task(
            database,
            task_id,
            models,
            now=now,
            _original_snapshot=_safe_object(
                payload.get("original_authorization_snapshot")
            ),
        )
        allowed_reasons = {"original_authorization_still_active"}
        blocking = [
            str(reason)
            for reason in inspected.get("reasons") or ()
            if reason not in allowed_reasons
        ]
        if blocking:
            raise PilotAuthorizationError(blocking[0])
        if tuple(int(item) for item in payload.get("task_paper_ids") or ()) != tuple(
            inspected["paper_ids"]
        ):
            raise PilotAuthorizationError("resume_task_paper_scope_changed")
        if _safe_object(payload.get("task_scope_snapshot")) != _task_scope_snapshot(
            inspected["source_config"]
        ):
            raise PilotAuthorizationError("resume_task_search_scope_changed")
        return authorization

    def bind_resume_use(self, authorization_id: str, task_id: int) -> bool:
        with _FILE_LOCK:
            payload = self._read()
            if payload.get("scope") != RESUME_EXISTING_TASK_SCOPE:
                raise PilotAuthorizationError("authorization_scope_mismatch")
            if payload.get("authorization_id") != authorization_id:
                raise PilotAuthorizationError("authorization_id_mismatch")
            if int(payload.get("existing_task_id") or 0) != task_id:
                raise PilotAuthorizationError("resume_authorization_task_mismatch")
            bindings = _safe_object(payload.get("resume_bindings"))
            key = str(task_id)
            if key in bindings:
                return False
            bindings[key] = {
                "task_id": task_id,
                "bound_at": datetime.now(UTC).isoformat(),
                "reason": "resume_after_interruption",
            }
            payload["resume_bindings"] = bindings
            payload["use_count"] = int(payload.get("use_count") or 0) + 1
            payload["status"] = "consumed"
            self._write(payload)
            return True

    def _resume_authorization_paths(self) -> list[Path]:
        if not self.resume_authorization_dir.is_dir():
            return []
        return sorted(
            self.resume_authorization_dir.glob("*.json"),
            key=lambda item: (item.stat().st_mtime_ns, item.name),
        )

    def _append_task_resume_reference(
        self,
        database: Database,
        task_id: int,
        authorization: PilotAuthorization,
        *,
        issuer_mode: str,
    ) -> None:
        with database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT source_config_json FROM workflow_batches WHERE id=?",
                (task_id,),
            ).fetchone()
            if row is None:
                raise PilotAuthorizationError("task_not_found")
            source = _safe_object(row["source_config_json"])
            references = list(source.get("pilot_resume_authorizations") or [])
            if not any(
                _safe_object(item).get("authorization_id")
                == authorization.authorization_id
                for item in references
            ):
                references.append(
                    {
                        "authorization_id": authorization.authorization_id,
                        "original_authorization_id": authorization.original_authorization_id,
                        "scope": RESUME_EXISTING_TASK_SCOPE,
                        "issued_at": authorization.created_at,
                        "expires_at": authorization.expires_at,
                        "issuer_mode": issuer_mode,
                        "reason": "resume_after_interruption",
                    }
                )
                source["pilot_resume_authorizations"] = references
                connection.execute(
                    """UPDATE workflow_batches SET source_config_json=?,updated_at=?
                    WHERE id=?""",
                    (
                        json.dumps(source, ensure_ascii=False, sort_keys=True),
                        datetime.now(UTC).isoformat(),
                        task_id,
                    ),
                )

    def revoke(self, authorization_id: str) -> None:
        with _FILE_LOCK:
            payload = self._read()
            if payload.get("authorization_id") != authorization_id:
                raise PilotAuthorizationError("authorization_id_mismatch")
            payload["status"] = "revoked"
            self._write(payload)

    def _authorization(self, payload: dict[str, Any]) -> PilotAuthorization:
        limits = PilotLimits(**dict(payload["limits"]))
        scope = str(payload.get("scope") or "create_tasks")
        limits.validate(allow_zero_tasks=scope == RESUME_EXISTING_TASK_SCOPE)
        status = str(payload["status"])
        if status not in {"active", "consumed", "revoked", "expired"}:
            raise ValueError("invalid authorization status")
        return PilotAuthorization(
            authorization_id=str(payload["authorization_id"]),
            created_at=str(payload["created_at"]),
            expires_at=str(payload["expires_at"]),
            status=status,
            operator_note=str(payload["operator_note"]),
            runtime_config_sha256=str(payload["runtime_config_sha256"]),
            prompt_bundle_sha256=str(payload["prompt_bundle_sha256"]),
            models={str(key): str(value) for key, value in dict(payload["models"]).items()},
            limits=limits,
            runtime_bundle_sha256=(
                str(payload["runtime_bundle_sha256"])
                if payload.get("runtime_bundle_sha256")
                else None
            ),
            scope=scope,
            existing_task_id=(
                int(payload["existing_task_id"])
                if payload.get("existing_task_id") is not None
                else None
            ),
            original_authorization_id=(
                str(payload["original_authorization_id"])
                if payload.get("original_authorization_id")
                else None
            ),
        )

    def _read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


def _safe_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _task_scope_snapshot(value: object) -> dict[str, object]:
    source = _safe_object(value)
    return {
        key: source.get(key)
        for key in (
            "journals",
            "date_from",
            "date_to",
            "since",
            "limit",
            "runtime_mode",
            "pilot_authorization_id",
            "runtime_config_sha256",
            "runtime_bundle_sha256",
            "prompt_bundle_sha256",
            "pilot_models_snapshot",
        )
    }


def _as_utc(value: datetime | None) -> datetime:
    current = value or datetime.now(UTC)
    if current.tzinfo is None:
        return current.replace(tzinfo=UTC)
    return current.astimezone(UTC)


def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise PilotAuthorizationError("authorization_timestamp_invalid") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
