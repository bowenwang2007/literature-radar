from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from eduecon_radar.db import Database, utcnow
from eduecon_radar.domain.semantic_models import (
    CompositeDecisionResult,
    EscalationDecision,
    SemanticMissedPriority,
    SemanticPaperInput,
)
from eduecon_radar.domain.semantic_prompting import PromptBundle, canonical_sha256


@dataclass(frozen=True)
class StoredSemanticRun:
    id: int
    reused: bool
    succeeded: bool


@dataclass(frozen=True)
class StoredCompositeDecision:
    id: int
    reused: bool


def semantic_schema_ready(database: Database) -> bool:
    if not database.path.is_file():
        return False
    uri = database.path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
                "('domain_semantic_runs','domain_composite_decisions')"
            )
        }
    return tables == {"domain_semantic_runs", "domain_composite_decisions"}


def apply_domain_semantic_migration(database: Database, migration_path: Path) -> None:
    if not database.path.is_file():
        raise FileNotFoundError(f"Database does not exist: {database.path}")
    if not _table_exists(database.path, "domain_evidence_runs"):
        raise RuntimeError("Apply domain evidence migration before semantic migration")
    with database.connect() as connection:
        connection.executescript(migration_path.read_text(encoding="utf-8"))


def _table_exists(database_path: Path, table: str) -> bool:
    uri = database_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        return (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            is not None
        )


class DomainSemanticRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def successful_by_fingerprint(self, fingerprint: str, role: str) -> dict | None:
        if not semantic_schema_ready(self.database):
            return None
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM domain_semantic_runs WHERE input_fingerprint=? "
                "AND run_role=? AND succeeded=1 ORDER BY id DESC LIMIT 1",
                (fingerprint, role),
            ).fetchone()
        return _semantic_row(row)

    def save_success(
        self,
        *,
        paper: SemanticPaperInput,
        role: str,
        fingerprint: str,
        prompt: PromptBundle,
        model_provider: str,
        model_name: str,
        engine_version: str,
        escalation_policy_version: str,
        result: object,
        input_tokens: int,
        output_tokens: int,
        estimated_cost: float | None,
        parent_fast_run_id: int | None = None,
    ) -> StoredSemanticRun:
        existing = self.successful_by_fingerprint(fingerprint, role)
        if existing:
            return StoredSemanticRun(int(existing["id"]), True, True)
        result_json = _model_json(result)
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """INSERT INTO domain_semantic_runs
                (paper_id,workflow_batch_id,domain_evidence_run_id,profile_id,
                 run_role,input_fingerprint,input_json,
                 prompt_id,prompt_version,system_prompt_sha256,user_prompt_sha256,
                 model_provider,model_name,engine_version,escalation_policy_version,
                 parent_fast_run_id,result_json,semantic_score,decision,confidence,
                 education_is_core,economics_mechanism_present,information_sufficiency,
                 title_evidence_sufficiency,suspected_false_friend,adjacent_domain,
                 succeeded,input_tokens,output_tokens,
                 estimated_cost,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    paper.paper_id,
                    paper.workflow_batch_id,
                    paper.domain_evidence_run_id,
                    paper.profile_id,
                    role,
                    fingerprint,
                    paper.model_dump_json(),
                    prompt.prompt_id,
                    prompt.prompt_version,
                    prompt.system_sha256,
                    prompt.user_sha256,
                    model_provider,
                    model_name,
                    engine_version,
                    escalation_policy_version,
                    parent_fast_run_id,
                    result_json,
                    getattr(result, "semantic_score", None),
                    getattr(result, "decision", None)
                    or getattr(result, "independent_decision", None),
                    getattr(result, "confidence", None),
                    _optional_bool(getattr(result, "education_is_core", None)),
                    _optional_bool(getattr(result, "economics_mechanism_present", None)),
                    getattr(result, "information_sufficiency", None)
                    or getattr(result, "decision_sufficiency", None),
                    getattr(result, "title_evidence_sufficiency", None),
                    _optional_bool(getattr(result, "suspected_false_friend", None)),
                    getattr(result, "adjacent_domain", None),
                    1,
                    input_tokens,
                    output_tokens,
                    estimated_cost,
                    utcnow(),
                ),
            )
        return StoredSemanticRun(int(cursor.lastrowid), False, True)

    def save_failure(
        self,
        *,
        paper: SemanticPaperInput,
        role: str,
        fingerprint: str,
        prompt: PromptBundle,
        model_provider: str,
        model_name: str,
        engine_version: str,
        escalation_policy_version: str,
        error: Exception,
        parent_fast_run_id: int | None = None,
    ) -> StoredSemanticRun:
        failure_payload = _failure_payload(
            error=error,
            prompt=prompt,
            model_provider=model_provider,
            model_name=model_name,
            engine_version=engine_version,
            escalation_policy_version=escalation_policy_version,
        )
        with self.database.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO domain_semantic_runs
                (paper_id,workflow_batch_id,domain_evidence_run_id,profile_id,
                 run_role,input_fingerprint,input_json,
                 prompt_id,prompt_version,system_prompt_sha256,user_prompt_sha256,
                 model_provider,model_name,engine_version,escalation_policy_version,
                 parent_fast_run_id,result_json,succeeded,failure_type,failure_message,created_at)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    paper.paper_id,
                    paper.workflow_batch_id,
                    paper.domain_evidence_run_id,
                    paper.profile_id,
                    role,
                    fingerprint,
                    paper.model_dump_json(),
                    prompt.prompt_id,
                    prompt.prompt_version,
                    prompt.system_sha256,
                    prompt.user_sha256,
                    model_provider,
                    model_name,
                    engine_version,
                    escalation_policy_version,
                    parent_fast_run_id,
                    json.dumps(
                        failure_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    0,
                    type(error).__name__,
                    _safe_failure_message(error),
                    utcnow(),
                ),
            )
        return StoredSemanticRun(int(cursor.lastrowid), False, False)

    def save_composite(
        self,
        *,
        paper: SemanticPaperInput,
        fast_run_id: int,
        strong_run_id: int | None,
        fingerprint: str,
        escalation: EscalationDecision,
        composite: CompositeDecisionResult,
        priority: SemanticMissedPriority,
    ) -> StoredCompositeDecision:
        with self.database.connect() as connection:
            existing = connection.execute(
                "SELECT id FROM domain_composite_decisions WHERE input_fingerprint=?",
                (fingerprint,),
            ).fetchone()
            if existing:
                return StoredCompositeDecision(int(existing["id"]), True)
            cursor = connection.execute(
                """INSERT INTO domain_composite_decisions
                (paper_id,workflow_batch_id,domain_evidence_run_id,
                 fast_semantic_run_id,strong_semantic_run_id,
                 input_fingerprint,escalation_json,composite_json,missed_priority_json,
                 policy_version,provisional_decision,uncertainty_level,
                 conflict_types_json,calibration_status,publishable,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)""",
                (
                    paper.paper_id,
                    paper.workflow_batch_id,
                    paper.domain_evidence_run_id,
                    fast_run_id,
                    strong_run_id,
                    fingerprint,
                    escalation.model_dump_json(),
                    composite.model_dump_json(),
                    priority.model_dump_json(),
                    composite.policy_version,
                    composite.provisional_decision,
                    composite.uncertainty_level,
                    json.dumps(composite.conflict_types, ensure_ascii=False),
                    composite.calibration_status,
                    utcnow(),
                ),
            )
        return StoredCompositeDecision(int(cursor.lastrowid), False)

    def latest_for_paper(self, paper_id: int) -> dict | None:
        return self.latest_for_papers((paper_id,)).get(paper_id)

    def latest_for_papers(
        self, paper_ids: list[int] | tuple[int, ...]
    ) -> dict[int, dict]:
        """Load latest composite plus Fast/Strong rows for a page in one query."""
        normalized = list(dict.fromkeys(int(paper_id) for paper_id in paper_ids))
        if not normalized or not semantic_schema_ready(self.database):
            return {}
        placeholders = ",".join("?" for _ in normalized)
        with self.database.connect() as connection:
            rows = connection.execute(
                f"""SELECT composite.*,
                    fast.input_json fast_input_json,
                    fast.result_json fast_result_json,
                    fast.succeeded fast_succeeded,
                    fast.failure_type fast_failure_type,
                    fast.failure_message fast_failure_message,
                    fast.decision fast_decision,
                    fast.confidence fast_confidence,
                    fast.model_name fast_model_name,
                    fast.created_at fast_created_at,
                    strong.input_json strong_input_json,
                    strong.result_json strong_result_json,
                    strong.succeeded strong_succeeded,
                    strong.failure_type strong_failure_type,
                    strong.failure_message strong_failure_message,
                    strong.decision strong_decision,
                    strong.confidence strong_confidence,
                    strong.model_name strong_model_name,
                    strong.created_at strong_created_at
                FROM domain_composite_decisions composite
                JOIN domain_semantic_runs fast ON fast.id=composite.fast_semantic_run_id
                LEFT JOIN domain_semantic_runs strong
                    ON strong.id=composite.strong_semantic_run_id
                WHERE composite.paper_id IN ({placeholders})
                AND composite.id=(
                    SELECT latest.id FROM domain_composite_decisions latest
                    WHERE latest.paper_id=composite.paper_id
                    ORDER BY latest.created_at DESC,latest.id DESC LIMIT 1
                )""",
                normalized,
            ).fetchall()
        output: dict[int, dict] = {}
        for row in rows:
            raw = dict(row)
            result = {
                key: value
                for key, value in raw.items()
                if not key.startswith(("fast_", "strong_"))
            }
            result["escalation"] = json.loads(result.pop("escalation_json"))
            result["composite"] = json.loads(result.pop("composite_json"))
            result["missed_priority"] = json.loads(result.pop("missed_priority_json"))
            result["fast"] = _prefixed_semantic_row(raw, "fast")
            result["strong"] = _prefixed_semantic_row(raw, "strong")
            output[int(result["paper_id"])] = result
        return output


def inspect_semantic_run_read_only(database_path: Path, paper_id: int) -> dict | None:
    if not database_path.is_file():
        raise FileNotFoundError(f"Database does not exist: {database_path}")
    uri = database_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
                "('domain_semantic_runs','domain_composite_decisions')"
            )
        }
        if tables != {"domain_semantic_runs", "domain_composite_decisions"}:
            return None
        row = connection.execute(
            "SELECT * FROM domain_composite_decisions WHERE paper_id=? "
            "ORDER BY created_at DESC,id DESC LIMIT 1",
            (paper_id,),
        ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["escalation"] = json.loads(result.pop("escalation_json"))
        result["composite"] = json.loads(result.pop("composite_json"))
        result["missed_priority"] = json.loads(result.pop("missed_priority_json"))
        fast = connection.execute(
            "SELECT * FROM domain_semantic_runs WHERE id=?",
            (result["fast_semantic_run_id"],),
        ).fetchone()
        strong = None
        if result["strong_semantic_run_id"]:
            strong = connection.execute(
                "SELECT * FROM domain_semantic_runs WHERE id=?",
                (result["strong_semantic_run_id"],),
            ).fetchone()
    result["fast"] = _semantic_row(fast)
    result["strong"] = _semantic_row(strong)
    return result


def _model_json(value: object) -> str:
    method = getattr(value, "model_dump", None)
    if not callable(method):
        raise TypeError("Semantic result must be a Pydantic model")
    return json.dumps(
        method(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _semantic_row(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    result = dict(row)
    result["input"] = json.loads(result.pop("input_json"))
    raw_result = result.pop("result_json")
    result["result"] = json.loads(raw_result) if raw_result else None
    return result


def _prefixed_semantic_row(row: dict, prefix: str) -> dict | None:
    if row.get(f"{prefix}_input_json") is None:
        return None
    raw_result = row.get(f"{prefix}_result_json")
    return {
        "input": json.loads(row[f"{prefix}_input_json"]),
        "result": json.loads(raw_result) if raw_result else None,
        "succeeded": row.get(f"{prefix}_succeeded"),
        "failure_type": row.get(f"{prefix}_failure_type"),
        "failure_message": row.get(f"{prefix}_failure_message"),
        "decision": row.get(f"{prefix}_decision"),
        "confidence": row.get(f"{prefix}_confidence"),
        "model_name": row.get(f"{prefix}_model_name"),
        "created_at": row.get(f"{prefix}_created_at"),
    }


def _optional_bool(value: object) -> int | None:
    return int(value) if isinstance(value, bool) else None


def _safe_failure_message(error: Exception) -> str:
    """Persist diagnostic category only; never persist secrets or response payloads."""
    message = str(error)
    message = re.sub(r"(?i)(authorization\s*[:=]\s*)([^\s,;]+)", r"\1[REDACTED]", message)
    message = re.sub(r"\bsk-[A-Za-z0-9_-]+\b", "[REDACTED]", message)
    return message[:300]


def _failure_payload(
    *,
    error: Exception,
    prompt: PromptBundle,
    model_provider: str,
    model_name: str,
    engine_version: str,
    escalation_policy_version: str,
) -> dict[str, object]:
    raw_response = getattr(error, "raw_response", None)
    validation_error = getattr(error, "validation_error", None)
    actual_model = getattr(error, "model_name", None) or model_name
    runtime_fingerprint = canonical_sha256(
        {
            "model_provider": model_provider,
            "model_name": actual_model,
            "engine_version": engine_version,
            "escalation_policy_version": escalation_policy_version,
            "system_prompt_sha256": prompt.system_sha256,
            "user_prompt_sha256": prompt.user_sha256,
        }
    )
    return {
        "execution_state": "retryable_error",
        "raw_response": _safe_raw_response(raw_response),
        "validation_error": str(validation_error)[:1000] if validation_error else None,
        "model_identity": {
            "provider": model_provider,
            "model": actual_model,
        },
        "prompt_hashes": {
            "system": prompt.system_sha256,
            "user": prompt.user_sha256,
        },
        "runtime_fingerprint": runtime_fingerprint,
        "failure_type": type(error).__name__,
        "failure_message": _safe_failure_message(error),
    }


def _safe_raw_response(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = re.sub(
        r"(?i)(authorization\s*[:=]\s*)([^\s,;]+)", r"\1[REDACTED]", value
    )
    cleaned = re.sub(r"\bsk-[A-Za-z0-9_-]+\b", "[REDACTED]", cleaned)
    return cleaned[:50_000]
