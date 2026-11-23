# --- not part of the product core: see docs/PRODUCT_CORE.md ---
# RESEARCH/GOVERNANCE (Gold pilot). CLI only; never on the product path.

from __future__ import annotations

import json
import random
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from eduecon_radar.db import Database, utcnow
from eduecon_radar.services.archive import sha256_file
from eduecon_radar.services.gold_governance import canonical_hash, canonical_json
from eduecon_radar.services.gold_metrics import evaluate_gold_predictions


@dataclass(frozen=True)
class GoldPilotBudget:
    max_calls: int
    max_input_tokens: int
    max_output_tokens: int
    max_cost_microunits: int


@dataclass(frozen=True)
class GoldPilotPreflight:
    snapshot_id: int
    snapshot_sha256: str
    runtime_config_sha256: str
    split_manifest_sha256: str
    split_name: str
    eligible_papers: int
    label_blinded: bool


def validate_gold_pilot_preflight(
    *,
    gold_database: Path,
    production_database: Path,
    snapshot_path: Path,
    snapshot_sha256: str,
    candidate_runtime_config: Path,
    candidate_runtime_config_sha256: str,
    split_name: str,
    budget: GoldPilotBudget,
    confirm_live_ai: bool,
    confirm_labels_not_in_prompt: bool,
    confirm_isolated_gold_database: bool,
    confirm_production_database_readonly: bool,
) -> GoldPilotPreflight:
    if split_name not in {"development", "calibration"}:
        raise ValueError("Gold AI Pilot cannot run on Holdout")
    confirmations = {
        "confirm_live_ai": confirm_live_ai,
        "confirm_labels_not_in_prompt": confirm_labels_not_in_prompt,
        "confirm_isolated_gold_database": confirm_isolated_gold_database,
        "confirm_production_database_readonly": confirm_production_database_readonly,
    }
    missing = [name for name, value in confirmations.items() if not value]
    if missing:
        raise ValueError("Gold AI Pilot confirmations missing: " + ", ".join(missing))
    if any(value <= 0 for value in asdict(budget).values()):
        raise ValueError("Every Gold AI Pilot budget must be explicitly positive")
    gold = gold_database.resolve()
    production = production_database.resolve()
    if gold == production or snapshot_path.resolve() == production:
        raise ValueError("Gold Pilot input/output must be isolated from the production database")
    if not gold.is_file() or not production.is_file():
        raise FileNotFoundError("Explicit Gold and production database paths must exist")
    if sha256_file(candidate_runtime_config) != candidate_runtime_config_sha256.upper():
        raise ValueError("Candidate runtime config SHA256 mismatch")
    config = json.loads(candidate_runtime_config.read_text(encoding="utf-8"))
    required_config = {
        "status",
        "provider",
        "fast_model",
        "strong_model",
        "prompt_bundle_sha256",
    }
    missing_config = sorted(required_config - set(config))
    if missing_config:
        raise ValueError(
            "Candidate runtime config fields missing: " + ", ".join(missing_config)
        )
    if config["status"] != "candidate":
        raise ValueError("Gold Pilot requires an inspected candidate runtime config")
    serialized_config = canonical_json(config).lower()
    if "gold_label" in serialized_config or "final_label" in serialized_config:
        raise ValueError("Candidate runtime config must not contain Gold labels")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    embedded_sha = str(snapshot.pop("snapshot_sha256", "")).upper()
    if embedded_sha != snapshot_sha256.upper() or canonical_hash(snapshot) != embedded_sha:
        raise ValueError("Gold snapshot SHA256 mismatch")
    snapshot["snapshot_sha256"] = embedded_sha
    if snapshot.get("status") != "frozen":
        raise ValueError("A frozen Gold snapshot is required")
    database = Database(gold)
    with database.connect() as connection:
        snapshot_row = connection.execute(
            """SELECT id,split_manifest_sha256 FROM gold_governance_snapshots
            WHERE snapshot_sha256=?""",
            (snapshot_sha256.upper(),),
        ).fetchone()
        if snapshot_row is None:
            raise ValueError("Gold snapshot is not registered in this Gold database")
        eligible = int(
            connection.execute(
                """SELECT COUNT(*) FROM gold_split_assignments a
                JOIN gold_split_manifests s ON s.id=a.split_manifest_id
                WHERE s.manifest_sha256=? AND a.split_name=?""",
                (str(snapshot_row["split_manifest_sha256"]), split_name),
            ).fetchone()[0]
        )
    if not eligible:
        raise ValueError("Selected Gold Pilot split has no papers")
    return GoldPilotPreflight(
        snapshot_id=int(snapshot_row["id"]),
        snapshot_sha256=snapshot_sha256.upper(),
        runtime_config_sha256=candidate_runtime_config_sha256.upper(),
        split_manifest_sha256=str(snapshot_row["split_manifest_sha256"]),
        split_name=split_name,
        eligible_papers=eligible,
        label_blinded=True,
    )


class GoldPilotRunner:
    """Budgeted, resumable runner. The processor receives source data, never Gold labels."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def run(
        self,
        *,
        run_uid: str,
        preflight: GoldPilotPreflight,
        budget: GoldPilotBudget,
        processor: Callable[[dict], dict],
        random_seed: int = 20260726,
        resume: bool = False,
    ) -> dict:
        run_id, owner, fencing = self._claim(
            run_uid, preflight, budget, random_seed=random_seed, resume=resume
        )
        rows = self._safe_inputs(preflight)
        rng = random.Random(random_seed)
        rng.shuffle(rows)
        stats = self.inspect(run_uid)["stats"]
        for item in rows:
            fingerprint = canonical_hash(
                {
                    "paper_content_sha256": item["paper_content_sha256"],
                    "runtime_config_sha256": preflight.runtime_config_sha256,
                    "split_manifest_sha256": preflight.split_manifest_sha256,
                }
            )
            if self._result_exists(run_id, fingerprint):
                continue
            if (
                stats["calls"] >= budget.max_calls
                or stats["input_tokens"] >= budget.max_input_tokens
                or stats["output_tokens"] >= budget.max_output_tokens
                or stats["cost_microunits"] >= budget.max_cost_microunits
            ):
                self._finish(run_id, owner, fencing, "paused", stats)
                return {"run_uid": run_uid, "status": "paused", "stats": stats}
            started = datetime.now(UTC)
            result: dict | None = None
            failure: Exception | None = None
            try:
                result = processor(_strict_ai_input(item))
            except Exception as exc:  # noqa: BLE001 - isolated and persisted
                failure = exc
            latency = round((datetime.now(UTC) - started).total_seconds() * 1000)
            usage = result.get("usage", {}) if result else {}
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            cost = int(usage.get("estimated_cost_microunits") or 0)
            stats["calls"] += 1
            stats["input_tokens"] += input_tokens
            stats["output_tokens"] += output_tokens
            stats["cost_microunits"] += cost
            stats["succeeded" if failure is None else "failed"] += 1
            self._save_result(
                run_id=run_id,
                item=item,
                request_fingerprint=fingerprint,
                result=result,
                failure=failure,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                latency_ms=latency,
                owner=owner,
                fencing=fencing,
            )
        self._finish(run_id, owner, fencing, "completed", stats)
        return {"run_uid": run_uid, "status": "completed", "stats": stats}

    def inspect(self, run_uid: str) -> dict:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM gold_pilot_runs_v2 WHERE run_uid=?", (run_uid,)
            ).fetchone()
            if row is None:
                return {
                    "run_uid": run_uid,
                    "status": "not_started",
                    "stats": {
                        "calls": 0,
                        "succeeded": 0,
                        "failed": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_microunits": 0,
                    },
                }
        return {**dict(row), "stats": json.loads(row["stats_json"])}

    def reconcile_usage(self, run_uid: str) -> dict:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT id FROM gold_pilot_runs_v2 WHERE run_uid=?", (run_uid,)
            ).fetchone()
            if row is None:
                raise ValueError("Gold Pilot run not found")
            totals = connection.execute(
                """SELECT COUNT(*) calls,SUM(status='succeeded') succeeded,
                SUM(status='failed') failed,COALESCE(SUM(input_tokens),0) input_tokens,
                COALESCE(SUM(output_tokens),0) output_tokens,
                COALESCE(SUM(estimated_cost_microunits),0) cost_microunits
                FROM gold_pilot_results_v2 WHERE pilot_run_id=?""",
                (int(row["id"]),),
            ).fetchone()
        fields = (
            "calls",
            "succeeded",
            "failed",
            "input_tokens",
            "output_tokens",
            "cost_microunits",
        )
        return {key: int(totals[key] or 0) for key in fields}

    def _claim(
        self,
        run_uid: str,
        preflight: GoldPilotPreflight,
        budget: GoldPilotBudget,
        *,
        random_seed: int,
        resume: bool,
    ) -> tuple[int, str, int]:
        owner = uuid.uuid4().hex
        now = datetime.now(UTC)
        expires = now + timedelta(minutes=5)
        connection = sqlite3.connect(self.database.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute(
                "SELECT * FROM gold_pilot_runs_v2 WHERE run_uid=?", (run_uid,)
            ).fetchone()
            if run and not resume:
                raise ValueError("Gold Pilot run exists; explicit resume is required")
            if run is None:
                cursor = connection.execute(
                    """INSERT INTO gold_pilot_runs_v2
                    (run_uid,snapshot_id,candidate_runtime_config_sha256,split_name,
                     budget_json,status,label_blinded,randomized_seed,stats_json,
                     created_at,updated_at)
                    VALUES (?,?,?,?,?,'running',1,?,?,?,?)""",
                    (
                        run_uid,
                        preflight.snapshot_id,
                        preflight.runtime_config_sha256,
                        preflight.split_name,
                        canonical_json(asdict(budget)),
                        random_seed,
                        canonical_json(
                            {
                                "calls": 0,
                                "succeeded": 0,
                                "failed": 0,
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "cost_microunits": 0,
                            }
                        ),
                        utcnow(),
                        utcnow(),
                    ),
                )
                run_id = int(cursor.lastrowid)
            else:
                run_id = int(run["id"])
                connection.execute(
                    "UPDATE gold_pilot_runs_v2 SET status='running',updated_at=? WHERE id=?",
                    (utcnow(), run_id),
                )
            active = connection.execute(
                """SELECT * FROM gold_pilot_leases
                WHERE pilot_run_id=? AND status='active' ORDER BY fencing_token DESC LIMIT 1""",
                (run_id,),
            ).fetchone()
            if active and datetime.fromisoformat(active["expires_at"]) > now:
                raise RuntimeError("Gold Pilot run already has an active executor")
            if active:
                connection.execute(
                    "UPDATE gold_pilot_leases SET status='stale' WHERE id=?",
                    (int(active["id"]),),
                )
            fencing = int(
                connection.execute(
                    "SELECT COALESCE(MAX(fencing_token),0)+1 FROM gold_pilot_leases"
                ).fetchone()[0]
            )
            connection.execute(
                """INSERT INTO gold_pilot_leases
                (pilot_run_id,owner_token,fencing_token,status,acquired_at,
                 heartbeat_at,expires_at) VALUES (?,?,?,'active',?,?,?)""",
                (run_id, owner, fencing, now.isoformat(), now.isoformat(), expires.isoformat()),
            )
            connection.commit()
            return run_id, owner, fencing
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _safe_inputs(self, preflight: GoldPilotPreflight) -> list[dict]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT p.id corpus_paper_id,p.gold_paper_id,p.content_sha256
                paper_content_sha256,p.original_title,p.abstract,p.authors_json,
                p.journal,p.publication_date,p.canonical_doi
                FROM gold_split_assignments a
                JOIN gold_split_manifests s ON s.id=a.split_manifest_id
                JOIN gold_corpus_papers p ON p.id=a.corpus_paper_id
                WHERE s.manifest_sha256=? AND a.split_name=? ORDER BY p.gold_paper_id""",
                (preflight.split_manifest_sha256, preflight.split_name),
            ).fetchall()
        return [{**dict(row), "authors": json.loads(row["authors_json"])} for row in rows]

    def _result_exists(self, run_id: int, fingerprint: str) -> bool:
        with self.database.connect() as connection:
            return (
                connection.execute(
                    """SELECT 1 FROM gold_pilot_results_v2
                    WHERE pilot_run_id=? AND request_fingerprint=?""",
                    (run_id, fingerprint),
                ).fetchone()
                is not None
            )

    def _save_result(
        self,
        *,
        run_id: int,
        item: dict,
        request_fingerprint: str,
        result: dict | None,
        failure: Exception | None,
        input_tokens: int,
        output_tokens: int,
        cost: int,
        latency_ms: int,
        owner: str,
        fencing: int,
    ) -> None:
        with self.database.connect() as connection:
            _assert_lease(connection, run_id, owner, fencing)
            safe_result = dict(result or {})
            safe_result.pop("gold_label", None)
            safe_result.pop("final_label", None)
            connection.execute(
                """INSERT INTO gold_pilot_results_v2
                (pilot_run_id,corpus_paper_id,paper_content_sha256,request_fingerprint,
                 model_provider,model_name,prompt_version,prompt_sha256,input_tokens,
                 output_tokens,estimated_cost_microunits,latency_ms,status,
                 raw_response_reference,parsed_result_json,failure_type,no_abstract,
                 limitations_json,label_blinded,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?)""",
                (
                    run_id,
                    item["corpus_paper_id"],
                    item["paper_content_sha256"],
                    request_fingerprint,
                    safe_result.get("provider"),
                    safe_result.get("model"),
                    safe_result.get("prompt_version"),
                    safe_result.get("prompt_sha256"),
                    input_tokens,
                    output_tokens,
                    cost,
                    latency_ms,
                    "failed" if failure else "succeeded",
                    safe_result.get("raw_response_reference"),
                    canonical_json(safe_result) if result else None,
                    type(failure).__name__ if failure else None,
                    int(not bool(item.get("abstract"))),
                    canonical_json(
                        ["no_abstract"]
                        if not item.get("abstract")
                        else safe_result.get("limitations", [])
                    ),
                    utcnow(),
                ),
            )

    def _finish(
        self, run_id: int, owner: str, fencing: int, status: str, stats: dict
    ) -> None:
        with self.database.connect() as connection:
            _assert_lease(connection, run_id, owner, fencing)
            connection.execute(
                """UPDATE gold_pilot_runs_v2 SET status=?,stats_json=?,updated_at=?
                WHERE id=?""",
                (status, canonical_json(stats), utcnow(), run_id),
            )
            connection.execute(
                """UPDATE gold_pilot_leases SET status='released',heartbeat_at=?
                WHERE pilot_run_id=? AND owner_token=? AND fencing_token=? AND status='active'""",
                (utcnow(), run_id, owner, fencing),
            )


def evaluate_holdout_once(
    *,
    database: Database,
    snapshot_sha256: str,
    candidate_bundle_sha256: str,
    split_manifest_sha256: str,
    predictions: list[dict],
    confirm_final_candidate: bool,
    confirm_holdout_one_time_evaluation: bool,
) -> dict:
    if not confirm_final_candidate or not confirm_holdout_one_time_evaluation:
        raise ValueError("Final candidate and one-time Holdout confirmations are required")
    with database.connect() as connection:
        snapshot = connection.execute(
            """SELECT id,manifest_json,split_manifest_sha256
            FROM gold_governance_snapshots WHERE snapshot_sha256=?""",
            (snapshot_sha256.upper(),),
        ).fetchone()
        if snapshot is None:
            raise ValueError("Gold snapshot not found")
        if str(snapshot["split_manifest_sha256"]) != split_manifest_sha256.upper():
            raise ValueError("Split manifest SHA256 mismatch")
        prior = connection.execute(
            """SELECT 1 FROM gold_holdout_evaluations
            WHERE snapshot_id=? AND candidate_bundle_sha256=?""",
            (int(snapshot["id"]), candidate_bundle_sha256.upper()),
        ).fetchone()
        if prior:
            raise ValueError("This candidate bundle already consumed its formal Holdout evaluation")
        split_rows = connection.execute(
            """SELECT p.gold_paper_id,a.gold_label,p.duplicate_cluster_key
            FROM gold_split_assignments a
            JOIN gold_split_manifests s ON s.id=a.split_manifest_id
            JOIN gold_corpus_papers p ON p.id=a.corpus_paper_id
            WHERE s.manifest_sha256=? AND a.split_name='holdout'""",
            (split_manifest_sha256.upper(),),
        ).fetchall()
    prediction_map = {str(item["gold_paper_id"]): item for item in predictions}
    joined = []
    for row in split_rows:
        prediction = prediction_map.get(str(row["gold_paper_id"]))
        if prediction is None:
            continue
        joined.append(
            {
                **prediction,
                "gold_paper_id": row["gold_paper_id"],
                "final_label": row["gold_label"],
                "duplicate_cluster_uid": row["duplicate_cluster_key"],
            }
        )
    result = evaluate_gold_predictions(joined)
    result_hash = canonical_hash(result)
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO gold_holdout_evaluations
            (snapshot_id,candidate_bundle_sha256,split_manifest_sha256,
             result_sha256,evaluated_at) VALUES (?,?,?,?,?)""",
            (
                int(snapshot["id"]),
                candidate_bundle_sha256.upper(),
                split_manifest_sha256.upper(),
                result_hash,
                utcnow(),
            ),
        )
    return {**result, "result_sha256": result_hash, "holdout_evaluation_count": 1}


def build_candidate_manifest(
    *,
    bundle_version: str,
    gold_snapshot_version: str,
    gold_snapshot_sha256: str,
    split_manifest_sha256: str,
    runtime_config_sha256: str,
    evidence_hash: str,
    ontology_hash: str,
    scoring_hash: str,
    fast_model: str,
    strong_model: str,
    prompt_hashes: dict[str, str],
    composite_hash: str,
    escalation_hash: str,
    missed_risk_hash: str,
    budget_policy_hash: str,
    development_metrics: dict,
    calibration_metrics: dict,
    confidence_intervals: dict,
    subgroup_metrics: dict,
    review_load: dict,
    strong_net_value: dict,
    token_cost: dict,
    red_team_status: dict,
    known_limitations: list[str],
    git_commit: str | None,
) -> dict:
    body = {
        "bundle_schema_version": "model-bundle-candidate-v1",
        "bundle_version": bundle_version,
        "status": "candidate",
        "release_status": "NOT RELEASED",
        "gold_snapshot_version": gold_snapshot_version,
        "gold_snapshot_sha256": gold_snapshot_sha256.upper(),
        "split_manifest_sha256": split_manifest_sha256.upper(),
        "candidate_runtime_config_sha256": runtime_config_sha256.upper(),
        "evidence_hash": evidence_hash.upper(),
        "ontology_hash": ontology_hash.upper(),
        "scoring_hash": scoring_hash.upper(),
        "fast_model": fast_model,
        "strong_model": strong_model,
        "prompt_hashes": prompt_hashes,
        "composite_hash": composite_hash.upper(),
        "escalation_hash": escalation_hash.upper(),
        "missed_risk_hash": missed_risk_hash.upper(),
        "budget_policy_hash": budget_policy_hash.upper(),
        "development_metrics": development_metrics,
        "calibration_metrics": calibration_metrics,
        "holdout_metrics": None,
        "confidence_intervals": confidence_intervals,
        "subgroup_metrics": subgroup_metrics,
        "review_load": review_load,
        "strong_net_value": strong_net_value,
        "token_cost": token_cost,
        "red_team_status": red_team_status,
        "known_limitations": known_limitations,
        "git_commit": git_commit,
        "created_at": utcnow(),
    }
    return {**body, "bundle_sha256": canonical_hash(body)}


def verify_candidate_isolation(candidate: dict) -> None:
    if candidate.get("status") != "candidate":
        raise ValueError("Phase 6.6D1 can create candidate manifests only")
    if candidate.get("holdout_metrics") is not None:
        raise ValueError(
            "Candidate manifest must not claim Holdout metrics before final evaluation"
        )
    if candidate.get("release_status") != "NOT RELEASED":
        raise ValueError("Candidate cannot be exposed as a released runtime bundle")


def _strict_ai_input(item: dict) -> dict:
    allowed = {
        "gold_paper_id",
        "original_title",
        "abstract",
        "authors",
        "journal",
        "publication_date",
        "canonical_doi",
    }
    safe = {key: item.get(key) for key in allowed}
    serialized = canonical_json(safe).lower()
    if "gold_label" in serialized or "final_label" in serialized:
        raise RuntimeError("Gold label leakage detected")
    return safe


def _assert_lease(
    connection: sqlite3.Connection, run_id: int, owner: str, fencing: int
) -> None:
    row = connection.execute(
        """SELECT 1 FROM gold_pilot_leases WHERE pilot_run_id=? AND owner_token=?
        AND fencing_token=? AND status='active'""",
        (run_id, owner, fencing),
    ).fetchone()
    if row is None:
        raise RuntimeError("Gold Pilot lease/fencing validation failed")
