from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from eduecon_radar.db import Database, utcnow
from eduecon_radar.domain.goldset_archive_gate import validate_archive_manifest
from eduecon_radar.domain.goldset_calibration import load_human_gold_snapshot
from eduecon_radar.domain.goldset_storage import AgreementReport, gold_set_schema_ready
from eduecon_radar.domain.models import DomainConfigBundle
from eduecon_radar.domain.semantic_models import SemanticEngineConfig
from eduecon_radar.domain.semantic_prompting import canonical_sha256


def build_model_bundle_candidate(
    *,
    database: Database,
    production_database: Path,
    archive_manifest: Path,
    snapshot_path: Path,
    snapshot_sha256: str,
    candidate_version: str,
    bundle: DomainConfigBundle,
    engine: SemanticEngineConfig,
    prompt_bundle_sha256: str,
    calibration_report: dict,
    agreement_report: AgreementReport,
    agreement_gold_set_id: int,
    red_team_report: dict,
    budget: dict,
    known_limitations: list[str],
    git_commit: str | None,
    output: Path,
) -> dict:
    if not gold_set_schema_ready(database):
        raise RuntimeError("Gold Set schema is not installed")
    if output.exists():
        raise FileExistsError(f"Model bundle candidate already exists: {output}")
    archive = validate_archive_manifest(
        archive_manifest,
        production_database=production_database,
        active_database=production_database,
    )
    if not archive.valid:
        raise ValueError("Production archive gate failed: " + ", ".join(archive.reasons))
    snapshot = load_human_gold_snapshot(snapshot_path, snapshot_sha256)
    if snapshot.get("source_db_sha256") != archive.source_sha256:
        raise ValueError("Gold snapshot source hash does not match the production archive")
    if len(snapshot["items"]) < 200:
        raise ValueError("A production candidate requires at least 200 real Gold items")
    if not agreement_report.ready_for_calibration:
        raise ValueError("Gold annotation agreement gate has not passed")
    if (
        not engine.fast.enabled
        or not engine.strong_review.enabled
        or engine.fast.model == "not-enabled"
        or engine.strong_review.model == "not-enabled"
    ):
        raise ValueError("Enabled Fast and Strong model versions are required for bundling")
    if calibration_report.get("warnings"):
        raise ValueError("Calibration readiness warnings must be resolved before bundling")
    if calibration_report.get("gold_set_snapshot_sha256") != snapshot_sha256.upper():
        raise ValueError("Calibration report belongs to a different Gold snapshot")
    if calibration_report.get("holdout_evaluation_count") != 1:
        raise ValueError("A final one-time holdout evaluation is required")
    if not red_team_report.get("critical_cases_passed"):
        raise ValueError("Critical prompt-injection red-team cases have not passed")
    with database.connect() as connection:
        snapshot_row = connection.execute(
            "SELECT id,gold_set_id FROM gold_set_snapshots WHERE snapshot_sha256=?",
            (snapshot_sha256.upper(),),
        ).fetchone()
    if snapshot_row is None:
        raise ValueError("Snapshot is not registered in the selected Gold Set database")
    if int(snapshot_row["gold_set_id"]) != agreement_gold_set_id:
        raise ValueError("Agreement report Gold Set does not own the selected snapshot")
    content = {
        "candidate_version": candidate_version,
        "status": "candidate",
        "release_status": "NOT RELEASED",
        "gold_set": {
            "snapshot_version": snapshot["snapshot_version"],
            "snapshot_sha256": snapshot_sha256.upper(),
            "sample_count": len(snapshot["items"]),
            "annotation_version": snapshot["annotation_version"],
            "agreement": asdict(agreement_report),
        },
        "versions": {
            "profile": bundle.profile.version,
            "ontology": bundle.ontology.version,
            "scoring": bundle.scoring.scoring_version,
            "semantic_engine": engine.version,
            "fast_model": engine.fast.model,
            "strong_model": engine.strong_review.model,
            "escalation_policy": engine.escalation.version,
            "decision_policy": calibration_report["selected_policy"],
        },
        "hashes": {
            "config_bundle_sha256": bundle.config_sha256,
            "prompt_bundle_sha256": prompt_bundle_sha256,
            "calibration_report_sha256": calibration_report["calibration_report_sha256"],
            "archive_source_sha256": archive.source_sha256,
        },
        "holdout_metrics": calibration_report["holdout"],
        "subgroup_metrics": calibration_report["subgroups"],
        "strong_review_incremental_value": calibration_report["strong_review_incremental_value"],
        "human_review_load": calibration_report["holdout"],
        "strategy_candidates": calibration_report["strategy_candidates"],
        "budget": budget,
        "red_team": red_team_report,
        "known_limitations": known_limitations,
        "production_archive_manifest": str(archive_manifest.resolve()),
        "git_commit": git_commit,
        "created_at": utcnow(),
    }
    candidate_hash = canonical_sha256(content)
    candidate = {**content, "bundle_sha256": candidate_hash}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        with database.connect() as connection:
            connection.execute(
                """INSERT INTO model_bundle_candidates
                (candidate_version,gold_set_snapshot_id,bundle_json,bundle_sha256,
                 status,created_at,approved_at) VALUES (?,?,?,?, 'candidate',?,NULL)""",
                (
                    candidate_version,
                    int(snapshot_row["id"]),
                    json.dumps(candidate, ensure_ascii=False, sort_keys=True),
                    candidate_hash,
                    content["created_at"],
                ),
            )
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return candidate


def inspect_model_bundle_candidate(path: Path) -> dict:
    candidate = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = candidate.pop("bundle_sha256", None)
    actual_hash = canonical_sha256(candidate)
    candidate["bundle_sha256"] = stored_hash
    candidate["hash_valid"] = stored_hash == actual_hash
    candidate["releasable_automatically"] = False
    return candidate
