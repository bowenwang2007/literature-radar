from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from eduecon_radar.db import Database, utcnow
from eduecon_radar.domain.evidence import DomainEvidenceResult
from eduecon_radar.domain.missed_risk import EvidenceReviewPriority
from eduecon_radar.domain.scoring import RuleScoringResult


@dataclass(frozen=True)
class StoredEvidenceRun:
    id: int
    reused: bool


def domain_evidence_schema_ready(database: Database) -> bool:
    if not database.path.is_file():
        return False
    uri = database.path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        return (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='domain_evidence_runs'"
            ).fetchone()
            is not None
        )


def apply_domain_evidence_migration(database: Database, migration_path: Path) -> None:
    if not database.path.is_file():
        raise FileNotFoundError(f"Database does not exist: {database.path}")
    sql = migration_path.read_text(encoding="utf-8")
    with database.connect() as connection:
        connection.executescript(sql)


class DomainEvidenceRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def save(
        self,
        *,
        evidence: DomainEvidenceResult,
        scoring: RuleScoringResult,
        priority: EvidenceReviewPriority,
        config_bundle_sha256: str,
        application_version: str,
        workflow_batch_id: int | None = None,
        force: bool = False,
    ) -> StoredEvidenceRun:
        if not domain_evidence_schema_ready(self.database):
            raise RuntimeError("Domain evidence migration has not been applied")
        with self.database.connect() as connection:
            existing = connection.execute(
                """SELECT id FROM domain_evidence_runs
                WHERE paper_id=? AND workflow_batch_id IS ? AND config_bundle_sha256=?
                AND scoring_version=? ORDER BY id DESC LIMIT 1""",
                (
                    evidence.paper_id,
                    workflow_batch_id,
                    config_bundle_sha256,
                    scoring.scoring_version,
                ),
            ).fetchone()
            if existing and not force:
                return StoredEvidenceRun(id=int(existing["id"]), reused=True)
            payload = {
                "evidence": evidence.model_dump(mode="json"),
                "scoring": scoring.model_dump(mode="json"),
                "priority": priority.model_dump(mode="json"),
            }
            cursor = connection.execute(
                """INSERT INTO domain_evidence_runs
                (paper_id,workflow_batch_id,profile_id,ontology_version,
                 subject_code_version,journal_prior_version,scoring_version,
                 config_bundle_sha256,application_version,ontology_evidence_score,
                 subject_code_evidence_score,journal_prior_evidence_score,
                 exclusion_evidence_penalty,concept_diversity_score,metadata_confidence,
                 provisional_rule_score,rule_evidence_tier,opaque_title,opacity_level,
                 review_priority_tier,review_priority_score,evidence_json,publishable,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    evidence.paper_id,
                    workflow_batch_id,
                    evidence.profile_id,
                    evidence.ontology_version,
                    evidence.subject_code_version,
                    evidence.journal_prior_version,
                    scoring.scoring_version,
                    config_bundle_sha256,
                    application_version,
                    scoring.ontology_evidence_score,
                    scoring.subject_code_evidence_score,
                    scoring.journal_prior_evidence_score,
                    scoring.exclusion_evidence_penalty,
                    scoring.concept_diversity_score,
                    scoring.metadata_confidence,
                    scoring.provisional_rule_score,
                    scoring.rule_evidence_tier,
                    int(evidence.opaque_title_features.opaque_title),
                    evidence.opaque_title_features.opacity_level,
                    priority.priority_tier,
                    priority.priority_score,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    int(scoring.publishable),
                    utcnow(),
                ),
            )
            return StoredEvidenceRun(id=int(cursor.lastrowid), reused=False)

    def latest_for_paper(
        self, paper_id: int, *, workflow_batch_id: int | None = None
    ) -> dict | None:
        return self.latest_for_papers(
            (paper_id,), workflow_batch_id=workflow_batch_id
        ).get(paper_id)

    def latest_for_papers(
        self,
        paper_ids: list[int] | tuple[int, ...],
        *,
        workflow_batch_id: int | None = None,
    ) -> dict[int, dict]:
        """Load the latest evidence for a visible page in one query."""
        normalized = list(dict.fromkeys(int(paper_id) for paper_id in paper_ids))
        if not normalized or not domain_evidence_schema_ready(self.database):
            return {}
        placeholders = ",".join("?" for _ in normalized)
        batch_condition = "" if workflow_batch_id is None else "AND workflow_batch_id=?"
        latest_batch_condition = (
            ""
            if workflow_batch_id is None
            else "AND latest.workflow_batch_id=candidate.workflow_batch_id"
        )
        params: list[object] = [*normalized]
        if workflow_batch_id is not None:
            params.append(workflow_batch_id)
        with self.database.connect() as connection:
            rows = connection.execute(
                f"""SELECT * FROM domain_evidence_runs candidate
                WHERE paper_id IN ({placeholders}) {batch_condition}
                AND id=(
                    SELECT latest.id FROM domain_evidence_runs latest
                    WHERE latest.paper_id=candidate.paper_id
                    {latest_batch_condition}
                    ORDER BY latest.created_at DESC,latest.id DESC LIMIT 1
                )""",
                params,
            ).fetchall()
        output: dict[int, dict] = {}
        for row in rows:
            result = dict(row)
            result["evidence_payload"] = json.loads(result.pop("evidence_json"))
            output[int(result["paper_id"])] = result
        return output


def inspect_domain_evidence_read_only(database_path: Path, paper_id: int) -> dict | None:
    database_path = database_path.resolve()
    if not database_path.is_file():
        raise FileNotFoundError(f"Database does not exist: {database_path}")
    uri = database_path.as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='domain_evidence_runs'"
        ).fetchone()
        if not table:
            return None
        row = connection.execute(
            "SELECT * FROM domain_evidence_runs WHERE paper_id=? "
            "ORDER BY created_at DESC,id DESC LIMIT 1",
            (paper_id,),
        ).fetchone()
    return dict(row) if row else None
