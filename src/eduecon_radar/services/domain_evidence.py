from __future__ import annotations

import json
from dataclasses import dataclass

from eduecon_radar import __version__
from eduecon_radar.db import Database
from eduecon_radar.domain.evidence import (
    DomainEvidenceResult,
    EvidenceExtractor,
    PaperEvidenceInput,
)
from eduecon_radar.domain.missed_risk import (
    EvidenceReviewPriority,
    MissedRiskPrioritizer,
)
from eduecon_radar.domain.models import DomainConfigBundle
from eduecon_radar.domain.scoring import RuleEvidenceScorer, RuleScoringResult
from eduecon_radar.domain.storage import DomainEvidenceRepository, StoredEvidenceRun


@dataclass(frozen=True)
class DomainEvidenceAnalysis:
    evidence: DomainEvidenceResult
    scoring: RuleScoringResult
    priority: EvidenceReviewPriority
    storage: StoredEvidenceRun


class DomainEvidenceService:
    def __init__(self, database: Database, bundle: DomainConfigBundle) -> None:
        self.database = database
        self.bundle = bundle

    def analyze_paper(
        self,
        paper_id: int,
        *,
        workflow_batch_id: int | None = None,
        force: bool = False,
    ) -> DomainEvidenceAnalysis:
        paper = load_paper_evidence_input(self.database, paper_id)
        evidence = EvidenceExtractor(self.bundle).extract(paper)
        scoring = RuleEvidenceScorer(self.bundle.scoring).score(evidence)
        priority = MissedRiskPrioritizer(
            self.bundle.scoring.scoring_version,
            self.bundle.scoring.evidence_parameters,
        ).prioritize(evidence, scoring)
        storage = DomainEvidenceRepository(self.database).save(
            evidence=evidence,
            scoring=scoring,
            priority=priority,
            config_bundle_sha256=self.bundle.config_sha256,
            application_version=__version__,
            workflow_batch_id=workflow_batch_id,
            force=force,
        )
        return DomainEvidenceAnalysis(evidence, scoring, priority, storage)


def load_paper_evidence_input(database: Database, paper_id: int) -> PaperEvidenceInput:
    with database.connect() as connection:
        row = connection.execute(
            """SELECT p.id,p.doi,p.title,p.abstract,p.authors_json,j.name journal,
            COALESCE(p.issue_date,p.online_date) publication_date,p.keywords_json,
            p.jel_codes_json,p.source_metadata_json
            FROM papers p JOIN journals j ON j.id=p.journal_id WHERE p.id=?""",
            (paper_id,),
        ).fetchone()
    if not row:
        raise ValueError(f"Paper not found: {paper_id}")
    try:
        authors = tuple(
            item for item in json.loads(row["authors_json"] or "[]") if isinstance(item, str)
        )
    except (json.JSONDecodeError, TypeError):
        authors = ()
    return PaperEvidenceInput(
        paper_id=int(row["id"]),
        doi=row["doi"],
        title=row["title"],
        abstract=row["abstract"],
        authors=authors,
        journal=row["journal"],
        publication_date=row["publication_date"],
        keywords=row["keywords_json"],
        jel_codes=row["jel_codes_json"],
        source_metadata=row["source_metadata_json"],
    )
