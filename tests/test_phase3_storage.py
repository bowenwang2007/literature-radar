from __future__ import annotations

from pathlib import Path

from test_phase3_evidence import paper, phase3_bundle

from eduecon_radar.db import Database
from eduecon_radar.domain.evidence import EvidenceExtractor
from eduecon_radar.domain.missed_risk import MissedRiskPrioritizer
from eduecon_radar.domain.scoring import RuleEvidenceScorer
from eduecon_radar.domain.storage import (
    DomainEvidenceRepository,
    apply_domain_evidence_migration,
    domain_evidence_schema_ready,
    inspect_domain_evidence_read_only,
)
from eduecon_radar.schemas import JournalConfig, PaperMetadata
from eduecon_radar.services.archive import sha256_file
from eduecon_radar.services.domain_evidence import DomainEvidenceService

ROOT = Path(__file__).resolve().parents[1]


def evidence_database(tmp_path):
    database = Database(tmp_path / "evidence.db")
    database.init_schema()
    journal_id = database.upsert_journal(
        JournalConfig(name="Fixture Economics Journal", short_name="FEJ", issn="0000-0000")
    )
    paper_id, _ = database.upsert_paper(
        PaperMetadata(title="School Choice", abstract="Student achievement"), journal_id
    )
    apply_domain_evidence_migration(
        database, ROOT / "migrations" / "0003_add_domain_evidence_runs.sql"
    )
    return database, paper_id


def results(paper_id):
    bundle = phase3_bundle(numeric=True)
    evidence = EvidenceExtractor(bundle).extract(paper(paper_id=paper_id))
    scoring = RuleEvidenceScorer(bundle.scoring).score(evidence)
    priority = MissedRiskPrioritizer(
        bundle.scoring.scoring_version, bundle.scoring.evidence_parameters
    ).prioritize(evidence, scoring)
    return bundle, evidence, scoring, priority


def test_evidence_migration_and_repository_are_versioned_idempotent_and_immutable(tmp_path):
    database, paper_id = evidence_database(tmp_path)
    bundle, evidence, scoring, priority = results(paper_id)
    repository = DomainEvidenceRepository(database)
    first = repository.save(
        evidence=evidence,
        scoring=scoring,
        priority=priority,
        config_bundle_sha256=bundle.config_sha256,
        application_version="0.5.0-phase3",
    )
    reused = repository.save(
        evidence=evidence,
        scoring=scoring,
        priority=priority,
        config_bundle_sha256=bundle.config_sha256,
        application_version="0.5.0-phase3",
    )
    forced = repository.save(
        evidence=evidence,
        scoring=scoring,
        priority=priority,
        config_bundle_sha256=bundle.config_sha256,
        application_version="0.5.0-phase3",
        force=True,
    )
    new_bundle = repository.save(
        evidence=evidence,
        scoring=scoring,
        priority=priority,
        config_bundle_sha256="B" * 64,
        application_version="0.5.0-phase3",
    )
    assert domain_evidence_schema_ready(database)
    assert not first.reused
    assert reused == type(reused)(id=first.id, reused=True)
    assert forced.id != first.id
    assert new_bundle.id not in {first.id, forced.id}
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM domain_evidence_runs").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM screenings").fetchone()[0] == 0
    recovered = repository.latest_for_paper(paper_id)
    assert recovered["evidence_payload"]["evidence"]["paper_id"] == paper_id
    assert recovered["publishable"] == 0


def test_read_only_inspection_and_migration_do_not_touch_another_database(tmp_path):
    database, paper_id = evidence_database(tmp_path)
    bundle, evidence, scoring, priority = results(paper_id)
    DomainEvidenceRepository(database).save(
        evidence=evidence,
        scoring=scoring,
        priority=priority,
        config_bundle_sha256=bundle.config_sha256,
        application_version="0.5.0-phase3",
    )
    before = (sha256_file(database.path), database.path.stat().st_mtime_ns)
    row = inspect_domain_evidence_read_only(database.path, paper_id)
    assert row["paper_id"] == paper_id
    assert (sha256_file(database.path), database.path.stat().st_mtime_ns) == before

    untouched = Database(tmp_path / "untouched.db")
    untouched.init_schema()
    untouched_hash = sha256_file(untouched.path)
    assert not domain_evidence_schema_ready(untouched)
    assert sha256_file(untouched.path) == untouched_hash


def test_domain_evidence_service_reads_only_source_metadata_and_never_writes_screening(
    tmp_path,
):
    database, paper_id = evidence_database(tmp_path)
    analysis = DomainEvidenceService(database, phase3_bundle(numeric=True)).analyze_paper(
        paper_id
    )

    assert analysis.evidence.paper_id == paper_id
    assert analysis.storage.id > 0
    assert not analysis.scoring.publishable
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM screenings").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 1
