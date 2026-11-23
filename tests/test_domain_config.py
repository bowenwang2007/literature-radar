from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from eduecon_radar.domain import DomainConfigError, DomainConfigLoader
from eduecon_radar.domain.models import (
    ExclusionContext,
    JournalPrior,
    OntologyConcept,
    ScoringDefinition,
    SubjectCodeDefinition,
)

ROOT = Path(__file__).resolve().parents[1]


def test_domain_loader_returns_frozen_versioned_uncalibrated_bundle():
    bundle = DomainConfigLoader(
        ROOT / "config" / "domains" / "education_economics"
    ).load()

    assert bundle.profile.profile_id == "education_economics"
    assert bundle.profile.ontology_version == bundle.ontology.version
    assert bundle.profile.scoring_version == bundle.scoring.scoring_version
    assert bundle.scoring.calibration_status == "uncalibrated"
    assert bundle.subject_codes.missing_signal_behavior == "neutral"
    assert len(bundle.config_sha256) == 64
    with pytest.raises(ValidationError):
        bundle.ontology.status = "released"


def test_invalid_domain_file_has_clear_filename(tmp_path):
    source = ROOT / "config" / "domains" / "education_economics"
    target = tmp_path / "domain"
    shutil.copytree(source, target)
    (target / "profile.yaml").write_text("profile:\n  unexpected: true\n", encoding="utf-8")

    with pytest.raises(DomainConfigError, match="profile.yaml"):
        DomainConfigLoader(target).load()


def test_mapping_key_order_does_not_change_canonical_hash(tmp_path):
    source = ROOT / "config" / "domains" / "education_economics"
    target = tmp_path / "domain"
    shutil.copytree(source, target)
    original = DomainConfigLoader(target).load().config_sha256
    profile_path = target / "profile.yaml"
    payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    reordered = {"profile": dict(reversed(list(payload["profile"].items())))}
    profile_path.write_text(
        yaml.safe_dump(reordered, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    assert DomainConfigLoader(target).load().config_sha256 == original


def test_configuration_content_change_changes_bundle_hash(tmp_path):
    source = ROOT / "config" / "domains" / "education_economics"
    target = tmp_path / "domain"
    shutil.copytree(source, target)
    original = DomainConfigLoader(target).load().config_sha256
    scoring_path = target / "scoring.yaml"
    payload = yaml.safe_load(scoring_path.read_text(encoding="utf-8"))
    payload["scoring"]["notes"] = "changed fixture note"
    scoring_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    assert DomainConfigLoader(target).load().config_sha256 != original


def test_ontology_supports_hierarchy_synonyms_and_phrases():
    concept = OntologyConcept(
        concept_id="child",
        preferred_term="test",
        display_name_zh="测试",
        description="schema capability test",
        synonyms=("synonym",),
        phrases=("test phrase",),
        parent_concept="parent",
        version_added="test",
    )

    assert concept.parent_concept == "parent"
    assert concept.synonyms == ("synonym",)
    assert concept.phrases == ("test phrase",)


def test_exclusion_context_cannot_be_hard_exclude():
    with pytest.raises(ValidationError, match="hard excludes"):
        ExclusionContext(
            context_id="test",
            preferred_term="test",
            phrases=(),
            description="test",
            signal_strength="weak",
            hard_exclude=True,
            version_added="test",
        )


def test_missing_subject_code_signal_must_be_neutral():
    with pytest.raises(ValidationError):
        SubjectCodeDefinition(
            profile_id="education_economics",
            version="test",
            status="draft",
            missing_signal_behavior="negative",
            codes=(),
        )


def test_unverified_journal_cannot_be_released():
    with pytest.raises(ValidationError, match="cannot be released"):
        JournalPrior(
            journal_id="test",
            canonical_name="Test Journal",
            group="reference",
            status="released",
            verified=False,
            version_added="test",
        )


def test_uncalibrated_scoring_cannot_publish_thresholds_or_weights():
    with pytest.raises(ValidationError, match="thresholds"):
        ScoringDefinition.model_validate(
            {
                "profile_id": "education_economics",
                "scoring_version": "test",
                "status": "draft",
                "calibration_status": "uncalibrated",
                "gold_set_version": None,
                "components": [],
                "thresholds": {"include_min": 0.5},
                "notes": "test",
            }
        )


def test_official_phase3_configuration_remains_uncalibrated_and_non_numeric():
    bundle = DomainConfigLoader(
        ROOT / "config" / "domains" / "education_economics"
    ).load()

    assert bundle.scoring.calibration_status == "uncalibrated"
    assert bundle.scoring.evidence_parameters is None
    assert bundle.scoring.thresholds.include_min is None
    assert not bundle.scoring.publishable
