from pathlib import Path

from eduecon_radar.domain.loader import DomainConfigLoader
from eduecon_radar.domain.semantic_prompting import (
    SemanticPromptBuilder,
    load_semantic_engine_config,
)
from eduecon_radar.domain.versioning import DomainVersionRegistry

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "domains" / "education_economics"


def test_phase6d_component_versions_are_consistent_and_hash_is_stable():
    first = DomainConfigLoader(CONFIG).load()
    second = DomainConfigLoader(CONFIG).load()
    assert first.profile.version == "0.2.0-phase6d-candidate"
    assert first.profile.ontology_version == first.ontology.version
    assert first.profile.exclusion_context_version == first.exclusions.version
    assert first.profile.scoring_version == first.scoring.scoring_version
    assert first.profile.prompt_version == first.metadata.prompt_version
    assert first.metadata.bundle_version == "0.2.0-phase6d-candidate"
    assert first.config_sha256 == second.config_sha256 and len(first.config_sha256) == 64


def test_prompt_and_semantic_bundle_versions_are_synchronized():
    semantic = load_semantic_engine_config(CONFIG)
    prompts = SemanticPromptBuilder(CONFIG / "prompts")
    status = prompts.validate()
    assert semantic.version == "0.2.0-phase6d-candidate"
    assert semantic.fast.prompt_version == status["version"]
    assert semantic.strong_review.prompt_version == status["version"]
    assert semantic.composite.version == "phase6d-composite-v1"


def test_old_version_descriptor_loads_and_is_rollback_target():
    registry = DomainVersionRegistry(CONFIG / "config_versions.yaml")
    current = registry.load()
    old = registry.load("0.1.0-foundation")
    assert current.status == "candidate" and current.immutable
    assert old.status == "draft" and old.immutable
    assert registry.rollback_target() == old
    assert len(registry.versions) >= 2
