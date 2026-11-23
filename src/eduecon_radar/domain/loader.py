from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from eduecon_radar.domain.models import (
    CandidateFile,
    DomainConfigBundle,
    ExclusionFile,
    JournalPriorFile,
    MetadataFile,
    OntologyFile,
    ProfileFile,
    ScoringFile,
    SubjectCodeFile,
)


class DomainConfigError(RuntimeError):
    """Raised when a domain configuration bundle is missing or inconsistent."""


ModelT = TypeVar("ModelT", bound=BaseModel)


class DomainConfigLoader:
    FILES = {
        "profile": ("profile.yaml", ProfileFile),
        "ontology": ("ontology.yaml", OntologyFile),
        "exclusions": ("exclusion_contexts.yaml", ExclusionFile),
        "subject_codes": ("subject_codes.yaml", SubjectCodeFile),
        "journal_priors": ("journal_priors.yaml", JournalPriorFile),
        "scoring": ("scoring.yaml", ScoringFile),
        "candidates": ("candidates.yaml", CandidateFile),
        "metadata": ("metadata.yaml", MetadataFile),
    }

    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir.resolve()

    def load(self) -> DomainConfigBundle:
        loaded: dict[str, BaseModel] = {}
        for key, (filename, model) in self.FILES.items():
            loaded[key] = self._load_file(filename, model)

        profile = loaded["profile"].profile
        ontology = loaded["ontology"].ontology
        exclusions = loaded["exclusions"].exclusion_contexts
        subject_codes = loaded["subject_codes"].subject_codes
        journal_priors = loaded["journal_priors"].journal_priors
        scoring = loaded["scoring"].scoring
        candidates = loaded["candidates"].ontology_candidates
        metadata = loaded["metadata"].metadata
        objects = (
            ontology,
            exclusions,
            subject_codes,
            journal_priors,
            scoring,
            candidates,
            metadata,
        )
        mismatches = [
            item.__class__.__name__ for item in objects if item.profile_id != profile.profile_id
        ]
        if mismatches:
            raise DomainConfigError("Domain profile_id mismatch in: " + ", ".join(mismatches))
        expected_versions = {
            "ontology": (profile.ontology_version, ontology.version),
            "exclusion_contexts": (
                profile.exclusion_context_version,
                exclusions.version,
            ),
            "subject_codes": (profile.subject_code_version, subject_codes.version),
            "journal_priors": (profile.journal_prior_version, journal_priors.version),
            "scoring": (profile.scoring_version, scoring.scoring_version),
            "prompt": (profile.prompt_version, metadata.prompt_version),
        }
        invalid_versions = [
            name for name, (expected, actual) in expected_versions.items() if expected != actual
        ]
        if invalid_versions:
            raise DomainConfigError(
                "Domain configuration version mismatch: " + ", ".join(invalid_versions)
            )
        self._validate_ontology_hierarchy(ontology.concepts)
        invalid_enabled = [
            item.concept_id
            for item in ontology.concepts
            if item.enabled and item.classification != "high_precision_anchor"
        ]
        if invalid_enabled:
            raise DomainConfigError(
                "Only high_precision_anchor concepts may be enabled: " + ", ".join(invalid_enabled)
            )
        payload = {
            "profile": profile.model_dump(mode="json"),
            "ontology": ontology.model_dump(mode="json"),
            "exclusions": exclusions.model_dump(mode="json"),
            "subject_codes": subject_codes.model_dump(mode="json"),
            "journal_priors": journal_priors.model_dump(mode="json"),
            "scoring": scoring.model_dump(mode="json"),
            "candidates": candidates.model_dump(mode="json"),
            "metadata": metadata.model_dump(mode="json"),
        }
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest().upper()
        return DomainConfigBundle(**payload, config_sha256=digest)

    def _load_file(self, filename: str, model: type[ModelT]) -> ModelT:
        path = self.config_dir / filename
        if not path.is_file():
            raise DomainConfigError(f"Domain configuration file is missing: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            return model.model_validate(raw)
        except (OSError, yaml.YAMLError, ValidationError, TypeError) as exc:
            raise DomainConfigError(f"Invalid domain configuration {path.name}: {exc}") from exc

    @staticmethod
    def _validate_ontology_hierarchy(concepts: tuple) -> None:
        by_id = {concept.concept_id: concept for concept in concepts}
        if len(by_id) != len(concepts):
            raise DomainConfigError("Ontology concept_id values must be unique")
        for concept in concepts:
            if concept.parent_concept and concept.parent_concept not in by_id:
                raise DomainConfigError(
                    f"Unknown parent concept {concept.parent_concept!r} for {concept.concept_id}"
                )
            missing_children = set(concept.child_concepts) - set(by_id)
            if missing_children:
                raise DomainConfigError(
                    f"Unknown child concepts for {concept.concept_id}: {sorted(missing_children)}"
                )
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(concept_id: str) -> None:
            if concept_id in visiting:
                raise DomainConfigError("Ontology hierarchy contains a cycle")
            if concept_id in visited:
                return
            visiting.add(concept_id)
            parent = by_id[concept_id].parent_concept
            if parent:
                visit(parent)
            visiting.remove(concept_id)
            visited.add(concept_id)

        for concept_id in by_id:
            visit(concept_id)
