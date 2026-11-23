from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from eduecon_radar.domain.models import DomainConfigBundle, OntologyConcept


class EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PaperEvidenceInput(EvidenceModel):
    paper_id: int
    doi: str | None = None
    title: str
    abstract: str | None = None
    authors: tuple[str, ...] = ()
    journal: str
    publication_date: str | None = None
    keywords: Any = ()
    jel_codes: Any = ()
    source_metadata: Any = None


class EvidenceProvenance(EvidenceModel):
    source_field: Literal["title", "abstract", "keywords", "jel_codes", "journal"]
    normalized_source: bool = True
    source_identifier: str | None = None


class MatchedConceptEvidence(EvidenceModel):
    concept_id: str
    preferred_term: str
    source_field: Literal["title", "abstract", "keywords"]
    matched_text: str
    match_type: Literal[
        "exact_phrase",
        "synonym_phrase",
        "normalized_phrase",
        "token_sequence",
        "parent_concept",
        "child_concept",
    ]
    concept_family: str
    signal_strength: Literal["neutral", "weak", "moderate", "strong"]
    evidence_span: str
    parent_concept: str | None = None
    is_negated_or_contextually_limited: bool = False
    provenance: EvidenceProvenance
    raw_match_count: int = Field(ge=1)


class ExclusionContextEvidence(EvidenceModel):
    context_id: str
    source_field: Literal["title", "abstract", "keywords"]
    matched_phrase: str
    negative_strength: Literal["weak", "moderate", "strong"]
    exception_triggered: bool
    exception_reason: str | None = None
    provenance: EvidenceProvenance


class SubjectCodeSignal(EvidenceModel):
    code: str
    source: str
    signal_type: Literal["positive", "adjacent", "context", "unknown"]
    strength: Literal["neutral", "weak", "moderate", "strong"]
    matched: bool
    provenance: EvidenceProvenance


class JournalPriorSignal(EvidenceModel):
    journal_id: str | None = None
    matched_name: str
    status: Literal["candidate", "draft", "released", "unmatched"]
    verified: bool
    prior_tier: Literal["strong_positive", "moderate_positive", "neutral"] | None = None
    signal_type: Literal["verified_prior_signal", "candidate_prior_signal", "neutral"]
    numeric_eligible: bool = False
    provenance: EvidenceProvenance


class MetadataEvidence(EvidenceModel):
    title_confidence: float = Field(ge=0, le=1)
    abstract_confidence: float = Field(ge=0, le=1)
    keyword_confidence: float = Field(ge=0, le=1)
    subject_code_confidence: float = Field(ge=0, le=1)
    journal_confidence: float = Field(ge=0, le=1)
    overall_metadata_confidence: float = Field(ge=0, le=1)
    limitations: tuple[str, ...] = ()


class OpaqueTitleFeatures(EvidenceModel):
    opaque_title: bool
    opacity_level: Literal["low", "medium", "high"]
    opacity_reasons: tuple[str, ...] = ()


class DomainEvidenceResult(EvidenceModel):
    paper_id: int
    profile_id: str
    ontology_version: str
    subject_code_version: str
    journal_prior_version: str
    scoring_version: str
    title_available: bool
    abstract_available: bool
    keywords_available: bool
    subject_codes_available: bool
    matched_concepts: tuple[MatchedConceptEvidence, ...]
    matched_exclusion_contexts: tuple[ExclusionContextEvidence, ...]
    subject_code_signals: tuple[SubjectCodeSignal, ...]
    journal_prior_signal: JournalPriorSignal
    metadata_evidence: MetadataEvidence
    opaque_title_features: OpaqueTitleFeatures
    concept_family_coverage: tuple[str, ...]
    positive_evidence_count: int
    negative_evidence_count: int
    extraction_warnings: tuple[str, ...]


def normalize_evidence_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[\-_‐‑‒–—]+", " ", normalized)
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def phrase_occurrences(text: str, phrase: str) -> int:
    normalized_text = normalize_evidence_text(text)
    normalized_phrase = normalize_evidence_text(phrase)
    if not normalized_text or not normalized_phrase:
        return 0
    escaped_phrase = re.escape(normalized_phrase).replace(r"\ ", r"\s+")
    pattern = rf"(?<!\w){escaped_phrase}(?!\w)"
    return len(re.findall(pattern, normalized_text, flags=re.UNICODE))


class EvidenceExtractor:
    def __init__(self, bundle: DomainConfigBundle) -> None:
        self.bundle = bundle
        self._concepts = {item.concept_id: item for item in bundle.ontology.concepts}

    def extract(self, paper: PaperEvidenceInput) -> DomainEvidenceResult:
        warnings: list[str] = []
        keywords = self._string_list(paper.keywords, "keywords", warnings)
        codes = self._string_list(paper.jel_codes, "jel_codes", warnings)
        source_metadata = self._mapping(paper.source_metadata, warnings)
        codes.extend(self._source_codes(source_metadata, warnings))
        codes = list(dict.fromkeys(code.strip().upper() for code in codes if code.strip()))
        fields = {
            "title": paper.title or "",
            "abstract": paper.abstract or "",
            "keywords": " ; ".join(keywords),
        }
        matches = self._concept_matches(fields)
        matches = self._suppress_parent_matches(matches)
        exclusions = self._exclusion_matches(fields)
        subject_signals = self._subject_signals(codes)
        journal_signal = self._journal_signal(paper.journal)
        metadata = self._metadata(paper, keywords, codes)
        opacity = self._title_opacity(paper, matches)
        families = tuple(sorted({item.concept_family for item in matches}))
        return DomainEvidenceResult(
            paper_id=paper.paper_id,
            profile_id=self.bundle.profile.profile_id,
            ontology_version=self.bundle.ontology.version,
            subject_code_version=self.bundle.subject_codes.version,
            journal_prior_version=self.bundle.journal_priors.version,
            scoring_version=self.bundle.scoring.scoring_version,
            title_available=bool(paper.title.strip()),
            abstract_available=bool((paper.abstract or "").strip()),
            keywords_available=bool(keywords),
            subject_codes_available=bool(codes),
            matched_concepts=tuple(matches),
            matched_exclusion_contexts=tuple(exclusions),
            subject_code_signals=tuple(subject_signals),
            journal_prior_signal=journal_signal,
            metadata_evidence=metadata,
            opaque_title_features=opacity,
            concept_family_coverage=families,
            positive_evidence_count=len({item.concept_id for item in matches})
            + sum(item.matched and item.signal_type != "unknown" for item in subject_signals)
            + int(journal_signal.signal_type != "neutral"),
            negative_evidence_count=sum(not item.exception_triggered for item in exclusions),
            extraction_warnings=tuple(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _string_list(value: Any, field: str, warnings: list[str]) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                warnings.append(f"malformed_{field}")
                return []
            value = decoded
        if not isinstance(value, (list, tuple)):
            warnings.append(f"malformed_{field}")
            return []
        result: list[str] = []
        for item in value:
            if isinstance(item, str):
                result.append(item)
            else:
                warnings.append(f"malformed_{field}_item")
        return result

    @staticmethod
    def _mapping(value: Any, warnings: list[str]) -> dict[str, Any]:
        if value is None or value == "":
            return {}
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                warnings.append("malformed_source_metadata")
                return {}
        if not isinstance(value, dict):
            warnings.append("malformed_source_metadata")
            return {}
        return value

    def _source_codes(self, metadata: dict[str, Any], warnings: list[str]) -> list[str]:
        values: list[str] = []
        for key in ("jel_codes", "JEL", "subject_codes"):
            if key in metadata:
                values.extend(self._string_list(metadata[key], f"source_metadata_{key}", warnings))
        return values

    def _concept_matches(self, fields: dict[str, str]) -> list[MatchedConceptEvidence]:
        matches: list[MatchedConceptEvidence] = []
        for concept in self.bundle.ontology.concepts:
            if not concept.enabled:
                continue
            terms = [(concept.preferred_term, "exact_phrase")]
            terms.extend((item, "synonym_phrase") for item in concept.synonyms)
            terms.extend((item, "normalized_phrase") for item in concept.phrases)
            for source_field, text in fields.items():
                term_hits: list[tuple[str, str, int]] = []
                for term, match_type in terms:
                    count = phrase_occurrences(text, term)
                    if count:
                        term_hits.append((term, match_type, count))
                if not term_hits:
                    continue
                # One concept/source contribution is retained; the most specific phrase wins.
                term, match_type, count = max(
                    term_hits, key=lambda hit: len(normalize_evidence_text(hit[0]).split())
                )
                matches.append(
                    MatchedConceptEvidence(
                        concept_id=concept.concept_id,
                        preferred_term=concept.preferred_term,
                        source_field=source_field,
                        matched_text=term,
                        match_type=match_type,
                        concept_family=self._concept_family(concept),
                        signal_strength=concept.signal_strength,
                        evidence_span=term,
                        parent_concept=concept.parent_concept,
                        provenance=EvidenceProvenance(source_field=source_field),
                        raw_match_count=count,
                    )
                )
        return matches

    def _concept_family(self, concept: OntologyConcept) -> str:
        current = concept
        while current.parent_concept and current.parent_concept in self._concepts:
            parent = self._concepts[current.parent_concept]
            if not parent.parent_concept:
                return current.concept_id
            current = parent
        return current.concept_id

    def _suppress_parent_matches(
        self, matches: list[MatchedConceptEvidence]
    ) -> list[MatchedConceptEvidence]:
        matched_ids = {item.concept_id for item in matches}
        ancestors: set[str] = set()
        for concept_id in matched_ids:
            parent_id = self._concepts[concept_id].parent_concept
            while parent_id:
                ancestors.add(parent_id)
                parent = self._concepts.get(parent_id)
                parent_id = parent.parent_concept if parent else None
        return [item for item in matches if item.concept_id not in ancestors]

    def _exclusion_matches(self, fields: dict[str, str]) -> list[ExclusionContextEvidence]:
        combined = " ".join(fields.values())
        matches: list[ExclusionContextEvidence] = []
        for context in self.bundle.exclusions.contexts:
            if not context.enabled:
                continue
            exception = next(
                (item for item in context.exceptions if phrase_occurrences(combined, item)), None
            )
            phrases = (context.preferred_term, *context.phrases)
            for source_field, text in fields.items():
                phrase = next((item for item in phrases if phrase_occurrences(text, item)), None)
                if phrase:
                    matches.append(
                        ExclusionContextEvidence(
                            context_id=context.context_id,
                            source_field=source_field,
                            matched_phrase=phrase,
                            negative_strength=context.signal_strength,
                            exception_triggered=exception is not None,
                            exception_reason=exception,
                            provenance=EvidenceProvenance(source_field=source_field),
                        )
                    )
                    break
        return matches

    def _subject_signals(self, codes: list[str]) -> list[SubjectCodeSignal]:
        configured = {
            normalize_evidence_text(item.code).upper(): item
            for item in self.bundle.subject_codes.codes
        }
        results: list[SubjectCodeSignal] = []
        for code in codes:
            normalized = normalize_evidence_text(code).upper()
            item = configured.get(normalized)
            results.append(
                SubjectCodeSignal(
                    code=code,
                    source=item.source if item else "unknown",
                    signal_type=item.signal_type if item and item.enabled else "unknown",
                    strength=item.strength if item and item.enabled else "neutral",
                    matched=bool(item and item.enabled),
                    provenance=EvidenceProvenance(source_field="jel_codes", source_identifier=code),
                )
            )
        return results

    def _journal_signal(self, journal: str) -> JournalPriorSignal:
        normalized = normalize_evidence_text(journal)
        for item in self.bundle.journal_priors.journals:
            names = (item.canonical_name, *item.aliases)
            if normalized not in {normalize_evidence_text(name) for name in names}:
                continue
            verified = item.verified and item.status == "released"
            return JournalPriorSignal(
                journal_id=item.journal_id,
                matched_name=journal,
                status=item.status,
                verified=item.verified,
                prior_tier=item.prior_tier,
                signal_type=(
                    "verified_prior_signal"
                    if verified and item.prior_tier
                    else "candidate_prior_signal"
                ),
                numeric_eligible=bool(verified and item.prior_tier),
                provenance=EvidenceProvenance(source_field="journal"),
            )
        return JournalPriorSignal(
            matched_name=journal,
            status="unmatched",
            verified=False,
            signal_type="neutral",
            provenance=EvidenceProvenance(source_field="journal"),
        )

    @staticmethod
    def _metadata(
        paper: PaperEvidenceInput, keywords: list[str], codes: list[str]
    ) -> MetadataEvidence:
        values = {
            "title": float(bool(paper.title.strip())),
            "abstract": float(bool((paper.abstract or "").strip())),
            "keywords": float(bool(keywords)),
            "subject_codes": float(bool(codes)),
            "journal": float(bool(paper.journal.strip())),
        }
        limitations = tuple(
            f"missing_{name}" for name, available in values.items() if not available
        )
        return MetadataEvidence(
            title_confidence=values["title"],
            abstract_confidence=values["abstract"],
            keyword_confidence=values["keywords"],
            subject_code_confidence=values["subject_codes"],
            journal_confidence=values["journal"],
            overall_metadata_confidence=sum(values.values()) / len(values),
            limitations=limitations,
        )

    @staticmethod
    def _title_opacity(
        paper: PaperEvidenceInput, matches: list[MatchedConceptEvidence]
    ) -> OpaqueTitleFeatures:
        title = normalize_evidence_text(paper.title)
        words = title.split()
        title_matches = [item for item in matches if item.source_field == "title"]
        abstract_matches = [item for item in matches if item.source_field == "abstract"]
        reasons: list[str] = []
        if not title_matches:
            reasons.append("no_domain_evidence_in_title")
        if len(words) <= 4 and not title_matches:
            reasons.append("short_generic_title_without_domain_evidence")
        if paper.title.strip().endswith("?"):
            reasons.append("question_style_title")
        proper_like = sum(len(word) > 1 and word.isupper() for word in paper.title.split())
        if len(words) >= 4 and proper_like >= max(3, len(words) - 2) and not title_matches:
            reasons.append("proper_names_without_topic_description")
        if not paper.abstract and not title_matches:
            reasons.append("no_abstract_and_no_title_evidence")
        if abstract_matches and not title_matches:
            reasons.append("abstract_evidence_not_visible_in_title")
        actionable_reasons = [
            reason for reason in reasons if reason != "no_domain_evidence_in_title"
        ]
        if len(actionable_reasons) >= 2:
            level = "high"
        elif actionable_reasons:
            level = "medium"
        else:
            level = "low"
        return OpaqueTitleFeatures(
            opaque_title=level in {"medium", "high"},
            opacity_level=level,
            opacity_reasons=tuple(reasons),
        )
