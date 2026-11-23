"""Deterministic education-economics screening.

Implements the overnight contract's Stage B model:

* one primary model call per paper with an English abstract;
* strict JSON schema/enum/evidence validation;
* dimension-consistency rules;
* a bounded conditional recheck (reviewer) call for uncertain, conflicting,
  weakly-confident, or evidence-invalid results;
* full separation between automatic and human decisions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from eduecon_radar.db import Database, utcnow
from eduecon_radar.services.deepseek import (
    DeepSeekClient,
    DeepSeekError,
    DeepSeekResponse,
    extract_first_json_object,
)
from eduecon_radar.services.migrations import (
    OvernightPipelineMigrationRequired,
    overnight_schema_ready,
)

LOGGER = logging.getLogger(__name__)

SCREENING_PROMPT_VERSION = "2026-07-31-screening-v1"
REVIEW_PROMPT_VERSION = "2026-07-31-screening-review-v1"
ONTOLOGY_VERSION = "education-economics-2026-07-31"
EVIDENCE_VALIDATION_VERSION = "2026-07-31-evidence-v1"
SCREENING_PARSER_VERSION = "2026-07-31-parser-v1"
#: The deterministic non-research rule records this version instead of a
#: profile screening prompt, because no model judgement was involved.
FRONT_MATTER_PROMPT_VERSION = "2026-08-02-front-matter-v1"

PRIMARY_DECISIONS = (
    "relevant",
    "irrelevant",
    "uncertain",
    "needs_abstract",
    "insufficient_evidence",
)
REVIEW_DECISIONS = ("confirm", "revise", "uncertain")
FINAL_DECISIONS = ("relevant", "irrelevant", "uncertain", "insufficient_evidence")
CENTRALITY = ("central", "secondary", "none")
ECONOMIC = ("substantive", "incidental", "none")
SUFFICIENCY = ("sufficient", "partial", "insufficient")
CONFIDENCE = ("high", "medium", "low")
PRIORITY = ("high", "medium", "low")
SUPPORTS = ("education", "economics", "final_decision")


class EvidenceItem(BaseModel):
    quote: str
    supports: Literal["education", "economics", "final_decision"]

    @field_validator("quote")
    @classmethod
    def quote_not_blank(cls, value: object) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("evidence quote must not be blank")
        return text

    @field_validator("supports")
    @classmethod
    def normalize_supports(cls, value: object) -> str:
        text = str(value or "").strip().casefold().replace("-", "_")
        if text not in SUPPORTS:
            raise ValueError("unsupported evidence.supports value")
        return text


class PrimaryScreeningOutput(BaseModel):
    decision: str
    education_centrality: str
    economic_relevance: str
    evidence_sufficiency: str
    confidence: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    reason: str = ""
    topic_tags: list[str] = Field(default_factory=list)
    review_priority: str = "medium"

    @field_validator(
        "decision",
        "education_centrality",
        "economic_relevance",
        "evidence_sufficiency",
        "confidence",
        "review_priority",
    )
    @classmethod
    def normalize_enums(cls, value: object) -> str:
        return str(value or "").strip().casefold().replace("-", "_")

    @field_validator("decision")
    @classmethod
    def decision_allowed(cls, value: str) -> str:
        if value not in PRIMARY_DECISIONS:
            raise ValueError("unsupported decision")
        return value

    @field_validator("education_centrality")
    @classmethod
    def centrality_allowed(cls, value: str) -> str:
        if value not in CENTRALITY:
            raise ValueError("unsupported centrality value")
        return value

    @field_validator("economic_relevance")
    @classmethod
    def economic_allowed(cls, value: str) -> str:
        if value not in ECONOMIC:
            raise ValueError("unsupported economic_relevance")
        return value

    @field_validator("evidence_sufficiency")
    @classmethod
    def sufficiency_allowed(cls, value: str) -> str:
        if value not in SUFFICIENCY:
            raise ValueError("unsupported evidence_sufficiency")
        return value

    @field_validator("confidence")
    @classmethod
    def confidence_allowed(cls, value: str) -> str:
        if value not in CONFIDENCE:
            raise ValueError("unsupported confidence")
        return value

    @field_validator("review_priority")
    @classmethod
    def priority_allowed(cls, value: str) -> str:
        if value not in PRIORITY:
            raise ValueError("unsupported review_priority")
        return value

    @field_validator("reason")
    @classmethod
    def reason_not_blank(cls, value: object) -> str:
        return str(value or "").strip()


class ReviewScreeningOutput(BaseModel):
    review_decision: str
    final_model_decision: str
    disagreement_reason: str = ""
    validated_evidence: list[EvidenceItem] = Field(default_factory=list)
    reason: str = ""

    @field_validator("review_decision", "final_model_decision")
    @classmethod
    def normalize_review_enums(cls, value: object) -> str:
        return str(value or "").strip().casefold().replace("-", "_")

    @field_validator("review_decision")
    @classmethod
    def review_decision_allowed(cls, value: str) -> str:
        if value not in REVIEW_DECISIONS:
            raise ValueError("unsupported review_decision")
        return value

    @field_validator("final_model_decision")
    @classmethod
    def final_decision_allowed(cls, value: str) -> str:
        if value not in FINAL_DECISIONS:
            raise ValueError("unsupported final_model_decision")
        return value


def parse_primary_payload(payload: object) -> PrimaryScreeningOutput:
    if not isinstance(payload, dict):
        raise ValueError("primary screening payload must be a JSON object")
    evidence = payload.get("evidence")
    if isinstance(evidence, str):
        evidence = [{"quote": evidence, "supports": "final_decision"}]
    elif isinstance(evidence, dict):
        evidence = [evidence]
    normalized = dict(payload)
    normalized["evidence"] = evidence
    return PrimaryScreeningOutput.model_validate(normalized)


def parse_review_payload(payload: object) -> ReviewScreeningOutput:
    if not isinstance(payload, dict):
        raise ValueError("review screening payload must be a JSON object")
    evidence = payload.get("validated_evidence")
    if isinstance(evidence, str):
        evidence = [{"quote": evidence, "supports": "final_decision"}]
    normalized = dict(payload)
    normalized["validated_evidence"] = evidence
    return ReviewScreeningOutput.model_validate(normalized)


def parse_primary_text(content: str) -> PrimaryScreeningOutput:
    cleaned = extract_first_json_object(content)
    cleaned = _strip_trailing_commas(cleaned)
    return parse_primary_payload(json.loads(cleaned))


def parse_review_text(content: str) -> ReviewScreeningOutput:
    cleaned = extract_first_json_object(content)
    cleaned = _strip_trailing_commas(cleaned)
    return parse_review_payload(json.loads(cleaned))


def _strip_trailing_commas(text: str) -> str:
    """Repair a common LLM quirk: trailing commas before } or ]."""
    return re.sub(r",(\s*[}\]])", r"\1", text)


def normalize_evidence_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "").casefold()
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def contains_cjk(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", value))


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...] = field(default_factory=tuple)


def validate_primary(
    output: PrimaryScreeningOutput,
    *,
    title: str,
    abstract: str,
) -> ValidationResult:
    """Deterministic schema, evidence, and dimension-consistency validation."""
    errors: list[str] = []
    has_abstract = bool((abstract or "").strip())
    source = normalize_evidence_text(f"{title or ''}\n{abstract or ''}")

    # Enum validation is enforced by the pydantic model; model-level rules below.
    if output.decision == "needs_abstract" and has_abstract:
        errors.append("needs_abstract_with_abstract_present")
    if output.decision == "insufficient_evidence" and output.confidence == "high":
        errors.append("insufficient_evidence_high_confidence")
    if output.decision == "relevant":
        if output.education_centrality != "central":
            errors.append("relevant_requires_central_subject")
        if output.economic_relevance != "substantive":
            errors.append("relevant_requires_substantive_economics")
        if output.evidence_sufficiency != "sufficient":
            errors.append("relevant_requires_sufficient_evidence")
        if not output.evidence:
            errors.append("relevant_requires_evidence")
    if output.education_centrality == "none" and output.decision == "relevant":
        errors.append("subject_none_cannot_be_relevant")
    if output.economic_relevance == "none" and output.decision == "relevant":
        errors.append("economics_none_cannot_be_relevant")

    for index, item in enumerate(output.evidence):
        quote = str(item.quote or "").strip()
        if contains_cjk(quote):
            errors.append(f"evidence[{index}].quote_is_chinese")
            continue
        normalized = normalize_evidence_text(quote)
        if len(normalized) < 12 or len(normalized.split()) < 2:
            errors.append(f"evidence[{index}].quote_too_short_or_keyword_only")
            continue
        if normalized not in source:
            errors.append(f"evidence[{index}].quote_not_found_in_source")
    return ValidationResult(not errors, tuple(errors))


def validate_review_evidence(
    evidence: list[EvidenceItem],
    *,
    title: str,
    abstract: str,
) -> ValidationResult:
    errors: list[str] = []
    source = normalize_evidence_text(f"{title or ''}\n{abstract or ''}")
    for index, item in enumerate(evidence):
        quote = str(item.quote or "").strip()
        if contains_cjk(quote):
            errors.append(f"evidence[{index}].quote_is_chinese")
            continue
        normalized = normalize_evidence_text(quote)
        if (
            len(normalized) < 12
            or len(normalized.split()) < 2
            or normalized not in source
        ):
            errors.append(f"evidence[{index}].quote_not_found_in_source")
    return ValidationResult(not errors, tuple(errors))


def needs_recheck(
    output: PrimaryScreeningOutput,
    validation: ValidationResult,
    *,
    has_abstract: bool,
    force: bool = False,
) -> bool:
    if force:
        return True
    if output.decision == "uncertain":
        return True
    if not validation.valid:
        return True
    if output.decision == "relevant" and (
        output.confidence != "high"
        or output.education_centrality != "central"
        or output.economic_relevance != "substantive"
        or output.evidence_sufficiency != "sufficient"
    ):
        return True
    return output.decision == "insufficient_evidence" and has_abstract


def resolve_final(
    primary: PrimaryScreeningOutput,
    review: ReviewScreeningOutput | None,
    *,
    validation: ValidationResult,
    recheck: bool,
    title: str,
    abstract: str,
) -> tuple[str, list[EvidenceItem], str, str]:
    """Return (final_decision, validated_evidence, review_decision, disagreement)."""
    if review is None:
        if not validation.valid:
            return "uncertain", [], "uncertain", "; ".join(validation.errors)
        return primary.decision, primary.evidence, "not_required", ""
    if review.review_decision == "confirm":
        evidence = review.validated_evidence or primary.evidence
        if primary.decision == "relevant":
            evidence_check = validate_review_evidence(
                evidence, title=title, abstract=abstract
            )
            if not evidence_check.valid:
                return (
                    "uncertain",
                    [],
                    "confirm",
                    "confirm_with_invalid_evidence: " + "; ".join(evidence_check.errors),
                )
        return primary.decision, evidence, "confirm", ""
    if review.review_decision == "revise":
        evidence = review.validated_evidence or primary.evidence
        evidence_check = validate_review_evidence(
            evidence, title=title, abstract=abstract
        )
        if review.final_model_decision == "relevant" and not evidence_check.valid:
            return "uncertain", [], "revise", "; ".join(evidence_check.errors)
        return review.final_model_decision, evidence, "revise", review.disagreement_reason
    return "uncertain", primary.evidence, "uncertain", review.disagreement_reason or review.reason


@dataclass(frozen=True)
class ScreeningOutcome:
    paper_id: int
    input_hash: str
    model_decision: str
    final_model_decision: str
    education_centrality: str | None
    economic_relevance: str | None
    evidence_sufficiency: str | None
    confidence: str | None
    evidence: tuple[dict[str, str], ...]
    reason: str | None
    topic_tags: tuple[str, ...]
    review_priority: str
    validation_status: str
    validation_errors: tuple[str, ...]
    review_decision: str
    disagreement_reason: str
    ai_calls: int
    input_tokens: int
    output_tokens: int
    reused: bool
    status: str
    screening_basis: str = "title_and_abstract"

    @property
    def classification(self) -> dict[str, object]:
        return {
            "decision": self.final_model_decision,
            "model_decision": self.model_decision,
            "screening_basis": self.screening_basis,
            "education_centrality": self.education_centrality or "",
            "economic_relevance": self.economic_relevance or "",
            "evidence_sufficiency": self.evidence_sufficiency or "",
            "confidence": self.confidence or "",
            "evidence": [item["quote"] for item in self.evidence],
            "reason": self.reason or "",
            "topic_tags": list(self.topic_tags),
            "review_priority": self.review_priority,
            "validation_status": self.validation_status,
            "validation_errors": list(self.validation_errors),
            "review_decision": self.review_decision,
            "disagreement_reason": self.disagreement_reason,
            "screening_ai_calls": self.ai_calls,
        }


#: Placeholders a profile prompt may use.  A template that does not mention one
#: is unaffected, and adding a field never requires editing existing prompts.
PROMPT_FIELDS = (
    "title",
    "abstract",
    "journal",
    "year",
    "primary",
    "jel_codes",
    "keywords",
    "subject_signals",
    "centrality_label",
    "centrality_definition",
    "economics_definition",
)


def render_prompt(template: str, values: Mapping[str, str]) -> str:
    """Substitute the placeholders a prompt actually uses.

    Deliberately *not* ``str.format``: system prompts contain literal JSON braces
    that ``format`` would try to interpret, and a missing value must never raise
    at call time.  Only known field names are replaced.
    """
    rendered = template
    for name in PROMPT_FIELDS:
        token = "{" + name + "}"
        if token in rendered:
            rendered = rendered.replace(token, str(values.get(name, "")))
    return rendered


def input_hash_for(title: str, abstract: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(title or "").encode("utf-8"))
    digest.update(b"\n")
    digest.update(str(abstract or "").encode("utf-8"))
    return digest.hexdigest()


class DeterministicScreeningService:
    """Persist one screening run per paper with deterministic validation."""

    def __init__(
        self,
        database: Database,
        *,
        client: DeepSeekClient | None = None,
        model: str = "",
        prompts_dir: Path | None = None,
        prompt_version: str = SCREENING_PROMPT_VERSION,
        prompt_context: Mapping[str, str] | None = None,
        primary_caller: Callable[[dict[str, object]], DeepSeekResponse] | None = None,
        review_caller: Callable[[dict[str, object]], DeepSeekResponse] | None = None,
        execution_guard: Callable[[], None] | None = None,
    ) -> None:
        if not overnight_schema_ready(database):
            raise OvernightPipelineMigrationRequired(
                "Overnight pipeline schema is required for deterministic screening"
            )
        self.database = database
        self.client = client
        self.model = model
        self.prompts_dir = prompts_dir
        #: Declared by the research profile; written to every assessment so a
        #: judgement can always be traced back to the prompt that produced it.
        self.prompt_version = prompt_version
        #: Profile-declared vocabulary the prompts may reference: what "central"
        #: means for this queue, and its JEL/subject signals.  A template that
        #: does not mention them is unaffected.
        self.prompt_context = dict(prompt_context or {})
        self.primary_caller = primary_caller
        self.review_caller = review_caller
        self.execution_guard = execution_guard

    def screen_paper(
        self,
        paper_id: int,
        *,
        task_id: int | None = None,
        force: bool = False,
    ) -> ScreeningOutcome:
        paper = self._load_paper(paper_id)
        if paper is None:
            raise ValueError(f"Paper {paper_id} not found")
        title = str(paper["title"] or "")
        abstract = str(paper["abstract"] or "").strip()
        digest = input_hash_for(title, abstract)
        if not abstract:
            return self._screen_title_only(
                paper_id=paper_id,
                task_id=task_id,
                paper=paper,
                title=title,
                digest=digest,
                force=force,
            )
        existing = self._latest_existing(paper_id, digest)
        if existing is not None and not force:
            return self._outcome_from_row(existing, reused=True)
        raw_responses: dict[str, object] = {}
        primary: PrimaryScreeningOutput | None = None
        primary_raw = ""
        # The provider reports token usage on every response; it was being
        # discarded, which left the run's audit row claiming a call that consumed
        # nothing.  Accumulate what the calls actually cost.
        input_tokens = 0
        output_tokens = 0
        try:
            self._guard()
            response = self._call_primary(paper)
            input_tokens += int(getattr(response, "input_tokens", 0) or 0)
            output_tokens += int(getattr(response, "output_tokens", 0) or 0)
            primary_raw = response.data.model_dump_json() if response.data else ""
            primary = response.data if isinstance(response.data, PrimaryScreeningOutput) else None
            if primary is None:
                primary = parse_primary_payload(response.data.model_dump(mode="json"))
        except Exception as exc:
            raw_text = getattr(exc, "raw_response", "") or ""
            if isinstance(exc, DeepSeekError) and exc.raw_response:
                raw_text = exc.raw_response
            parsed = self._try_parse_primary(raw_text)
            if parsed is None:
                return self._record(
                    paper_id=paper_id,
                    task_id=task_id,
                    input_hash=digest,
                    output=None,
                    validation=ValidationResult(False, ("parse_failed", _safe_failure(exc))),
                    recheck=False,
                    review=None,
                    raw_payload={"primary_raw": raw_text, "error": _safe_failure(exc)},
                    ai_calls=1,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reused=False,
                    status="needs_retry",
                    title=title,
                    abstract=abstract,
                )
            primary = parsed
            raw_responses["primary_recovered"] = True
        raw_responses["primary"] = primary.model_dump(mode="json")
        raw_responses["primary_raw"] = primary_raw
        validation = validate_primary(primary, title=title, abstract=abstract)
        recheck = needs_recheck(
            primary,
            validation,
            has_abstract=True,
            force=force,
        )
        review: ReviewScreeningOutput | None = None
        review_raw = ""
        if recheck:
            try:
                self._guard()
                review_response = self._call_review(paper, primary)
                input_tokens += int(getattr(review_response, "input_tokens", 0) or 0)
                output_tokens += int(getattr(review_response, "output_tokens", 0) or 0)
                review_raw = (
                    review_response.data.model_dump_json()
                    if review_response.data
                    else ""
                )
                review = (
                    review_response.data
                    if isinstance(review_response.data, ReviewScreeningOutput)
                    else parse_review_payload(
                        review_response.data.model_dump(mode="json")
                    )
                )
            except Exception as exc:
                raw_text = getattr(exc, "raw_response", "") or ""
                if isinstance(exc, DeepSeekError) and exc.raw_response:
                    raw_text = exc.raw_response
                try:
                    review = parse_review_text(raw_text)
                except Exception:
                    review = None
                    raw_responses["review_error"] = _safe_failure(exc)
            if review is not None:
                raw_responses["review"] = review.model_dump(mode="json")
            if review_raw:
                raw_responses["review_raw"] = review_raw
        final_decision, evidence, review_decision, disagreement = resolve_final(
            primary,
            review,
            validation=validation,
            recheck=recheck,
            title=title,
            abstract=abstract,
        )
        validation_status = "valid"
        if not validation.valid:
            validation_status = "needs_recheck"
        if recheck and review is None:
            validation_status = "recheck_failed"
            final_decision = "uncertain"
        if (
            review is not None
            and review.review_decision == "revise"
            and final_decision == "uncertain"
            and review.final_model_decision == "relevant"
        ):
            validation_status = "recheck_failed"
        if (
            review is not None
            and review.review_decision == "confirm"
            and final_decision == "uncertain"
            and primary.decision == "relevant"
        ):
            validation_status = "recheck_failed"
        status = "classified"
        if final_decision == "uncertain" and validation_status == "recheck_failed":
            status = "needs_retry"
        return self._record(
            paper_id=paper_id,
            task_id=task_id,
            input_hash=digest,
            output=primary,
            validation=validation,
            recheck=recheck,
            review=review,
            raw_payload=raw_responses,
            ai_calls=1 + int(recheck and review is not None),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reused=False,
            status=status,
            title=title,
            abstract=abstract,
            final_decision=final_decision,
            evidence=evidence,
            review_decision=review_decision,
            disagreement=disagreement,
            validation_status=validation_status,
        )

    def _call_primary(self, paper: dict[str, object]) -> DeepSeekResponse:
        if self.primary_caller is not None:
            return self.primary_caller(paper)
        if self.client is None or self.prompts_dir is None:
            raise DeepSeekError("Primary screening client is not configured")
        system = render_prompt(
            (self.prompts_dir / "screening_primary_system.txt").read_text(encoding="utf-8"),
            self._prompt_values(paper),
        )
        user = render_prompt(
            (self.prompts_dir / "screening_primary_user.txt").read_text(encoding="utf-8"),
            self._prompt_values(paper),
        )
        return self.client.request_json(
            model=self.model,
            system_prompt=system + "\nWrite the reason in one concise Chinese sentence. "
            "Keep evidence quotations verbatim and preserve the specified enum values.",
            user_prompt=user,
            schema=PrimaryScreeningOutput,
        )

    def _call_review(
        self, paper: dict[str, object], primary: PrimaryScreeningOutput
    ) -> DeepSeekResponse:
        if self.review_caller is not None:
            return self.review_caller({**paper, "primary": primary})
        if self.client is None or self.prompts_dir is None:
            raise DeepSeekError("Review screening client is not configured")
        values = {
            **self._prompt_values(paper),
            "primary": json.dumps(primary.model_dump(mode="json"), ensure_ascii=False),
        }
        system = render_prompt(
            (self.prompts_dir / "screening_review_system.txt").read_text(encoding="utf-8"),
            values,
        )
        user = render_prompt(
            (self.prompts_dir / "screening_review_user.txt").read_text(encoding="utf-8"),
            values,
        )
        return self.client.request_json(
            model=self.model,
            system_prompt=system + "\nWrite the reason in one concise Chinese sentence. "
            "Keep evidence quotations verbatim and preserve the specified enum values.",
            user_prompt=user,
            schema=ReviewScreeningOutput,
        )

    def _prompt_values(self, paper: dict[str, object]) -> dict[str, str]:
        """Everything a profile prompt may reference.

        The profile's own vocabulary (what "central" means here, its JEL and
        subject signals) travels with the service, and the paper's own JEL codes
        and keywords are passed through when the source provided them.
        """
        return {
            "title": str(paper.get("title") or ""),
            "abstract": str(paper.get("abstract") or ""),
            "journal": str(paper.get("journal") or ""),
            "year": str(paper.get("year") or ""),
            "jel_codes": ", ".join(str(item) for item in (paper.get("jel_codes") or [])),
            "keywords": ", ".join(str(item) for item in (paper.get("keywords") or [])),
            **self.prompt_context,
        }

    @staticmethod
    def _try_parse_primary(raw_text: str) -> PrimaryScreeningOutput | None:
        if not raw_text:
            return None
        try:
            return parse_primary_text(raw_text)
        except Exception:
            return None

    def _record(
        self,
        *,
        paper_id: int,
        task_id: int | None,
        input_hash: str,
        output: PrimaryScreeningOutput | None,
        validation: ValidationResult,
        recheck: bool,
        review: ReviewScreeningOutput | None,
        raw_payload: dict[str, object],
        ai_calls: int,
        input_tokens: int,
        output_tokens: int,
        reused: bool,
        status: str,
        title: str,
        abstract: str,
        final_decision: str | None = None,
        evidence: list[EvidenceItem] | None = None,
        review_decision: str = "not_required",
        disagreement: str = "",
        validation_status: str | None = None,
        screening_basis: str = "title_and_abstract",
    ) -> ScreeningOutcome:
        now = utcnow()
        model_decision = output.decision if output else "needs_retry"
        final = final_decision or model_decision
        evidence_items = [item.model_dump(mode="json") for item in (evidence or [])]
        if not evidence_items and output is not None:
            evidence_items = [item.model_dump(mode="json") for item in output.evidence]
        valid = validation.valid
        final_status = validation_status or ("valid" if valid else "invalid")
        if output is None:
            final_status = "parse_failed"
        with self.database.connect() as connection:
            connection.execute(
                """UPDATE paper_screenings SET stale=1 WHERE paper_id=? AND stale=0""",
                (paper_id,),
            )
            connection.execute(
                """INSERT INTO paper_screenings
                (paper_id,task_id,input_hash,model_decision,education_centrality,
                 economic_relevance,evidence_sufficiency,confidence,evidence_json,
                 reason,topic_tags_json,review_priority,model_name,model_identity,
                 prompt_version,prompt_hash,ontology_version,screened_at,
                 raw_response_json,parser_version,evidence_validation_version,
                 validation_status,validation_errors_json,stale,review_decision,
                 final_model_decision,disagreement_reason,validated_evidence_json,
                 review_reason,review_model_name,review_prompt_version,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,?,?,?,?,?,?,?)""",
                (
                    paper_id,
                    task_id,
                    input_hash,
                    model_decision,
                    output.education_centrality if output else None,
                    output.economic_relevance if output else None,
                    output.evidence_sufficiency if output else None,
                    output.confidence if output else None,
                    json.dumps(evidence_items, ensure_ascii=False),
                    (output.reason if output else None),
                    json.dumps(output.topic_tags if output else [], ensure_ascii=False),
                    output.review_priority if output else "medium",
                    self.model or None,
                    self.model or None,
                    self.prompt_version,
                    hashlib.sha256(
                        (self.prompts_dir / "screening_primary_system.txt")
                        .read_bytes()
                        if self.prompts_dir
                        else b""
                    ).hexdigest()
                    if self.prompts_dir
                    else None,
                    ONTOLOGY_VERSION,
                    now,
                    json.dumps(raw_payload, ensure_ascii=False),
                    SCREENING_PARSER_VERSION,
                    EVIDENCE_VALIDATION_VERSION,
                    final_status,
                    json.dumps(list(validation.errors), ensure_ascii=False),
                    review_decision,
                    final,
                    disagreement,
                    json.dumps(evidence_items, ensure_ascii=False),
                    review.reason if review else None,
                    self.model if review is not None else None,
                    REVIEW_PROMPT_VERSION if review is not None else None,
                    now,
                    now,
                ),
            )
        return ScreeningOutcome(
            paper_id=paper_id,
            input_hash=input_hash,
            model_decision=model_decision,
            final_model_decision=final,
            education_centrality=output.education_centrality if output else None,
            economic_relevance=output.economic_relevance if output else None,
            evidence_sufficiency=output.evidence_sufficiency if output else None,
            confidence=output.confidence if output else None,
            evidence=tuple(evidence_items),
            reason=(output.reason if output else None),
            topic_tags=tuple(output.topic_tags if output else []),
            review_priority=output.review_priority if output else "medium",
            validation_status=final_status,
            validation_errors=tuple(validation.errors),
            review_decision=review_decision,
            disagreement_reason=disagreement,
            ai_calls=ai_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reused=reused,
            status=status,
            screening_basis=screening_basis,
        )

    def _screen_title_only(
        self,
        *,
        paper_id: int,
        task_id: int | None,
        paper: dict[str, object],
        title: str,
        digest: str,
        force: bool,
    ) -> ScreeningOutcome:
        """Produce a preliminary judgment from title + bibliographic metadata.

        An empty abstract no longer means "no judgment": the model is asked
        once with the title and journal context, and the result is explicitly
        labelled ``title_only`` so a later full screening can replace it.
        """
        existing = self._latest_existing(paper_id, digest)
        if (
            existing is not None
            and not force
            and str(existing["model_decision"]) != "needs_abstract"
            and str(existing["validation_status"]) not in {"parse_failed", "recheck_failed"}
        ):
            return self._outcome_from_row(existing, reused=True)
        raw_responses: dict[str, object] = {
            "screening_basis": "title_only",
        }
        primary: PrimaryScreeningOutput | None = None
        primary_raw = ""
        input_tokens = 0
        output_tokens = 0
        try:
            self._guard()
            response = self._call_primary(paper)
            input_tokens += int(getattr(response, "input_tokens", 0) or 0)
            output_tokens += int(getattr(response, "output_tokens", 0) or 0)
            primary_raw = response.data.model_dump_json() if response.data else ""
            primary = (
                response.data
                if isinstance(response.data, PrimaryScreeningOutput)
                else None
            )
            if primary is None:
                primary = parse_primary_payload(response.data.model_dump(mode="json"))
        except Exception as exc:
            raw_text = getattr(exc, "raw_response", "") or ""
            if isinstance(exc, DeepSeekError) and exc.raw_response:
                raw_text = exc.raw_response
            parsed = self._try_parse_primary(raw_text)
            if parsed is None:
                return self._record(
                    paper_id=paper_id,
                    task_id=task_id,
                    input_hash=digest,
                    output=None,
                    validation=ValidationResult(
                        False, ("parse_failed", _safe_failure(exc))
                    ),
                    recheck=False,
                    review=None,
                    raw_payload={
                        "primary_raw": raw_text,
                        "error": _safe_failure(exc),
                        "screening_basis": "title_only",
                    },
                    ai_calls=1,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reused=False,
                    status="needs_retry",
                    title=title,
                    abstract="",
                    screening_basis="title_only",
                )
            primary = parsed
            raw_responses["primary_recovered"] = True
        raw_responses["primary"] = primary.model_dump(mode="json")
        raw_responses["primary_raw"] = primary_raw
        if primary.decision in {"needs_abstract", "insufficient_evidence"}:
            reason = "标题级初判：模型请求摘要确认，暂记不确定。"
            if primary.reason:
                reason += " " + primary.reason
            primary = primary.model_copy(
                update={
                    "decision": "uncertain",
                    "confidence": "low",
                    "evidence_sufficiency": "insufficient",
                    "evidence": [],
                    "reason": reason,
                }
            )
        validation = validate_primary(primary, title=title, abstract="")
        if not validation.valid:
            reason = "标题级初判（验证未通过）：" + "; ".join(validation.errors)
            if primary.reason:
                reason += " " + primary.reason
            primary = primary.model_copy(
                update={
                    "decision": "uncertain",
                    "confidence": "low",
                    "evidence_sufficiency": "insufficient",
                    "evidence": [],
                    "reason": reason,
                }
            )
            validation = ValidationResult(True, ())
        return self._record(
            paper_id=paper_id,
            task_id=task_id,
            input_hash=digest,
            output=primary,
            validation=validation,
            recheck=False,
            review=None,
            raw_payload=raw_responses,
            ai_calls=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reused=False,
            status="classified",
            title=title,
            abstract="",
            screening_basis="title_only",
        )

    def _latest_existing(self, paper_id: int, input_hash: str) -> dict[str, object] | None:
        # Reuse is scoped to *this profile's* prompt version.  Looking up the
        # module default instead let one profile's judgement be reused as
        # another profile's result for the same paper.
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM paper_screenings
                WHERE paper_id=? AND input_hash=? AND stale=0 AND prompt_version=?
                ORDER BY id DESC LIMIT 1""",
                (paper_id, input_hash, self.prompt_version),
            ).fetchall()
        return dict(rows[0]) if rows else None

    @staticmethod
    def _outcome_from_row(row: dict[str, object], *, reused: bool) -> ScreeningOutcome:
        status = "classified"
        if str(row["model_decision"]) == "needs_abstract":
            status = "needs_abstract"
        elif str(row["validation_status"]) in {"parse_failed", "recheck_failed"}:
            status = "needs_retry"
        basis = "title_and_abstract"
        raw = row.get("raw_response_json")
        if raw:
            try:
                payload = json.loads(str(raw))
                if isinstance(payload, dict):
                    basis = str(
                        payload.get("screening_basis") or "title_and_abstract"
                    )
            except (TypeError, ValueError):
                basis = "title_and_abstract"
        return ScreeningOutcome(
            paper_id=int(row["paper_id"]),
            input_hash=str(row["input_hash"]),
            model_decision=str(row["model_decision"]),
            final_model_decision=str(row["final_model_decision"] or row["model_decision"]),
            education_centrality=row["education_centrality"],
            economic_relevance=row["economic_relevance"],
            evidence_sufficiency=row["evidence_sufficiency"],
            confidence=row["confidence"],
            evidence=tuple(
                json.loads(row["validated_evidence_json"] or "[]")
                or json.loads(row["evidence_json"] or "[]")
            ),
            reason=row["reason"],
            topic_tags=tuple(json.loads(row["topic_tags_json"] or "[]")),
            review_priority=str(row["review_priority"] or "medium"),
            validation_status=str(row["validation_status"]),
            validation_errors=tuple(json.loads(row["validation_errors_json"] or "[]")),
            review_decision=str(row["review_decision"] or "not_required"),
            disagreement_reason=str(row["disagreement_reason"] or ""),
            ai_calls=0,
            input_tokens=0,
            output_tokens=0,
            reused=reused,
            status=status,
            screening_basis=basis,
        )

    def _load_paper(self, paper_id: int) -> dict[str, object] | None:
        with self.database.connect() as connection:
            row = connection.execute(
                """SELECT p.id,p.title,p.abstract,p.authors_json,j.name journal,
                COALESCE(p.issue_date,p.online_date) publication_date,
                p.jel_codes_json,p.keywords_json
                FROM papers p JOIN journals j ON j.id=p.journal_id WHERE p.id=?""",
                (paper_id,),
            ).fetchone()
        if not row:
            return None
        paper = dict(row)
        paper["year"] = str(paper.get("publication_date") or "")[:4] or None
        paper["authors"] = json.loads(paper.get("authors_json") or "[]")
        paper["jel_codes"] = json.loads(paper.pop("jel_codes_json", None) or "[]")
        paper["keywords"] = json.loads(paper.pop("keywords_json", None) or "[]")
        return paper

    def _guard(self) -> None:
        if self.execution_guard is not None:
            self.execution_guard()


def _safe_failure(exc: Exception) -> str:
    return str(exc).replace("Authorization", "[redacted]")[:500]
