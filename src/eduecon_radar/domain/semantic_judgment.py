from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

from eduecon_radar.domain.semantic_models import (
    EscalationDecision,
    FastSemanticResult,
    SemanticEngineConfig,
    SemanticPaperInput,
    StrongSemanticReviewResult,
)
from eduecon_radar.domain.semantic_prompting import PromptBundle, SemanticPromptBuilder
from eduecon_radar.services.deepseek import DeepSeekClient

ResultT = TypeVar("ResultT", bound=BaseModel)


@dataclass(frozen=True)
class SemanticAIResponse:
    data: BaseModel
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float | None = None


class SemanticAIClient(Protocol):
    def request_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: type[ResultT],
    ) -> SemanticAIResponse: ...


class DeepSeekSemanticAdapter:
    """Explicit adapter only; construction and use are never automatic in Phase 4."""

    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client

    def request_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: type[ResultT],
    ) -> SemanticAIResponse:
        response = self.client.request_json(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
        )
        return SemanticAIResponse(
            data=response.data,
            model_name=response.model_name,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )


@dataclass(frozen=True)
class SemanticJudgment:
    result: FastSemanticResult | StrongSemanticReviewResult
    prompt: PromptBundle
    input_tokens: int
    output_tokens: int
    estimated_cost: float | None


class FastSemanticJudge:
    def __init__(
        self,
        *,
        config: SemanticEngineConfig,
        prompts: SemanticPromptBuilder,
        client: SemanticAIClient,
    ) -> None:
        self.config = config
        self.prompts = prompts
        self.client = client

    def judge(self, paper: SemanticPaperInput) -> SemanticJudgment:
        prompt = self.prompts.build_fast(paper)
        response = self.client.request_json(
            model=self.config.fast.model,
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            schema=FastSemanticResult,
        )
        result = FastSemanticResult.model_validate(response.data)
        _validate_identity(
            result.paper_id,
            response.model_name,
            paper,
            expected_model_name=(
                self.config.fast.model if self.config.status == "released" else None
            ),
        )
        result = result.model_copy(update={"model_name": response.model_name})
        _validate_no_abstract_safety(paper, result, self.config)
        return SemanticJudgment(
            result=result,
            prompt=prompt,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            estimated_cost=response.estimated_cost,
        )


class StrongSemanticReviewer:
    def __init__(
        self,
        *,
        config: SemanticEngineConfig,
        prompts: SemanticPromptBuilder,
        client: SemanticAIClient,
    ) -> None:
        self.config = config
        self.prompts = prompts
        self.client = client

    def review(
        self,
        paper: SemanticPaperInput,
        fast: FastSemanticResult,
        escalation: EscalationDecision,
    ) -> SemanticJudgment:
        prompt = self.prompts.build_strong(paper, fast, escalation)
        response = self.client.request_json(
            model=self.config.strong_review.model,
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            schema=StrongSemanticReviewResult,
        )
        result = StrongSemanticReviewResult.model_validate(response.data)
        _validate_identity(
            result.paper_id,
            response.model_name,
            paper,
            expected_model_name=(
                self.config.strong_review.model if self.config.status == "released" else None
            ),
        )
        result = result.model_copy(update={"model_name": response.model_name})
        if not paper.original_abstract and "abstract_missing" not in result.limitations:
            raise ValueError("Strong review without abstract must disclose abstract_missing")
        return SemanticJudgment(
            result=result,
            prompt=prompt,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            estimated_cost=response.estimated_cost,
        )


def _validate_identity(
    paper_id: int,
    model_name: str,
    paper: SemanticPaperInput,
    *,
    expected_model_name: str | None = None,
) -> None:
    if paper_id != paper.paper_id:
        raise ValueError("Semantic response paper_id does not match its input")
    if not model_name.strip():
        raise ValueError("Semantic response model_name cannot be empty")
    if expected_model_name is not None and model_name != expected_model_name:
        raise ValueError("Semantic response model_name does not match requested model")


def _validate_no_abstract_safety(
    paper: SemanticPaperInput,
    result: FastSemanticResult,
    config: SemanticEngineConfig,
) -> None:
    if paper.original_abstract:
        return
    if "abstract_missing" not in result.limitations:
        raise ValueError("No-abstract result must disclose abstract_missing")
    if result.abstract_evidence_sufficiency != "insufficient":
        raise ValueError("No-abstract result must mark abstract evidence insufficient")
    if result.information_sufficiency == "high":
        raise ValueError("No-abstract result cannot claim high information sufficiency")
    cap = config.escalation.no_abstract_confidence_cap
    if cap is not None and result.confidence > cap:
        raise ValueError("No-abstract result exceeds configured confidence cap")
