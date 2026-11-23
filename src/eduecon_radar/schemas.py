from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JournalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    short_name: str
    issn: str
    eissn: str | None = None
    rank: int | None = None
    rank_source: str | None = None
    priority: Literal["high", "medium", "low"] = "medium"
    enabled: bool = True


class PaperMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    doi: str | None = None
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    publisher: str | None = None
    article_type: str | None = None
    online_date: date | None = None
    issue_date: date | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    source_url: str | None = None
    open_access_url: str | None = None
    keywords: list[str] = Field(default_factory=list)
    jel_codes: list[str] = Field(default_factory=list)
    source_metadata: dict = Field(default_factory=dict)

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("title cannot be blank")
        return value


class RuleResult(BaseModel):
    score: float
    positive_hits: list[str] = Field(default_factory=list)
    risk_hits: list[str] = Field(default_factory=list)
    should_screen: bool


class AIScreeningResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["include", "borderline", "exclude"]
    confidence: float = Field(ge=0, le=1)
    reason: str
    topics: list[str] = Field(default_factory=list)
    education_level: str | None = None
    country_or_region: str | None = None
    method: str | None = None
    research_question: str | None = None
    main_result: str | None = None
    missing_information: list[str] = Field(default_factory=list)


class SummaryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chinese_title: str
    research_question: str | None = None
    data_and_sample: str | None = None
    identification_strategy: str | None = None
    main_result: str | None = None
    heterogeneity: str | None = None
    policy_implication: str | None = None
    chinese_summary: str
    missing_information: list[str] = Field(default_factory=list)


class TitleTranslationResult(BaseModel):
    """Strict structured result for a faithful title translation."""

    model_config = ConfigDict(extra="forbid")

    chinese_title: str

    @field_validator("chinese_title")
    @classmethod
    def title_translation_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("chinese_title must not be blank")
        return value


class AbstractTranslationResult(BaseModel):
    """Strict structured result for a faithful abstract translation."""

    model_config = ConfigDict(extra="forbid")

    chinese_abstract: str

    @field_validator("chinese_abstract")
    @classmethod
    def abstract_translation_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("chinese_abstract must not be blank")
        return value


class PersonalStrongDecision(BaseModel):
    """Strict Phase 7A single-pass education-economics judgment."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["relevant", "irrelevant", "uncertain", "needs_abstract"]
    education_centrality: Literal["central", "secondary", "none"]
    economic_relevance: Literal["substantive", "incidental", "none"]
    confidence: Literal["high", "medium", "low"]
    evidence: list[str] = Field(default_factory=list)
    reason: str

    @field_validator(
        "decision",
        "education_centrality",
        "economic_relevance",
        "confidence",
        mode="before",
    )
    @classmethod
    def normalize_personal_enum(cls, value: object) -> object:
        return value.strip().casefold() if isinstance(value, str) else value

    @field_validator("evidence")
    @classmethod
    def evidence_is_nonblank(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]

    @field_validator("reason")
    @classmethod
    def reason_is_nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("reason must not be blank")
        return value
