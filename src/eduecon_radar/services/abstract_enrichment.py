"""Reliable, identity-safe multi-source abstract enrichment."""

from __future__ import annotations

import json
import logging
import random
import re
import unicodedata
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from enum import StrEnum
from pathlib import Path
from urllib.parse import quote

import httpx

from eduecon_radar.db import Database, utcnow
from eduecon_radar.services.abstract_normalization import (
    PARSER_VERSION,
    content_sha256,
    normalize_abstract,
    validate_abstract_quality,
)
from eduecon_radar.services.abstract_source_config import AbstractSourceConfig
from eduecon_radar.services.doi import canonicalize_doi
from eduecon_radar.services.metadata_provenance import (
    abstract_quality_flags,
    load_source_priorities,
    provenance_schema_ready,
    record_provenance,
)
from eduecon_radar.services.migrations import overnight_schema_ready
from eduecon_radar.services.resilient_http import (
    ResilientHttpClient,
    parse_datetime,
    parse_retry_after_seconds,
)

LOGGER = logging.getLogger(__name__)


class AbstractEnrichmentError(RuntimeError):
    error_category = "unknown"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        retry_after: int | float | str | None = None,
        failure_kind: str | None = None,
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.retry_after = retry_after
        self.failure_kind = failure_kind


class AbstractSourceUnavailable(AbstractEnrichmentError):
    error_category = "provider_unavailable"
    retryable = True


class AbstractRateLimited(AbstractEnrichmentError):
    error_category = "rate_limited"
    retryable = True


class AbstractAuthenticationError(AbstractEnrichmentError):
    error_category = "authentication"


class AbstractIdentityMismatch(AbstractEnrichmentError):
    error_category = "identity_mismatch"


class AbstractInvalidContent(AbstractEnrichmentError):
    error_category = "invalid_content"


class AbstractTemporaryFailure(AbstractEnrichmentError):
    error_category = "temporary_failure"
    retryable = True


class AbstractPermanentFailure(AbstractEnrichmentError):
    error_category = "permanent_rejection"


class AbstractNotFound(AbstractEnrichmentError):
    error_category = "not_found"


class AbstractWorkerInterrupted(AbstractEnrichmentError):
    error_category = "worker_interrupted"
    retryable = True


class CandidateMatchDecision(StrEnum):
    ACCEPTED = "accepted"
    REJECTED_LOW_CONFIDENCE = "rejected_low_confidence"
    REJECTED_CONFLICT = "rejected_conflict"
    AMBIGUOUS_REQUIRES_MANUAL_REVIEW = "ambiguous_requires_manual_review"


MATCH_RULE_VERSION = "abstract-identity-h2-v1"


@dataclass(frozen=True)
class CandidateMatchEvaluation:
    decision: CandidateMatchDecision
    confidence: float
    reason: str
    evidence: dict[str, object]


@dataclass(frozen=True)
class AbstractFetchContext:
    workflow_batch_id: int | None = None
    timeout: float = 20.0
    attempt_number: int = 1
    request_id: str = ""
    user_agent: str = "LiteratureRadar/1.0"
    allow_title_fallback: bool = True
    cache_policy: str = "default"
    shared_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AbstractCandidate:
    text: str
    source: str
    source_url: str | None = None
    source_record_id: str | None = None
    retrieved_at: str | None = None
    content_sha256: str | None = None
    language: str | None = "en"
    quality_status: str = "clean"
    parser_version: str = PARSER_VERSION
    identity_match_method: str | None = None
    identity_confidence: float | None = None
    raw_title: str | None = None
    raw_doi: str | None = None
    raw_year: int | None = None
    raw_authors: tuple[str, ...] = ()
    raw_journal: str | None = None

    def __post_init__(self) -> None:
        if self.retrieved_at is None:
            object.__setattr__(self, "retrieved_at", datetime.now(UTC).isoformat())
        if self.content_sha256 is None:
            object.__setattr__(self, "content_sha256", content_sha256(self.text))


@dataclass(frozen=True)
class EnrichmentPaperInput:
    paper_id: int
    doi: str | None
    title: str
    authors: list[str]
    journal: str
    year: int | None
    issn: str | None
    volume: str | None = None
    issue: str | None = None
    source_metadata: dict[str, object] = field(default_factory=dict)


# Backward-compatible name for callers and fixtures; implementation is authoritative.
normalize_doi = canonicalize_doi


def normalize_title(value: str | None) -> str:
    text = unicodedata.normalize("NFKC", value or "").casefold()
    return " ".join(re.findall(r"[\w]+", text, re.UNICODE))


def normalize_author_name(value: str | None) -> str:
    raw = unicodedata.normalize("NFKC", value or "").strip()
    if "," in raw:
        parts = normalize_title(raw.split(",", 1)[0]).split()
        return parts[-1] if parts else ""
    parts = normalize_title(raw).split()
    return parts[-1] if parts else ""


def _parse_year(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        year = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return year if 1000 <= year <= 9999 else None


def evaluate_candidate_match(
    paper: EnrichmentPaperInput,
    candidate: AbstractCandidate,
    *,
    title_fallback: bool,
    strict_threshold: float = 0.94,
) -> CandidateMatchEvaluation:
    """Return a deterministic identity decision without mutating paper metadata."""

    expected_doi = normalize_doi(paper.doi)
    actual_doi = normalize_doi(candidate.raw_doi)
    normalized_expected_title = normalize_title(paper.title)
    normalized_actual_title = normalize_title(candidate.raw_title)
    title_score = (
        SequenceMatcher(None, normalized_expected_title, normalized_actual_title).ratio()
        if normalized_expected_title and normalized_actual_title
        else 0.0
    )
    title_length_ratio = (
        min(len(normalized_expected_title), len(normalized_actual_title))
        / max(len(normalized_expected_title), len(normalized_actual_title))
        if normalized_expected_title and normalized_actual_title
        else 0.0
    )
    expected_authors = {normalize_author_name(item) for item in paper.authors}
    actual_authors = {normalize_author_name(item) for item in candidate.raw_authors}
    expected_authors.discard("")
    actual_authors.discard("")
    author_overlap = sorted(expected_authors & actual_authors)
    normalized_expected_journal = normalize_title(paper.journal)
    normalized_actual_journal = normalize_title(candidate.raw_journal)
    journal_score = (
        SequenceMatcher(
            None, normalized_expected_journal, normalized_actual_journal
        ).ratio()
        if normalized_expected_journal and normalized_actual_journal
        else 0.0
    )
    expected_year = _parse_year(paper.year)
    returned_year = _parse_year(candidate.raw_year)
    year_delta = (
        abs(expected_year - returned_year)
        if expected_year is not None and returned_year is not None
        else None
    )
    evidence: dict[str, object] = {
        "rule_version": MATCH_RULE_VERSION,
        "expected_doi": expected_doi,
        "returned_doi": actual_doi,
        "doi_match": bool(expected_doi and actual_doi and expected_doi == actual_doi),
        "title_score": round(title_score, 6),
        "title_length_ratio": round(title_length_ratio, 6),
        "author_overlap": author_overlap,
        "year_delta": year_delta,
        "journal_score": round(journal_score, 6),
    }
    if expected_doi and actual_doi and expected_doi != actual_doi:
        return CandidateMatchEvaluation(
            CandidateMatchDecision.REJECTED_CONFLICT,
            0.0,
            "candidate DOI conflicts with paper DOI",
            evidence,
        )
    if not title_fallback and expected_doi and actual_doi == expected_doi:
        return CandidateMatchEvaluation(
            CandidateMatchDecision.ACCEPTED,
            1.0,
            "exact DOI match",
            evidence,
        )
    if expected_doi and not actual_doi:
        return CandidateMatchEvaluation(
            CandidateMatchDecision.AMBIGUOUS_REQUIRES_MANUAL_REVIEW,
            title_score,
            "title candidate did not return a DOI",
            evidence,
        )
    if not candidate.raw_title or title_score < strict_threshold or title_length_ratio < 0.75:
        return CandidateMatchEvaluation(
            CandidateMatchDecision.REJECTED_LOW_CONFIDENCE,
            title_score,
            "strict title match failed",
            evidence,
        )
    if expected_year is None or returned_year is None:
        return CandidateMatchEvaluation(
            CandidateMatchDecision.AMBIGUOUS_REQUIRES_MANUAL_REVIEW,
            title_score,
            "publication year evidence missing",
            evidence,
        )
    if year_delta is not None and year_delta > 1:
        return CandidateMatchEvaluation(
            CandidateMatchDecision.REJECTED_CONFLICT,
            min(title_score, 0.5),
            "publication year conflicts",
            evidence,
        )
    if not expected_authors or not actual_authors:
        return CandidateMatchEvaluation(
            CandidateMatchDecision.AMBIGUOUS_REQUIRES_MANUAL_REVIEW,
            title_score,
            "author evidence missing",
            evidence,
        )
    if not author_overlap:
        return CandidateMatchEvaluation(
            CandidateMatchDecision.REJECTED_CONFLICT,
            min(title_score, 0.5),
            "author match failed",
            evidence,
        )
    if not normalized_expected_journal or not normalized_actual_journal:
        return CandidateMatchEvaluation(
            CandidateMatchDecision.AMBIGUOUS_REQUIRES_MANUAL_REVIEW,
            min(title_score, 0.8),
            "journal evidence missing",
            evidence,
        )
    if journal_score < 0.85:
        return CandidateMatchEvaluation(
            CandidateMatchDecision.REJECTED_CONFLICT,
            min(title_score, journal_score),
            "journal match failed",
            evidence,
        )
    return CandidateMatchEvaluation(
        CandidateMatchDecision.ACCEPTED,
        min(title_score, journal_score),
        "strict DOI/title/author/year/journal match",
        evidence,
    )


def validate_candidate_identity(
    paper: EnrichmentPaperInput,
    candidate: AbstractCandidate,
    *,
    strict_threshold: float = 0.94,
    require_title_corroboration: bool = False,
) -> AbstractCandidate:
    evaluation = evaluate_candidate_match(
        paper,
        candidate,
        title_fallback=require_title_corroboration,
        strict_threshold=strict_threshold,
    )
    if evaluation.decision is not CandidateMatchDecision.ACCEPTED:
        raise AbstractIdentityMismatch(evaluation.reason)
    return replace(
        candidate,
        identity_match_method=(
            "doi"
            if not require_title_corroboration
            else "doi_plus_strict_title_author_year_journal"
            if normalize_doi(paper.doi) and normalize_doi(candidate.raw_doi)
            else "strict_title_author_year_journal"
        ),
        identity_confidence=evaluation.confidence,
    )


class AbstractSourceAdapter(ABC):
    @property
    @abstractmethod
    def source_name(self) -> str: ...

    @abstractmethod
    def fetch(
        self, paper: EnrichmentPaperInput, context: AbstractFetchContext | None = None
    ) -> AbstractCandidate | None: ...

    def fetch_title_candidates(
        self, paper: EnrichmentPaperInput, context: AbstractFetchContext | None = None
    ) -> tuple[AbstractCandidate, ...]:
        return ()


class _JsonAdapter(AbstractSourceAdapter):
    def __init__(self, client: ResilientHttpClient | httpx.Client | None = None) -> None:
        self.http = (
            client if isinstance(client, ResilientHttpClient) else ResilientHttpClient(client)
        )

    def _json(self, url: str, **kwargs: object) -> dict:
        try:
            return self.http.get_json(url, **kwargs)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            retry_after = _retry_after_seconds(exc.response.headers.get("retry-after"))
            if status == 404:
                raise AbstractNotFound(
                    "source record not found", http_status=status
                ) from exc
            if exc.response.status_code == 429:
                raise AbstractRateLimited(
                    "source rate limited", http_status=status, retry_after=retry_after
                ) from exc
            if exc.response.status_code in {401, 403}:
                raise AbstractAuthenticationError(
                    "source authentication failed", http_status=status
                ) from exc
            if exc.response.status_code >= 500:
                raise AbstractTemporaryFailure(
                    "source server failure",
                    http_status=status,
                    retry_after=retry_after,
                    failure_kind="provider_error",
                ) from exc
            if exc.response.status_code in {400, 405, 409, 422}:
                raise AbstractPermanentFailure(
                    "source request is invalid",
                    http_status=status,
                    failure_kind="invalid_request",
                ) from exc
            raise AbstractPermanentFailure(
                "source request rejected",
                http_status=status,
                failure_kind="provider_rejection",
            ) from exc
        except httpx.TimeoutException as exc:
            raise AbstractTemporaryFailure(
                "source request timed out", failure_kind="timeout"
            ) from exc
        except httpx.TransportError as exc:
            raise AbstractTemporaryFailure(
                "source temporarily unavailable", failure_kind="connection"
            ) from exc
        except ValueError as exc:
            raise AbstractInvalidContent("source returned invalid JSON") from exc


class CrossrefEnrichmentAdapter(_JsonAdapter):
    source_name = "crossref"

    def fetch(self, paper: EnrichmentPaperInput, context: AbstractFetchContext | None = None):
        doi = normalize_doi(paper.doi)
        if not doi:
            return None
        url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
        # Crossref's single-work route returns the complete record and rejects
        # the list-route ``select`` parameter with HTTP 400.
        item = self._json(url).get("message", {})
        if not item:
            return None
        if context is not None:
            alternative_ids = item.get("alternative-id") or []
            if isinstance(alternative_ids, str):
                alternative_ids = [alternative_ids]
            context.shared_metadata["crossref_alternative_ids"] = [
                str(value) for value in alternative_ids if value
            ]
        return _candidate(item.get("abstract") or "", self.source_name, url, item)

    def fetch_title_candidates(self, paper, context=None):
        payload = self._json(
            "https://api.crossref.org/works",
            params={
                "query.title": paper.title,
                "rows": 5,
                "select": "DOI,title,abstract,author,published,container-title",
            },
        )
        items = payload.get("message", {}).get("items", [])
        return tuple(
            _candidate(item.get("abstract") or "", self.source_name, None, item)
            for item in items
        )


class OpenAlexEnrichmentAdapter(_JsonAdapter):
    source_name = "openalex"

    def __init__(self, api_key: str | None = None, client=None) -> None:
        super().__init__(client)
        self.api_key = api_key

    def fetch(self, paper: EnrichmentPaperInput, context: AbstractFetchContext | None = None):
        doi = normalize_doi(paper.doi)
        if not doi:
            return None
        params = {
            "filter": f"doi:{doi}",
            "select": (
                "id,doi,title,display_name,publication_year,authorships,"
                "primary_location,abstract_inverted_index"
            ),
        }
        if self.api_key:
            params["api_key"] = self.api_key
        items = self._json("https://api.openalex.org/works", params=params).get("results", [])
        if not items:
            return None
        item = items[0]
        return _candidate(
            item.get("abstract_inverted_index") or "",
            self.source_name,
            item.get("id"),
            item,
        )

    def fetch_title_candidates(self, paper, context=None):
        params: dict[str, object] = {
            "search": paper.title,
            "per-page": 5,
            "select": (
                "id,doi,title,display_name,publication_year,authorships,"
                "primary_location,abstract_inverted_index"
            ),
        }
        if self.api_key:
            params["api_key"] = self.api_key
        items = self._json("https://api.openalex.org/works", params=params).get("results", [])
        return tuple(
            _candidate(
                item.get("abstract_inverted_index") or "",
                self.source_name,
                item.get("id"),
                item,
            )
            for item in items
        )


class SemanticScholarEnrichmentAdapter(_JsonAdapter):
    source_name = "semantic_scholar"

    def __init__(self, api_key: str | None = None, client=None) -> None:
        super().__init__(client)
        self.api_key = api_key

    def fetch(self, paper: EnrichmentPaperInput, context: AbstractFetchContext | None = None):
        doi = normalize_doi(paper.doi)
        if not doi:
            return None
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/DOI:"
            + quote(doi, safe="")
            + "?fields=title,abstract,year,authors,externalIds"
        )
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        item = self._json(url, headers=headers)
        if not item:
            return None
        return _candidate(item.get("abstract") or "", self.source_name, url, item)

    def fetch_title_candidates(self, paper, context=None):
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        payload = self._json(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": paper.title,
                "limit": 5,
                "fields": "title,abstract,year,authors,externalIds,journal",
            },
            headers=headers,
        )
        return tuple(
            _candidate(item.get("abstract") or "", self.source_name, None, item)
            for item in payload.get("data", [])
        )


class DOAJEnrichmentAdapter(_JsonAdapter):
    source_name = "doaj"

    def fetch(self, paper: EnrichmentPaperInput, context: AbstractFetchContext | None = None):
        doi = normalize_doi(paper.doi)
        if not doi:
            return None
        encoded_doi = quote(doi, safe="")
        url = f"https://doaj.org/api/search/articles/doi:{encoded_doi}"
        results = self._json(url).get("results", [])
        if not results:
            return None
        item = results[0].get("bibjson", {})
        return _candidate(item.get("abstract") or "", self.source_name, url, item)

    def fetch_title_candidates(self, paper, context=None):
        url = "https://doaj.org/api/search/articles/title:" + quote(
            f'"{paper.title}"', safe=""
        )
        results = self._json(url, params={"pageSize": 5}).get("results", [])
        return tuple(
            _candidate(
                item.get("bibjson", {}).get("abstract") or "",
                self.source_name,
                url,
                item.get("bibjson", {}),
            )
            for item in results
            if item.get("bibjson")
        )


class OpenAIREEnrichmentAdapter(_JsonAdapter):
    source_name = "openaire"

    def fetch(self, paper: EnrichmentPaperInput, context: AbstractFetchContext | None = None):
        doi = normalize_doi(paper.doi)
        if not doi:
            return None
        payload = self._json(
            "https://api.openaire.eu/graph/v3/research-products",
            params={"pid": doi, "type": "publication", "pageSize": 5},
        )
        records = payload.get("results") or []
        if not records:
            return None
        record = records[0]
        returned_doi = next(
            (
                str(item.get("value"))
                for item in record.get("pids") or []
                if isinstance(item, dict)
                and str(item.get("scheme") or "").casefold() == "doi"
                and item.get("value")
            ),
            None,
        )
        descriptions = [
            str(value).strip()
            for value in record.get("descriptions") or []
            if str(value).strip()
        ]
        authors = [
            {"name": item.get("fullName")}
            for item in record.get("authors") or []
            if isinstance(item, dict) and item.get("fullName")
        ]
        publication_date = str(record.get("publicationDate") or "")
        item = {
            "DOI": returned_doi,
            "title": record.get("mainTitle"),
            "abstract": max(descriptions, key=len, default=""),
            "authors": authors,
            "year": publication_date[:4] if publication_date else None,
            "id": record.get("id"),
        }
        return _candidate(
            item.get("abstract") or "",
            self.source_name,
            f"https://api.openaire.eu/graph/v3/research-products/{record.get('id')}",
            item,
        )


class RePEcOAIAdapter(AbstractSourceAdapter):
    """Retrieve publisher-supplied RePEc metadata without scraping web pages.

    RePEc article handles for Elsevier journals contain the publisher's PII.  We
    only query OAI-PMH after an exact DOI lookup at Crossref supplies that PII,
    then corroborate the returned title and authors before the service accepts it.
    Unsupported journals cause no network traffic.
    """

    source_name = "repec"
    _DC = "{http://purl.org/dc/elements/1.1/}"
    _SERIES_BY_ISSN = {
        "0272-7757": ("eee", "ecoedu"),
        "1873-7382": ("eee", "ecoedu"),
    }

    def __init__(
        self,
        client: ResilientHttpClient | httpx.Client | None = None,
        *,
        crossref_client: ResilientHttpClient | httpx.Client | None = None,
        endpoint: str = "http://oai.repec.org/oai.php",
        series_by_issn: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        self.http = (
            client if isinstance(client, ResilientHttpClient) else ResilientHttpClient(client)
        )
        self.crossref_http = (
            crossref_client
            if isinstance(crossref_client, ResilientHttpClient)
            else ResilientHttpClient(crossref_client)
        )
        self.endpoint = endpoint
        self.series_by_issn = series_by_issn or dict(self._SERIES_BY_ISSN)

    def fetch(
        self, paper: EnrichmentPaperInput, context: AbstractFetchContext | None = None
    ) -> AbstractCandidate | None:
        doi = normalize_doi(paper.doi)
        series = self._series_for(paper)
        if not doi or series is None or not paper.volume or not paper.year:
            return None
        alternative_ids = self._alternative_ids(paper, context)
        pii = next(
            (
                value.upper()
                for value in alternative_ids
                if re.fullmatch(r"S[0-9A-Z]{16}", value.strip().upper())
            ),
            None,
        )
        if pii is None:
            return None
        issue = (paper.issue or "c").strip().casefold()
        archive, series_name = series
        handle = (
            f"RePEc:{archive}:{series_name}:v:{paper.volume}:y:{paper.year}:"
            f"i:{issue}:s{pii.casefold().removeprefix('s')}"
        )
        response = self._get(
            self.endpoint,
            params={
                "verb": "GetRecord",
                "metadataPrefix": "oai_dc",
                "identifier": f"oai:{handle}",
            },
        )
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            raise AbstractInvalidContent("RePEc returned invalid OAI-PMH XML") from exc
        error = root.find("{http://www.openarchives.org/OAI/2.0/}error")
        if error is not None:
            if error.attrib.get("code") == "idDoesNotExist":
                return None
            raise AbstractPermanentFailure(
                f"RePEc OAI-PMH rejected the request: {error.attrib.get('code') or 'unknown'}"
            )
        title = self._first(root, "title")
        authors = tuple(self._all(root, "creator"))
        descriptions = self._all(root, "description")
        abstract = max(descriptions, key=len, default="")
        if not title and not abstract:
            return None
        candidate = AbstractCandidate(
            abstract,
            self.source_name,
            source_url=str(response.url),
            source_record_id=handle,
            identity_match_method="doi_via_crossref_pii_and_repec_handle",
            raw_title=title,
            raw_doi=doi,
            raw_year=paper.year,
            raw_authors=authors,
            raw_journal=paper.journal,
        )
        evaluation = evaluate_candidate_match(
            paper, candidate, title_fallback=True
        )
        if evaluation.decision is not CandidateMatchDecision.ACCEPTED:
            raise AbstractIdentityMismatch(
                f"RePEc identity corroboration failed: {evaluation.reason}"
            )
        return candidate

    def _alternative_ids(
        self, paper: EnrichmentPaperInput, context: AbstractFetchContext | None
    ) -> list[str]:
        shared_has_value = bool(
            context is not None
            and "crossref_alternative_ids" in context.shared_metadata
        )
        values: object = (
            context.shared_metadata.get("crossref_alternative_ids")
            if shared_has_value and context is not None
            else None
        )
        if values is None:
            values = paper.source_metadata.get("alternative-id")
        if values is None:
            doi = normalize_doi(paper.doi)
            if not doi:
                return []
            try:
                response = self.crossref_http.get_json(
                    f"https://api.crossref.org/works/{quote(doi, safe='')}"
                )
            except (httpx.HTTPError, ValueError) as exc:
                raise AbstractTemporaryFailure(
                    "Crossref identifier lookup for RePEc failed",
                    failure_kind="connection",
                ) from exc
            item = response.get("message") or {}
            values = item.get("alternative-id") or []
            if context is not None:
                context.shared_metadata["crossref_alternative_ids"] = values
        if isinstance(values, str):
            return [values]
        return [str(value) for value in values or [] if value]

    def _get(self, url: str, **kwargs: object) -> httpx.Response:
        try:
            return self.http.get(url, **kwargs)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:
                raise AbstractRateLimited(
                    "RePEc rate limited the request",
                    http_status=status,
                    retry_after=exc.response.headers.get("retry-after"),
                ) from exc
            if status >= 500:
                raise AbstractTemporaryFailure(
                    "RePEc server failure", http_status=status, failure_kind="provider_error"
                ) from exc
            raise AbstractPermanentFailure(
                "RePEc rejected the request", http_status=status
            ) from exc
        except httpx.TimeoutException as exc:
            raise AbstractTemporaryFailure(
                "RePEc request timed out", failure_kind="timeout"
            ) from exc
        except httpx.TransportError as exc:
            raise AbstractTemporaryFailure(
                "RePEc temporarily unavailable", failure_kind="connection"
            ) from exc

    @classmethod
    def _first(cls, root: ET.Element, field: str) -> str | None:
        values = cls._all(root, field)
        return values[0] if values else None

    @classmethod
    def _all(cls, root: ET.Element, field: str) -> list[str]:
        return [
            str(item.text).strip()
            for item in root.iter(f"{cls._DC}{field}")
            if item.text and str(item.text).strip()
        ]

    def _series_for(self, paper: EnrichmentPaperInput) -> tuple[str, str] | None:
        issn = str(paper.issn or "").strip()
        if issn in self.series_by_issn:
            return self.series_by_issn[issn]
        if normalize_title(paper.journal) == "economics of education review":
            return "eee", "ecoedu"
        return None


def load_repec_series_registry(path: Path) -> dict[str, tuple[str, str]]:
    """Load a small reviewed ISSN-to-RePEc-series registry."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("series")
    if not isinstance(entries, list):
        raise ValueError("RePEc series registry requires a series list")
    result: dict[str, tuple[str, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid RePEc series registry entry")
        archive = str(entry.get("archive") or "").strip().casefold()
        series = str(entry.get("series") or "").strip().casefold()
        issns = entry.get("issns") or []
        if not re.fullmatch(r"[a-z0-9]{3}", archive):
            raise ValueError("invalid RePEc archive code")
        if not re.fullmatch(r"[a-z0-9]{6}", series):
            raise ValueError("invalid RePEc series code")
        for issn in issns:
            normalized = str(issn).strip()
            if not re.fullmatch(r"\d{4}-[\dXx]{4}", normalized):
                raise ValueError(f"invalid ISSN in RePEc registry: {normalized}")
            result[normalized] = (archive, series)
    return result


@dataclass
class EnrichmentStats:
    total: int = 0
    enriched: int = 0
    failed: int = 0
    skipped_have_abstract: int = 0
    skipped_manual: int = 0
    skipped_not_abstractable: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    by_result_state: dict[str, int] = field(default_factory=dict)
    papers_without_abstract: int = 0
    papers_without_doi: int = 0
    provider_queries: int = 0
    cache_hits: int = 0


@dataclass(frozen=True)
class _QueryCacheEntry:
    attempt: dict[str, object]
    accepted: AbstractCandidate | None
    expires_at: datetime


class AbstractEnrichmentService:
    def __init__(
        self,
        database: Database,
        *,
        adapters: list[AbstractSourceAdapter] | None = None,
        crossref_adapter=None,
        openalex_adapter=None,
        semantic_scholar_adapter=None,
        source_configs: dict[str, AbstractSourceConfig] | None = None,
        policy_version: str = "6c.2",
        clock: Callable[[], datetime | str | None] | None = None,
        jitter: Callable[[], float] = random.random,
        negative_cache_ttl_seconds: float = 6 * 3600,
        temporary_cache_ttl_seconds: float = 60,
        success_cache_ttl_seconds: float = 30 * 24 * 3600,
        max_backoff_seconds: float = 3600,
        max_total_queries_per_paper: int = 16,
    ) -> None:
        self.database = database
        self._adapters = adapters or [
            item for item in (crossref_adapter, openalex_adapter, semantic_scholar_adapter) if item
        ]
        self.source_configs = source_configs or {}
        self._adapters.sort(
            key=lambda item: (
                self.source_configs[item.source_name].priority
                if item.source_name in self.source_configs
                else 999
            )
        )
        self.policy_version = policy_version
        self.clock = clock or (lambda: datetime.now(UTC))
        self.jitter = jitter
        self.negative_cache_ttl_seconds = float(negative_cache_ttl_seconds)
        self.temporary_cache_ttl_seconds = float(temporary_cache_ttl_seconds)
        self.success_cache_ttl_seconds = float(success_cache_ttl_seconds)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self.max_total_queries_per_paper = int(max_total_queries_per_paper)
        self._query_cache: dict[str, _QueryCacheEntry] = {}
        self._provider_cooldowns: dict[str, datetime] = {}
        self.run_key = (
            f"personal-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-"
            f"{int(random.random() * 10_000_000)}"
        )
        self._call_log_ready: bool | None = None

    def _now(self) -> datetime:
        return parse_datetime(self.clock()) or datetime.now(UTC)

    def enrich_papers(
        self,
        paper_ids: tuple[int, ...],
        *,
        workflow_batch_id: int | None = None,
        retry_failed_only: bool = False,
        force_refresh: bool = False,
    ) -> EnrichmentStats:
        stats = EnrichmentStats()
        if self._call_log_ready is None:
            try:
                self._call_log_ready = overnight_schema_ready(self.database)
            except Exception:
                self._call_log_ready = False
        for paper_id in dict.fromkeys(paper_ids):
            stats.total += 1
            paper = self._load_paper(paper_id)
            if not paper:
                stats.failed += 1
                continue
            if not normalize_doi(paper.doi):
                stats.papers_without_doi += 1
            with self.database.connect() as connection:
                row = connection.execute(
                    "SELECT abstract,source_metadata_json FROM papers WHERE id=?", (paper_id,)
                ).fetchone()
            meta = json.loads(row["source_metadata_json"] or "{}")
            if row["abstract"]:
                stats.skipped_have_abstract += 1
                provenance = meta.get("abstract_provenance")
                source = str(
                    meta.get("abstract_source")
                    or (
                        provenance.get("source")
                        if isinstance(provenance, dict)
                        else ""
                    )
                    or ""
                )
                if source == "manual":
                    stats.skipped_manual += 1
                    self._count_result(stats, "skipped_manual_provenance")
                else:
                    self._count_result(stats, "skipped_existing_abstract")
                continue
            prior_audit = meta.get("abstract_enrichment")
            prior_attempts = (
                list(prior_audit.get("attempts") or [])
                if isinstance(prior_audit, dict)
                else []
            )
            attempts: list[dict[str, object]] = []
            accepted: AbstractCandidate | None = None
            interrupted = False
            disabled_sources = sorted(
                name for name, config in self.source_configs.items() if not config.enabled
            )
            if is_non_research_front_matter(paper.title):
                audit = {
                    "status": "not_applicable",
                    "last_attempt_at": self._now().isoformat(),
                    "workflow_batch_id": workflow_batch_id,
                    "successful_source": None,
                    "attempts": [],
                    "candidate_hash": None,
                    "parser_version": PARSER_VERSION,
                    "config_version": self.policy_version,
                    "match_rule_version": MATCH_RULE_VERSION,
                    "failed_sources": [],
                    "retryable": False,
                    "strategy_exhausted": True,
                    "provider_queries": 0,
                    "cache_hits": 0,
                    "execution_state": "not_applicable",
                    "final_reason": "non_research_front_matter_no_abstract_expected",
                    "normalized_doi": normalize_doi(paper.doi),
                    "doi_status": "valid" if normalize_doi(paper.doi) else "missing_doi",
                    "disabled_sources": disabled_sources,
                }
                if isinstance(prior_audit, dict):
                    history = meta.get("abstract_enrichment_history")
                    history = list(history) if isinstance(history, list) else []
                    history.append(prior_audit)
                    meta["abstract_enrichment_history"] = history[-20:]
                meta["abstract_enrichment"] = audit
                self._save_metadata(paper_id, meta)
                stats.skipped_not_abstractable += 1
                self._count_result(stats, "not_applicable")
                continue
            enabled: list[
                tuple[AbstractSourceAdapter, AbstractSourceConfig | None, AbstractFetchContext]
            ] = []
            shared_metadata: dict[str, object] = {}
            for adapter in self._adapters:
                config = self.source_configs.get(adapter.source_name)
                if config is not None and not config.enabled:
                    continue
                enabled.append(
                    (
                        adapter,
                        config,
                        AbstractFetchContext(
                            workflow_batch_id=workflow_batch_id,
                            timeout=config.timeout_seconds if config else 20.0,
                            cache_policy=(
                                "refresh" if force_refresh else "default"
                            ),
                            allow_title_fallback=(
                                config.allow_title_fallback if config else True
                            ),
                            shared_metadata=shared_metadata,
                        ),
                    )
                )
            for query_mode in ("doi", "title"):
                if query_mode == "doi" and not normalize_doi(paper.doi):
                    continue
                for adapter, _config, context in enabled:
                    if query_mode == "title" and not context.allow_title_fallback:
                        continue
                    if len(attempts) >= self.max_total_queries_per_paper:
                        attempts.append(
                            self._budget_exhausted_attempt(
                                adapter.source_name, query_mode, paper
                            )
                        )
                        interrupted = True
                        break
                    attempt, accepted = self._attempt(
                        adapter,
                        paper,
                        context,
                        query_mode=query_mode,
                        prior_attempts=prior_attempts,
                    )
                    attempts.append(attempt)
                    self._count_result(stats, str(attempt["result_state"]))
                    stats.cache_hits += int(bool(attempt.get("cache_hit")))
                    stats.provider_queries += int(
                        not bool(attempt.get("cache_hit"))
                        and not bool(attempt.get("fused_skip"))
                    )
                    self._log_provider_call(paper_id, attempt)
                    if accepted:
                        break
                    if attempt["result_state"] == "interrupted":
                        interrupted = True
                        break
                if accepted or interrupted:
                    break
            retryable = any(bool(item.get("retryable")) for item in attempts)
            result_states = {str(item.get("result_state")) for item in attempts}
            strategy_exhausted = bool(attempts) and not retryable and not interrupted
            final_reason = self._final_reason(
                accepted=accepted,
                result_states=result_states,
                strategy_exhausted=strategy_exhausted,
            )
            audit = {
                "status": "completed" if accepted else "failed",
                "last_attempt_at": self._now().isoformat(),
                "workflow_batch_id": workflow_batch_id,
                "successful_source": accepted.source if accepted else None,
                "attempts": attempts,
                "candidate_hash": accepted.content_sha256 if accepted else None,
                "parser_version": PARSER_VERSION,
                "config_version": self.policy_version,
                "match_rule_version": MATCH_RULE_VERSION,
                "failed_sources": [
                    a["source"]
                    for a in attempts
                    if a["result_state"]
                    in {
                        "rate_limited",
                        "timeout",
                        "temporary_provider_error",
                        "permanent_provider_rejection",
                        "invalid_request",
                        "interrupted",
                    }
                ],
                "retryable": retryable,
                "strategy_exhausted": strategy_exhausted,
                "provider_queries": sum(
                    not bool(item.get("cache_hit")) for item in attempts
                ),
                "cache_hits": sum(bool(item.get("cache_hit")) for item in attempts),
                "execution_state": (
                    "completed"
                    if accepted
                    else "pending"
                    if final_reason == "provider_budget_exhausted_pending"
                    else "retryable_error"
                    if retryable
                    else "needs_manual_review"
                ),
                "final_reason": final_reason,
                "normalized_doi": normalize_doi(paper.doi),
                "doi_status": (
                    "valid"
                    if normalize_doi(paper.doi)
                    else "invalid_doi"
                    if paper.doi
                    else "missing_doi"
                ),
                "disabled_sources": disabled_sources,
            }
            if isinstance(prior_audit, dict):
                history = meta.get("abstract_enrichment_history")
                history = list(history) if isinstance(history, list) else []
                history.append(prior_audit)
                meta["abstract_enrichment_history"] = history[-20:]
            meta["abstract_enrichment"] = audit
            if accepted:
                self._save_abstract(paper_id, accepted, meta)
                stats.enriched += 1
                stats.by_source[accepted.source] = stats.by_source.get(accepted.source, 0) + 1
            else:
                self._save_metadata(paper_id, meta)
                stats.failed += 1
                stats.papers_without_abstract += 1
        return stats

    @staticmethod
    def _count_result(stats: EnrichmentStats, result_state: str) -> None:
        stats.by_result_state[result_state] = (
            stats.by_result_state.get(result_state, 0) + 1
        )

    @staticmethod
    def _final_reason(
        *,
        accepted: AbstractCandidate | None,
        result_states: set[str],
        strategy_exhausted: bool,
    ) -> str | None:
        if accepted:
            return None
        if "interrupted" in result_states:
            return "interrupted_recoverable"
        if result_states & {
            "rate_limited",
            "timeout",
            "temporary_provider_error",
            "provider_fused_skipped",
        }:
            return "retryable_provider_errors"
        if result_states & {"unauthorized", "forbidden"}:
            return "provider_access_denied"
        if "parse_error" in result_states:
            return "provider_response_parse_error"
        if "provider_attempt_budget_exhausted" in result_states:
            return "provider_budget_exhausted_pending"
        if "matched_without_abstract" in result_states:
            return "matched_records_without_abstract"
        if "low_confidence_match" in result_states:
            return "title_candidates_require_manual_review"
        if strategy_exhausted:
            return "strategy_exhausted_no_match"
        return "needs_manual_abstract"

    def _attempt(
        self,
        adapter: AbstractSourceAdapter,
        paper: EnrichmentPaperInput,
        context: AbstractFetchContext,
        *,
        query_mode: str,
        prior_attempts: list[dict[str, object]],
    ) -> tuple[dict[str, object], AbstractCandidate | None]:
        attempted_at = self._now()
        normalized_query = self._normalized_query(paper, query_mode)
        cache_key = (
            f"{adapter.source_name}|{query_mode}|{normalized_query}|{self.policy_version}"
        )
        cached = None
        prior = None
        if context.cache_policy != "refresh":
            cached = self._query_cache.get(cache_key)
            if cached and cached.expires_at > attempted_at:
                attempt = dict(cached.attempt)
                attempt["cache_hit"] = True
                return attempt, cached.accepted
            prior = self._cached_prior_attempt(
                prior_attempts, cache_key, attempted_at
            )
        if prior is not None:
            return prior, None
        attempt_count = (
            sum(
                item.get("source") == adapter.source_name
                and item.get("query_mode", "doi") == query_mode
                and item.get("policy_version") == self.policy_version
                for item in prior_attempts
            )
            + 1
        )
        attempt: dict[str, object] = {
            "source": adapter.source_name,
            "provider": adapter.source_name,
            "paper_id": paper.paper_id,
            "query_mode": query_mode,
            "query_identifier": normalized_query,
            "attempt_count": attempt_count,
            "last_attempt_at": attempted_at.isoformat(),
            "started_at": attempted_at.isoformat(),
            "finished_at": attempted_at.isoformat(),
            "policy_version": self.policy_version,
            "cache_key": cache_key,
            "normalized_query": normalized_query,
            "cache_hit": False,
            "fused_skip": False,
            "workflow_batch_id": context.workflow_batch_id,
            "worker_interrupted": False,
            "http_status": None,
            "retry_after": None,
            "next_retry_at": None,
            "cache_until": None,
            "abstract_text": None,
            "abstract_language": None,
            "matched_title": None,
            "matched_doi": None,
            "matched_authors": [],
            "matched_year": None,
            "confidence": None,
            "error_category": None,
            "error_message": None,
            "retryable": False,
        }
        cooldown_until = self._provider_cooldowns.get(adapter.source_name)
        if cooldown_until is not None and cooldown_until > attempted_at:
            attempt.update(
                status="rate_limited",
                result_state="provider_fused_skipped",
                error_category="provider_fused_skip",
                error_message=(
                    "Provider-level cooldown active from an earlier 429; "
                    "skipped for the remainder of this run"
                ),
                retryable=True,
                fused_skip=True,
                next_retry_at=cooldown_until.isoformat(),
                provider_cooldown_until=cooldown_until.isoformat(),
                rejected_candidates=[],
            )
            return attempt, None
        try:
            if query_mode == "doi":
                candidate = adapter.fetch(
                    paper, replace(context, attempt_number=attempt_count)
                )
                candidates = (candidate,) if candidate is not None else ()
            elif query_mode == "title":
                candidates = adapter.fetch_title_candidates(
                    paper, replace(context, attempt_number=attempt_count)
                )
            else:
                raise ValueError(f"unsupported abstract query mode: {query_mode}")
            candidates = _deduplicate_candidates(candidates)
            if not candidates:
                attempt.update(
                    status="no_match",
                    result_state="no_match",
                    error_category="not_found",
                    retryable=False,
                    rejected_candidates=[],
                )
                attempt["finished_at"] = self._now().isoformat()
                return self._cache_attempt(attempted_at, attempt, None)
            rejected: list[dict[str, object]] = []
            matched_without_abstract: list[dict[str, object]] = []
            for candidate in candidates:
                evaluation = evaluate_candidate_match(
                    paper, candidate, title_fallback=query_mode == "title"
                )
                if evaluation.decision is not CandidateMatchDecision.ACCEPTED:
                    rejected.append(
                        {
                            "source_record_id": candidate.source_record_id,
                            "returned_doi": candidate.raw_doi,
                            "returned_title": candidate.raw_title,
                            "returned_abstract": bool(candidate.text.strip()),
                            "reason": evaluation.reason,
                            "error_category": "identity_mismatch",
                            "match_decision": evaluation.decision.value,
                            "match_confidence": evaluation.confidence,
                            "match_evidence": evaluation.evidence,
                        }
                    )
                    continue
                checked = replace(
                    candidate,
                    identity_match_method=(
                        candidate.identity_match_method
                        or "doi"
                        if query_mode == "doi"
                        else "doi_plus_strict_title_author_year_journal"
                        if normalize_doi(paper.doi)
                        else "strict_title_author_year_journal"
                    ),
                    identity_confidence=evaluation.confidence,
                )
                match_record = {
                    "source_record_id": checked.source_record_id,
                    "returned_doi": checked.raw_doi,
                    "returned_title": checked.raw_title,
                    "match_decision": evaluation.decision.value,
                    "match_confidence": evaluation.confidence,
                    "match_evidence": evaluation.evidence,
                }
                if not checked.text.strip():
                    matched_without_abstract.append(match_record)
                    continue
                quality = validate_abstract_quality(checked.text, title=paper.title)
                if not quality.accepted:
                    rejected.append(
                        {
                            **match_record,
                            "reason": ",".join(quality.rejection_reasons),
                            "error_category": "invalid_content",
                        }
                    )
                    continue
                accepted = replace(
                    checked,
                    text=quality.normalized_text,
                    content_sha256=quality.content_sha256,
                    language=quality.language,
                    quality_status=quality.quality_status,
                )
                attempt.update(
                    status="success",
                    result_state="success_with_abstract",
                    error_category=None,
                    retryable=False,
                    returned_record=True,
                    returned_abstract=True,
                    matched=True,
                    match_decision=CandidateMatchDecision.ACCEPTED.value,
                    match_evidence=evaluation.evidence,
                    match_rule_version=MATCH_RULE_VERSION,
                    identity_result=accepted.identity_match_method,
                    identity_confidence=accepted.identity_confidence,
                    confidence=accepted.identity_confidence,
                    abstract_text=accepted.text,
                    matched_title=accepted.raw_title,
                    matched_doi=accepted.raw_doi,
                    matched_authors=list(accepted.raw_authors),
                    matched_year=accepted.raw_year,
                    raw_payload_reference=accepted.source_record_id,
                    quality_result=list(quality.rejection_reasons),
                    source_record_id=accepted.source_record_id,
                    rejected_candidates=rejected,
                    abstract_language=accepted.language,
                )
                attempt["finished_at"] = self._now().isoformat()
                return self._cache_attempt(attempted_at, attempt, accepted)
            if matched_without_abstract:
                attempt.update(
                    status="matched_without_abstract",
                    result_state="matched_without_abstract",
                    error_category=None,
                    retryable=False,
                    returned_record=True,
                    returned_abstract=False,
                    matched=True,
                    match_rule_version=MATCH_RULE_VERSION,
                    matched_records=matched_without_abstract,
                    rejected_candidates=rejected,
                )
                attempt["finished_at"] = self._now().isoformat()
                return self._cache_attempt(attempted_at, attempt, None)
            attempt.update(
                status="low_confidence_match",
                result_state="low_confidence_match",
                error_category="identity_or_quality_rejected",
                retryable=False,
                returned_record=True,
                returned_abstract=any(
                    bool(item.get("returned_abstract")) for item in rejected
                ),
                matched=False,
                match_rule_version=MATCH_RULE_VERSION,
                rejected_candidates=rejected,
            )
            attempt["finished_at"] = self._now().isoformat()
            return self._cache_attempt(attempted_at, attempt, None)
        except AbstractEnrichmentError as exc:
            if isinstance(exc, AbstractNotFound):
                attempt.update(
                    status="not_found",
                    result_state="not_found",
                    error_type=type(exc).__name__,
                    error_category=exc.error_category,
                    error_message=str(exc)[:500],
                    http_status=exc.http_status,
                    retryable=False,
                    rejected_candidates=[],
                )
                attempt["finished_at"] = self._now().isoformat()
                return self._cache_attempt(attempted_at, attempt, None)
            retry_after = parse_retry_after_seconds(exc.retry_after, now=attempted_at)
            retry_delay = (
                retry_after
                if retry_after is not None
                else min(
                    self.max_backoff_seconds,
                    float(2 ** min(max(0, attempt_count - 1), 10))
                    + float(self.jitter()),
                )
            )
            next_retry = (
                attempted_at + timedelta(seconds=retry_delay)
                if exc.retryable
                else None
            )
            result_state = self._error_result_state(exc)
            attempt.update(
                status=_canonical_provider_status(result_state),
                result_state=result_state,
                error_type=type(exc).__name__,
                error_category=exc.error_category,
                error_message=str(exc)[:500],
                http_status=exc.http_status,
                retry_after=retry_after,
                retryable=exc.retryable,
                next_retry_at=next_retry.isoformat() if next_retry else None,
                worker_interrupted=isinstance(exc, AbstractWorkerInterrupted),
            )
            attempt["finished_at"] = self._now().isoformat()
            if isinstance(exc, AbstractRateLimited) and attempt.get("next_retry_at"):
                next_retry = parse_datetime(attempt["next_retry_at"])
                if next_retry is not None:
                    # The contract requires a provider-level fuse for the rest of
                    # this run. The parsed Retry-After is kept for audit, but the
                    # cooldown must not expire mid-run (a 1s backoff is too short
                    # when network latency exceeds it).
                    cooldown_end = max(
                        next_retry,
                        attempted_at + timedelta(days=1),
                    )
                    self._provider_cooldowns[adapter.source_name] = cooldown_end
                    attempt["provider_cooldown_until"] = cooldown_end.isoformat()
            return self._cache_attempt(attempted_at, attempt, None)
        except Exception as exc:
            LOGGER.exception(
                "Unexpected abstract provider error for %s/%s",
                adapter.source_name,
                query_mode,
            )
            next_retry = attempted_at + timedelta(
                seconds=self.temporary_cache_ttl_seconds
            )
            attempt.update(
                status="temporary_error",
                result_state="temporary_provider_error",
                error_type=type(exc).__name__,
                error_category="unexpected_provider_error",
                error_message=str(exc)[:500],
                retryable=True,
                next_retry_at=next_retry.isoformat(),
            )
            attempt["finished_at"] = self._now().isoformat()
            return self._cache_attempt(attempted_at, attempt, None)

    def _cache_attempt(
        self,
        attempted_at: datetime,
        attempt: dict[str, object],
        accepted: AbstractCandidate | None,
    ) -> tuple[dict[str, object], AbstractCandidate | None]:
        if bool(attempt.get("fused_skip")):
            attempt["cache_until"] = None
            return attempt, accepted
        result_state = str(attempt["result_state"])
        expires_at: datetime | None
        if result_state == "success_with_abstract":
            expires_at = attempted_at + timedelta(seconds=self.success_cache_ttl_seconds)
        elif result_state in {
            "not_found",
            "no_match",
            "matched_without_abstract",
            "low_confidence_match",
            "unauthorized",
            "forbidden",
            "parse_error",
            "permanent_provider_rejection",
            "invalid_request",
        }:
            expires_at = attempted_at + timedelta(seconds=self.negative_cache_ttl_seconds)
        elif result_state in {"rate_limited", "timeout", "temporary_provider_error"}:
            value = attempt.get("next_retry_at")
            expires_at = parse_datetime(value) if isinstance(value, (datetime, str)) else None
            if expires_at is None:
                expires_at = attempted_at + timedelta(
                    seconds=self.temporary_cache_ttl_seconds
                )
        else:
            expires_at = None
        if expires_at is not None:
            attempt["cache_until"] = expires_at.isoformat()
            self._query_cache[str(attempt["cache_key"])] = _QueryCacheEntry(
                dict(attempt), accepted, expires_at
            )
        return attempt, accepted

    def _log_provider_call(self, paper_id: int, attempt: dict[str, object]) -> None:
        if not self._call_log_ready:
            return
        try:
            with self.database.connect() as connection:
                connection.execute(
                    """INSERT INTO provider_call_log
                    (run_key,paper_id,provider,query_mode,query_identifier,status,
                     abstract_text,abstract_language,matched_title,matched_doi,
                     matched_authors_json,matched_year,confidence,http_status,
                     retryable,retry_after,next_retry_at,error_category,error_message,
                     started_at,finished_at,cache_hit,fused_skip,created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        self.run_key,
                        paper_id,
                        attempt.get("provider") or attempt.get("source") or "",
                        attempt.get("query_mode") or "",
                        attempt.get("query_identifier"),
                        attempt.get("status") or attempt.get("result_state") or "",
                        attempt.get("abstract_text"),
                        attempt.get("abstract_language"),
                        attempt.get("matched_title"),
                        attempt.get("matched_doi"),
                        json.dumps(attempt.get("matched_authors") or [], ensure_ascii=False),
                        attempt.get("matched_year"),
                        attempt.get("confidence"),
                        attempt.get("http_status"),
                        int(bool(attempt.get("retryable"))),
                        attempt.get("retry_after"),
                        attempt.get("next_retry_at"),
                        attempt.get("error_category"),
                        attempt.get("error_message"),
                        attempt.get("started_at") or attempt.get("last_attempt_at"),
                        attempt.get("finished_at") or attempt.get("last_attempt_at"),
                        int(bool(attempt.get("cache_hit"))),
                        int(bool(attempt.get("fused_skip"))),
                        utcnow(),
                    ),
                )
        except Exception:
            LOGGER.exception("Failed to record provider call log for paper %s", paper_id)

    def _cached_prior_attempt(
        self,
        prior_attempts: list[dict[str, object]],
        cache_key: str,
        now: datetime,
    ) -> dict[str, object] | None:
        for prior in reversed(prior_attempts):
            if prior.get("cache_key") != cache_key:
                continue
            cache_until = prior.get("cache_until")
            if not cache_until or prior.get("result_state") == "success_with_abstract":
                continue
            try:
                expires_at = parse_datetime(
                    cache_until if isinstance(cache_until, (datetime, str)) else None
                )
            except (TypeError, ValueError):
                continue
            if expires_at is None:
                continue
            if expires_at <= now:
                return None
            cached = dict(prior)
            cached["cache_hit"] = True
            return cached
        return None

    def _normalized_query(self, paper: EnrichmentPaperInput, query_mode: str) -> str:
        if query_mode == "doi":
            return normalize_doi(paper.doi) or ""
        return "|".join(
            (
                normalize_title(paper.title),
                normalize_doi(paper.doi) or "",
                str(paper.year or ""),
                normalize_author_name(paper.authors[0] if paper.authors else ""),
                normalize_title(paper.journal),
            )
        )

    @staticmethod
    def _error_result_state(exc: AbstractEnrichmentError) -> str:
        if isinstance(exc, AbstractWorkerInterrupted):
            return "interrupted"
        if isinstance(exc, AbstractRateLimited):
            return "rate_limited"
        if isinstance(exc, AbstractTemporaryFailure):
            return "timeout" if exc.failure_kind == "timeout" else "temporary_provider_error"
        if isinstance(exc, AbstractAuthenticationError):
            return "unauthorized" if exc.http_status == 401 else "forbidden"
        if isinstance(exc, AbstractInvalidContent):
            return "parse_error"
        if isinstance(exc, AbstractPermanentFailure):
            return (
                "invalid_request"
                if exc.failure_kind == "invalid_request"
                else "permanent_provider_rejection"
            )
        return "temporary_provider_error" if exc.retryable else "permanent_provider_rejection"

    def _budget_exhausted_attempt(
        self, provider: str, query_mode: str, paper: EnrichmentPaperInput
    ) -> dict[str, object]:
        return {
            "source": provider,
            "provider": provider,
            "paper_id": None,
            "query_mode": query_mode,
            "query_identifier": None,
            "result_state": "provider_attempt_budget_exhausted",
            "status": "temporary_error",
            "error_category": "attempt_budget_exhausted",
            "retryable": True,
            "cache_hit": False,
            "fused_skip": False,
            "policy_version": self.policy_version,
            "normalized_query": self._normalized_query(paper, query_mode),
            "last_attempt_at": self._now().isoformat(),
            "next_retry_at": None,
            "worker_interrupted": False,
        }

    def _load_paper(self, paper_id: int) -> EnrichmentPaperInput | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT p.id,p.doi,p.title,p.authors_json,"
                "p.source_metadata_json,p.volume,p.issue,"
                "COALESCE(p.issue_date,p.online_date) pub_date,j.name journal,"
                "j.issn issn FROM papers p JOIN journals j ON j.id=p.journal_id "
                "WHERE p.id=?",
                (paper_id,),
            ).fetchone()
        if not row:
            return None
        metadata = json.loads(row["source_metadata_json"] or "{}")
        raw_doi = metadata.get("DOI") or metadata.get("doi")
        return EnrichmentPaperInput(
            int(row["id"]),
            row["doi"] or (str(raw_doi) if raw_doi else None),
            row["title"] or "",
            list(json.loads(row["authors_json"] or "[]")),
            row["journal"] or "",
            int(row["pub_date"][:4]) if row["pub_date"] else None,
            row["issn"],
            row["volume"],
            row["issue"],
            metadata,
        )

    def _save_metadata(self, paper_id: int, metadata: dict) -> None:
        with self.database.connect() as connection:
            connection.execute(
                "UPDATE papers SET source_metadata_json=?,updated_at=? WHERE id=?",
                (json.dumps(metadata, ensure_ascii=False), datetime.now(UTC).isoformat(), paper_id),
            )

    def _save_abstract(self, paper_id: int, candidate: AbstractCandidate, metadata: dict) -> None:
        metadata["abstract_source"] = candidate.source
        metadata["abstract_provenance"] = {
            key: value for key, value in asdict(candidate).items() if key != "text"
        }
        with self.database.connect() as connection:
            cursor = connection.execute(
                "UPDATE papers SET abstract=?,source_metadata_json=?,updated_at=? "
                "WHERE id=? AND (abstract IS NULL OR abstract='')",
                (
                    candidate.text,
                    json.dumps(metadata, ensure_ascii=False),
                    datetime.now(UTC).isoformat(),
                    paper_id,
                ),
            )
            if cursor.rowcount and provenance_schema_ready(connection):
                _, priorities = load_source_priorities()
                source = candidate.source.casefold().replace(" ", "_")
                record_provenance(
                    connection,
                    paper_id=paper_id,
                    field_name="abstract",
                    value=candidate.text,
                    source_name=source,
                    source_record_id=candidate.source_record_id,
                    source_priority=priorities.get(source, 50),
                    decision="supplemented",
                    reason="abstract_enrichment_filled_missing_value",
                    quality_flags=abstract_quality_flags(candidate.text),
                    is_current=True,
                    observed_at=candidate.retrieved_at,
                )


def _candidate(value, source: str, url: str | None, item: dict) -> AbstractCandidate:
    authors = item.get("author") or item.get("authors") or item.get("authorships") or []
    author_names = tuple(
        str(
            a.get("name")
            or a.get("author", {}).get("display_name")
            or " ".join(
                part
                for part in (str(a.get("given") or ""), str(a.get("family") or ""))
                if part
            )
        )
        for a in authors
        if isinstance(a, dict)
    )
    doi = item.get("DOI") or item.get("doi") or item.get("externalIds", {}).get("DOI")
    if not doi:
        doi = next(
            (
                identifier.get("id")
                for identifier in item.get("identifier", [])
                if isinstance(identifier, dict) and str(identifier.get("type", "")).lower() == "doi"
            ),
            None,
        )
    year = (
        item.get("year")
        or item.get("publication_year")
        or (
            item.get("published", {}).get("date-parts", [[None]])[0][0]
            if isinstance(item.get("published"), dict)
            else None
        )
    )
    title = item.get("title") or item.get("display_name")
    if isinstance(title, list):
        title = title[0] if title else None
    journal = item.get("container-title")
    journal_field = item.get("journal")
    if not journal:
        journal = (
            journal_field.get("name") or journal_field.get("title")
            if isinstance(journal_field, dict)
            else journal_field
        )
    if isinstance(journal, list):
        journal = journal[0] if journal else None
    if not journal:
        primary_location = item.get("primary_location")
        if isinstance(primary_location, dict):
            source_info = primary_location.get("source")
            if isinstance(source_info, dict):
                journal = source_info.get("display_name")
    return AbstractCandidate(
        normalize_abstract(value),
        source,
        url,
        str(item.get("id") or doi or ""),
        raw_title=title,
        raw_doi=doi,
        raw_year=_parse_year(year),
        raw_authors=author_names,
        raw_journal=str(journal) if journal else None,
    )


clean_html = normalize_abstract


def is_non_research_front_matter(title: str | None) -> bool:
    """Identify unambiguous journal furniture for which no abstract is expected."""

    normalized = normalize_title(title)
    return bool(
        re.fullmatch(
            r"(?:editorial board|editorial board and consulting editorial board|"
            r"front matter|back matter|table of contents|contents|subject index|index)",
            normalized,
        )
        or re.match(r"^(?:corrigendum|erratum|retraction notice)\b", normalized)
    )


def _rebuild_inverted_index(index: dict[str, list[int]]) -> str:
    return normalize_abstract(index)


def _deduplicate_candidates(
    candidates: tuple[AbstractCandidate, ...],
) -> tuple[AbstractCandidate, ...]:
    unique: list[AbstractCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        identity = (
            normalize_doi(candidate.raw_doi)
            or candidate.source_record_id
            or normalize_title(candidate.raw_title)
        )
        key = f"{candidate.source}:{identity}"
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)


def _retry_after_seconds(value: int | float | str | None) -> float | None:
    return parse_retry_after_seconds(value)


def _canonical_provider_status(result_state: str) -> str:
    return {
        "success_with_abstract": "success",
        "temporary_provider_error": "temporary_error",
        "permanent_provider_rejection": "permanent_error",
        "invalid_request": "permanent_error",
        "interrupted": "temporary_error",
    }.get(result_state, result_state)
