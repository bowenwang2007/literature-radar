from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode, urlparse

from eduecon_radar.services.doi import canonicalize_doi

REPEC_HANDLE_RE = re.compile(r"^RePEc:([A-Za-z0-9_-]+):([A-Za-z0-9_-]+):(.+)$", re.IGNORECASE)
OPENALEX_WORK_RE = re.compile(r"^W\d+$", re.IGNORECASE)
PUBLISHER_BLOCKED_HOSTS = {
    "sciencedirect.com",
    "www.sciencedirect.com",
    "elsevier.com",
    "www.elsevier.com",
    "linkinghub.elsevier.com",
}


@dataclass(frozen=True)
class PaperAccess:
    """Separate the authoritative DOI from one honest auxiliary access route."""

    doi_url: str | None
    preferred_url: str | None = None
    preferred_label: str | None = None
    preferred_kind: str | None = None


def build_doi_url(doi: str | None) -> str | None:
    normalized = canonicalize_doi(doi)
    if not normalized:
        return None
    return f"https://doi.org/{quote(normalized, safe='/')}"


def build_crossref_url(doi: str | None) -> str | None:
    normalized = canonicalize_doi(doi)
    if not normalized:
        return None
    return f"https://api.crossref.org/works/{quote(normalized, safe='')}"


def build_google_scholar_url(title: str | None) -> str | None:
    if not title or not title.strip():
        return None
    return f"https://scholar.google.com/scholar?{urlencode({'q': title})}"


def build_openalex_work_url(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip().rstrip("/")
    work_id = candidate.rsplit("/", 1)[-1]
    if not OPENALEX_WORK_RE.fullmatch(work_id):
        return None
    return f"https://openalex.org/{work_id.upper()}"


def build_repec_landing_url(handle: str | None) -> str | None:
    """Turn a verified RePEc article handle into its public IDEAS landing page."""
    if not handle:
        return None
    match = REPEC_HANDLE_RE.fullmatch(handle.strip())
    if not match:
        return None
    archive, series, remainder = match.groups()
    slug = remainder.replace(":", "")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", slug):
        return None
    return (
        f"https://ideas.repec.org/a/{archive.casefold()}/{series.casefold()}/{slug.casefold()}.html"
    )


def _find_repec_handle(value: Any) -> str | None:
    if isinstance(value, str):
        return value if REPEC_HANDLE_RE.fullmatch(value.strip()) else None
    if isinstance(value, dict):
        for key in ("source_record_id", "raw_payload_reference"):
            found = _find_repec_handle(value.get(key))
            if found:
                return found
        for nested in value.values():
            found = _find_repec_handle(nested)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found = _find_repec_handle(nested)
            if found:
                return found
    return None


def _safe_public_url(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip()
    parsed = urlparse(candidate)
    host = (parsed.hostname or "").casefold()
    if parsed.scheme not in {"http", "https"} or not host:
        return None
    if host in {"doi.org", "dx.doi.org"} or host in PUBLISHER_BLOCKED_HOSTS:
        return None
    return candidate


def resolve_paper_access(
    *,
    doi: str | None,
    open_access_url: str | None = None,
    source_url: str | None = None,
    source_metadata: dict[str, Any] | None = None,
    openalex_id: str | None = None,
) -> PaperAccess:
    """Choose one concise access route without relabeling it as a DOI.

    Publisher URLs known to fail in the current environment are deliberately
    excluded.  DOI resolution remains available independently as the canonical
    identifier and final fallback.
    """
    doi_url = build_doi_url(doi)
    public_oa = _safe_public_url(open_access_url)
    if public_oa:
        return PaperAccess(doi_url, public_oa, "开放版本", "open_access")

    repec_url = build_repec_landing_url(_find_repec_handle(source_metadata or {}))
    if repec_url:
        return PaperAccess(doi_url, repec_url, "RePEc 页面", "repec")

    openalex_url = build_openalex_work_url(openalex_id)
    if not openalex_url and source_url:
        openalex_url = build_openalex_work_url(source_url)
    if openalex_url:
        return PaperAccess(doi_url, openalex_url, "OpenAlex 记录", "openalex")

    public_source = _safe_public_url(source_url)
    if public_source:
        host = (urlparse(public_source).hostname or "").casefold()
        if host == "ideas.repec.org" or host.endswith(".ideas.repec.org"):
            return PaperAccess(doi_url, public_source, "RePEc 页面", "repec")
    return PaperAccess(doi_url)


def screening_basis_message(
    *,
    has_screening: bool,
    model_name: str | None,
    has_abstract: bool,
) -> tuple[str, str]:
    if not has_screening:
        return "No screening assessment available.", "尚无筛选判断。"
    if (model_name or "").casefold() == "rules-only":
        return (
            "Decision produced by rules-only screening; no AI semantic assessment was performed.",
            "该判断由纯规则筛选产生，未执行AI语义判断。",
        )
    if has_abstract:
        return (
            "AI assessment based on available bibliographic metadata and the Crossref abstract. "
            "Full text was not verified.",
            "AI判断基于现有书目信息和Crossref摘要，未核验论文全文。",
        )
    return (
        "AI assessment based on the title and available bibliographic metadata. "
        "No abstract or full text was verified.",
        "AI判断基于题名和现有书目信息，未获得Crossref摘要，也未核验全文。",
    )
