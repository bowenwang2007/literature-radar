from __future__ import annotations

import html
import json
import re
import time
from datetime import date
from typing import Any
from urllib.parse import quote

import httpx

from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.doi import canonicalize_doi

TAG_RE = re.compile(r"<[^>]+>")
CROSSREF_SELECT_FIELDS = ",".join(
    (
        "DOI",
        "title",
        "abstract",
        "author",
        "publisher",
        "type",
        "published-online",
        "created",
        "published-print",
        "issued",
        "volume",
        "issue",
        "page",
        "URL",
        "subject",
        "container-title",
        "ISSN",
        "alternative-id",
    )
)


class CrossrefRequestError(RuntimeError):
    """Raised when Crossref rejects an otherwise completed API request."""


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code != 400:
        response.raise_for_status()
        return
    try:
        payload = response.json()
        detail = payload.get("message", payload) if isinstance(payload, dict) else payload
        rendered = json.dumps(detail, ensure_ascii=False)
    except (ValueError, TypeError):
        rendered = response.text.strip() or "Crossref did not provide an error body"
    raise CrossrefRequestError(f"Crossref API rejected the request with HTTP 400: {rendered[:500]}")


def _date_from_parts(value: dict[str, Any] | None) -> date | None:
    if not value:
        return None
    groups = value.get("date-parts") or [[]]
    parts = groups[0]
    if not parts:
        timestamp = value.get("date-time")
        try:
            return date.fromisoformat(str(timestamp)[:10]) if timestamp else None
        except ValueError:
            return None
    padded = [*parts, 1, 1]
    try:
        return date(int(padded[0]), int(padded[1]), int(padded[2]))
    except (TypeError, ValueError):
        return None


def _clean_abstract(value: str | None) -> str | None:
    if not value:
        return None
    return " ".join(html.unescape(TAG_RE.sub(" ", value)).split()) or None


def parse_crossref_item(item: dict[str, Any]) -> PaperMetadata:
    titles = item.get("title") or []
    title = titles[0] if titles else ""
    authors = []
    for author in item.get("author") or []:
        name = " ".join(part for part in [author.get("given"), author.get("family")] if part)
        if name:
            authors.append(name)
    return PaperMetadata(
        doi=item.get("DOI"),
        title=title,
        abstract=_clean_abstract(item.get("abstract")),
        authors=authors,
        publisher=item.get("publisher"),
        article_type=item.get("type"),
        online_date=_date_from_parts(item.get("published-online")),
        issue_date=_date_from_parts(item.get("published-print") or item.get("issued")),
        volume=item.get("volume"),
        issue=item.get("issue"),
        pages=item.get("page"),
        source_url=item.get("URL"),
        keywords=item.get("subject") or [],
        source_metadata=item,
    )


def canonical_paper_key(paper: PaperMetadata) -> str:
    doi = canonicalize_doi(paper.doi)
    if doi:
        return f"doi:{doi}"
    normalized_title = " ".join(paper.title.casefold().split())
    return f"title:{normalized_title}"


def _retrieval_activity_date(paper: PaperMetadata) -> date:
    metadata = paper.source_metadata
    return (
        _date_from_parts(metadata.get("published-online"))
        or _date_from_parts(metadata.get("created"))
        or paper.issue_date
        or date.min
    )


class CrossrefClient:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(30),
            headers={"User-Agent": "LiteratureRadar/1.0 (metadata research tool)"},
        )
        self.last_fetch_stats: dict[str, object] | None = None

    def fetch(
        self,
        issn: str,
        since: date,
        limit: int | None = None,
        *,
        until: date | None = None,
    ) -> list[PaperMetadata]:
        if until is not None and until < since:
            raise ValueError("Crossref end date cannot be earlier than start date")
        # Crossref's publication-date filter follows deposited publication
        # fields.  Some online-first records are created during the requested
        # window but already carry a future print/issued date, so a bounded
        # product sync also queries the creation-date index and unions both
        # result sets.  This is still the public Crossref API; no publisher
        # page is fetched.
        date_bases = ("pub", "created") if until is not None else ("pub",)
        merged: dict[str, PaperMetadata] = {}
        aggregate = {
            "requested_limit": limit,
            "pages_requested": 0,
            "provider_returned_count": 0,
            "raw_candidate_count": 0,
            "stopped_reason": "provider_exhausted",
            "error_summary": None,
            "date_bases": list(date_bases),
        }
        errors: list[str] = []
        last_error: Exception | None = None
        successful_queries = 0
        for date_basis in date_bases:
            try:
                batch, stats = self._fetch_date_basis(
                    issn,
                    since,
                    limit,
                    until=until,
                    date_basis=date_basis,
                )
            except (httpx.HTTPStatusError, httpx.TransportError, CrossrefRequestError) as exc:
                last_error = exc
                errors.append(f"{date_basis}: Crossref query failed")
                continue
            successful_queries += 1
            for field in (
                "pages_requested",
                "provider_returned_count",
                "raw_candidate_count",
            ):
                aggregate[field] = int(aggregate[field]) + int(stats[field])
            if stats.get("error_summary"):
                errors.append(f"{date_basis}: {stats['error_summary']}")
                aggregate["stopped_reason"] = str(
                    stats.get("stopped_reason") or "provider_error"
                )
            for paper in batch:
                key = canonical_paper_key(paper)
                existing = merged.get(key)
                if existing is None:
                    paper.source_metadata["eduecon_retrieval_date_bases"] = [date_basis]
                    merged[key] = paper
                else:
                    bases = existing.source_metadata.setdefault(
                        "eduecon_retrieval_date_bases", []
                    )
                    if date_basis not in bases:
                        bases.append(date_basis)

        if successful_queries == 0 and last_error is not None:
            raise last_error
        papers = sorted(merged.values(), key=_retrieval_activity_date, reverse=True)
        if limit is not None:
            papers = papers[:limit]
        if errors:
            if aggregate["stopped_reason"] == "provider_exhausted":
                aggregate["stopped_reason"] = "provider_error"
            aggregate["error_summary"] = "; ".join(errors)[:500]
        elif limit is not None and len(merged) >= limit:
            aggregate["stopped_reason"] = "requested_limit_reached"
        self.last_fetch_stats = aggregate
        return papers

    def _get_page(self, issn: str, params: dict) -> httpx.Response:
        from eduecon_radar.services.resilient_http import parse_retry_after_seconds

        url = f"https://api.crossref.org/journals/{quote(issn, safe='')}/works"
        for attempt in range(3):
            try:
                response = self.client.get(url, params=params)
                if response.status_code == 404:
                    response = self.client.get(
                        "https://api.crossref.org/works",
                        params={**params, "filter": f"issn:{issn},{params['filter']}"},
                    )
                if response.status_code not in {429, 500, 502, 503, 504} or attempt == 2:
                    _raise_for_status(response)
                    return response
                delay = parse_retry_after_seconds(response.headers.get("Retry-After"))
                if delay is not None and delay > 30:
                    _raise_for_status(response)
            except httpx.TransportError:
                if attempt == 2:
                    raise
                delay = None
            time.sleep(delay if delay is not None else 2 ** attempt)
        raise AssertionError("unreachable")

    def _fetch_date_basis(
        self,
        issn: str,
        since: date,
        limit: int | None,
        *,
        until: date | None,
        date_basis: str,
    ) -> tuple[list[PaperMetadata], dict[str, object]]:
        rows = min(limit or 100, 1000)
        cursor = "*"
        papers: list[PaperMetadata] = []
        pages_requested = 0
        provider_returned_count = 0
        stopped_reason = "provider_exhausted"
        error_summary: str | None = None
        filter_name = "pub-date" if date_basis == "pub" else "created-date"
        date_filter = f"from-{filter_name}:{since.isoformat()}"
        if until is not None:
            date_filter += f",until-{filter_name}:{until.isoformat()}"
        seen_pages = set()
        while True:
            pages_requested += 1
            try:
                params = {
                    "filter": date_filter, "rows": rows, "cursor": cursor,
                    "select": CROSSREF_SELECT_FIELDS,
                }
                response = self._get_page(issn, params)
                message = response.json().get("message", {})
            except (httpx.HTTPStatusError, httpx.TransportError, CrossrefRequestError) as exc:
                if isinstance(exc, httpx.HTTPStatusError):
                    status = exc.response.status_code
                    stopped_reason = (
                        "rate_limited" if status == 429 else "provider_error"
                    )
                elif isinstance(exc, CrossrefRequestError):
                    stopped_reason = "provider_error"
                else:
                    stopped_reason = "timeout"
                error_summary = str(exc)[:500]
                if not papers:
                    raise
                break
            items = message.get("items", [])
            provider_returned_count += len(items)
            if len(items) == rows:
                signature = tuple((item.get("DOI"), str(item.get("title"))) for item in items)
                if signature in seen_pages:
                    stopped_reason = "provider_error"
                    error_summary = "Crossref repeated a full page without advancing"
                    break
                seen_pages.add(signature)
            for item in items:
                try:
                    papers.append(parse_crossref_item(item))
                except ValueError:
                    continue
                if limit and len(papers) >= limit:
                    stopped_reason = "requested_limit_reached"
                    stats = {
                        "requested_limit": limit,
                        "pages_requested": pages_requested,
                        "provider_returned_count": provider_returned_count,
                        "raw_candidate_count": len(papers),
                        "stopped_reason": stopped_reason,
                        "error_summary": None,
                    }
                    return papers, stats
            next_cursor = message.get("next-cursor")
            if (
                not items
                or len(items) < rows
                or not next_cursor
                or (limit and len(papers) >= limit)
            ):
                break
            cursor = next_cursor
        stats = {
            "requested_limit": limit,
            "pages_requested": pages_requested,
            "provider_returned_count": provider_returned_count,
            "raw_candidate_count": len(papers),
            "stopped_reason": stopped_reason,
            "error_summary": error_summary,
        }
        return papers, stats

    def fetch_work(self, doi: str) -> dict[str, Any] | None:
        """Fetch one Crossref work record via the single-work route."""
        from eduecon_radar.services.doi import canonicalize_doi

        normalized = canonicalize_doi(doi)
        if not normalized:
            return None
        response = self.client.get(
            f"https://api.crossref.org/works/{quote(normalized, safe='')}"
        )
        _raise_for_status(response)
        message = response.json().get("message", {})
        return message if message else None
