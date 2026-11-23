from __future__ import annotations

from datetime import date
from typing import Any

import httpx

from eduecon_radar.schemas import PaperMetadata


def _rebuild_abstract(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    words = [(position, word) for word, positions in index.items() for position in positions]
    return " ".join(word for _, word in sorted(words))


def parse_openalex_work(item: dict[str, Any]) -> PaperMetadata:
    primary = item.get("primary_location") or {}
    source = primary.get("source") or {}
    biblio = item.get("biblio") or {}
    authors = [
        authorship.get("author", {}).get("display_name")
        for authorship in item.get("authorships") or []
    ]
    authors = [name for name in authors if name]
    publication_date = item.get("publication_date")
    return PaperMetadata(
        doi=item.get("doi"),
        title=item.get("title") or item.get("display_name") or "",
        abstract=_rebuild_abstract(item.get("abstract_inverted_index")),
        authors=authors,
        article_type=item.get("type"),
        issue_date=date.fromisoformat(publication_date) if publication_date else None,
        volume=biblio.get("volume"),
        issue=biblio.get("issue"),
        pages="-".join(
            value for value in [biblio.get("first_page"), biblio.get("last_page")] if value
        )
        or None,
        source_url=primary.get("landing_page_url") or item.get("id"),
        open_access_url=(item.get("best_oa_location") or {}).get("landing_page_url"),
        keywords=[keyword.get("display_name") for keyword in item.get("keywords") or []],
        source_metadata={
            "openalex_id": item.get("id"),
            "source": source,
            "title": item.get("title") or item.get("display_name"),
            "publication_date": publication_date,
            "volume": biblio.get("volume"),
            "issue": biblio.get("issue"),
            "type": item.get("type"),
            "has_inverted_index": bool(item.get("abstract_inverted_index")),
        },
    )


class OpenAlexClient:
    def __init__(self, api_key: str | None = None, client: httpx.Client | None = None) -> None:
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=httpx.Timeout(30))

    def fetch(self, issn: str, since: date, limit: int | None = None) -> list[PaperMetadata]:
        params = {
            "filter": f"primary_location.source.issn:{issn},from_updated_date:{since.isoformat()}",
            "per-page": min(limit or 100, 200),
            "cursor": "*",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        response = self.client.get("https://api.openalex.org/works", params=params)
        response.raise_for_status()
        items = response.json().get("results", [])
        parsed = []
        for item in items:
            try:
                parsed.append(parse_openalex_work(item))
            except ValueError:
                continue
        return parsed[:limit] if limit else parsed

    def fetch_by_doi(self, doi: str) -> PaperMetadata | None:
        """Fetch one OpenAlex work by exact DOI."""
        from eduecon_radar.services.doi import canonicalize_doi

        normalized = canonicalize_doi(doi)
        if not normalized:
            return None
        params: dict[str, object] = {
            "filter": f"doi:{normalized}",
            "select": (
                "id,doi,title,display_name,publication_year,publication_date,"
                "authorships,primary_location,biblio,type,abstract_inverted_index"
            ),
        }
        if self.api_key:
            params["api_key"] = self.api_key
        items = self.client.get(
            "https://api.openalex.org/works", params=params
        ).json().get("results", [])
        if not items:
            return None
        return parse_openalex_work(items[0])

    def fetch_by_id(self, source_id: str) -> PaperMetadata | None:
        """Fetch one OpenAlex work by its canonical source id (e.g. W123…)."""
        from eduecon_radar.services.source_traceability import (
            build_openalex_work_url,
        )

        url = build_openalex_work_url(source_id)
        if not url:
            return None
        params: dict[str, object] = {
            "select": (
                "id,doi,title,display_name,publication_year,publication_date,"
                "authorships,primary_location,biblio,type,abstract_inverted_index"
            ),
        }
        if self.api_key:
            params["api_key"] = self.api_key
        response = self.client.get(url, params=params)
        response.raise_for_status()
        item = response.json()
        if not item or not isinstance(item, dict):
            return None
        try:
            return parse_openalex_work(item)
        except ValueError:
            return None
