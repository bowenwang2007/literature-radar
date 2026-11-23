"""Deterministic retrieval classification for journal-article verification."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

NON_RESEARCH_TITLE = re.compile(
    r"^(editorial board|contents|list of reviewers|announcement|"
    r"corrigendum|erratum|retraction notice)\b",
    re.I,
)


def _date_parts(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, dict):
        return None
    parts = value.get("date-parts", [[]])[0]
    if not parts:
        return None
    parts = [*parts, 1, 1][:3]
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except (TypeError, ValueError):
        return None


def date_in_range(value: Any, since: date, until: date) -> bool:
    parts = _date_parts(value)
    if parts is None:
        return False
    try:
        parsed = date(*parts)
    except (TypeError, ValueError):
        return False
    return since <= parsed <= until


def classify_item(item: dict[str, Any]) -> dict[str, Any]:
    """Classify one Crossref-style item into stable audit fields."""
    online = item.get("published-online") or item.get("created")
    issue = item.get("published-print") or item.get("issued")
    volume = item.get("volume")
    article_type = str(item.get("type") or "")
    title = str(item.get("title") or "")
    if isinstance(title, list):
        title = title[0] if title else ""
    doi = str(item.get("DOI") or item.get("doi") or "").casefold()
    return {
        "doi": doi,
        "title": title,
        "volume": volume,
        "volume_class": (
            "articles_in_press" if not volume else str(volume)
        ),
        "online_date": _date_parts(online),
        "issue_date": _date_parts(issue),
        "article_type": article_type,
        "is_research": bool(
            article_type
            in {"journal-article", "research-article", "proceedings-article"}
            and not NON_RESEARCH_TITLE.match(title)
        ),
        "duplicate_key": f"doi:{doi}" if doi else f"title:{title.casefold()}",
    }


def count_paths(
    items: list[dict[str, Any]], since: date, until: date
) -> dict[str, Any]:
    online_in_range = 0
    issue_in_range = 0
    seen: set[str] = set()
    duplicates = 0
    non_research = 0
    volumes: dict[str, int] = {}
    for item in items:
        classified = classify_item(item)
        if date_in_range(
            item.get("published-online") or item.get("created"),
            since,
            until,
        ):
            online_in_range += 1
        if date_in_range(
            item.get("published-print") or item.get("issued"),
            since,
            until,
        ):
            issue_in_range += 1
        key = classified["duplicate_key"]
        if key in seen:
            duplicates += 1
        seen.add(key)
        if not classified["is_research"]:
            non_research += 1
        volumes[classified["volume_class"]] = (
            volumes.get(classified["volume_class"], 0) + 1
        )
    return {
        "unique_items": len(items) - duplicates,
        "online_published_in_range": online_in_range,
        "issue_dated_in_range": issue_in_range,
        "duplicates": duplicates,
        "non_research_count": non_research,
        "volumes": volumes,
    }
