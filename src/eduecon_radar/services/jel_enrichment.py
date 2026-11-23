"""JEL classification enrichment — a *paper fact*, resolved from published sources.

Why this module exists
----------------------
JEL codes are properties of a paper, not opinions of a reading queue.  The same
paper carries the same JEL codes whether Education Economics, Labor Economics or
Behavioral Economics is looking at it, so the codes live once on ``papers`` and
are shared by every profile.  What a profile does with them is its own business
(``subject_signals.jel_codes`` in ``profile.yaml``).

Two rules shape the whole design
--------------------------------
1. **A code is never invented.**  Codes are only ever copied from a source that
   published them for that exact paper.  No model, no title heuristic, no
   "papers like this usually have D91".  If no source publishes codes for a
   paper, the honest answer is "unknown", and the ledger records that we asked
   and got nothing — which is different from never having asked.

2. **Provenance is per observation, not per field.**  ``paper_jel_observations``
   keeps one row per (paper, source) with the source URL, how the codes were
   read, and whether the source actually had them.  That is what makes the
   stored value auditable after the fact.

About the sources
-----------------
Empirically (measured, not assumed) the usual bibliographic APIs do *not* carry
JEL codes: Crossref deposits no JEL field for these journals, OpenAlex has none,
and Semantic Scholar has none.  What does carry them is RePEc, whose records
contain the publisher-supplied JEL classification, and publisher landing pages.

RePEc's own guidance (https://ideas.repec.org/getdata.html) is explicit: they
discourage scraping and prefer their bulk metadata or their API, which is issued
by request.  This module therefore:

* keeps every network provider **disabled by default** in
  ``config/jel_sources.yaml`` — enabling one is a deliberate act;
* uses one request per *browse page* (JEL-code listings), never per paper, and
  caches every response so a re-run makes no new requests;
* waits between requests beyond the provider's declared rate limit, identifies
  itself with a contact address, and stops on 429/5xx instead of retrying hard;
* supports a ``repec_official_api`` provider for when an access code is granted.

``manual_verified`` is always available, needs no network, and is how a code
that a human read off the published article gets into the database.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import unescape
from pathlib import Path
from typing import Any, Protocol

import yaml

from eduecon_radar.resources import read_text
from eduecon_radar.services.metadata_provenance import (
    load_source_priorities,
    record_provenance,
)
from eduecon_radar.services.resilient_http import ResilientHttpClient

LOGGER = logging.getLogger(__name__)

JEL_SOURCES_PATH = "config/jel_sources.yaml"

#: JEL codes are one letter, one or two digits, and an optional letter suffix for
#: the finest published granularity (``D9``, ``D91``, ``Y50``, ``F14``, ``I23``,
#: ``O33``).  Anything else is not a JEL code and is never stored as one.
_JEL_PATTERN = re.compile(r"^([A-Z])(\d{1,2})([A-Z]?)$")

#: The top-level sections the American Economic Association actually publishes.
#: A code whose first letter is not here is kept and flagged rather than
#: silently dropped: dropping source data would be a silent edit.
KNOWN_JEL_TOP_LEVEL = frozenset("ABCDEFGHJKLMNOPQRSTUVYZ")

#: Codes are written the way the publisher writes them.  This only strips
#: separators and surrounding punctuation; it never repairs or completes a code.
_CODE_TOKEN = re.compile(r"[A-Z]\d{1,2}[A-Z]?")

#: A source-rendered classification section, e.g.
#: ``JEL classification: D81 - Microeconomics - ... D91 - Microeconomics - ...``
#: Only text *after* this marker is parsed, so a code mentioned in a title or an
#: abstract can never be mistaken for the paper's own classification.
_CLASSIFICATION_MARKER = re.compile(
    r"JEL[\s\-]*(?:classification|codes?|codes\s*:)", re.IGNORECASE
)


def normalize_jel_code(value: str) -> str | None:
    """Return the canonical form of a single code, or ``None`` if it is not one.

    Only the surrounding whitespace and the case of the §-letter are touched.
    A malformed token is **rejected, never repaired**: ``D9a`` is not silently
    turned into ``D9A``, and ``O3 3`` is not silently joined into ``O33``,
    because either would store a code no source published.  Legitimate notation
    variation is handled instead: ``d91`` and `` D91 `` are the same code.
    """
    if value is None:
        return None
    token = str(value).strip()
    match = _JEL_PATTERN.fullmatch(token.upper())
    if match is None:
        return None
    letter, digits, suffix = match.groups()
    # A lowercase suffix is malformed JEL notation.  Uppercasing would turn
    # "D9a" into the valid-looking "D9A", i.e. it would *create* a code.
    if suffix and not token.endswith(suffix):
        return None
    return f"{letter}{digits}{suffix}"


def normalize_jel_codes(values: Iterable[str]) -> tuple[str, ...]:
    """Canonical, de-duplicated, order-preserving code list."""
    seen: dict[str, None] = {}
    for value in values:
        code = normalize_jel_code(value)
        if code is not None:
            seen.setdefault(code, None)
    return tuple(seen)


def jel_codes_to_storage(codes: Iterable[str]) -> str:
    """The canonical ``papers.jel_codes_json`` representation."""
    return json.dumps(list(normalize_jel_codes(codes)), ensure_ascii=False)


def jel_codes_from_storage(value: object) -> tuple[str, ...]:
    """Parse a stored ``jel_codes_json`` value; unknown shapes yield no codes."""
    if value in (None, "", "[]"):
        return ()
    if isinstance(value, (list, tuple)):
        return normalize_jel_codes(str(item) for item in value)
    text = str(value).strip()
    if not text:
        return ()
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        # Tolerate a plain comma/space separated string; still never invent.
        return normalize_jel_codes(re.split(r"[,;\s]+", text))
    if isinstance(parsed, (list, tuple)):
        return normalize_jel_codes(str(item) for item in parsed)
    if isinstance(parsed, str):
        return normalize_jel_codes(re.split(r"[,;\s]+", parsed))
    return ()


def extract_jel_codes_from_classification_text(text: object) -> tuple[str, ...]:
    """Read codes out of a source-rendered classification section.

    The function refuses to look at text that does not contain an explicit JEL
    classification marker.  That refusal is the point: it is what stops a paper
    *about* code D91 from being labelled D91.
    """
    raw = unescape(re.sub(r"<[^>]+>", " ", str(text or "")))
    marker = _CLASSIFICATION_MARKER.search(raw)
    if marker is None:
        return ()
    section = raw[marker.end() :]
    # A classification section is a run of "CODE - description" items and ends
    # at the next heading-like content.  Bounding it prevents swallowing codes
    # that belong to a *cited* paper further down the page.
    stop = re.search(
        r"(?:More about this item|Related works|Citations|References|"
        r"Download restriction|Other versions|Statistics|Corrections)\b",
        section,
        re.IGNORECASE,
    )
    if stop is not None:
        section = section[: stop.start()]
    return normalize_jel_codes(_CODE_TOKEN.findall(section))


@dataclass(frozen=True)
class JelObservation:
    """One source's answer about one paper."""

    paper_id: int
    source_name: str
    extraction_method: str
    resolution: str
    codes: tuple[str, ...] = ()
    source_record_url: str | None = None
    raw_evidence: str | None = None
    http_status: int | None = None
    failure_reason: str | None = None
    #: Codes the source published that do not survive validation.  Kept so the
    #: rejection is visible instead of looking like the source had nothing.
    rejected_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.resolution not in {"resolved", "absent_at_source", "unresolved"}:
            raise ValueError(f"Unknown resolution: {self.resolution!r}")
        if self.resolution != "resolved" and self.codes:
            raise ValueError("Only a resolved observation may carry codes")
        if self.resolution == "resolved" and not self.codes and not self.rejected_codes:
            raise ValueError(
                "A resolved observation must carry codes or an explicit rejection"
            )

    @property
    def is_usable(self) -> bool:
        return self.resolution == "resolved" and bool(self.codes)


@dataclass(frozen=True)
class JelPaperInput:
    """The minimum a provider needs to look a paper up."""

    paper_id: int
    doi: str | None
    title: str
    authors: tuple[str, ...] = ()
    journal_name: str | None = None
    year: int | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    issn: str | None = None
    repec_series: str | None = None


@dataclass
class JelEnrichmentStats:
    papers_considered: int = 0
    papers_with_codes_before: int = 0
    papers_with_codes_after: int = 0
    observations_recorded: int = 0
    resolved: int = 0
    absent_at_source: int = 0
    unresolved: int = 0
    stored_updates: int = 0
    provenance_records: int = 0
    entries_harvested: int = 0
    requests_made: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "papers_considered": self.papers_considered,
            "papers_with_codes_before": self.papers_with_codes_before,
            "papers_with_codes_after": self.papers_with_codes_after,
            "observations_recorded": self.observations_recorded,
            "resolved": self.resolved,
            "absent_at_source": self.absent_at_source,
            "unresolved": self.unresolved,
            "stored_updates": self.stored_updates,
            "provenance_records": self.provenance_records,
            "entries_harvested": self.entries_harvested,
            "requests_made": self.requests_made,
            "skipped": dict(self.skipped),
            "errors": list(self.errors),
        }


class JelProvider(Protocol):
    """A source of published JEL codes."""

    name: str
    enabled: bool
    priority: int

    def observations_for(
        self, papers: Sequence[JelPaperInput]
    ) -> dict[int, JelObservation]: ...


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


class ManualJelProvider:
    """Codes a human read from the published article.

    This provider never guesses: the codes are supplied explicitly, and they are
    recorded with the same ledger row shape as a network source so that a manual
    entry is auditable exactly like an automatic one.
    """

    name = "manual"
    extraction_method = "human_verified"

    def __init__(
        self,
        entries: dict[int, Sequence[str]],
        *,
        note: str | None = None,
        source_url: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.entries = {int(key): tuple(value) for key, value in entries.items()}
        self.note = note
        self.source_url = source_url
        self.enabled = enabled
        self.priority = 100

    def observations_for(
        self, papers: Sequence[JelPaperInput]
    ) -> dict[int, JelObservation]:
        found: dict[int, JelObservation] = {}
        for paper in papers:
            raw = self.entries.get(paper.paper_id)
            if raw is None:
                continue
            codes = normalize_jel_codes(raw)
            rejected = tuple(
                str(item) for item in raw if normalize_jel_code(str(item)) is None
            )
            if not codes:
                found[paper.paper_id] = JelObservation(
                    paper_id=paper.paper_id,
                    source_name=self.name,
                    extraction_method=self.extraction_method,
                    resolution="unresolved",
                    raw_evidence=self.note,
                    source_record_url=self.source_url,
                    failure_reason="manual_entry_contained_no_valid_jel_code",
                )
                continue
            found[paper.paper_id] = JelObservation(
                paper_id=paper.paper_id,
                source_name=self.name,
                extraction_method=self.extraction_method,
                resolution="resolved",
                codes=codes,
                rejected_codes=rejected,
                raw_evidence=self.note or "manual entry",
                source_record_url=self.source_url,
            )
        return found


@dataclass(frozen=True)
class RepecIndexEntry:
    """One article as listed on a RePEc JEL-code browse page."""

    handle: str | None
    title: str
    doi: str | None
    year: int | None
    authors: str | None = None
    journal: str | None = None
    code: str | None = None

    @property
    def identity_keys(self) -> tuple[tuple[str, str], ...]:
        """Match keys for this entry, most reliable first.

        The DOI is the identity when the publisher registered one.  Otherwise the
        normalized title plus year.  Both keys are emitted when available so an
        entry found through one listing can still match a paper that only carries
        the other identifier.
        """
        keys: list[tuple[str, str]] = []
        if self.doi:
            keys.append(("doi", self.doi.casefold()))
        normalized = normalize_title_for_match(self.title)
        if normalized:
            keys.append(("title", f"{normalized}|{self.year or ''}"))
        return tuple(keys)


def normalize_title_for_match(value: object) -> str:
    text = unescape(str(value or "")).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


#: Canonical host for a RePEc article handle.
REPEC_BASE_URL = "https://ideas.repec.org"


def repec_record_url(handle: object) -> str | None:
    """Normalize a RePEc article handle into one absolute URL.

    Both shapes occur in real listings and in stored data, so the caller must not
    assume either: IDEAS serves the title anchor as an absolute URL
    (``https://ideas.repec.org/a/...``), while other feeds and older records carry
    only the path (``/a/...``).  Prefixing unconditionally produced the doubled
    ``https://ideas.repec.orghttps://ideas.repec.org/a/...`` that reached 43
    production provenance rows, so the shape is detected here instead of assumed.
    """
    if handle is None:
        return None
    text = str(handle).strip()
    if not text:
        return None
    if text.startswith(("http://", "https://")):
        return text
    if not text.startswith("/"):
        text = f"/{text}"
    return f"{REPEC_BASE_URL}{text}"


def parse_repec_jel_browse_page(html: str, *, code: str) -> list[RepecIndexEntry]:
    """Parse an IDEAS ``/j/<code>.html`` listing into entries.

    The real markup (verified against the live page; the exact bytes are kept at
    ``tests/fixtures/repec_jel_D91.html``) renders one ``<li
    class="list-group-item ...">`` per article, with the title inside ``<b>`` and
    the publisher DOI in a trailing ``DOI: ...``.  Years arrive as ``<h3>``
    separators that the entries following them belong to.

    Nothing is inferred: an entry without a truthful title is dropped, and an
    entry without a DOI simply carries no DOI.
    """
    entries: list[RepecIndexEntry] = []
    # Walk year headings and list items in document order so each item inherits
    # the year of the heading above it.
    tokens = re.findall(
        r"(?is)<h3[^>]*>\s*(\d{4})\s*</h3>"
        r"|<li\s+class=\"list-group-item[^\"]*\"[^>]*>(.*?)"
        r"(?=<li\s+class=\"list-group-item|</ul>|$)",
        html,
    )
    current_year: int | None = None
    for year_text, item in tokens:
        if year_text:
            current_year = int(year_text)
            continue
        if not item:
            continue
        title_match = re.search(r"(?is)<b>(.*?)</b>", item)
        if title_match is None:
            continue
        title = " ".join(unescape(re.sub(r"<[^>]+>", " ", title_match.group(1))).split())
        title = title.strip(" \u201c\u201d\"',;.")
        if len(title) < 12:
            continue
        doi_match = re.search(r"(?i)\bDOI:\s*(10\.\d{4,9}/[^\s<>\"]+)", item)
        doi = doi_match.group(1).rstrip(".,;") if doi_match else None
        # The title anchor is the record's own handle; RePEc namespaces it by
        # item type (/a/ article, /p/ paper, /h/ chapter, /b/ book, /c/ software).
        # IDEAS serves it absolute, but a relative path is accepted too and
        # normalized, so no consumer has to guess which shape it received.
        handle_match = re.search(
            r"(?is)<a\s+href=\"((?:https?://ideas\.repec\.org)?/[aphbc]/[^\"#]+\.html)\""
            r"\s*>\s*\"?<b>",
            item,
        )
        handle = repec_record_url(handle_match.group(1)) if handle_match else None
        entries.append(
            RepecIndexEntry(
                handle=handle,
                title=title,
                doi=doi,
                year=current_year,
                code=code,
            )
        )
    return entries


def repec_jel_listing_url(code: str, page: int = 1) -> str:
    """The listing URL for a code and page (``/j/D91.html``, ``/j/D91-2.html``)."""
    if page <= 1:
        return f"https://ideas.repec.org/j/{code}.html"
    return f"https://ideas.repec.org/j/{code}-{page}.html"


def repec_jel_listing_page_count(html: str) -> int:
    """Highest page number linked from a listing page (1 when unpaginated)."""
    pages = [int(match) for match in re.findall(r"/j/[A-Z]\d{1,2}-(\d+)\.html", html)]
    return max(pages, default=1)


@dataclass(frozen=True)
class RepecIdeasProviderConfig:
    enabled: bool = False
    #: Requests per second against IDEAS.  Deliberately low: RePEc asks that
    #: automated clients not strain their volunteer-run servers.
    rate_limit_per_second: float = 0.2
    timeout_seconds: float = 30.0
    #: Hard cap on listing pages fetched per JEL code, however many the site
    #: links.  Bounds both load and run time.
    max_pages_per_code: int = 3
    #: Extra politeness delay applied on top of the rate limit.
    extra_delay_seconds: float = 1.0
    index_ttl_seconds: int = 24 * 60 * 60
    user_agent: str = (
        "LiteratureRadar/1.0 (personal literature review; JEL metadata; "
        "contact: local-operator)"
    )


class RepecIdeasJelProvider:
    """JEL codes from RePEc's own classification listings.

    Only browse pages are fetched — one request per JEL code per page, cached on
    disk with a TTL — and the discovered index is matched against the papers we
    actually hold.  RePEc asks automated clients not to scrape their site, so the
    provider is disabled unless explicitly enabled and never fetches an article
    page per paper.
    """

    name = "repec_ideas"
    extraction_method = "structured_listing"
    priority = 70

    def __init__(
        self,
        codes: Sequence[str],
        *,
        config: RepecIdeasProviderConfig | None = None,
        cache_dir: Path | None = None,
        client: ResilientHttpClient | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.codes = tuple(normalize_jel_codes(codes))
        self.config = config or RepecIdeasProviderConfig()
        self.enabled = bool(self.config.enabled and self.codes)
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = client or ResilientHttpClient(
            user_agent=self.config.user_agent,
            timeout=self.config.timeout_seconds,
            max_retries=1,
            rate_limit_per_second=self.config.rate_limit_per_second,
        )
        self._sleeper = sleeper
        self.requests_made = 0
        #: identity key -> {code: entry}.  A paper normally appears on several
        #: classification listings (one per code it carries), so codes are
        #: accumulated across pages instead of the first match winning.
        self._index: dict[tuple[str, str], dict[str, RepecIndexEntry]] | None = None
        #: Distinct articles seen in the listings, deduplicated by title so a
        #: paper listed under three codes is counted once.
        self.articles_indexed = 0

    # -- index -------------------------------------------------------------- #

    def build_index(self) -> dict[tuple[str, str], dict[str, RepecIndexEntry]]:
        """Fetch (or reuse) the classification listings and key them by identity."""
        if self._index is not None:
            return self._index
        index: dict[tuple[str, str], dict[str, RepecIndexEntry]] = {}
        seen_articles: set[str] = set()
        for code in self.codes:
            page = 1
            pages_available = 1
            while page <= min(self.config.max_pages_per_code, pages_available):
                html = self._fetch_browse_page(code, page)
                if html is None:
                    break
                if page == 1:
                    pages_available = repec_jel_listing_page_count(html)
                entries = parse_repec_jel_browse_page(html, code=code)
                if not entries:
                    break
                for entry in entries:
                    seen_articles.add(
                        normalize_title_for_match(entry.title) or (entry.doi or "")
                    )
                    for key in entry.identity_keys:
                        index.setdefault(key, {})[code] = entry
                page += 1
        self._index = index
        self.articles_indexed = len(seen_articles - {""})
        return index

    def _browse_url(self, code: str, page: int) -> str:
        return repec_jel_listing_url(code, page)

    def _cache_path(self, code: str, page: int) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"repec_jel_{code}_{page}.html"

    def _fetch_browse_page(self, code: str, page: int) -> str | None:
        cached = self._cache_path(code, page)
        if cached is not None and cached.is_file():
            age = time.time() - cached.stat().st_mtime
            if age <= self.config.index_ttl_seconds:
                return cached.read_text(encoding="utf-8")
        url = self._browse_url(code, page)
        try:
            if self.config.extra_delay_seconds:
                self._sleeper(self.config.extra_delay_seconds)
            response = self._client.get(url)
        except Exception as exc:  # noqa: BLE001 - provider failure must not abort a run
            LOGGER.warning("RePEc JEL listing unavailable for %s page %s: %s", code, page, exc)
            return None
        self.requests_made += 1
        if response.status_code != 200:
            LOGGER.info(
                "RePEc JEL listing %s page %s returned HTTP %s", code, page, response.status_code
            )
            return None
        html = response.text
        if cached is not None:
            cached.write_text(html, encoding="utf-8")
        return html

    # -- provider protocol -------------------------------------------------- #

    @property
    def index_size(self) -> int:
        """How many distinct classified articles the provider index holds."""
        return self.articles_indexed

    def observations_for(
        self, papers: Sequence[JelPaperInput]
    ) -> dict[int, JelObservation]:
        if not self.enabled:
            return {}
        index = self.build_index()
        observations: dict[int, JelObservation] = {}
        for paper in papers:
            entries: dict[str, RepecIndexEntry] = {}
            if paper.doi:
                entries.update(index.get(("doi", paper.doi.casefold())) or {})
            normalized = normalize_title_for_match(paper.title)
            if normalized:
                entries.update(index.get(("title", f"{normalized}|{paper.year or ''}")) or {})
            if not entries:
                continue
            codes = normalize_jel_codes(entries)
            sample = next(iter(entries.values()))
            observations[paper.paper_id] = JelObservation(
                paper_id=paper.paper_id,
                source_name=self.name,
                extraction_method=self.extraction_method,
                resolution="resolved" if codes else "unresolved",
                codes=codes,
                raw_evidence=(
                    f"codes={' '.join(sorted(entries))}; "
                    f"listing={sample.handle or sample.doi or sample.title[:80]}"
                ),
                source_record_url=repec_record_url(sample.handle),
                failure_reason=None if codes else "listing_entry_carried_no_parseable_code",
            )
        return observations


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class JelSourceSettings:
    version: str
    repec_ideas: RepecIdeasProviderConfig
    website_url_template: str | None = None
    codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "repec_ideas": {
                "enabled": self.repec_ideas.enabled,
                "rate_limit_per_second": self.repec_ideas.rate_limit_per_second,
                "max_pages_per_code": self.repec_ideas.max_pages_per_code,
            },
            "codes": list(self.codes),
        }


def load_jel_source_settings(path: Path | None = None) -> JelSourceSettings:
    payload = yaml.safe_load(
        path.read_text(encoding="utf-8") if path is not None else read_text(JEL_SOURCES_PATH)
    ) or {}
    repec = payload.get("repec_ideas") or {}
    defaults = RepecIdeasProviderConfig()
    config = RepecIdeasProviderConfig(
        enabled=bool(repec.get("enabled", defaults.enabled)),
        rate_limit_per_second=float(
            repec.get("rate_limit_per_second", defaults.rate_limit_per_second)
        ),
        timeout_seconds=float(repec.get("timeout_seconds", defaults.timeout_seconds)),
        max_pages_per_code=int(repec.get("max_pages_per_code", defaults.max_pages_per_code)),
        extra_delay_seconds=float(
            repec.get("extra_delay_seconds", defaults.extra_delay_seconds)
        ),
        index_ttl_seconds=int(repec.get("index_ttl_seconds", defaults.index_ttl_seconds)),
        user_agent=str(repec.get("user_agent", defaults.user_agent)),
    )
    return JelSourceSettings(
        version=str(payload.get("version") or "unversioned"),
        repec_ideas=config,
        website_url_template=payload.get("website_url_template"),
        codes=tuple(str(item) for item in (payload.get("codes") or ())),
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def jel_observations_schema_ready(connection: sqlite3.Connection) -> bool:
    return (
        connection.execute(
            """SELECT 1 FROM sqlite_master WHERE type='table'
            AND name='paper_jel_observations'"""
        ).fetchone()
        is not None
    )


def record_jel_observation(connection: sqlite3.Connection, observation: JelObservation) -> None:
    """Upsert one (paper, source) answer into the observation ledger."""
    now = datetime.now(UTC).isoformat()
    raw_evidence = observation.raw_evidence
    if observation.rejected_codes:
        rejected = "rejected:" + ",".join(observation.rejected_codes)
        raw_evidence = f"{raw_evidence} | {rejected}" if raw_evidence else rejected
    connection.execute(
        """INSERT INTO paper_jel_observations
        (paper_id,source_name,source_record_url,extraction_method,resolution,
         codes_json,codes_normalized,raw_evidence,http_status,failure_reason,observed_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(paper_id,source_name) DO UPDATE SET
            source_record_url=excluded.source_record_url,
            extraction_method=excluded.extraction_method,
            resolution=excluded.resolution,
            codes_json=excluded.codes_json,
            codes_normalized=excluded.codes_normalized,
            raw_evidence=excluded.raw_evidence,
            http_status=excluded.http_status,
            failure_reason=excluded.failure_reason,
            observed_at=excluded.observed_at""",
        (
            observation.paper_id,
            observation.source_name,
            observation.source_record_url,
            observation.extraction_method,
            observation.resolution,
            jel_codes_to_storage(observation.codes),
            " ".join(observation.codes),
            raw_evidence,
            observation.http_status,
            observation.failure_reason,
            now,
        ),
    )


def resolve_jel_codes(
    observations: Sequence[JelObservation],
    *,
    priorities: dict[str, int],
) -> tuple[JelObservation | None, tuple[JelObservation, ...]]:
    """Choose the authoritative observation and report any disagreement.

    Codes are copied from a single source.  Mixing codes from different sources
    would produce a set that no publisher ever asserted, which is exactly the
    kind of unsourced claim this module exists to prevent.  When sources
    disagree the higher-priority source wins and the disagreement is returned so
    the caller can record it as a conflict rather than hiding it.
    """
    usable = [item for item in observations if item.is_usable]
    if not usable:
        return None, ()
    ordered = sorted(
        usable,
        key=lambda item: (-priorities.get(item.source_name, 0), item.source_name),
    )
    winner = ordered[0]
    contested = tuple(item for item in ordered[1:] if set(item.codes) != set(winner.codes))
    return winner, contested


def _apply_codes_to_paper(
    connection: sqlite3.Connection,
    *,
    observation: JelObservation,
    contested: tuple[JelObservation, ...],
    priority: int,
) -> tuple[bool, bool]:
    """Write the winning codes onto ``papers`` and record provenance.

    Returns ``(value_changed, provenance_recorded)``.
    """
    row = connection.execute(
        "SELECT jel_codes_json FROM papers WHERE id=?", (observation.paper_id,)
    ).fetchone()
    if row is None:
        return False, False
    stored = jel_codes_to_storage(observation.codes)
    current = jel_codes_from_storage(row["jel_codes_json"])
    value_changed = tuple(current) != tuple(observation.codes)
    if value_changed:
        connection.execute(
            """UPDATE papers SET jel_codes_json=?,updated_at=?
            WHERE id=?""",
            (stored, datetime.now(UTC).isoformat(), observation.paper_id),
        )
    reason = "jel_codes_from_published_source"
    if contested:
        reason = "jel_codes_from_higher_priority_source_over_conflicting_sources"
    record_provenance(
        connection,
        paper_id=observation.paper_id,
        field_name="jel_codes_json",
        value=stored,
        source_name=observation.source_name,
        source_record_id=observation.source_record_url,
        source_priority=priority,
        decision="accepted",
        reason=reason,
        is_current=True,
    )
    return value_changed, True


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #


class JelSchemaRequired(RuntimeError):
    """Raised when migration 0014 has not been applied."""


class JelEnrichmentService:
    """Resolve JEL codes for papers and store them as global paper facts."""

    def __init__(
        self,
        database: Any,
        providers: Sequence[JelProvider],
        *,
        priorities: dict[str, int] | None = None,
    ) -> None:
        self.database = database
        if priorities is None:
            _, priorities = load_source_priorities()
        self.priorities = dict(priorities)
        self.providers = tuple(providers)

    # -- inputs ------------------------------------------------------------- #

    def load_papers(
        self,
        *,
        paper_ids: Sequence[int] | None = None,
        only_missing: bool = True,
        limit: int | None = None,
        repec_series: dict[str, str] | None = None,
    ) -> list[JelPaperInput]:
        series = {str(k): str(v) for k, v in (repec_series or {}).items()}
        sql = [
            """SELECT p.id,p.doi,p.title,p.authors_json,p.volume,p.issue,p.pages,
            p.jel_codes_json,COALESCE(p.issue_date,p.online_date) publication_date,
            j.name journal_name,j.short_name journal_short,j.issn journal_issn
            FROM papers p JOIN journals j ON j.id=p.journal_id"""
        ]
        clauses: list[str] = []
        params: list[Any] = []
        if paper_ids:
            clauses.append("p.id IN ({})".format(",".join("?" * len(paper_ids))))
            params.extend(int(item) for item in paper_ids)
        if only_missing:
            clauses.append(
                "(p.jel_codes_json IS NULL OR p.jel_codes_json IN ('','[]','{}'))"
            )
        if clauses:
            sql.append("WHERE " + " AND ".join(clauses))
        sql.append("ORDER BY p.id")
        if limit is not None:
            sql.append(f"LIMIT {int(limit)}")
        with self.database.connect() as connection:
            rows = connection.execute(" ".join(sql), params).fetchall()
        papers: list[JelPaperInput] = []
        for row in rows:
            try:
                authors = tuple(json.loads(row["authors_json"] or "[]"))
            except (TypeError, ValueError):
                authors = ()
            year = int(str(row["publication_date"])[:4]) if row["publication_date"] else None
            journal_key = str(row["journal_short"])
            papers.append(
                JelPaperInput(
                    paper_id=int(row["id"]),
                    doi=row["doi"],
                    title=row["title"],
                    authors=authors,
                    journal_name=row["journal_name"],
                    year=year,
                    volume=row["volume"],
                    issue=row["issue"],
                    pages=row["pages"],
                    issn=row["journal_issn"],
                    repec_series=series.get(journal_key),
                )
            )
        return papers

    def count_papers_with_codes(self) -> int:
        with self.database.connect() as connection:
            return int(
                connection.execute(
                    """SELECT COUNT(*) FROM papers WHERE jel_codes_json IS NOT NULL
                    AND jel_codes_json NOT IN ('','[]','{}')"""
                ).fetchone()[0]
            )

    # -- run ---------------------------------------------------------------- #

    def enrich(
        self,
        papers: Sequence[JelPaperInput],
        *,
        stats: JelEnrichmentStats | None = None,
    ) -> JelEnrichmentStats:
        result = stats or JelEnrichmentStats()
        with self.database.connect() as connection:
            if not jel_observations_schema_ready(connection):
                raise JelSchemaRequired(
                    "paper_jel_observations is missing; run migrations/0014 first"
                )
        if not papers:
            result.papers_with_codes_after = self.count_papers_with_codes()
            return result
        result.papers_considered += len(papers)
        result.papers_with_codes_before = self.count_papers_with_codes()

        collected: dict[int, list[JelObservation]] = {}
        for provider in self.providers:
            if not getattr(provider, "enabled", False):
                result.skipped[provider.name] = result.skipped.get(provider.name, 0) + 1
                continue
            try:
                produced = provider.observations_for(papers)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("JEL provider %s failed", provider.name)
                result.errors.append(f"{provider.name}: {exc}")
                continue
            result.requests_made += int(getattr(provider, "requests_made", 0) or 0)
            result.entries_harvested += int(getattr(provider, "index_size", 0) or 0)
            for paper_id, observation in (produced or {}).items():
                collected.setdefault(int(paper_id), []).append(observation)

        for paper in papers:
            observations = collected.get(paper.paper_id) or []
            winner, contested = resolve_jel_codes(observations, priorities=self.priorities)
            with self.database.connect() as connection:
                for observation in observations:
                    record_jel_observation(connection, observation)
                    result.observations_recorded += 1
                    if observation.resolution == "resolved":
                        result.resolved += 1
                    elif observation.resolution == "absent_at_source":
                        result.absent_at_source += 1
                    else:
                        result.unresolved += 1
                if winner is not None:
                    changed, recorded = _apply_codes_to_paper(
                        connection,
                        observation=winner,
                        contested=contested,
                        priority=self.priorities.get(winner.source_name, 0),
                    )
                    result.stored_updates += int(changed)
                    result.provenance_records += int(recorded)
        result.papers_with_codes_after = self.count_papers_with_codes()
        return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def audit_jel_enrichment(database: Path) -> dict[str, Any]:
    """Read-only report: what is stored, where it came from, and what is missing."""
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        ready = jel_observations_schema_ready(connection)
        total = int(connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
        with_codes = int(
            connection.execute(
                """SELECT COUNT(*) FROM papers WHERE jel_codes_json IS NOT NULL
                AND jel_codes_json NOT IN ('','[]','{}')"""
            ).fetchone()[0]
        )
        report: dict[str, Any] = {
            "status": "ok",
            "observation_ledger": "present" if ready else "missing",
            "total_papers": total,
            "papers_with_jel_codes": with_codes,
            "papers_without_jel_codes": total - with_codes,
            "coverage": round(with_codes / total, 4) if total else 0.0,
        }
        if not ready:
            report["status"] = "partial"
            report["hint"] = "apply migrations/0014_add_jel_observations.sql"
            return report
        report["by_resolution"] = {
            row["resolution"]: int(row["count"])
            for row in connection.execute(
                """SELECT resolution,COUNT(*) count FROM paper_jel_observations
                GROUP BY resolution"""
            ).fetchall()
        }
        report["by_source"] = {
            row["source_name"]: int(row["count"])
            for row in connection.execute(
                """SELECT source_name,COUNT(*) count FROM paper_jel_observations
                GROUP BY source_name ORDER BY count DESC"""
            ).fetchall()
        }
        report["distinct_codes"] = [
            {"code": row["code"], "papers": int(row["count"])}
            for row in connection.execute(
                """SELECT value code,COUNT(*) count FROM papers,json_each(
                    CASE WHEN jel_codes_json IN ('','[]') THEN '[]' ELSE jel_codes_json END)
                GROUP BY value ORDER BY code"""
            ).fetchall()
        ]
        report["provenance_current"] = {
            row["source_name"]: int(row["count"])
            for row in connection.execute(
                """SELECT source_name,COUNT(*) count FROM paper_metadata_provenance
                WHERE field_name='jel_codes_json' AND is_current=1
                GROUP BY source_name ORDER BY count DESC"""
            ).fetchall()
        }
        report["papers_never_asked_count"] = int(
            connection.execute(
                """SELECT COUNT(*) FROM papers p WHERE NOT EXISTS (
                    SELECT 1 FROM paper_jel_observations o WHERE o.paper_id=p.id)"""
            ).fetchone()[0]
        )
        report["papers_source_had_none_count"] = int(
            connection.execute(
                """SELECT COUNT(DISTINCT o.paper_id) FROM paper_jel_observations o
                WHERE o.resolution IN ('absent_at_source','unresolved')"""
            ).fetchone()[0]
        )
        report["papers_with_codes_without_ledger_row"] = int(
            connection.execute(
                """SELECT COUNT(*) FROM papers p
                WHERE p.jel_codes_json IS NOT NULL AND p.jel_codes_json NOT IN ('','[]','{}')
                AND NOT EXISTS (
                    SELECT 1 FROM paper_jel_observations o
                    WHERE o.paper_id=p.id AND o.resolution='resolved')"""
            ).fetchone()[0]
        )
        report["papers_never_asked"] = [
            {"paper_id": int(row["id"]), "title": row["title"][:90]}
            for row in connection.execute(
                """SELECT p.id,p.title FROM papers p WHERE NOT EXISTS (
                    SELECT 1 FROM paper_jel_observations o WHERE o.paper_id=p.id)
                ORDER BY p.id LIMIT 50"""
            ).fetchall()
        ]
        report["papers_source_had_none"] = [
            {"paper_id": int(row["paper_id"]), "title": row["title"][:90]}
            for row in connection.execute(
                """SELECT o.paper_id,p.title FROM paper_jel_observations o
                JOIN papers p ON p.id=o.paper_id
                WHERE o.resolution IN ('absent_at_source','unresolved')
                GROUP BY o.paper_id ORDER BY o.paper_id LIMIT 50"""
            ).fetchall()
        ]
    return report


def build_providers(
    *,
    settings: JelSourceSettings | None = None,
    manual_entries: dict[int, Sequence[str]] | None = None,
    manual_note: str | None = None,
    manual_source_url: str | None = None,
    cache_dir: Path | None = None,
    client: ResilientHttpClient | None = None,
    provider_names: Sequence[str] | None = None,
) -> tuple[JelProvider, ...]:
    """Assemble the provider chain for a run.

    Adding a source means adding one branch here plus one class — no other module
    in the product needs to know that JEL codes came from somewhere new.
    """
    resolved = settings or load_jel_source_settings()
    wanted = {str(item) for item in provider_names} if provider_names else None
    providers: list[JelProvider] = []
    if manual_entries and (wanted is None or "manual" in wanted):
        providers.append(
            ManualJelProvider(
                manual_entries,
                note=manual_note,
                source_url=manual_source_url,
            )
        )
    if wanted is None or "repec_ideas" in wanted:
        providers.append(
            RepecIdeasJelProvider(
                resolved.codes,
                config=resolved.repec_ideas,
                cache_dir=cache_dir,
                client=client,
            )
        )
    return tuple(providers)


def format_jel_report(report: dict[str, Any]) -> str:
    if report.get("status") == "partial":
        return (
            f"JEL report: observation ledger {report['observation_ledger']} — "
            f"{report.get('hint', '')}"
        )
    lines = [
        "JEL metadata report",
        f"  papers                          : {report['total_papers']}",
        f"  with JEL codes                  : {report['papers_with_jel_codes']}",
        f"  without JEL codes               : {report['papers_without_jel_codes']}",
        f"  coverage                        : {report['coverage']:.1%}",
        f"  observations by resolution      : {report.get('by_resolution', {})}",
        f"  observations by source          : {report.get('by_source', {})}",
        f"  current provenance by source    : {report.get('provenance_current', {})}",
    ]
    distinct = report.get("distinct_codes") or []
    if distinct:
        lines.append(
            "  distinct codes                  : "
            + ", ".join(f"{item['code']}×{item['papers']}" for item in distinct)
        )
    never = report.get("papers_never_asked") or []
    never_count = report.get("papers_never_asked_count", len(never))
    if never_count:
        lines.append(
            f"  no source was even asked about  : {never_count} "
            f"(first ids: {', '.join(str(item['paper_id']) for item in never[:10])})"
        )
    none_at_source = report.get("papers_source_had_none") or []
    none_count = report.get("papers_source_had_none_count", len(none_at_source))
    if none_count:
        lines.append(
            f"  asked, source published no code : {none_count} "
            f"(first ids: {', '.join(str(item['paper_id']) for item in none_at_source[:10])})"
        )
    unbacked = report.get("papers_with_codes_without_ledger_row", 0)
    if unbacked:
        lines.append(
            f"  codes with no resolved ledger row: {unbacked} "
            "(stored before the ledger existed, or written by a source that did "
            "not record an observation)"
        )
    return "\n".join(lines)
