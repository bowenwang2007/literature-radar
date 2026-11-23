"""JEL enrichment: validation, source parsing, merge rules and provenance.

The tests are written against the *rules* the feature promises, not against its
implementation:

* a code is never invented — only copied from a source that published it;
* an unrecognizable token is dropped rather than repaired;
* a paper's codes are shared by every profile and owned by none;
* "the source has no code for this paper" is recorded as a fact, not as silence.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from eduecon_radar.db import Database
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.jel_enrichment import (
    JelEnrichmentService,
    JelObservation,
    JelPaperInput,
    JelSchemaRequired,
    ManualJelProvider,
    RepecIdeasJelProvider,
    audit_jel_enrichment,
    build_providers,
    extract_jel_codes_from_classification_text,
    jel_codes_from_storage,
    jel_codes_to_storage,
    load_jel_source_settings,
    normalize_jel_code,
    normalize_jel_codes,
    parse_repec_jel_browse_page,
    repec_jel_listing_page_count,
    repec_jel_listing_url,
    repec_record_url,
    resolve_jel_codes,
)
from eduecon_radar.services.journal_registry import RegistryJournal
from eduecon_radar.services.metadata_provenance import value_is_empty

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "repec_jel_D91.html"
MIGRATION = ROOT / "migrations" / "0014_add_jel_observations.sql"


# --------------------------------------------------------------------------- #
# Validation: a code is recognized or rejected, never repaired
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("D91", "D91"),
        ("d91", "D91"),
        (" d03 ", "D03"),
        ("O33", "O33"),
        ("Y50", "Y50"),
        ("F14", "F14"),
        ("J24", "J24"),
        ("A10", "A10"),
    ],
)
def test_valid_codes_are_canonicalized_by_case_and_whitespace_only(raw, expected):
    assert normalize_jel_code(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "  ",
        "D",
        "D9a",
        "91",
        "D910",
        "DD91",
        "D-91",
        "D 9 1 extra",
        "very relevant paper",
        "1234",
        "O3 3",
    ],
)
def test_invalid_tokens_are_rejected_rather_than_repaired(raw):
    assert normalize_jel_code(raw) is None


def test_normalize_jel_codes_deduplicates_and_preserves_order():
    assert normalize_jel_codes(["D91", "d03", "D91", "nonsense", "J24"]) == (
        "D91",
        "D03",
        "J24",
    )


def test_storage_round_trip_is_stable():
    stored = jel_codes_to_storage(["D91", "D03"])
    assert stored == '["D91", "D03"]'
    assert jel_codes_from_storage(stored) == ("D91", "D03")
    assert jel_codes_from_storage(None) == ()
    assert jel_codes_from_storage("[]") == ()
    assert jel_codes_from_storage("") == ()
    assert jel_codes_from_storage("not json") == ()
    assert jel_codes_from_storage('"D91, D03"') == ("D91", "D03")


def test_empty_json_containers_count_as_no_fact():
    """An empty list is the absence of a value, not a value."""
    assert value_is_empty("[]") is True
    assert value_is_empty("{}") is True
    assert value_is_empty("") is True
    assert value_is_empty(None) is True
    assert value_is_empty('["D91"]') is False
    assert value_is_empty("D91") is False


# --------------------------------------------------------------------------- #
# Reading codes out of a source-rendered classification
# --------------------------------------------------------------------------- #


def test_classification_text_requires_an_explicit_jel_marker():
    """A paper *about* D91 must not be labelled D91."""
    abstract = "We contribute to the D91 literature on present bias and G40 frictions."
    assert extract_jel_codes_from_classification_text(abstract) == ()


def test_classification_text_reads_codes_after_the_marker():
    text = (
        "Keywords Accountability ; JEL classification: D81 - Microeconomics - - "
        "Criteria for Decision-Making D91 - Microeconomics - - Role and Effects "
        "L26 - Industrial Organization - - Entrepreneurship"
    )
    assert extract_jel_codes_from_classification_text(text) == ("D81", "D91", "L26")


def test_classification_text_stops_at_the_next_section():
    text = (
        "JEL classification: D91 - Micro-Based Behavioral Economics "
        "More about this item Keywords D84 - Expectations Citations L26 - Firm"
    )
    assert extract_jel_codes_from_classification_text(text) == ("D91",)


# --------------------------------------------------------------------------- #
# RePEc listing parsing (fixture is the exact live page)
# --------------------------------------------------------------------------- #


pytestmark_fixture = pytest.mark.skipif(
    not FIXTURE.is_file(), reason="RePEc fixture page is not present"
)


@pytest.fixture(scope="module")
def repec_fixture_html() -> str:
    if not FIXTURE.is_file():
        pytest.skip("RePEc fixture page is not present")
    return FIXTURE.read_text(encoding="utf-8")


def test_repec_fixture_parses_titles_and_dois(repec_fixture_html):
    entries = parse_repec_jel_browse_page(repec_fixture_html, code="D91")
    assert len(entries) > 100
    assert all(entry.code == "D91" for entry in entries)
    # No entry may carry a markup or script fragment as its title.
    for entry in entries:
        assert "<" not in entry.title
        assert "function" not in entry.title
        assert len(entry.title) >= 12
    with_doi = [entry for entry in entries if entry.doi]
    assert len(with_doi) > 20
    assert all(entry.doi.startswith("10.") for entry in with_doi)
    handles = [entry for entry in entries if entry.handle]
    assert len(handles) > 100
    # RePEc namespaces records by item type; every handle must be one of them.
    assert all(
        re.search(r"ideas\.repec\.org/[aphbc]/", entry.handle) for entry in handles
    )


def test_repec_fixture_years_are_attached_from_headings(repec_fixture_html):
    entries = parse_repec_jel_browse_page(repec_fixture_html, code="D91")
    years = {entry.year for entry in entries}
    assert years and all(isinstance(year, int) and 1900 < year < 2100 for year in years)


def test_repec_pagination_url_and_page_count(repec_fixture_html):
    assert repec_jel_listing_url("D91") == "https://ideas.repec.org/j/D91.html"
    assert repec_jel_listing_url("D91", 2) == "https://ideas.repec.org/j/D91-2.html"
    assert repec_jel_listing_page_count(repec_fixture_html) > 1
    assert repec_jel_listing_page_count("<html>no pagination</html>") == 1


def test_parser_returns_nothing_for_a_page_without_listings():
    assert parse_repec_jel_browse_page("<html><body>empty</body></html>", code="D91") == []


def test_identity_keys_prefer_doi_but_always_offer_title():
    entries = parse_repec_jel_browse_page(
        '<h3>2026</h3><ul class="list-group paper-list">'
        '<li class="list-group-item downnone"> A Author, 2026, '
        '<a href="https://ideas.repec.org/a/eee/ebehvo/v1.html">"<b>'
        "A Study of Consideration Sets and Peer Effects</b></a>,\" Journal, "
        "DOI: 10.1016/j.jebo.2026.01.001."
        "</ul>",
        code="D91",
    )
    assert len(entries) == 1
    keys = dict(entries[0].identity_keys)
    assert keys["doi"] == "10.1016/j.jebo.2026.01.001"
    assert keys["title"].startswith("a study of consideration sets")


# --------------------------------------------------------------------------- #
# Evidence URL normalization
#
# IDEAS serves the title anchor as an absolute URL.  Prefixing it unconditionally
# produced `https://ideas.repec.orghttps://ideas.repec.org/a/...` in 43
# production provenance rows, so both shapes are pinned here.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # The shape IDEAS actually returns.
        (
            "https://ideas.repec.org/a/eee/ebehvo/v1.html",
            "https://ideas.repec.org/a/eee/ebehvo/v1.html",
        ),
        # A relative path must still be resolved, not left bare.
        ("/a/eee/ebehvo/v1.html", "https://ideas.repec.org/a/eee/ebehvo/v1.html"),
        # The path without a leading slash is accepted too.
        ("a/eee/ebehvo/v1.html", "https://ideas.repec.org/a/eee/ebehvo/v1.html"),
        # Other RePEc item types are namespaced the same way.
        ("https://ideas.repec.org/p/nbr/nberwo/12345.html",
         "https://ideas.repec.org/p/nbr/nberwo/12345.html"),
        ("/h/abc/def/1.html", "https://ideas.repec.org/h/abc/def/1.html"),
        # Nothing to point at.
        (None, None),
        ("", None),
        ("   ", None),
    ],
)
def test_repec_record_url_normalizes_both_shapes(raw, expected):
    assert repec_record_url(raw) == expected


def test_repec_record_url_never_produces_a_doubled_host():
    for raw in (
        "https://ideas.repec.org/a/eee/ebehvo/v1.html",
        "/a/eee/ebehvo/v1.html",
    ):
        url = repec_record_url(raw)
        assert url is not None
        assert url.count("ideas.repec.org") == 1
        assert "http" not in url[4:], f"second scheme embedded in {url!r}"


def test_browse_parser_normalizes_absolute_and_relative_handles():
    absolute = parse_repec_jel_browse_page(
        '<h3>2026</h3><ul class="list-group paper-list">'
        '<li class="list-group-item downnone"> A Author, 2026, '
        '<a href="https://ideas.repec.org/a/eee/ebehvo/v1.html">"<b>'
        "A Study of Consideration Sets and Peer Effects</b></a>,\" Journal."
        "</ul>",
        code="D91",
    )
    relative = parse_repec_jel_browse_page(
        '<h3>2026</h3><ul class="list-group paper-list">'
        '<li class="list-group-item downnone"> A Author, 2026, '
        '<a href="/a/eee/ebehvo/v1.html">"<b>'
        "A Study of Consideration Sets and Peer Effects</b></a>,\" Journal."
        "</ul>",
        code="D91",
    )
    assert absolute[0].handle == "https://ideas.repec.org/a/eee/ebehvo/v1.html"
    assert relative[0].handle == absolute[0].handle


def _repec_listing_html(doi: str, title: str) -> str:
    """A listing shaped like IDEAS' own output: absolute title-anchor href."""
    return (
        '<h3>2026</h3><ul class="list-group paper-list">'
        '<li class="list-group-item downnone"> A Author, 2026, '
        f'<a href="https://ideas.repec.org/a/eee/ebehvo/v1.html">"<b>{title}</b></a>," '
        f"Journal, DOI: {doi}."
        "</ul>"
    )


def test_provider_observation_carries_one_absolute_evidence_url(monkeypatch):
    """The stored provenance URL must be a single, absolute, clickable address.

    Regression for the defect that put
    ``https://ideas.repec.orghttps://ideas.repec.org/a/...`` into 43 production
    provenance rows: the parser's handle is already absolute, so it must not be
    prefixed again.
    """
    from eduecon_radar.services.jel_enrichment import RepecIndexEntry  # noqa: F401

    settings = load_jel_source_settings()

    # Ships disabled by configuration; this test drives the matching path
    # directly and cannot reach the network because browse fetching is stubbed.
    provider = RepecIdeasJelProvider(
        ["D91"], config=replace(settings.repec_ideas, enabled=True)
    )
    title = "A Study of Consideration Sets and Peer Effects"
    html = _repec_listing_html("10.1016/j.jebo.2026.01.001", title)

    monkeypatch.setattr(
        RepecIdeasJelProvider, "_fetch_browse_page", lambda self, code, page: html
    )
    observations = provider.observations_for(
        [JelPaperInput(1, "10.1016/j.jebo.2026.01.001", title, year=2026)]
    )
    assert set(observations) == {1}
    observation = observations[1]
    assert observation.codes == ("D91",)
    assert observation.resolution == "resolved"
    assert observation.source_record_url == (
        "https://ideas.repec.org/a/eee/ebehvo/v1.html"
    )
    assert observation.source_record_url.count("ideas.repec.org") == 1

    # The same guarantee from a relative handle in the source page.
    relative_html = html.replace(
        'href="https://ideas.repec.org/a/', 'href="/a/'
    )
    provider._index = None
    monkeypatch.setattr(
        RepecIdeasJelProvider,
        "_fetch_browse_page",
        lambda self, code, page: relative_html,
    )
    relative = provider.observations_for(
        [JelPaperInput(1, "10.1016/j.jebo.2026.01.001", title, year=2026)]
    )[1]
    assert relative.source_record_url == observation.source_record_url
    assert relative.codes == observation.codes

    # A listing without a title anchor carries no URL rather than a bare host.
    bare = html.replace(
        '<a href="https://ideas.repec.org/a/eee/ebehvo/v1.html">', "<a>"
    )
    provider._index = None
    monkeypatch.setattr(
        RepecIdeasJelProvider, "_fetch_browse_page", lambda self, code, page: bare
    )
    without_handle = provider.observations_for(
        [JelPaperInput(1, "10.1016/j.jebo.2026.01.001", title, year=2026)]
    )[1]
    assert without_handle.source_record_url is None
    assert without_handle.codes == ("D91",)


# --------------------------------------------------------------------------- #
# Provider configuration and chain
# --------------------------------------------------------------------------- #


def test_repec_provider_ships_disabled_and_does_nothing_when_disabled():
    settings = load_jel_source_settings()
    assert settings.repec_ideas.enabled is False
    provider = RepecIdeasJelProvider(["D91"], config=settings.repec_ideas)
    assert provider.enabled is False
    assert provider.observations_for([JelPaperInput(1, "10.1/x", "T")]) == {}
    assert provider.requests_made == 0


def test_build_providers_only_enables_what_was_asked_for(tmp_path):
    settings = load_jel_source_settings()
    providers = build_providers(settings=settings, cache_dir=tmp_path)
    assert [provider.name for provider in providers] == ["repec_ideas"]
    assert providers[0].enabled is False

    manual_only = build_providers(
        settings=settings,
        manual_entries={1: ["D91"]},
        provider_names=["manual"],
    )
    assert [provider.name for provider in manual_only] == ["manual"]
    assert manual_only[0].enabled is True


def test_manual_provider_reports_invalid_entries_instead_of_accepting_them():
    provider = ManualJelProvider({7: ["D91", "not-a-code"]})
    observation = provider.observations_for([JelPaperInput(7, None, "T")])[7]
    assert observation.codes == ("D91",)
    assert observation.rejected_codes == ("not-a-code",)
    assert observation.resolution == "resolved"

    empty = ManualJelProvider({8: ["nonsense"]}).observations_for(
        [JelPaperInput(8, None, "T")]
    )[8]
    assert empty.resolution == "unresolved"
    assert empty.codes == ()


def test_observation_invariants_are_enforced():
    with pytest.raises(ValueError):
        JelObservation(1, "s", "m", "resolved")
    with pytest.raises(ValueError):
        JelObservation(1, "s", "m", "absent_at_source", codes=("D91",))
    with pytest.raises(ValueError):
        JelObservation(1, "s", "m", "unknown")
    assert JelObservation(1, "s", "m", "absent_at_source").is_usable is False
    assert JelObservation(1, "s", "m", "resolved", codes=("D91",)).is_usable is True


# --------------------------------------------------------------------------- #
# Merge rules
# --------------------------------------------------------------------------- #


def test_higher_priority_source_wins_and_disagreement_is_reported():
    low = JelObservation(1, "repec_ideas", "structured_listing", "resolved", ("D91",))
    high = JelObservation(1, "manual", "human_verified", "resolved", ("D03",))
    winner, contested = resolve_jel_codes(
        [low, high], priorities={"manual": 100, "repec_ideas": 70}
    )
    assert winner is high
    assert contested == (low,)

    agree = JelObservation(1, "manual", "human_verified", "resolved", ("D91",))
    winner, contested = resolve_jel_codes(
        [low, agree], priorities={"manual": 100, "repec_ideas": 70}
    )
    assert winner is agree
    assert contested == ()


def test_codes_are_never_mixed_across_sources():
    """The stored set must be something a single source actually asserted."""
    first = JelObservation(1, "a", "m", "resolved", ("D91",))
    second = JelObservation(1, "b", "m", "resolved", ("D03",))
    winner, contested = resolve_jel_codes([first, second], priorities={"a": 10, "b": 20})
    assert winner is second
    assert winner.codes == ("D03",)
    assert contested == (first,)


def test_absent_observations_cannot_become_the_chosen_value():
    absent = JelObservation(1, "repec_ideas", "structured_listing", "absent_at_source")
    winner, contested = resolve_jel_codes([absent], priorities={"repec_ideas": 70})
    assert winner is None
    assert contested == ()


# --------------------------------------------------------------------------- #
# End-to-end persistence against a temporary database
# --------------------------------------------------------------------------- #


def _journal() -> RegistryJournal:
    return RegistryJournal(
        journal_key="journal_of_economic_behavior_and_organization",
        canonical_name="Journal of Economic Behavior & Organization",
        short_name="JEBO",
        aliases=(),
        print_issn="0167-2681",
        online_issn=None,
        publisher="Elsevier",
        journal_group="behavioral_economics_core",
        subject_tags=(),
        enabled=True,
        display_order=1,
        retrieval_priority=1,
        metadata_source="crossref",
        registry_version="test",
    )


@pytest.fixture()
def jel_database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "jel.db")
    database.init_schema()
    with database.connect() as connection:
        connection.executescript(MIGRATION.read_text(encoding="utf-8"))
        # The provenance table is created by migration 0008, which this focused
        # fixture does not apply; add just enough of it for the field-level
        # provenance assertions below.
        connection.execute(
            """CREATE TABLE IF NOT EXISTS paper_metadata_provenance (
                id INTEGER PRIMARY KEY,
                paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                field_name TEXT NOT NULL,
                active_value_hash TEXT,
                source_name TEXT NOT NULL,
                source_record_id TEXT,
                source_priority INTEGER NOT NULL DEFAULT 0,
                source_observed_at TEXT,
                fetched_at TEXT,
                is_current INTEGER NOT NULL DEFAULT 0,
                quality_flags_json TEXT NOT NULL DEFAULT '[]',
                decision TEXT,
                replacement_reason TEXT,
                previous_provenance_id INTEGER,
                created_at TEXT NOT NULL
            )"""
        )
    journal_id = database.upsert_journal(_journal().as_journal_config())
    database.upsert_paper(
        PaperMetadata(
            doi="10.1016/j.jebo.2026.01.001",
            title="Attention and Consideration Sets in Risky Choice",
            abstract="An experiment.",
            authors=["A. Author"],
            source_metadata={"_source_name": "crossref", "DOI": "10.1016/j.jebo.2026.01.001"},
        ),
        journal_id,
    )
    database.upsert_paper(
        PaperMetadata(
            doi="10.1016/j.jebo.2026.01.002",
            title="Institutional Persistence and Growth",
            abstract=None,
            authors=["B. Author"],
            source_metadata={"_source_name": "crossref", "DOI": "10.1016/j.jebo.2026.01.002"},
        ),
        journal_id,
    )
    return database


def test_service_requires_the_observation_ledger(tmp_path):
    database = Database(tmp_path / "bare.db")
    database.init_schema()
    service = JelEnrichmentService(database, [])
    with pytest.raises(JelSchemaRequired):
        service.enrich([JelPaperInput(1, None, "T")])


def test_manual_codes_are_stored_with_provenance(jel_database):
    service = JelEnrichmentService(
        jel_database,
        [ManualJelProvider({1: ["D91", "D03"]}, source_url="https://example.invalid/a")],
    )
    papers = service.load_papers()
    assert len(papers) == 2
    stats = service.enrich(papers)
    assert stats.stored_updates == 1
    assert stats.provenance_records == 1
    assert stats.observations_recorded == 1

    with jel_database.connect() as connection:
        stored = connection.execute(
            "SELECT jel_codes_json FROM papers WHERE id=1"
        ).fetchone()[0]
        assert jel_codes_from_storage(stored) == ("D91", "D03")
        ledger = connection.execute(
            "SELECT * FROM paper_jel_observations WHERE paper_id=1"
        ).fetchone()
        assert ledger["source_name"] == "manual"
        assert ledger["extraction_method"] == "human_verified"
        assert ledger["resolution"] == "resolved"
        assert ledger["source_record_url"] == "https://example.invalid/a"
        provenance = connection.execute(
            """SELECT source_name,decision,is_current FROM paper_metadata_provenance
            WHERE paper_id=1 AND field_name='jel_codes_json' ORDER BY id"""
        ).fetchall()
    current = [row for row in provenance if row["is_current"]]
    assert len(current) == 1
    assert current[0]["source_name"] == "manual"
    assert current[0]["decision"] == "accepted"


def test_only_missing_papers_are_selected_by_default(jel_database):
    service = JelEnrichmentService(jel_database, [ManualJelProvider({1: ["D91"]})])
    service.enrich(service.load_papers())
    assert [paper.paper_id for paper in service.load_papers()] == [2]
    assert [paper.paper_id for paper in service.load_papers(only_missing=False)] == [1, 2]


def test_repeated_enrichment_is_idempotent_for_the_ledger(jel_database):
    provider = ManualJelProvider({1: ["D91"]})
    service = JelEnrichmentService(jel_database, [provider])
    service.enrich(service.load_papers())
    second = service.enrich(service.load_papers(only_missing=False))
    with jel_database.connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM paper_jel_observations WHERE paper_id=1"
        ).fetchone()[0]
    assert count == 1
    # The value did not change, so nothing was rewritten as a new fact.
    assert second.stored_updates == 0


def test_a_later_higher_priority_source_replaces_the_codes_and_keeps_history(jel_database):
    JelEnrichmentService(jel_database, [ManualJelProvider({1: ["D91"]})]).enrich(
        [JelPaperInput(1, "10.1016/j.jebo.2026.01.001", "Attention")]
    )
    # A source with lower priority must not overwrite the manual value.
    with jel_database.connect() as connection:
        from eduecon_radar.services.jel_enrichment import record_jel_observation

        record_jel_observation(
            connection,
            JelObservation(1, "repec_ideas", "structured_listing", "resolved", ("D03",)),
        )
        row = connection.execute("SELECT jel_codes_json FROM papers WHERE id=1").fetchone()
    assert jel_codes_from_storage(row[0]) == ("D91",)
    # ...but the disagreement is on record.
    with jel_database.connect() as connection:
        sources = {
            item["source_name"]
            for item in connection.execute(
                "SELECT source_name FROM paper_jel_observations WHERE paper_id=1"
            ).fetchall()
        }
    assert sources == {"manual", "repec_ideas"}


def test_absent_at_source_is_recorded_as_an_answer_not_silence(jel_database):
    class AbsentProvider:
        name = "repec_ideas"
        enabled = True
        priority = 70

        def observations_for(self, papers):
            return {
                paper.paper_id: JelObservation(
                    paper.paper_id,
                    self.name,
                    "structured_listing",
                    "absent_at_source",
                    failure_reason="no listing entry matched this paper",
                )
                for paper in papers
            }

    service = JelEnrichmentService(jel_database, [AbsentProvider()])
    stats = service.enrich(service.load_papers())
    assert stats.absent_at_source == 2
    assert stats.stored_updates == 0
    report = audit_jel_enrichment(jel_database.path)
    assert report["papers_with_jel_codes"] == 0
    assert len(report["papers_source_had_none"]) == 2
    assert report["papers_never_asked"] == []
    assert report["by_resolution"]["absent_at_source"] == 2


def test_report_separates_never_asked_from_source_had_none(jel_database):
    report = audit_jel_enrichment(jel_database.path)
    assert report["status"] == "ok"
    assert report["observation_ledger"] == "present"
    assert report["papers_with_jel_codes"] == 0
    assert len(report["papers_never_asked"]) == 2
    assert report["papers_source_had_none"] == []


def test_report_groups_codes_and_counts_provenance(jel_database):
    service = JelEnrichmentService(
        jel_database, [ManualJelProvider({1: ["D91", "D03"], 2: ["J24"]})]
    )
    service.enrich(service.load_papers())
    report = audit_jel_enrichment(jel_database.path)
    assert report["papers_with_jel_codes"] == 2
    assert report["coverage"] == 1.0
    assert {item["code"] for item in report["distinct_codes"]} == {"D91", "D03", "J24"}
    assert report["provenance_current"] == {"manual": 2}


def test_report_degrades_gracefully_without_the_ledger(tmp_path):
    database = Database(tmp_path / "bare.db")
    database.init_schema()
    report = audit_jel_enrichment(database.path)
    assert report["status"] == "partial"
    assert report["observation_ledger"] == "missing"


def test_migration_is_additive_and_idempotent(jel_database):
    from eduecon_radar.services.jel_enrichment import jel_observations_schema_ready

    with jel_database.connect() as connection:
        assert jel_observations_schema_ready(connection)
        connection.executescript(MIGRATION.read_text(encoding="utf-8"))
        assert jel_observations_schema_ready(connection)
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(paper_jel_observations)"
            ).fetchall()
        }
    assert {
        "paper_id",
        "source_name",
        "source_record_url",
        "extraction_method",
        "resolution",
        "codes_json",
        "codes_normalized",
        "raw_evidence",
        "http_status",
        "failure_reason",
        "observed_at",
    } <= columns


def test_observation_ledger_cascades_with_the_paper(jel_database):
    service = JelEnrichmentService(jel_database, [ManualJelProvider({1: ["D91"]})])
    service.enrich([JelPaperInput(1, "10.1016/j.jebo.2026.01.001", "Attention")])
    with jel_database.connect() as connection:
        connection.execute("DELETE FROM papers WHERE id=1")
        remaining = connection.execute(
            "SELECT COUNT(*) FROM paper_jel_observations WHERE paper_id=1"
        ).fetchone()[0]
    assert remaining == 0


def test_provider_failure_does_not_abort_the_run(jel_database):
    class Exploding:
        name = "broken"
        enabled = True
        priority = 1

        def observations_for(self, papers):
            raise RuntimeError("provider down")

    service = JelEnrichmentService(
        jel_database, [Exploding(), ManualJelProvider({2: ["J24"]})]
    )
    stats = service.enrich(service.load_papers())
    assert stats.errors and stats.errors[0].startswith("broken:")
    assert stats.stored_updates == 1


def test_disabled_providers_are_reported_as_skipped(jel_database):
    settings = load_jel_source_settings()
    service = JelEnrichmentService(
        jel_database, list(build_providers(settings=settings))
    )
    stats = service.enrich(service.load_papers())
    assert stats.skipped == {"repec_ideas": 1}
    assert stats.observations_recorded == 0


def test_jel_codes_are_a_paper_fact_not_a_profile_field(jel_database):
    """The stored codes must not reference any profile."""
    JelEnrichmentService(jel_database, [ManualJelProvider({1: ["D91"]})]).enrich(
        jel_database and [JelPaperInput(1, "10.1016/j.jebo.2026.01.001", "Attention")]
    )
    with jel_database.connect() as connection:
        paper_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(papers)").fetchall()
        }
        ledger_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(paper_jel_observations)"
            ).fetchall()
        }
    assert "jel_codes_json" in paper_columns
    assert "profile_id" not in ledger_columns
    assert not any(name.startswith("profile") for name in ledger_columns)


def test_configured_search_codes_are_all_valid_and_canonical():
    settings = load_jel_source_settings()
    assert settings.codes
    assert set(settings.codes) == set(normalize_jel_codes(settings.codes))
    assert all(normalize_jel_code(code) == code for code in settings.codes)


def test_source_settings_surface_the_disabled_by_default_policy():
    settings = load_jel_source_settings()
    assert settings.repec_ideas.enabled is False
    assert settings.repec_ideas.rate_limit_per_second <= 0.5
    assert settings.repec_ideas.extra_delay_seconds > 0
    assert "contact" in settings.repec_ideas.user_agent
    assert settings.version


def test_observation_ledger_rejects_unknown_resolutions(jel_database):
    with jel_database.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """INSERT INTO paper_jel_observations
            (paper_id,source_name,extraction_method,resolution,codes_json,codes_normalized,
             observed_at) VALUES (1,'x','m','invented','[]','','now')"""
        )


def test_ledger_keeps_one_row_per_paper_and_source(jel_database):
    service = JelEnrichmentService(
        jel_database, [ManualJelProvider({1: ["D91"], 2: ["J24"]})]
    )
    service.enrich(service.load_papers())
    with jel_database.connect() as connection:
        rows = connection.execute(
            "SELECT paper_id,source_name,COUNT(*) c FROM paper_jel_observations "
            "GROUP BY paper_id,source_name HAVING c > 1"
        ).fetchall()
        raw = connection.execute(
            "SELECT codes_json FROM paper_jel_observations ORDER BY paper_id"
        ).fetchall()
    assert rows == []
    assert [json.loads(row[0]) for row in raw] == [["D91"], ["J24"]]
