import json
from pathlib import Path

import httpx
import pytest

from eduecon_radar.db import Database, utcnow
from eduecon_radar.services.abstract_enrichment import (
    AbstractCandidate,
    AbstractEnrichmentService,
    AbstractFetchContext,
    AbstractIdentityMismatch,
    AbstractInvalidContent,
    AbstractNotFound,
    AbstractRateLimited,
    AbstractSourceAdapter,
    AbstractTemporaryFailure,
    AbstractWorkerInterrupted,
    CrossrefEnrichmentAdapter,
    DOAJEnrichmentAdapter,
    EnrichmentPaperInput,
    OpenAlexEnrichmentAdapter,
    RePEcOAIAdapter,
    SemanticScholarEnrichmentAdapter,
    normalize_author_name,
    normalize_doi,
    normalize_title,
    validate_candidate_identity,
)
from eduecon_radar.services.abstract_source_config import load_abstract_sources
from eduecon_radar.services.resilient_http import ResilientHttpClient


@pytest.fixture
def paper():
    return EnrichmentPaperInput(
        1,
        "https://doi.org/10.1000/ABC",
        "Returns to Education",
        ["Jane Smith"],
        "JEE",
        2024,
        "1234",
    )


def test_identity_normalizers_and_doi_match(paper):
    assert normalize_doi(" DOI: 10.1000/ABC. ") == "10.1000/abc."
    assert normalize_title("Returns: to Education!") == "returns to education"
    assert normalize_author_name("Jane Smith") == "smith"
    result = validate_candidate_identity(
        paper, AbstractCandidate("word " * 30, "x", raw_doi="10.1000/abc")
    )
    assert result.identity_match_method == "doi" and result.identity_confidence == 1


def test_identity_rejects_doi_year_author_and_fuzzy_conflicts(paper):
    with pytest.raises(AbstractIdentityMismatch):
        validate_candidate_identity(
            paper, AbstractCandidate("word " * 30, "x", raw_doi="10.2/wrong")
        )


def _adapter(kind, handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    reliable = ResilientHttpClient(
        client, max_retries=0, sleeper=lambda _: None, rate_limit_per_second=0
    )
    return {
        "crossref": CrossrefEnrichmentAdapter,
        "openalex": OpenAlexEnrichmentAdapter,
        "semantic_scholar": SemanticScholarEnrichmentAdapter,
        "doaj": DOAJEnrichmentAdapter,
    }[kind](client=reliable)


def _payload(kind, *, abstract=True):
    text = "This study estimates returns to education using longitudinal student data. " * 5
    if kind == "crossref":
        return {
            "message": {
                "DOI": "10.1000/abc",
                "title": ["Returns to Education"],
                "abstract": text if abstract else None,
            }
        }
    if kind == "openalex":
        index = {word: [position] for position, word in enumerate(text.split())}
        return {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "doi": "https://doi.org/10.1000/abc",
                    "display_name": "Returns to Education",
                    "abstract_inverted_index": index if abstract else None,
                }
            ]
        }
    if kind == "semantic_scholar":
        return {
            "paperId": "p1",
            "externalIds": {"DOI": "10.1000/abc"},
            "title": "Returns to Education",
            "abstract": text if abstract else None,
        }
    return {
        "results": [
            {
                "bibjson": {
                    "title": "Returns to Education",
                    "identifier": [{"type": "doi", "id": "10.1000/abc"}],
                    "abstract": text if abstract else None,
                }
            }
        ]
    }


@pytest.mark.parametrize("kind", ["crossref", "openalex", "semantic_scholar", "doaj"])
def test_each_enabled_adapter_hit_missing_and_provenance(kind, paper):
    adapter = _adapter(
        kind, lambda request: httpx.Response(200, json=_payload(kind), request=request)
    )
    candidate = adapter.fetch(paper)
    checked = validate_candidate_identity(paper, candidate)
    assert checked.source == kind and checked.source_url
    assert checked.content_sha256 and checked.identity_match_method == "doi"

    missing = _adapter(
        kind,
        lambda request: httpx.Response(200, json=_payload(kind, abstract=False), request=request),
    )
    missing_candidate = missing.fetch(paper)
    assert missing_candidate is not None
    assert missing_candidate.text == ""


@pytest.mark.parametrize("kind", ["crossref", "openalex", "semantic_scholar", "doaj"])
@pytest.mark.parametrize(
    ("status", "error"),
    [
        (404, AbstractNotFound),
        (429, AbstractRateLimited),
        (500, AbstractTemporaryFailure),
        (502, AbstractTemporaryFailure),
        (503, AbstractTemporaryFailure),
    ],
)
def test_each_enabled_adapter_http_failures(kind, status, error, paper):
    adapter = _adapter(kind, lambda request: httpx.Response(status, json={}, request=request))
    with pytest.raises(error):
        adapter.fetch(paper)


@pytest.mark.parametrize("kind", ["crossref", "openalex", "semantic_scholar", "doaj"])
def test_each_enabled_adapter_timeout_and_invalid_json(kind, paper):
    timeout = _adapter(kind, lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("timeout")))
    with pytest.raises(AbstractTemporaryFailure):
        timeout.fetch(paper)
    invalid = _adapter(
        kind,
        lambda request: httpx.Response(
            200, content=b"not-json", headers={"content-type": "application/json"}, request=request
        ),
    )
    with pytest.raises(AbstractInvalidContent):
        invalid.fetch(paper)


@pytest.mark.parametrize("kind", ["crossref", "openalex", "semantic_scholar", "doaj"])
def test_each_enabled_adapter_missing_required_shape(kind, paper):
    adapter = _adapter(kind, lambda request: httpx.Response(200, json={}, request=request))
    assert adapter.fetch(paper) is None


def test_repec_oai_uses_crossref_pii_and_exact_handle_without_page_scraping() -> None:
    shared: dict[str, object] = {}
    context = AbstractFetchContext(shared_metadata=shared)
    paper = EnrichmentPaperInput(
        8,
        "10.1016/j.econedurev.2026.102785",
        "Tracking when ranking matters",
        ["Fanny Landaud", "Éric Maurin"],
        "Economics of Education Review",
        2026,
        "0272-7757",
        "112",
        "C",
    )
    crossref_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                request=request,
                json={
                    "message": {
                        "DOI": paper.doi,
                        "title": [paper.title],
                        "abstract": None,
                        "alternative-id": ["S0272775726000270"],
                    }
                },
            )
        )
    )
    CrossrefEnrichmentAdapter(client=crossref_client).fetch(paper, context)
    requests: list[httpx.Request] = []
    abstract = (
        "This paper investigates the effect of grouping students by prior achievement "
        "into different classes and estimates consequences for performance and inequality. "
        * 3
    )
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
      <GetRecord><record><metadata>
        <dc xmlns="http://www.openarchives.org/OAI/2.0/oai_dc/"
            xmlns:d="http://purl.org/dc/elements/1.1/">
          <d:title>{paper.title}</d:title>
          <d:description>{abstract}</d:description>
          <d:description>Ability tracking; Higher education</d:description>
          <d:creator>Landaud, Fanny</d:creator>
          <d:creator>Maurin, Éric</d:creator>
        </dc>
      </metadata></record></GetRecord>
    </OAI-PMH>"""

    def repec_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            text=xml,
            headers={"content-type": "text/xml"},
        )

    adapter = RePEcOAIAdapter(
        client=httpx.Client(transport=httpx.MockTransport(repec_handler)),
        crossref_client=crossref_client,
    )
    candidate = adapter.fetch(paper, context)
    assert candidate is not None
    assert candidate.text == abstract.strip()
    assert candidate.raw_doi == paper.doi
    assert candidate.identity_match_method == "doi_via_crossref_pii_and_repec_handle"
    assert candidate.source_record_id == (
        "RePEc:eee:ecoedu:v:112:y:2026:i:c:s0272775726000270"
    )
    assert requests[0].url.host == "oai.repec.org"
    assert requests[0].url.params["identifier"] == f"oai:{candidate.source_record_id}"


def test_repec_oai_skips_unsupported_journal_without_network() -> None:
    def forbidden(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("unsupported journals must not trigger a request")

    adapter = RePEcOAIAdapter(
        client=httpx.Client(transport=httpx.MockTransport(forbidden)),
        crossref_client=httpx.Client(transport=httpx.MockTransport(forbidden)),
    )
    paper = EnrichmentPaperInput(
        1,
        "10.1000/example",
        "A sufficiently descriptive paper title",
        ["Jane Smith"],
        "Unmapped Journal",
        2026,
        "0000-0000",
        "1",
        "1",
    )
    assert adapter.fetch(paper, AbstractFetchContext()) is None


def test_adapter_candidate_doi_and_title_conflicts_are_rejected(paper):
    for candidate in (
        AbstractCandidate("word " * 30, "crossref", raw_doi="10.9/conflict"),
        AbstractCandidate(
            "word " * 30,
            "crossref",
            raw_title="Unrelated auction theory",
            raw_year=2024,
            raw_authors=("Jane Smith",),
        ),
    ):
        with pytest.raises(AbstractIdentityMismatch):
            validate_candidate_identity(paper, candidate)
    with pytest.raises(AbstractIdentityMismatch):
        validate_candidate_identity(
            paper,
            AbstractCandidate(
                "word " * 30, "x", raw_title=paper.title, raw_year=2020, raw_authors=("Jane Smith",)
            ),
        )


def _paper_database(tmp_path, *, abstract=None, manual=False):
    database = Database(tmp_path / "enrichment.db")
    database.init_schema()
    now = utcnow()
    metadata = (
        '{"abstract_source":"manual","abstract_manual_url":"https://example.org/a"}'
        if manual
        else "{}"
    )
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO journals(id,name,short_name,issn,priority,created_at,updated_at) "
            "VALUES(1,'Journal','JEE','1234','high',?,?)",
            (now, now),
        )
        connection.execute(
            "INSERT INTO papers(id,doi,fingerprint,title,abstract,authors_json,journal_id,"
            "issue_date,source_metadata_json,first_seen_at,last_seen_at,created_at,updated_at) "
            "VALUES(1,'10.1000/abc','fp','Returns to Education',?,? ,1,"
            "'2024-01-01',?,?,?,?,?)",
            (abstract, '["Jane Smith"]', metadata, now, now, now, now),
        )
    return database


def test_single_source_failure_continues_and_success_provenance_is_saved(tmp_path):
    class Failing(AbstractSourceAdapter):
        source_name = "failing"

        def fetch(self, paper, context=None):
            raise AbstractTemporaryFailure("offline")

    class Success(AbstractSourceAdapter):
        source_name = "crossref"

        def fetch(self, paper, context=None):
            return AbstractCandidate(
                "This study estimates returns to education with longitudinal student data. " * 5,
                self.source_name,
                source_url="https://api.crossref.org/works/x",
                raw_doi="10.1000/abc",
            )

    database = _paper_database(tmp_path)
    stats = AbstractEnrichmentService(database, adapters=[Failing(), Success()]).enrich_papers((1,))
    assert stats.enriched == 1 and stats.by_source == {"crossref": 1}
    with database.connect() as connection:
        row = connection.execute(
            "SELECT abstract,source_metadata_json FROM papers WHERE id=1"
        ).fetchone()
    assert row["abstract"].startswith("This study")
    assert '"source": "crossref"' in row["source_metadata_json"]
    assert '"error_type": "AbstractTemporaryFailure"' in row["source_metadata_json"]


def test_doi_not_found_uses_strict_title_fallback_and_records_query_modes(tmp_path):
    class TitleFallback(AbstractSourceAdapter):
        source_name = "crossref"

        def fetch(self, paper, context=None):
            return None

        def fetch_title_candidates(self, paper, context=None):
            return (
                AbstractCandidate(
                    "This study estimates returns to education with longitudinal student data. "
                    * 5,
                    self.source_name,
                    raw_title=paper.title,
                    raw_doi=paper.doi,
                    raw_year=paper.year,
                    raw_authors=tuple(paper.authors),
                    raw_journal=paper.journal,
                ),
            )

    database = _paper_database(tmp_path)
    stats = AbstractEnrichmentService(database, adapters=[TitleFallback()]).enrich_papers((1,))

    assert stats.enriched == 1
    with database.connect() as connection:
        row = connection.execute(
            "SELECT abstract,source_metadata_json FROM papers WHERE id=1"
        ).fetchone()
    audit = json.loads(row["source_metadata_json"])["abstract_enrichment"]
    assert [item["query_mode"] for item in audit["attempts"]] == ["doi", "title"]
    assert (
        audit["attempts"][1]["identity_result"]
        == "doi_plus_strict_title_author_year_journal"
    )


def test_real_crossref_adapter_404_then_calls_title_endpoint(tmp_path):
    requests: list[httpx.Request] = []
    abstract = (
        "This study estimates returns to education with longitudinal student data. " * 5
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params.get("query.title"):
            return httpx.Response(
                200,
                json={
                    "message": {
                        "items": [
                            {
                                "DOI": "10.1000/abc",
                                "title": ["Returns to Education"],
                                "abstract": abstract,
                                "author": [{"given": "Jane", "family": "Smith"}],
                                "container-title": ["Journal"],
                                "published": {"date-parts": [[2024]]},
                            }
                        ]
                    }
                },
                request=request,
            )
        return httpx.Response(404, json={"message": "not found"}, request=request)

    database = _paper_database(tmp_path)
    adapter = _adapter("crossref", handler)
    stats = AbstractEnrichmentService(database, adapters=[adapter]).enrich_papers((1,))

    assert stats.enriched == 1
    assert len(requests) == 2
    assert requests[1].url.params["query.title"] == "Returns to Education"


def test_title_fallback_rejects_low_confidence_candidate_without_saving(tmp_path):
    class BadTitleFallback(AbstractSourceAdapter):
        source_name = "crossref"

        def fetch(self, paper, context=None):
            return None

        def fetch_title_candidates(self, paper, context=None):
            return (
                AbstractCandidate(
                    "This is a sufficiently long abstract about an unrelated auction model. " * 5,
                    self.source_name,
                    raw_title="Unrelated auction theory",
                    raw_doi=paper.doi,
                    raw_year=paper.year,
                    raw_authors=tuple(paper.authors),
                    raw_journal=paper.journal,
                ),
            )

    database = _paper_database(tmp_path)
    stats = AbstractEnrichmentService(database, adapters=[BadTitleFallback()]).enrich_papers((1,))

    assert stats.failed == 1
    with database.connect() as connection:
        row = connection.execute(
            "SELECT abstract,source_metadata_json FROM papers WHERE id=1"
        ).fetchone()
    audit = json.loads(row["source_metadata_json"])["abstract_enrichment"]
    assert row["abstract"] is None
    assert audit["attempts"][1]["status"] == "low_confidence_match"
    assert audit["attempts"][1]["rejected_candidates"][0]["error_category"] == "identity_mismatch"


@pytest.mark.parametrize(
    ("error", "category", "worker_interrupted"),
    [
        (AbstractRateLimited("slow down", http_status=429, retry_after=30), "rate_limited", False),
        (AbstractTemporaryFailure("timeout"), "temporary_failure", False),
        (AbstractWorkerInterrupted("page closed"), "worker_interrupted", True),
    ],
)
def test_retryable_failures_keep_actionable_audit(
    tmp_path, error, category, worker_interrupted
):
    class Retryable(AbstractSourceAdapter):
        source_name = "crossref"

        def fetch(self, paper, context=None):
            raise error

    database = _paper_database(tmp_path)
    stats = AbstractEnrichmentService(database, adapters=[Retryable()]).enrich_papers((1,))

    assert stats.failed == 1
    with database.connect() as connection:
        metadata = json.loads(
            connection.execute(
                "SELECT source_metadata_json FROM papers WHERE id=1"
            ).fetchone()[0]
        )
    audit = metadata["abstract_enrichment"]
    attempt = audit["attempts"][0]
    assert audit["execution_state"] == "retryable_error"
    assert attempt["error_category"] == category
    assert attempt["retryable"] is True
    assert attempt["worker_interrupted"] is worker_interrupted
    assert attempt["next_retry_at"]
    if category == "rate_limited":
        assert attempt["http_status"] == 429
        assert attempt["retry_after"] == 30


def test_manual_abstract_and_url_are_never_overwritten_by_retry(tmp_path):
    database = _paper_database(tmp_path, abstract="Human supplied English abstract", manual=True)

    class Unexpected(AbstractSourceAdapter):
        source_name = "unexpected"

        def fetch(self, paper, context=None):
            raise AssertionError("adapter must not be called")

    stats = AbstractEnrichmentService(database, adapters=[Unexpected()]).enrich_papers((1,))
    assert stats.skipped_have_abstract == 1
    with database.connect() as connection:
        row = connection.execute(
            "SELECT abstract,source_metadata_json FROM papers WHERE id=1"
        ).fetchone()
    assert row["abstract"] == "Human supplied English abstract"
    assert "https://example.org/a" in row["source_metadata_json"]


def test_shipped_config_has_no_retired_or_disabled_provider() -> None:
    """One provider list, all of it live: retired shims are gone, not silenced."""
    root = Path(__file__).resolve().parents[1]
    config = load_abstract_sources(root / "config" / "abstract_sources.yaml")
    assert [item.name for item in config.sources] == [
        "manual",
        "crossref",
        "repec",
        "openalex",
        "semantic_scholar",
        "doaj",
        "openaire",
    ]
    assert all(item.enabled for item in config.sources)
    assert all(item.feature_status == "stable" for item in config.sources)
    import eduecon_radar.services.abstract_enrichment as enrichment

    assert not hasattr(enrichment, "PublisherMetadataAdapter")
