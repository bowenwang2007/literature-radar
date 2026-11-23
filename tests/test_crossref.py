from datetime import date

import httpx
import pytest

from eduecon_radar.services.crossref import (
    CROSSREF_SELECT_FIELDS,
    CrossrefClient,
    CrossrefRequestError,
)


def test_crossref_fetch_parses_successful_mock_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "message": {
                    "items": [
                        {
                            "DOI": "10.1234/example",
                            "title": ["Education and earnings"],
                            "author": [{"given": "Ada", "family": "Scholar"}],
                            "published-online": {"date-parts": [[2025, 8, 1]]},
                        }
                    ],
                    "next-cursor": "next-page",
                }
            },
        )

    client = CrossrefClient(httpx.Client(transport=httpx.MockTransport(handler)))
    papers = client.fetch("0002-8282", date(2025, 7, 18))

    assert len(papers) == 1
    assert papers[0].doi == "10.1234/example"
    assert papers[0].title == "Education and earnings"
    assert papers[0].authors == ["Ada Scholar"]


def test_crossref_request_uses_only_legal_pagination_and_publication_date_params():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            request=request,
            json={"message": {"items": [], "next-cursor": "unused"}},
        )

    client = CrossrefClient(httpx.Client(transport=httpx.MockTransport(handler)))
    client.fetch("0033-5533", date(2025, 7, 18))

    params = captured[0].url.params
    assert set(params.keys()) == {"filter", "rows", "cursor", "select"}
    assert params["filter"] == "from-pub-date:2025-07-18"
    assert params["rows"] == "100"
    assert params["cursor"] == "*"
    assert params["select"] == CROSSREF_SELECT_FIELDS
    assert "abstract" in params["select"].split(",")
    assert "cursor-max" not in params


def test_crossref_request_supports_a_bounded_publication_date_range():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200, request=request, json={"message": {"items": [], "next-cursor": None}}
        )

    client = CrossrefClient(httpx.Client(transport=httpx.MockTransport(handler)))
    client.fetch("0033-5533", date(2026, 1, 1), until=date(2026, 1, 31))

    assert captured[0].url.params["filter"] == (
        "from-pub-date:2026-01-01,until-pub-date:2026-01-31"
    )
    assert captured[1].url.params["filter"] == (
        "from-created-date:2026-01-01,until-created-date:2026-01-31"
    )


def test_bounded_fetch_unions_created_date_records_and_deduplicates_doi():
    def handler(request: httpx.Request) -> httpx.Response:
        common = {
            "DOI": "10.1016/j.econedurev.2026.102818",
            "title": ["Online-first record with a future print issue"],
            "created": {"date-parts": [[2026, 7, 25]]},
            "published-print": {"date-parts": [[2026, 12]]},
        }
        items = (
            [common]
            if "created-date" in request.url.params["filter"]
            else []
        )
        return httpx.Response(
            200,
            request=request,
            json={"message": {"items": items, "next-cursor": None}},
        )

    client = CrossrefClient(httpx.Client(transport=httpx.MockTransport(handler)))
    papers = client.fetch(
        "0272-7757",
        date(2026, 5, 2),
        until=date(2026, 8, 2),
        limit=100,
    )

    assert [paper.doi for paper in papers] == [
        "10.1016/j.econedurev.2026.102818"
    ]
    assert papers[0].online_date is None  # DOI creation is not online publication.
    assert papers[0].issue_date == date(2026, 12, 1)
    assert papers[0].source_metadata["eduecon_retrieval_date_bases"] == ["created"]
    assert client.last_fetch_stats["date_bases"] == ["pub", "created"]


def test_crossref_http_400_includes_api_error_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            request=request,
            json={"status": "failed", "message": "filter parameter is invalid"},
        )

    client = CrossrefClient(httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(CrossrefRequestError, match="HTTP 400.*filter parameter is invalid"):
        client.fetch("0002-8282", date(2025, 7, 18))


def test_created_query_survives_publication_query_failure():
    def handler(request):
        if "from-pub-date" in request.url.params["filter"]:
            raise httpx.ReadTimeout("publication index unavailable", request=request)
        return httpx.Response(200, json={"message": {"items": [{
            "DOI": "10.1000/created-only", "title": ["Created-only record"],
            "created": {"date-time": "2026-07-25T12:00:00Z"},
        }]}}, request=request)

    client = CrossrefClient(httpx.Client(transport=httpx.MockTransport(handler)))
    papers = client.fetch("0272-7757", date(2026, 7, 1), until=date(2026, 7, 31))
    assert len(papers) == 1
    assert papers[0].online_date is None
    assert client.last_fetch_stats["stopped_reason"] == "provider_error"
    assert client.last_fetch_stats["error_summary"]
