from urllib.parse import parse_qs, urlparse

import pytest

from eduecon_radar.services.deduplication import normalize_doi
from eduecon_radar.services.source_traceability import (
    build_crossref_url,
    build_doi_url,
    build_google_scholar_url,
    build_openalex_work_url,
    build_repec_landing_url,
    resolve_paper_access,
    screening_basis_message,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10.1257/AER.12345", "10.1257/aer.12345"),
        ("https://doi.org/10.1257/AER.12345", "10.1257/aer.12345"),
        ("http://doi.org/10.1257/AER.12345", "10.1257/aer.12345"),
        ("http://dx.doi.org/10.1257/AER.12345", "10.1257/aer.12345"),
        ("https://dx.doi.org/10.1257/AER.12345", "10.1257/aer.12345"),
        ("doi:10.1257/AER.12345", "10.1257/aer.12345"),
        ("DOI:10.1257/AER.12345", "10.1257/aer.12345"),
        ("  DOI:10.1257/AER.12345  ", "10.1257/aer.12345"),
        (None, None),
        ("", None),
        ("   ", None),
    ],
)
def test_normalize_doi_supported_inputs(value, expected):
    assert normalize_doi(value) == expected


def test_doi_url_uses_normalized_and_encoded_path():
    assert build_doi_url(" DOI:10.1257/AER.12345 ") == (
        "https://doi.org/10.1257/aer.12345"
    )
    assert build_doi_url("https://doi.org/10.1000/a b?c#d") is None
    assert "doi.org/https" not in build_doi_url("https://doi.org/10.1000/test")


@pytest.mark.parametrize("value", [None, "", "doi:   "])
def test_doi_url_is_absent_without_a_doi(value):
    assert build_doi_url(value) is None


def test_crossref_url_uses_fully_encoded_normalized_doi():
    assert build_crossref_url("https://dx.doi.org/10.1000/a b?c") is None
    assert build_crossref_url(None) is None
    assert build_crossref_url("") is None


@pytest.mark.parametrize(
    ("doi", "doi_path", "crossref_path"),
    [
        ("10.1000/abc(def)", "10.1000/abc%28def%29", "10.1000%2Fabc%28def%29"),
        ("10.1000/abc.", "10.1000/abc.", "10.1000%2Fabc."),
        ("10.1000/a?b=c", "10.1000/a%3Fb%3Dc", "10.1000%2Fa%3Fb%3Dc"),
        ("10.1000/a#fragment", "10.1000/a%23fragment", "10.1000%2Fa%23fragment"),
        (
            "10.1000/中文测试",
            "10.1000/%E4%B8%AD%E6%96%87%E6%B5%8B%E8%AF%95",
            "10.1000%2F%E4%B8%AD%E6%96%87%E6%B5%8B%E8%AF%95",
        ),
    ],
)
def test_doi_and_crossref_urls_encode_without_losing_suffix_characters(
    doi, doi_path, crossref_path
):
    assert build_doi_url(doi) == f"https://doi.org/{doi_path}"
    assert build_crossref_url(doi) == f"https://api.crossref.org/works/{crossref_path}"


@pytest.mark.parametrize(
    "title",
    [
        "The Economics of Education",
        "Education, Wages & Growth: Evidence?",
        'A "Quoted" Paper Title',
        "教育经济学：政策与结果",
    ],
)
def test_google_scholar_url_round_trips_the_original_title(title):
    url = build_google_scholar_url(title)
    assert url is not None
    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "scholar.google.com"
    assert parsed.path == "/scholar"
    assert parse_qs(parsed.query) == {"q": [title]}
    assert "#" not in parsed.query
    assert '"' not in parsed.query
    assert "教育" not in parsed.query


@pytest.mark.parametrize("title", [None, "", "   "])
def test_google_scholar_url_is_absent_without_a_title(title):
    assert build_google_scholar_url(title) is None


def test_missing_doi_still_allows_google_scholar_search():
    assert build_doi_url(None) is None
    assert build_crossref_url(None) is None
    assert build_google_scholar_url("Original paper title") is not None


def test_access_keeps_doi_separate_and_prefers_verified_repec_landing():
    access = resolve_paper_access(
        doi="10.1016/j.econedurev.2026.102776",
        source_url="https://doi.org/10.1016/j.econedurev.2026.102776",
        source_metadata={
            "abstract_enrichment": {
                "attempts": [
                    {
                        "source": "repec",
                        "source_record_id": (
                            "RePEc:eee:ecoedu:v:112:y:2026:i:c:"
                            "s027277572600018x"
                        ),
                    }
                ]
            }
        },
    )

    assert access.doi_url == "https://doi.org/10.1016/j.econedurev.2026.102776"
    assert access.preferred_label == "RePEc 页面"
    assert access.preferred_url == (
        "https://ideas.repec.org/a/eee/ecoedu/"
        "v112y2026ics027277572600018x.html"
    )


def test_access_never_promotes_blocked_publisher_url_over_doi():
    access = resolve_paper_access(
        doi="10.1016/j.econedurev.2026.102818",
        open_access_url=(
            "https://www.sciencedirect.com/science/article/pii/"
            "S0272775726000609/pdf"
        ),
        source_url="https://doi.org/10.1016/j.econedurev.2026.102818",
    )

    assert access.doi_url == "https://doi.org/10.1016/j.econedurev.2026.102818"
    assert access.preferred_url is None


def test_access_uses_open_repository_then_openalex_record():
    repository = resolve_paper_access(
        doi="10.1000/example",
        open_access_url="https://repository.example.edu/paper/1",
        openalex_id="W7171263386",
    )
    assert repository.preferred_label == "开放版本"
    assert repository.preferred_url == "https://repository.example.edu/paper/1"

    openalex = resolve_paper_access(
        doi="10.1000/example",
        openalex_id="https://openalex.org/W7171263386",
    )
    assert openalex.preferred_label == "OpenAlex 记录"
    assert openalex.preferred_url == "https://openalex.org/W7171263386"
    assert build_openalex_work_url("not-a-work") is None
    assert build_repec_landing_url("RePEc:unsafe") is None


def test_screening_basis_without_a_screening():
    assert screening_basis_message(
        has_screening=False,
        model_name=None,
        has_abstract=True,
    ) == ("No screening assessment available.", "尚无筛选判断。")


def test_screening_basis_for_rules_only_does_not_claim_ai_input():
    english, chinese = screening_basis_message(
        has_screening=True,
        model_name="rules-only",
        has_abstract=True,
    )
    assert "no AI semantic assessment" in english
    assert "未执行AI语义判断" in chinese
    assert "Crossref abstract" not in english


def test_screening_basis_for_ai_with_crossref_abstract():
    english, chinese = screening_basis_message(
        has_screening=True,
        model_name="deepseek-model",
        has_abstract=True,
    )
    assert "available bibliographic metadata and the Crossref abstract" in english
    assert "现有书目信息和Crossref摘要" in chinese
    assert "Full text was not verified" in english


def test_screening_basis_for_ai_without_abstract_mentions_other_metadata():
    english, chinese = screening_basis_message(
        has_screening=True,
        model_name="deepseek-model",
        has_abstract=False,
    )
    assert "title and available bibliographic metadata" in english
    assert "No abstract or full text was verified" in english
    assert "题名和现有书目信息" in chinese
