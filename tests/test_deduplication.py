import pytest

from eduecon_radar.services.deduplication import ascii_lower, fingerprint_paper, normalize_doi


def test_normalize_doi():
    assert normalize_doi(" DOI: 10.1000/ABC. ") == "10.1000/abc."
    assert normalize_doi("https://doi.org/10.1000/ABC)") == "10.1000/abc)"
    assert normalize_doi(None) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        (" 10.1000/xyz123 ", "10.1000/xyz123"),
        ("https://doi.org/10.1000/xyz123", "10.1000/xyz123"),
        ("http://doi.org/10.1000/xyz123", "10.1000/xyz123"),
        ("https://dx.doi.org/10.1000/xyz123", "10.1000/xyz123"),
        ("http://dx.doi.org/10.1000/xyz123", "10.1000/xyz123"),
        ("DOI: 10.1000/xyz123", "10.1000/xyz123"),
        ("doi:10.1000/xyz123", "10.1000/xyz123"),
    ],
)
def test_normalize_doi_whitespace_and_prefixes(value, expected):
    assert normalize_doi(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "10.1000/abc(def)",
        "10.1000/abc)",
        "10.1000/abc.",
        "10.1000/foo:",
        "10.1000/a;b",
        "10.1000/a?b=c",
        "10.1000/a#fragment",
        "10.1000/a]",
        "10.1000/a}",
    ],
)
def test_normalize_doi_preserves_suffix_punctuation(value):
    assert normalize_doi(value) == value


def test_ascii_lower_changes_only_ascii_letters():
    assert ascii_lower("10.1000/ABC.Def") == "10.1000/abc.def"
    assert normalize_doi("DOI:10.1000/ABC") == "10.1000/abc"
    assert normalize_doi("10.1000/ÄBC") == "10.1000/Äbc"
    assert normalize_doi("10.1000/中文测试") == "10.1000/中文测试"


def test_fingerprint_is_stable_under_title_formatting():
    left = fingerprint_paper("School Choice: Evidence!", "Jane Doe", 2025, "AER")
    right = fingerprint_paper(" school choice — evidence ", "JANE DOE", 2025, "aer")
    assert left == right
