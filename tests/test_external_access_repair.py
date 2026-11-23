from datetime import date

import pytest

from eduecon_radar.services.retrieval_verification import _audit_crossref_gap


class _CrossrefRecords:
    def fetch_work(self, doi: str):
        created = {
            "10.1016/j.econedurev.2026.102818": [2026, 7, 25],
            "10.1016/j.econedurev.2026.102819": [2026, 7, 28],
        }[doi]
        return {
            "DOI": doi,
            "title": [f"Title for {doi}"],
            "published-print": {"date-parts": [[2026, 12]]},
            "issued": {"date-parts": [[2026, 12]]},
            "created": {"date-parts": [created]},
        }


@pytest.mark.parametrize(
    ("doi", "openalex_date", "created"),
    [
        ("10.1016/j.econedurev.2026.102818", "2026-07-25", "2026-07-25"),
        ("10.1016/j.econedurev.2026.102819", "2026-07-28", "2026-07-28"),
    ],
)
def test_known_openalex_gaps_are_explained_by_crossref_date_basis(
    doi: str, openalex_date: str, created: str
) -> None:
    audited = _audit_crossref_gap(
        {
            "doi": doi,
            "title": f"Title for {doi}",
            "publication_date": openalex_date,
            "reason": "possible gap",
        },
        since=date(2026, 5, 2),
        until=date(2026, 8, 2),
        crossref=_CrossrefRecords(),
    )

    assert audited["gap_classification"] == "date_basis_difference"
    assert audited["crossref_dates"] == {
        "published_online": None,
        "published_print": "2026-12",
        "issued": "2026-12",
        "created": created,
    }
    assert "创建日期补充检索修复" in audited["reason"]
