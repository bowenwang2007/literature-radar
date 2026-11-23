from __future__ import annotations

from urllib.parse import quote

import httpx


class UnpaywallClient:
    def __init__(self, email: str, client: httpx.Client | None = None) -> None:
        self.email = email
        self.client = client or httpx.Client(timeout=httpx.Timeout(20))

    def find_open_access_url(self, doi: str) -> str | None:
        response = self.client.get(
            f"https://api.unpaywall.org/v2/{quote(doi, safe='')}",
            params={"email": self.email},
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        best = data.get("best_oa_location") or {}
        return best.get("url_for_landing_page") or best.get("url_for_pdf")
