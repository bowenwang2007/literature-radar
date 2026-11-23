from __future__ import annotations

import hashlib
import re
import unicodedata

from eduecon_radar.services.doi import ascii_lower as ascii_lower
from eduecon_radar.services.doi import canonicalize_doi

# Backward-compatible import name. There is one implementation.
normalize_doi = canonicalize_doi


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def fingerprint_paper(title: str, first_author: str, year: int | None, journal_key: str) -> str:
    parts = [
        normalize_title(title),
        normalize_title(first_author),
        str(year or ""),
        journal_key.lower(),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
