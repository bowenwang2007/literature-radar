from __future__ import annotations

import html
import re
import unicodedata
from urllib.parse import quote, unquote

DOI_PREFIX = re.compile(
    r"^(?:https?://(?:(?:dx\.)?doi\.org)/|doi\s*:\s*)",
    re.IGNORECASE,
)
DOI_SHAPE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
ASCII_LOWER_TRANSLATION = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
)


def ascii_lower(value: str) -> str:
    """Lower only ASCII letters; DOI suffixes may contain case-sensitive Unicode."""
    return value.translate(ASCII_LOWER_TRANSLATION)


def canonicalize_doi(value: str | None) -> str | None:
    """Return the storage/lookup DOI without guessing away legal suffix characters.

    Surrounding Unicode whitespace, HTML escaping, URL encoding and recognized
    resolver prefixes are representation details. Punctuation inside the DOI,
    including a final period or parenthesis, is preserved because it can be legal.
    """
    if value is None:
        return None
    normalized = unicodedata.normalize("NFC", html.unescape(str(value)))
    normalized = unquote(normalized).strip()
    normalized = DOI_PREFIX.sub("", normalized, count=1).strip()
    normalized = ascii_lower(normalized)
    return normalized if validate_doi_syntax(normalized) else None


def validate_doi_syntax(value: str | None) -> bool:
    if not value or len(value) > 2048 or any(character.isspace() for character in value):
        return False
    return DOI_SHAPE.fullmatch(value) is not None


def normalize_doi_for_lookup(value: str | None) -> str | None:
    canonical = canonicalize_doi(value)
    if canonical is not None:
        return canonical
    if value is None:
        return None
    # Link builders may safely percent-encode a legacy value for manual lookup,
    # while canonical storage still rejects embedded whitespace.
    normalized = unicodedata.normalize("NFC", html.unescape(str(value)))
    normalized = unquote(normalized).strip()
    normalized = DOI_PREFIX.sub("", normalized, count=1).strip()
    normalized = ascii_lower(normalized)
    return normalized if normalized.casefold().startswith("10.") and "/" in normalized else None


def doi_url(value: str | None) -> str | None:
    canonical = canonicalize_doi(value)
    if canonical is None:
        return None
    return f"https://doi.org/{quote(canonical, safe='/')}"
