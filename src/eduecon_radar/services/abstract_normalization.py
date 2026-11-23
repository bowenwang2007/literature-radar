"""Safe abstract parsing, normalization, and quality validation."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass

PARSER_VERSION = "6c.1"
_SPACE = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]*>")
_SCRIPT = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.I | re.S)
_HEADING = re.compile(r"^\s*(abstract|summary)\s*[:.\-]?\s*", re.I)
_NOISE = re.compile(
    r"(?i)(©|copyright\b|all rights reserved|privacy policy|cookie policy|"
    r"navigation|skip to (main )?content|author bio(?:graphy)?|references?\s*$)"
)


@dataclass(frozen=True)
class AbstractQualityResult:
    accepted: bool
    quality_status: str
    rejection_reasons: tuple[str, ...]
    normalized_text: str
    language: str
    word_count: int
    content_sha256: str


def rebuild_inverted_index(index: dict[str, list[int]]) -> str:
    positions = [(position, word) for word, values in index.items() for position in values]
    return " ".join(word for _, word in sorted(positions))


def normalize_abstract(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        if value and all(
            isinstance(positions, list)
            and all(isinstance(position, int) for position in positions)
            for positions in value.values()
        ):
            value = rebuild_inverted_index(value)
        else:
            value = next(
                (
                    value[key]
                    for key in (
                        "abstract",
                        "abstract_en",
                        "description",
                        "summary",
                        "indexed_abstract",
                        "inverted_index",
                        "jats",
                    )
                    if key in value and value[key]
                ),
                None,
            )
            return normalize_abstract(value)
    if isinstance(value, (list, tuple)):
        value = "\n\n".join(
            part for item in value if (part := normalize_abstract(item))
        )
    raw = unicodedata.normalize("NFKC", html.unescape(str(value)))
    raw = _SCRIPT.sub(" ", raw)
    if "<" in raw and ">" in raw:
        try:
            # ElementTree does not resolve external entities or fetch external DTDs.
            root = ET.fromstring(raw)
            raw = " ".join(part for part in root.itertext())
        except ET.ParseError:
            raw = _TAG.sub(" ", raw)
    raw = _HEADING.sub("", raw)
    return _SPACE.sub(" ", raw).strip()


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_abstract_quality(
    value: object | None,
    *,
    title: str | None = None,
    duplicate_hashes: set[str] | None = None,
    minimum_words: int = 20,
    maximum_words: int = 10_000,
) -> AbstractQualityResult:
    text = normalize_abstract(value)
    words = text.split()
    digest = content_sha256(text)
    reasons: list[str] = []
    if not text:
        reasons.append("empty")
    elif len(words) < minimum_words:
        reasons.append("too_short")
    if len(words) > maximum_words:
        reasons.append("too_long")
    if title and _SPACE.sub(" ", title).strip().casefold() == text.casefold():
        reasons.append("title_duplicate")
    if re.fullmatch(r"(?:https?://|doi:\s*|10\.\d{4,9}/)\S+", text, re.I):
        reasons.append("url_or_doi")
    if _NOISE.search(text):
        reasons.append("navigation_or_boilerplate")
    if re.search(r"(?i)^keywords?\s*:", text) and len(words) < 80:
        reasons.append("keyword_list")
    if re.search(r"<[^>]*$|^[^<]*>", text):
        reasons.append("truncated_markup")
    latin = sum(character.isascii() and character.isalpha() for character in text)
    letters = sum(character.isalpha() for character in text)
    language = "en" if not letters or latin / letters >= 0.75 else "und"
    if text and language != "en":
        reasons.append("not_primarily_english")
    if duplicate_hashes and digest in duplicate_hashes:
        reasons.append("duplicate_content")
    accepted = not reasons
    return AbstractQualityResult(
        accepted=accepted,
        quality_status="clean" if accepted else "rejected",
        rejection_reasons=tuple(reasons),
        normalized_text=text,
        language=language,
        word_count=len(words),
        content_sha256=digest,
    )
