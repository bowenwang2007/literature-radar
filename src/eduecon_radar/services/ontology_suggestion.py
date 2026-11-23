from __future__ import annotations

import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
STOPWORDS = {
    "and",
    "for",
    "from",
    "into",
    "of",
    "on",
    "the",
    "to",
    "with",
    "evidence",
    "effects",
    "impact",
}


def suggest_ontology_candidates(
    false_negative_items: list[dict],
    *,
    minimum_frequency: int = 2,
    existing_terms: set[str] | None = None,
) -> list[dict]:
    """Suggest repeated title phrases only; never modifies the ontology."""
    existing = {item.lower().strip() for item in (existing_terms or set())}
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for item in false_negative_items:
        title = str(item.get("title") or item.get("original_title") or "").strip()
        tokens = [
            token.lower()
            for token in TOKEN_RE.findall(title)
            if token.lower() not in STOPWORDS
        ]
        phrases = {
            " ".join(tokens[index : index + width])
            for width in (2, 3)
            for index in range(len(tokens) - width + 1)
        }
        for phrase in phrases:
            if phrase in existing:
                continue
            counts[phrase] += 1
            if len(examples[phrase]) < 3:
                examples[phrase].append(title)
    return [
        {
            "phrase": phrase,
            "frequency": frequency,
            "examples": examples[phrase],
            "status": "candidate_requires_human_verification",
        }
        for phrase, frequency in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
        if frequency >= minimum_frequency
    ]
