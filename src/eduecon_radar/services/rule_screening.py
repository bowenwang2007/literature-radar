from __future__ import annotations

import re

from eduecon_radar.schemas import PaperMetadata, RuleResult

POSITIVE_TERMS = (
    "education",
    "school",
    "student",
    "teacher",
    "classroom",
    "college",
    "university",
    "admission",
    "enrollment",
    "tuition",
    "scholarship",
    "financial aid",
    "curriculum",
    "test score",
    "academic achievement",
    "school choice",
    "peer effect",
    "early childhood",
    "vocational education",
    "human capital",
    "training",
    "degree",
    "graduation",
    "student debt",
)

RISK_PATTERNS = {
    "student_sample_only": r"(?:sample|participants?) (?:of|consisted of|were) .*students?",
    "education_as_control": r"(?:control(?:led)? for|covariates? include).*education",
    "generic_workplace_training": r"(?:employee|workplace|firm).*training",
    "college_affiliation": r"college of [a-z]",
    "human_capital_background": r"human capital (?:theory|framework|background)",
}


def screen_by_rules(paper: PaperMetadata) -> RuleResult:
    text = " ".join(
        [paper.title, paper.abstract or "", " ".join(paper.keywords), " ".join(paper.jel_codes)]
    ).lower()
    positives = [term for term in POSITIVE_TERMS if re.search(rf"\b{re.escape(term)}\b", text)]
    risks = [name for name, pattern in RISK_PATTERNS.items() if re.search(pattern, text)]
    score = min(10.0, len(positives) * 1.25) - len(risks) * 0.5
    # Missing abstracts are retained: a title-only record should not be silently discarded.
    should_screen = bool(positives) or paper.abstract is None
    return RuleResult(
        score=round(score, 2),
        positive_hits=positives,
        risk_hits=risks,
        should_screen=should_screen,
    )
