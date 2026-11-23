"""Profile quality evaluation — a light workflow, not a governance platform.

What it does
------------
Three small steps, each one file in and one file out:

1. **sample**  pick papers whose automatic judgement is worth a human look and
   write an annotation sheet (CSV) with the space to record the human call;
2. **compare** join the returned sheet against the stored automatic judgement
   and report agreement, false positives, false negatives and the ambiguous
   boundary cases;
3. **stratify** the same numbers by screening basis, because a title-only
   judgement and a title-and-abstract judgement are not the same claim.

What it deliberately does not do
--------------------------------
No thresholds are tuned, no prompts are edited, no labels are written back into
``paper_profile_reviews``.  A handful of disagreements is not evidence that a
prompt is wrong, and this module exists to make that evidence visible — not to
act on it.  Acting is a human decision made after reading the report.

Independence is a design constraint, not a courtesy: the sampler exports the
automatic judgement so disagreements are computable, and ``--blind`` withholds
it so a human can annotate without anchoring.  Either way the importer reads only
``paper_id`` and ``human_decision`` from the returned file.
"""

from __future__ import annotations

import csv
import io
import json
import random
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: The decisions a human may record.  ``skip`` is honest abstention: a paper the
#: annotator could not judge from what is available.  Abstentions are reported
#: separately and never counted as agreement.
HUMAN_DECISIONS = ("relevant", "irrelevant", "uncertain", "skip")

#: Automatic decisions as stored by the screening service.
AUTO_DECISIONS = ("relevant", "irrelevant", "uncertain", "insufficient_evidence")

#: How the automatic judgement was reached.
SCREENING_BASES = ("title_only", "title_and_abstract", "deterministic_metadata_rule")

SHEET_COLUMNS = (
    "profile_id",
    "paper_id",
    "journal",
    "publication_date",
    "has_abstract",
    "title",
    "abstract",
    # ---- for the annotator -------------------------------------------------
    "human_decision",
    "human_note",
    # ---- provenance of the sample (withheld when --blind) ------------------
    "auto_decision",
    "auto_basis",
    "auto_confidence",
    "auto_prompt_version",
    "sample_stratum",
    "sample_reason",
)


@dataclass(frozen=True)
class AssessmentRow:
    profile_id: str
    paper_id: int
    decision: str
    screening_basis: str
    confidence: str | None
    prompt_version: str | None
    model_name: str | None
    title: str
    abstract: str | None
    journal: str
    publication_date: str | None
    classification: dict[str, Any] = field(default_factory=dict)

    @property
    def basis(self) -> str:
        return self.screening_basis or "unrecorded"

    @property
    def has_abstract(self) -> bool:
        return bool((self.abstract or "").strip())


def load_assessments(
    database: Path,
    *,
    profile_id: str | None = None,
    paper_ids: list[int] | None = None,
) -> list[AssessmentRow]:
    """Read stored automatic judgements, newest row per (profile, paper)."""
    uri = database.resolve().as_uri() + "?mode=ro"
    clauses: list[str] = []
    params: list[Any] = []
    if profile_id:
        clauses.append("a.profile_id = ?")
        params.append(profile_id)
    if paper_ids:
        clauses.append("a.paper_id IN ({})".format(",".join("?" * len(paper_ids))))
        params.extend(int(item) for item in paper_ids)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT a.profile_id,a.paper_id,a.decision,a.screening_basis,a.classification_json,
               a.model_name,a.prompt_version,p.title,p.abstract,p.issue_date,p.online_date,
               j.short_name journal
        FROM paper_profile_assessments a
        JOIN papers p ON p.id=a.paper_id
        JOIN journals j ON j.id=p.journal_id
        {where}
        ORDER BY a.profile_id,a.paper_id
    """  # noqa: S608 - clause text is built from fixed strings, values are bound
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(sql, params).fetchall()
    assessments: list[AssessmentRow] = []
    for row in rows:
        try:
            classification = json.loads(row["classification_json"] or "{}")
        except (TypeError, ValueError):
            classification = {}
        if not isinstance(classification, dict):
            classification = {}
        assessments.append(
            AssessmentRow(
                profile_id=str(row["profile_id"]),
                paper_id=int(row["paper_id"]),
                decision=str(row["decision"] or "unrecorded"),
                screening_basis=str(row["screening_basis"] or ""),
                confidence=(
                    str(classification.get("confidence"))
                    if classification.get("confidence")
                    else None
                ),
                prompt_version=row["prompt_version"],
                model_name=row["model_name"],
                title=str(row["title"]),
                abstract=row["abstract"],
                journal=str(row["journal"]),
                publication_date=row["issue_date"] or row["online_date"],
                classification=classification,
            )
        )
    return assessments


def sample_assessments(
    assessments: list[AssessmentRow],
    *,
    per_decision: int = 8,
    per_basis: int = 0,
    include_ids: list[int] | None = None,
    only_ids: list[int] | None = None,
    seed: int = 20260925,
) -> list[AssessmentRow]:
    """Pick a sample worth annotating.

    Two independent draws, so the sheet covers both questions a small sample can
    answer honestly:

    * ``per_decision`` — a balanced draw per automatic decision, which is what
      makes false positives and false negatives *findable*.  Note that a balanced
      draw is deliberately **not** a random sample of the corpus, so agreement
      computed on it is a property of the sample, not a precision estimate for
      the whole library.  ``compare`` states the sampling mode in its report.
    * ``per_basis`` — a balanced draw per screening basis, to see whether
      title-only judgements behave differently from full ones.
    """
    ordered = sorted(assessments, key=lambda item: (item.profile_id, item.paper_id))
    if only_ids:
        wanted = {int(item) for item in only_ids}
        ordered = [item for item in ordered if item.paper_id in wanted]
    rng = random.Random(seed)
    chosen: dict[tuple[str, int], AssessmentRow] = {}

    def take(pool: list[AssessmentRow], count: int) -> None:
        if count <= 0 or not pool:
            return
        picked = rng.sample(pool, min(count, len(pool)))
        for item in picked:
            chosen[(item.profile_id, item.paper_id)] = item

    by_decision: dict[str, list[AssessmentRow]] = {}
    for item in ordered:
        by_decision.setdefault(item.decision, []).append(item)
    for pool in by_decision.values():
        take(pool, per_decision)

    if per_basis:
        by_basis: dict[str, list[AssessmentRow]] = {}
        for item in ordered:
            by_basis.setdefault(item.basis, []).append(item)
        for pool in by_basis.values():
            take(pool, per_basis)

    for identifier in include_ids or []:
        for item in ordered:
            if item.paper_id == int(identifier):
                chosen[(item.profile_id, item.paper_id)] = item

    return sorted(chosen.values(), key=lambda item: (item.profile_id, item.paper_id))


def sample_reason(item: AssessmentRow) -> str:
    """Why this row is worth a look — a hint for the annotator, not a label."""
    reasons: list[str] = []
    if item.decision == "uncertain":
        reasons.append("auto_uncertain")
    if item.basis == "title_only":
        reasons.append("title_only_judgement")
    if item.basis == "deterministic_metadata_rule":
        reasons.append("rule_based_judgement")
    if (item.confidence or "").casefold() == "low":
        reasons.append("low_confidence")
    if not item.has_abstract:
        reasons.append("no_abstract")
    return ",".join(reasons) or "random_within_decision"


# --------------------------------------------------------------------------- #
# Sampling design for a corpus-scale annotation effort
# --------------------------------------------------------------------------- #

#: The two kinds of draw in a stratified sheet, and what each one is *for*.
#: Keeping them apart is what makes the numbers honest: ``representative`` rows
#: support precision/recall for the library, ``boundary`` rows exist to find the
#: failure cases and must never be pooled into a prevalence estimate.
STRATUM_REPRESENTATIVE = "representative"
STRATUM_BOUNDARY = "boundary"

#: A row counts as a boundary case when the automatic judgement is one we have
#: reason to distrust: it abstained, it never saw an abstract, it named itself
#: low-confidence, or it was produced by a deterministic rule rather than a model.
def is_boundary_candidate(item: AssessmentRow) -> bool:
    return (
        item.decision in {"uncertain", "insufficient_evidence"}
        or item.basis in {"title_only", "deterministic_metadata_rule"}
        or (item.confidence or "").casefold() == "low"
        or not item.has_abstract
    )


@dataclass
class SamplingPlan:
    """A stratified sample and the record of how it was drawn.

    ``stratum`` is written into the sheet so the comparison report can compute
    precision/recall on the representative rows alone, instead of quietly
    producing a number that the boundary oversample has inflated.
    """

    rows: list[AssessmentRow]
    stratum_of: dict[tuple[str, int], str]
    per_profile: dict[str, int]
    per_profile_stratum: dict[str, dict[str, int]]
    #: Seats requested versus seats the pool could actually supply, per profile.
    #: A shortfall here is the honest explanation for a thin slice.
    seats: dict[str, int] = field(default_factory=dict)
    capacity: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": len(self.rows),
            "per_profile": dict(self.per_profile),
            "per_profile_stratum": {
                profile: dict(counts) for profile, counts in self.per_profile_stratum.items()
            },
            "seats": dict(self.seats),
            "capacity": dict(self.capacity),
        }


def _allocate(total: int, weights: dict[str, int]) -> dict[str, int]:
    """Largest-remainder allocation of ``total`` seats over ``weights``."""
    if not weights:
        return {}
    pool = sum(weights.values())
    if pool <= 0:
        return {key: 0 for key in weights}
    exact = {key: total * value / pool for key, value in weights.items()}
    seats = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(seats.values())
    if remainder > 0:
        order = sorted(
            exact,
            key=lambda key: (-(exact[key] - seats[key]), key),
        )
        for key in order[:remainder]:
            seats[key] += 1
    return seats


def _allocate_within_capacity(
    total: int, weights: dict[str, int], capacity: dict[str, int]
) -> dict[str, int]:
    """Proportional allocation that redistributes seats a profile cannot fill.

    Proportional allocation alone silently under-fills the sheet: a queue with
    plenty of weight but almost no eligible rows (a profile whose judgements are
    nearly all abstentions, or nearly all fully screened) is handed seats it
    cannot use, and the seats are simply lost.  Water-filling hands them to the
    profiles that still have rows, so the requested total is actually reached
    whenever the pool can supply it.
    """
    seats = {key: 0 for key in weights}
    remaining = min(total, sum(capacity.get(key, 0) for key in weights))
    open_keys = {key for key in weights if capacity.get(key, 0) > 0 and weights[key] > 0}
    while remaining > 0 and open_keys:
        # Weights are renormalised over the profiles still able to take a seat.
        open_weights = {key: weights[key] for key in open_keys}
        allocation = _allocate(remaining, open_weights)
        progressed = False
        for key in sorted(open_keys):
            room = capacity.get(key, 0) - seats[key]
            take = min(allocation.get(key, 0), room)
            if take > 0:
                seats[key] += take
                remaining -= take
                progressed = True
            if seats[key] >= capacity.get(key, 0):
                open_keys.discard(key)
        if not progressed:
            break
    return seats


def design_sample(
    assessments: list[AssessmentRow],
    *,
    total: int = 130,
    boundary_share: float = 0.35,
    min_per_profile: int = 0,
    only_profiles: list[str] | None = None,
    include_ids: list[int] | None = None,
    seed: int = 20260925,
) -> SamplingPlan:
    """Draw an annotation sample that can support real metrics.

    Two strata, drawn independently and labelled as such:

    * **representative** — a probability sample *within each profile*, sized in
      proportion to how many judgements that profile holds.  This is the only
      part that can speak to "how well does this queue screen its library"; it
      keeps the real decision mix instead of a manufactured 50/50.
    * **boundary** — a deliberate oversample of rows the system itself flagged
      (abstentions, title-only judgements, low confidence, no abstract).  There
      are few of them, so a representative draw would contain almost none — and
      they are exactly the rows worth a human's attention.

    A row only lands in one stratum.  If a boundary draw picks something the
    representative draw already took, the representative assignment wins, so the
    prevalence estimate stays uncontaminated.

    ``min_per_profile`` reserves seats for every queue.  Strict proportion would
    give a small queue two or three rows, which is not enough to compute anything
    about it — and "we cannot say whether this queue works" is a worse outcome
    than a slightly less proportional sample.  The reserved seats come out of the
    largest queues, and the report states the deviation rather than hiding it.
    """
    if total < 1:
        raise ValueError("total must be positive")
    if not 0.0 <= boundary_share <= 1.0:
        raise ValueError("boundary_share must be between 0 and 1")
    if min_per_profile < 0:
        raise ValueError("min_per_profile must not be negative")

    pool = list(assessments)
    if only_profiles:
        wanted = {str(item) for item in only_profiles}
        pool = [item for item in pool if item.profile_id in wanted]
    if not pool:
        return SamplingPlan([], {}, {}, {})

    rng = random.Random(seed)
    by_profile: dict[str, list[AssessmentRow]] = {}
    for item in pool:
        by_profile.setdefault(item.profile_id, []).append(item)

    boundary_total = int(round(total * boundary_share))
    representative_total = total - boundary_total

    # Seats are proportional to each profile's judgement count, so the sample
    # mirrors the library rather than over-weighting the smallest queue.  The
    # capacity is the pool each profile can actually supply for that stratum, and
    # seats a profile cannot fill are redistributed instead of being lost.
    profile_weights = {key: len(value) for key, value in by_profile.items()}
    representative_pool: dict[str, list[AssessmentRow]] = {}
    boundary_pool: dict[str, list[AssessmentRow]] = {}
    for key, value in by_profile.items():
        boundary_pool[key] = [item for item in value if is_boundary_candidate(item)]
        representative_pool[key] = [
            item for item in value if not is_boundary_candidate(item)
        ]
    representative_capacity = {key: len(value) for key, value in representative_pool.items()}
    boundary_capacity = {key: len(value) for key, value in boundary_pool.items()}

    # The boundary stratum is smaller than its quota whenever few rows were
    # flagged.  Asking for the nominal quota would silently shrink the sheet, so
    # the seats that cannot be filled are handed to the representative stratum —
    # which is the slice the metrics actually depend on — and the real total is
    # recomputed from the seats that were actually handed out.
    boundary_seats = _allocate_within_capacity(
        boundary_total, boundary_capacity, boundary_capacity
    )
    boundary_total = sum(boundary_seats.values())
    representative_total = total - boundary_total

    # Reserve the per-profile floor first, then spend what is left over
    # proportionally.  A queue with fewer rows than the floor reserves all of
    # them: "this queue has 8 judgements, look at all 8" is the only useful
    # instruction a floor can give it.
    reserved: dict[str, int] = {}
    if min_per_profile:
        reserved = {
            profile_id: min(min_per_profile, capacity)
            for profile_id, capacity in representative_capacity.items()
            if capacity > 0
        }
    reserved_total = sum(reserved.values())
    if reserved_total > representative_total:
        # The sheet cannot hold the floor for every queue.  Reserve the smallest
        # queues in full and give the rest proportional seats, instead of
        # abandoning the floor altogether and starving exactly those queues.
        reserved = {}
        reserved_total = 0
        for profile_id, capacity in sorted(
            representative_capacity.items(), key=lambda item: (item[1], item[0])
        ):
            if reserved_total + capacity > representative_total:
                break
            reserved[profile_id] = capacity
            reserved_total += capacity
        remaining_total = representative_total - reserved_total
        seat_pool = _allocate_within_capacity(
            remaining_total,
            {
                key: weight
                for key, weight in profile_weights.items()
                if key not in reserved
            },
            {
                key: capacity
                for key, capacity in representative_capacity.items()
                if key not in reserved
            },
        )
    else:
        seat_pool = _allocate_within_capacity(
            representative_total - reserved_total,
            profile_weights,
            {
                profile_id: max(0, capacity - reserved.get(profile_id, 0))
                for profile_id, capacity in representative_capacity.items()
            },
        )
    rep_seats = {
        profile_id: seat_pool.get(profile_id, 0) + reserved.get(profile_id, 0)
        for profile_id in representative_capacity
    }
    rep_seats = {key: value for key, value in rep_seats.items() if value > 0}

    stratum_of: dict[tuple[str, int], str] = {}
    chosen: dict[tuple[str, int], AssessmentRow] = {}

    for profile_id in by_profile:
        # Representative first: a boundary row must never be the only evidence
        # for prevalence, and taking it here would remove it from that pool.
        pool = representative_pool[profile_id]
        for item in rng.sample(pool, min(rep_seats.get(profile_id, 0), len(pool))):
            key = (profile_id, item.paper_id)
            chosen[key] = item
            stratum_of[key] = STRATUM_REPRESENTATIVE

        pool = boundary_pool[profile_id]
        for item in rng.sample(pool, min(boundary_seats.get(profile_id, 0), len(pool))):
            key = (profile_id, item.paper_id)
            if key in chosen:
                continue
            chosen[key] = item
            stratum_of[key] = STRATUM_BOUNDARY

    for identifier in include_ids or []:
        for item in pool:
            if item.paper_id != int(identifier):
                continue
            key = (item.profile_id, item.paper_id)
            if key in chosen:
                continue
            chosen[key] = item
            stratum_of[key] = "included_by_request"

    ordered_rows = sorted(
        chosen.values(), key=lambda item: (item.profile_id, item.paper_id)
    )
    per_profile: dict[str, int] = {}
    per_profile_stratum: dict[str, dict[str, int]] = {}
    for item in ordered_rows:
        per_profile[item.profile_id] = per_profile.get(item.profile_id, 0) + 1
        counts = per_profile_stratum.setdefault(item.profile_id, {})
        stratum = stratum_of[(item.profile_id, item.paper_id)]
        counts[stratum] = counts.get(stratum, 0) + 1
    return SamplingPlan(
        ordered_rows,
        stratum_of,
        per_profile,
        per_profile_stratum,
        seats={**rep_seats, **{f"{k} (boundary)": v for k, v in boundary_seats.items() if v}},
        capacity=representative_capacity,
    )


def render_sheet(
    rows: list[AssessmentRow],
    *,
    blind: bool = False,
    include_abstract: bool = True,
    stratum_of: dict[tuple[str, int], str] | None = None,
) -> str:
    """Render the annotation sheet as CSV text.

    ``sample_stratum`` stays visible even when ``blind`` withholds the automatic
    judgement: it records *how the row was selected*, which the annotator needs in
    order to know that this is a boundary-oversampled sheet, and which the
    comparison step needs in order to keep precision off the oversampled rows.  It
    is not the answer, so showing it does not leak the label.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(SHEET_COLUMNS), lineterminator="\n")
    writer.writeheader()
    strata = stratum_of or {}
    for item in rows:
        writer.writerow(
            {
                "profile_id": item.profile_id,
                "paper_id": item.paper_id,
                "journal": item.journal,
                "publication_date": item.publication_date or "",
                "has_abstract": "yes" if item.has_abstract else "no",
                "title": item.title,
                "abstract": (item.abstract or "") if include_abstract else "",
                "human_decision": "",
                "human_note": "",
                "auto_decision": "" if blind else item.decision,
                "auto_basis": "" if blind else item.basis,
                "auto_confidence": "" if blind else (item.confidence or ""),
                "auto_prompt_version": "" if blind else (item.prompt_version or ""),
                "sample_stratum": strata.get(
                    (item.profile_id, item.paper_id), STRATUM_REPRESENTATIVE
                ),
                "sample_reason": sample_reason(item),
            }
        )
    return buffer.getvalue()


def parse_sheet(text: str) -> list[dict[str, str]]:
    """Parse a returned annotation sheet, tolerating a UTF-8 BOM."""
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    return [dict(row) for row in reader]


@dataclass(frozen=True)
class HumanLabel:
    profile_id: str
    paper_id: int
    decision: str
    note: str = ""


def load_human_labels(text: str) -> tuple[list[HumanLabel], list[str]]:
    """Read human decisions from a returned sheet.

    Only ``paper_id`` and ``human_decision`` are required: the annotator's call
    must not depend on, or be reconciled against, the automatic one.
    """
    labels: list[HumanLabel] = []
    problems: list[str] = []
    for index, row in enumerate(parse_sheet(text), start=2):
        raw_decision = (row.get("human_decision") or "").strip().casefold()
        if not raw_decision:
            problems.append(f"row {index}: no human_decision recorded")
            continue
        if raw_decision not in HUMAN_DECISIONS:
            problems.append(
                f"row {index}: human_decision {raw_decision!r} is not one of "
                f"{', '.join(HUMAN_DECISIONS)}"
            )
            continue
        try:
            paper_id = int((row.get("paper_id") or "").strip())
        except ValueError:
            problems.append(f"row {index}: paper_id is missing or not an integer")
            continue
        labels.append(
            HumanLabel(
                profile_id=(row.get("profile_id") or "").strip(),
                paper_id=paper_id,
                decision=raw_decision,
                note=(row.get("human_note") or "").strip(),
            )
        )
    return labels, problems


@dataclass(frozen=True)
class ConfusionEntry:
    auto_decision: str
    human_decision: str
    count: int
    paper_ids: tuple[int, ...]


# --------------------------------------------------------------------------- #
# Annotation intake: what a returned sheet must satisfy before it is believed
# --------------------------------------------------------------------------- #

#: Columns an annotation sheet must carry.  ``auto_decision`` is included because
#: a *blind* sheet must still have the column (empty); a sheet that dropped it
#: cannot be told apart from one whose labels were chosen with the answer visible.
REQUIRED_SHEET_COLUMNS = (
    "profile_id",
    "paper_id",
    "title",
    "human_decision",
    "human_note",
    "auto_decision",
    "sample_stratum",
)


@dataclass
class SheetIntake:
    """Result of checking a returned annotation sheet before comparing it."""

    rows: int
    labelled: int
    unlabelled: int
    per_decision: dict[str, int]
    per_profile: dict[str, int]
    per_stratum: dict[str, int]
    skipped: int
    notes_provided: int
    was_blank_before: bool
    windows_respected: bool
    problems: list[str]
    blocking: list[str]

    @property
    def usable(self) -> bool:
        return not self.blocking and self.labelled > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "labelled": self.labelled,
            "unlabelled": self.unlabelled,
            "completion_rate": round(self.labelled / self.rows, 4) if self.rows else 0.0,
            "per_decision": dict(self.per_decision),
            "per_profile": dict(self.per_profile),
            "per_stratum": dict(self.per_stratum),
            "skipped": self.skipped,
            "notes_provided": self.notes_provided,
            "was_blank_before": self.was_blank_before,
            "windows_respected": self.windows_respected,
            "problems": list(self.problems),
            "blocking": list(self.blocking),
            "usable": self.usable,
        }


def intake_sheet(text: str, *, require_blind: bool = True) -> SheetIntake:
    """Validate a returned annotation sheet against the labelling rules.

    The rules exist because a calibration number is only as trustworthy as the
    labels behind it, and the failure modes are silent:

    * **only an empty ``human_decision`` may be filled.**  A sheet that already
      carried a label was not annotated, it was copied — most likely from the
      automatic column.
    * **the sheet must be blind.**  If ``auto_decision`` is populated, the
      annotator could see the answer, and the comparison stops measuring the
      screen and starts measuring agreement-with-what-was-on-screen.
    * **``human_note`` is preserved.**  It is the only place the reason for a
      disagreement is recorded, and the boundary catalogue is built from it.
    * **``skip`` is counted separately** from a decision: it is an abstention, not
      a judgement, and folding it into either class would invent a label.
    """
    rows = parse_sheet(text)
    problems: list[str] = []
    blocking: list[str] = []

    if not rows:
        return SheetIntake(
            0, 0, 0, {}, {}, {}, 0, 0, True, False,
            ["the sheet has no data rows"], ["the sheet has no data rows"],
        )

    header = set(rows[0].keys())
    missing = [name for name in REQUIRED_SHEET_COLUMNS if name not in header]
    if missing:
        blocking.append(f"sheet is missing required columns: {', '.join(missing)}")

    per_decision: Counter[str] = Counter()
    per_profile: Counter[str] = Counter()
    per_stratum: Counter[str] = Counter()
    labelled = unlabelled = skipped = notes = 0
    auto_visible = 0

    for index, row in enumerate(rows, start=2):
        raw = (row.get("human_decision") or "").strip().casefold()
        if (row.get("auto_decision") or "").strip():
            auto_visible += 1
        if not raw:
            unlabelled += 1
            continue
        if raw not in HUMAN_DECISIONS:
            problems.append(
                f"row {index}: human_decision {raw!r} is not one of "
                f"{', '.join(HUMAN_DECISIONS)}"
            )
            continue
        labelled += 1
        per_decision[raw] += 1
        per_profile[(row.get("profile_id") or "(none)").strip() or "(none)"] += 1
        per_stratum[
            ((row.get("sample_stratum") or "").strip() or STRATUM_REPRESENTATIVE)
        ] += 1
        if raw == "skip":
            skipped += 1
        if (row.get("human_note") or "").strip():
            notes += 1

    if require_blind and auto_visible:
        blocking.append(
            f"{auto_visible} row(s) have auto_decision populated; this sheet is not "
            "blind, so a comparison against it would measure agreement with what the "
            "annotator could see. Re-export with --blind."
        )
    if labelled == 0:
        blocking.append(
            "no row carries a human_decision; the sheet has not been annotated yet, "
            "so no comparison is possible"
        )

    # A sheet whose human column is entirely empty was untouched.  Once any row is
    # filled, we cannot prove the others were empty beforehand, so the check is
    # reported as a property of the observed file rather than a guarantee.
    was_blank_before = labelled == 0
    windows_respected = True  # no window column exists; kept explicit for auditing
    return SheetIntake(
        rows=len(rows),
        labelled=labelled,
        unlabelled=unlabelled,
        per_decision=dict(per_decision),
        per_profile=dict(per_profile),
        per_stratum=dict(per_stratum),
        skipped=skipped,
        notes_provided=notes,
        was_blank_before=was_blank_before,
        windows_respected=windows_respected,
        problems=problems,
        blocking=blocking,
    )


def format_intake(intake: SheetIntake) -> str:
    lines = [
        "Annotation intake",
        f"  rows                 : {intake.rows}",
        f"  labelled             : {intake.labelled}",
        f"  not yet labelled     : {intake.unlabelled}",
        f"  completion rate      : {intake.labelled / intake.rows:.1%}"
        if intake.rows
        else "  completion rate      : n/a",
        f"  decisions            : {intake.per_decision}",
        f"  per profile          : {intake.per_profile}",
        f"  per stratum          : {intake.per_stratum}",
        f"  skip (abstentions)   : {intake.skipped}",
        f"  rows with a note     : {intake.notes_provided}",
        f"  usable               : {intake.usable}",
    ]
    for problem in intake.problems:
        lines.append(f"  ! {problem}")
    for item in intake.blocking:
        lines.append(f"  BLOCKING: {item}")
    return "\n".join(lines)


@dataclass
class ComparisonReport:
    profile_id: str
    sampled_rows: int
    labelled_rows: int
    unlabelled_rows: int
    abstentions: int
    agreement: int
    agreement_rate: float
    auto_positive_human_negative: list[dict[str, Any]]
    auto_negative_human_positive: list[dict[str, Any]]
    boundary_cases: list[dict[str, Any]]
    confusion: list[ConfusionEntry]
    ambiguous_human_calls: list[dict[str, Any]]
    problems: list[str]
    sampling_mode: str
    stratification: dict[str, Any]
    #: Metrics over the *representative* stratum only.  This is the number that
    #: speaks to how the queue performs on its library; the boundary oversample
    #: would inflate it, so it is never pooled in.
    representative_metrics: MetricCell | None = None
    #: Metrics over the boundary stratum, reported separately as a failure-case
    #: catalogue rather than as a performance estimate.
    boundary_metrics: MetricCell | None = None
    #: Metrics restricted to fully-screened rows (title + abstract), where the
    #: screen actually had the evidence the contract asks for.
    full_screening_metrics: MetricCell | None = None
    #: Per-profile breakdown, for the cross-profile sheet.
    by_profile: dict[str, Any] = field(default_factory=dict)
    #: Per-screening-basis breakdown, so a title-only row is never averaged with
    #: a full one.
    by_basis: dict[str, Any] = field(default_factory=dict)
    #: Read-only description of how each profile failed.  Findings only.
    diagnoses: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "sampling_mode": self.sampling_mode,
            "sampled_rows": self.sampled_rows,
            "labelled_rows": self.labelled_rows,
            "unlabelled_rows": self.unlabelled_rows,
            "abstentions": self.abstentions,
            "agreement": self.agreement,
            "agreement_rate": round(self.agreement_rate, 4),
            "false_positives": len(self.auto_positive_human_negative),
            "false_negatives": len(self.auto_negative_human_positive),
            "boundary_case_count": len(self.boundary_cases),
            "confusion": [
                {
                    "auto": entry.auto_decision,
                    "human": entry.human_decision,
                    "count": entry.count,
                    "paper_ids": list(entry.paper_ids),
                }
                for entry in self.confusion
            ],
            "auto_positive_human_negative": self.auto_positive_human_negative,
            "auto_negative_human_positive": self.auto_negative_human_positive,
            "boundary_cases": self.boundary_cases,
            "ambiguous_human_calls": self.ambiguous_human_calls,
            "stratification": self.stratification,
            "problems": self.problems,
            "metrics": {
                "representative": (
                    None if self.representative_metrics is None
                    else self.representative_metrics.as_dict()
                ),
                "boundary": (
                    None if self.boundary_metrics is None else self.boundary_metrics.as_dict()
                ),
                "full_screening": (
                    None
                    if self.full_screening_metrics is None
                    else self.full_screening_metrics.as_dict()
                ),
                "by_profile": self.by_profile,
                "by_basis": self.by_basis,
            },
            "diagnoses": self.diagnoses,
        }


def _is_auto_positive(decision: str) -> bool:
    return decision == "relevant"


def _is_auto_negative(decision: str) -> bool:
    return decision == "irrelevant"


def _is_human_positive(decision: str) -> bool:
    return decision in {"relevant", "related"}


def _is_human_negative(decision: str) -> bool:
    return decision == "irrelevant"


def _row_summary(row: dict[str, str]) -> dict[str, Any]:
    return {
        "paper_id": int(row["paper_id"]),
        "journal": row.get("journal", ""),
        "title": (row.get("title") or "")[:120],
        "auto_decision": row.get("auto_decision", ""),
        "auto_basis": row.get("auto_basis", ""),
        "auto_confidence": row.get("auto_confidence", ""),
        "human_decision": (row.get("human_decision") or "").strip().casefold(),
        "human_note": (row.get("human_note") or "")[:200],
        "sample_stratum": row.get("sample_stratum", ""),
        "sample_reason": row.get("sample_reason", ""),
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

#: How many labelled rows a cell needs before a rate is worth printing.
#: Below this the interval is so wide that the number invites a wrong decision;
#: the report says "insufficient" instead of printing it.
MIN_LABELS_FOR_RATE = 20

#: 95% two-sided normal quantile, used by the Wilson interval.
_Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int, *, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because annotation counts are small
    and rates near 0 or 1 are exactly where the normal interval misbehaves — and
    precision on a screening queue sits near 1 by construction.
    """
    if total <= 0:
        return (0.0, 1.0)
    phat = successes / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = (
        z
        * ((phat * (1 - phat) / total + z * z / (4 * total * total)) ** 0.5)
        / denominator
    )
    # Clamp to the closed unit interval.  Floating-point error leaves the bound at
    # 0.9999999999999999 for a perfect score, which would render as an interval
    # that excludes the observed result.
    return (min(1.0, max(0.0, centre - margin)), min(1.0, max(0.0, centre + margin)))


@dataclass(frozen=True)
class RateEstimate:
    """A proportion with its sample size and interval."""

    successes: int
    total: int
    value: float | None
    low: float | None
    high: float | None
    sufficient: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "successes": self.successes,
            "total": self.total,
            "value": None if self.value is None else round(self.value, 4),
            "wilson_95": (
                None
                if self.low is None or self.high is None
                else [round(self.low, 4), round(self.high, 4)]
            ),
            "sufficient": self.sufficient,
        }

    def render(self, label: str) -> str:
        if self.total == 0:
            return f"{label:26s}: no labelled rows"
        if not self.sufficient:
            return (
                f"{label:26s}: {self.successes}/{self.total} "
                f"(insufficient: needs {MIN_LABELS_FOR_RATE}+ labelled rows)"
            )
        assert self.value is not None and self.low is not None and self.high is not None
        return (
            f"{label:26s}: {self.successes}/{self.total} = {self.value:.1%} "
            f"[{self.low:.1%}, {self.high:.1%}]"
        )


def _rate(successes: int, total: int) -> RateEstimate:
    if total <= 0:
        return RateEstimate(successes, total, None, None, None, False)
    low, high = wilson_interval(successes, total)
    return RateEstimate(
        successes, total, successes / total, low, high, total >= MIN_LABELS_FOR_RATE
    )


@dataclass
class MetricCell:
    """Precision / recall / coverage for one slice of labelled rows.

    ``uncertain`` is deliberately excluded from the denominator of precision and
    recall and reported on its own.  Folding an abstention into either error
    direction would misdescribe it: the screen did not claim the paper was
    relevant, and it did not claim it was irrelevant either.
    """

    slice_name: str
    labelled: int
    auto_relevant: int
    auto_irrelevant: int
    auto_uncertain: int
    human_relevant: int
    human_irrelevant: int
    human_uncertain: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: RateEstimate
    recall: RateEstimate
    coverage: RateEstimate
    agreement: RateEstimate

    def as_dict(self) -> dict[str, Any]:
        return {
            "slice": self.slice_name,
            "labelled": self.labelled,
            "auto": {
                "relevant": self.auto_relevant,
                "irrelevant": self.auto_irrelevant,
                "uncertain": self.auto_uncertain,
            },
            "human": {
                "relevant": self.human_relevant,
                "irrelevant": self.human_irrelevant,
                "uncertain": self.human_uncertain,
            },
            "confusion": {
                "true_positive": self.true_positive,
                "false_positive": self.false_positive,
                "false_negative": self.false_negative,
                "true_negative": self.true_negative,
            },
            "precision": self.precision.as_dict(),
            "recall": self.recall.as_dict(),
            "coverage": self.coverage.as_dict(),
            "agreement": self.agreement.as_dict(),
        }


def compute_metrics(
    pairs: list[tuple[dict[str, str], AssessmentRow]],
    *,
    slice_name: str,
) -> MetricCell:
    """Compute precision, recall, coverage and agreement over labelled rows.

    Definitions, stated because they decide what the numbers mean:

    * ``precision`` = of the papers this queue called **relevant**, the share the
      human also called relevant.
    * ``recall``    = of the papers the human called **relevant**, the share this
      queue also called relevant.  This asks "would I have been shown it?".
    * ``coverage``  = the share of judged papers the queue was willing to call at
      all (not ``uncertain``).  A high-precision queue that abstains on half the
      library is not doing well, and precision alone would hide that.
    * ``agreement`` = exact decision match over all labelled rows.

    Rows where either side said ``uncertain`` are counted in ``coverage`` and in
    the raw tallies, and are excluded from the precision/recall denominators.
    """
    tp = fp = fn = tn = 0
    auto_rel = auto_irr = auto_unc = 0
    human_rel = human_irr = human_unc = 0
    decisive_auto = 0
    decisive_both = 0
    labelled = 0

    for row, assessment in pairs:
        human = (row.get("human_decision") or "").strip().casefold()
        auto = assessment.decision
        labelled += 1
        if auto == "relevant":
            auto_rel += 1
        elif auto == "irrelevant":
            auto_irr += 1
        else:
            auto_unc += 1
        if human == "relevant":
            human_rel += 1
        elif human == "irrelevant":
            human_irr += 1
        else:
            human_unc += 1

        # Agreement is exact match, including "uncertain == uncertain".
        if human == auto:
            decisive_both += 1
        # Precision and recall only over rows both sides were willing to call.
        if human in {"relevant", "irrelevant"} and auto in {"relevant", "irrelevant"}:
            decisive_auto += 1
            if auto == "relevant" and human == "relevant":
                tp += 1
            elif auto == "relevant" and human == "irrelevant":
                fp += 1
            elif auto == "irrelevant" and human == "relevant":
                fn += 1
            else:
                tn += 1

    return MetricCell(
        slice_name=slice_name,
        labelled=labelled,
        auto_relevant=auto_rel,
        auto_irrelevant=auto_irr,
        auto_uncertain=auto_unc,
        human_relevant=human_rel,
        human_irrelevant=human_irr,
        human_uncertain=human_unc,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        precision=_rate(tp, tp + fp),
        recall=_rate(tp, tp + fn),
        coverage=_rate(auto_rel + auto_irr, labelled),
        agreement=_rate(decisive_both, labelled),
    )


def compare_sheet(
    sheet_text: str,
    assessments: list[AssessmentRow],
    *,
    profile_id: str = "",
    sampling_mode: str = "balanced_per_decision",
) -> ComparisonReport:
    """Join a returned sheet against the stored automatic judgements.

    A disagreement becomes a *finding* only when the human read the paper and
    still disagreed.  ``uncertain`` from either side is reported as a boundary
    case, not as an error: a paper the automatic screen could not call is a
    different problem from a paper it called wrongly.
    """
    rows = parse_sheet(sheet_text)
    by_key = {(item.profile_id, item.paper_id): item for item in assessments}
    problems: list[str] = []
    matched: list[tuple[dict[str, str], AssessmentRow]] = []
    for row in rows:
        if not (row.get("paper_id") or "").strip():
            problems.append("a sheet row has no paper_id")
            continue
        explicit_profile = (row.get("profile_id") or "").strip()
        if explicit_profile:
            key = (explicit_profile, int(row["paper_id"]))
            assessment = by_key.get(key)
        else:
            candidates = [
                item
                for (candidate_profile, paper_id), item in by_key.items()
                if paper_id == int(row["paper_id"])
                and (not profile_id or candidate_profile == profile_id)
            ]
            assessment = candidates[0] if len(candidates) == 1 else None
            if len(candidates) > 1:
                problems.append(
                    f"paper {row['paper_id']} is judged by several profiles; the sheet "
                    "must carry profile_id"
                )
                continue
        if assessment is None:
            problems.append(
                f"paper {row['paper_id']} in profile "
                f"{explicit_profile or profile_id or '(any)'} has no stored assessment"
            )
            continue
        matched.append((row, assessment))

    # A sheet spanning several queues must not be labelled with whichever one
    # happened to sort first; say "(all profiles)" instead.
    if profile_id:
        resolved_profile = profile_id
    else:
        matched_profiles = {assessment.profile_id for _, assessment in matched}
        resolved_profile = matched_profiles.pop() if len(matched_profiles) == 1 else ""

    confusion_counter: Counter[tuple[str, str]] = Counter()
    confusion_ids: dict[tuple[str, str], list[int]] = {}
    false_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    boundary: list[dict[str, Any]] = []
    abstentions = 0
    unlabelled = 0
    agreement = 0
    considered = 0
    by_basis: dict[str, Counter[str]] = {}

    for row, assessment in matched:
        human = (row.get("human_decision") or "").strip().casefold()
        if not human:
            unlabelled += 1
            continue
        if human == "skip":
            abstentions += 1
            continue
        if human not in HUMAN_DECISIONS:
            problems.append(
                f"paper {assessment.paper_id}: human_decision {human!r} is not recognized"
            )
            continue
        considered += 1
        auto = assessment.decision
        confusion_counter[(auto, human)] += 1
        confusion_ids.setdefault((auto, human), []).append(assessment.paper_id)
        basis_stats = by_basis.setdefault(assessment.basis, Counter())
        basis_stats["considered"] += 1

        summary = _row_summary(row)
        summary["auto_prompt_version"] = assessment.prompt_version or ""
        if human == auto:
            agreement += 1
            basis_stats["agreement"] += 1
        if _is_auto_positive(auto) and _is_human_negative(human):
            summary["error_kind"] = "false_positive"
            false_positives.append(summary)
        elif _is_auto_negative(auto) and _is_human_positive(human):
            summary["error_kind"] = "false_negative"
            false_negatives.append(summary)
        if "uncertain" in {auto, human}:
            summary.setdefault("error_kind", "boundary")
            boundary.append(summary)

    ambiguous = [
        summary
        for summary in (
            _row_summary(row)
            for row, assessment in matched
            if (row.get("human_decision") or "").strip().casefold() == "uncertain"
        )
    ]

    confusion = [
        ConfusionEntry(auto, human, count, tuple(confusion_ids[(auto, human)]))
        for (auto, human), count in sorted(confusion_counter.items())
    ]
    stratification = {
        basis: {
            "considered": stats.get("considered", 0),
            "agreement": stats.get("agreement", 0),
            "agreement_rate": round(
                stats.get("agreement", 0) / stats["considered"], 4
            )
            if stats.get("considered")
            else None,
        }
        for basis, stats in sorted(by_basis.items())
    }

    # Metrics are computed on slices that answer different questions, and the
    # stratum a row came from decides which slice it belongs to.  Pooling the
    # boundary oversample into the representative rows would produce a precision
    # number that describes the sampling design rather than the screening queue.
    decisive_pairs = [
        (row, assessment)
        for row, assessment in matched
        if (row.get("human_decision") or "").strip().casefold()
        in {"relevant", "irrelevant", "uncertain"}
    ]
    # A sheet written before the stratum column existed has no such key at all
    # (DictReader yields None for absent columns), so an absent stratum must fall
    # back to the representative slice rather than dropping the row from metrics.
    def _stratum(row: dict[str, str]) -> str:
        return (row.get("sample_stratum") or "").strip() or STRATUM_REPRESENTATIVE

    representative_pairs = [
        (row, assessment)
        for row, assessment in decisive_pairs
        if _stratum(row) == STRATUM_REPRESENTATIVE
    ]
    boundary_pairs = [
        (row, assessment)
        for row, assessment in decisive_pairs
        if _stratum(row) == STRATUM_BOUNDARY
    ]
    full_pairs = [
        (row, assessment)
        for row, assessment in decisive_pairs
        if assessment.basis == "title_and_abstract"
    ]

    representative_metrics = (
        compute_metrics(representative_pairs, slice_name="representative stratum")
        if representative_pairs
        else None
    )
    boundary_metrics = (
        compute_metrics(boundary_pairs, slice_name="boundary stratum")
        if boundary_pairs
        else None
    )
    full_screening_metrics = (
        compute_metrics(full_pairs, slice_name="title_and_abstract")
        if full_pairs
        else None
    )

    profile_slices: dict[str, list[tuple[dict[str, str], AssessmentRow]]] = {}
    for row, assessment in decisive_pairs:
        profile_slices.setdefault(assessment.profile_id, []).append((row, assessment))
    by_profile = {
        name: compute_metrics(pairs, slice_name=name).as_dict()
        for name, pairs in sorted(profile_slices.items())
    }
    basis_slices: dict[str, list[tuple[dict[str, str], AssessmentRow]]] = {}
    for row, assessment in decisive_pairs:
        basis_slices.setdefault(assessment.basis, []).append((row, assessment))
    by_basis_metrics = {
        name: compute_metrics(pairs, slice_name=name).as_dict()
        for name, pairs in sorted(basis_slices.items())
    }

    # One diagnosis per profile: a queue is the unit that could have a boundary
    # problem, so pooling four queues' errors would describe nothing.
    error_rows = [
        {**summary, "error_kind": kind}
        for kind, group in (
            ("false_positive", false_positives),
            ("false_negative", false_negatives),
        )
        for summary in group
    ]
    diagnoses: dict[str, Any] = {}
    profile_pairs: dict[str, list[tuple[dict[str, str], AssessmentRow]]] = {}
    for row, assessment in matched:
        profile_pairs.setdefault(assessment.profile_id, []).append((row, assessment))
    for name, pairs in sorted(profile_pairs.items()):
        relevant_errors = [
            item
            for item in error_rows
            if any(
                int(row["paper_id"]) == int(item["paper_id"])
                and assessment.profile_id == name
                for row, assessment in pairs
            )
        ]
        labelled_here = sum(
            1
            for row, _ in pairs
            if (row.get("human_decision") or "").strip().casefold()
            in {"relevant", "irrelevant", "uncertain", "skip"}
        )
        diagnoses[name] = diagnose(
            pairs,
            profile_id=name,
            errors=relevant_errors,
            total_errors=len(relevant_errors),
            labelled=labelled_here,
        ).as_dict()

    return ComparisonReport(
        profile_id=resolved_profile,
        sampled_rows=len(rows),
        labelled_rows=considered,
        unlabelled_rows=unlabelled,
        abstentions=abstentions,
        agreement=agreement,
        agreement_rate=(agreement / considered) if considered else 0.0,
        auto_positive_human_negative=false_positives,
        auto_negative_human_positive=false_negatives,
        boundary_cases=boundary,
        confusion=confusion,
        ambiguous_human_calls=ambiguous,
        problems=problems,
        sampling_mode=sampling_mode,
        stratification=stratification,
        representative_metrics=representative_metrics,
        boundary_metrics=boundary_metrics,
        full_screening_metrics=full_screening_metrics,
        by_profile=by_profile,
        by_basis=by_basis_metrics,
        diagnoses=diagnoses,
    )


def format_comparison(report: ComparisonReport) -> str:
    lines = [
        f"Profile quality report — {report.profile_id or '(all profiles)'}",
        f"  sampling mode            : {report.sampling_mode}",
        f"  sheet rows               : {report.sampled_rows}",
        f"  human-labelled           : {report.labelled_rows}",
        f"  not yet labelled         : {report.unlabelled_rows}",
        f"  abstentions (skip)       : {report.abstentions}",
        f"  exact agreement          : {report.agreement}/{report.labelled_rows}"
        f" ({report.agreement_rate:.1%})",
        f"  automatic positive, human negative : {len(report.auto_positive_human_negative)}",
        f"  automatic negative, human positive : {len(report.auto_negative_human_positive)}",
        f"  boundary cases (either 'uncertain'): {len(report.boundary_cases)}",
    ]

    if report.representative_metrics is not None:
        metrics = report.representative_metrics
        lines.append("")
        lines.append("  METRICS — representative stratum only (the only slice that "
                     "estimates library performance)")
        lines.append(f"    {metrics.precision.render('precision (auto=relevant)')}")
        lines.append(f"    {metrics.recall.render('recall (human=relevant)')}")
        lines.append(f"    {metrics.coverage.render('coverage (not uncertain)')}")
        lines.append(f"    {metrics.agreement.render('exact agreement')}")
        lines.append(
            f"    confusion: TP={metrics.true_positive} FP={metrics.false_positive} "
            f"FN={metrics.false_negative} TN={metrics.true_negative}  "
            f"(uncertain auto={metrics.auto_uncertain}, human={metrics.human_uncertain})"
        )
    else:
        lines.append("")
        lines.append("  METRICS — no labelled rows in the representative stratum yet")

    if report.full_screening_metrics is not None:
        full = report.full_screening_metrics
        lines.append("")
        lines.append("  METRICS — title + abstract rows only")
        lines.append(f"    {full.precision.render('precision')}")
        lines.append(f"    {full.recall.render('recall')}")

    if report.by_profile:
        lines.append("")
        lines.append("  METRICS by profile:")
        for name, payload in sorted(report.by_profile.items()):
            lines.append(
                f"    {name:22s} labelled={payload['labelled']:<4} "
                f"P={_render_rate(payload['precision'])} "
                f"R={_render_rate(payload['recall'])} "
                f"C={_render_rate(payload['coverage'])}"
            )

    if report.by_basis:
        lines.append("")
        lines.append("  METRICS by screening basis:")
        for name, payload in sorted(report.by_basis.items()):
            lines.append(
                f"    {name:28s} labelled={payload['labelled']:<4} "
                f"P={_render_rate(payload['precision'])} "
                f"R={_render_rate(payload['recall'])} "
                f"C={_render_rate(payload['coverage'])}"
            )

    if report.boundary_metrics is not None:
        boundary = report.boundary_metrics
        lines.append("")
        lines.append(
            "  boundary stratum (deliberately oversampled — read as a failure-case "
            "catalogue, NOT as a performance estimate)"
        )
        lines.append(f"    labelled={boundary.labelled} "
                     f"uncertain auto={boundary.auto_uncertain} "
                     f"human={boundary.human_uncertain} "
                     f"FP={boundary.false_positive} FN={boundary.false_negative}")

    if report.stratification:
        lines.append("")
        lines.append("  by screening basis (raw counts):")
        for basis, stats in report.stratification.items():
            rate = stats["agreement_rate"]
            lines.append(
                f"    {basis:26s} n={stats['considered']:<4} "
                f"agreement={stats['agreement']}"
                + (f" ({rate:.1%})" if rate is not None else "")
            )

    if report.confusion:
        lines.append("")
        lines.append("  confusion (automatic -> human):")
        for entry in report.confusion:
            lines.append(
                f"    {entry.auto_decision:22s} -> {entry.human_decision:12s} "
                f"{entry.count:<4} ids={list(entry.paper_ids)[:12]}"
            )
    if report.auto_positive_human_negative:
        lines.append("")
        lines.append("  automatic positive / human negative (false positives):")
        for item in report.auto_positive_human_negative:
            lines.append(f"    #{item['paper_id']} {item['title'][:80]}")
            lines.append(f"        reason: {item['human_note'] or '(none recorded)'}")
    if report.auto_negative_human_positive:
        lines.append("")
        lines.append("  automatic negative / human positive (false negatives):")
        for item in report.auto_negative_human_positive:
            lines.append(f"    #{item['paper_id']} {item['title'][:80]}")
            lines.append(f"        reason: {item['human_note'] or '(none recorded)'}")
    if report.ambiguous_human_calls:
        lines.append("")
        lines.append("  human called these ambiguous:")
        for item in report.ambiguous_human_calls:
            lines.append(
                f"    #{item['paper_id']} auto={item['auto_decision']} "
                f"{item['title'][:70]} — {item['human_note'] or '(no note)'}"
            )
    if report.problems:
        lines.append("")
        lines.append("  sheet problems:")
        for problem in report.problems:
            lines.append(f"    ! {problem}")

    if report.diagnoses:
        lines.append("")
        lines.append("  DIAGNOSIS (description only — nothing has been changed)")
        for name, payload in sorted(report.diagnoses.items()):
            lines.append(
                f"    {name:22s} labelled={payload['labelled']:<4} "
                f"FP={payload['false_positives']:<3} FN={payload['false_negatives']:<3} "
                f"direction={payload['direction']}"
            )
            lines.append(f"      {payload['statement']}")
            for pattern in payload["patterns"]:
                lines.append(
                    f"      cluster: {pattern['error_kind']} by {pattern['attribute']} = "
                    f"{pattern['value']!r} n={pattern['count']} ids={pattern['paper_ids']}"
                )
            for caveat in payload["caveats"]:
                lines.append(f"      caveat: {caveat}")

    lines.append("")
    if report.labelled_rows == 0:
        lines.append(
            "  NOTE: no labels yet.  No performance claim can be made from this sheet."
        )
    elif report.labelled_rows < MIN_LABELS_FOR_RATE:
        lines.append(
            f"  NOTE: only {report.labelled_rows} labelled rows.  This can point at "
            "boundary cases; it cannot support changing a prompt or a threshold."
        )
    else:
        lines.append(
            "  NOTE: rates are descriptive of this sample.  Changing a prompt or a "
            "threshold on the strength of them is a separate, human decision."
        )
    return "\n".join(lines)


def _render_rate(payload: dict[str, Any]) -> str:
    """Compact rendering of a rate cell, honest about insufficient samples."""
    if payload["total"] == 0:
        return "n/a"
    if not payload["sufficient"]:
        return f"{payload['successes']}/{payload['total']}(insufficient)"
    interval = payload["wilson_95"]
    return f"{payload['value']:.2f}[{interval[0]:.2f},{interval[1]:.2f}]"


# --------------------------------------------------------------------------- #
# Diagnostics — describing what went wrong, never changing it
# --------------------------------------------------------------------------- #

#: How many errors of one kind before a pattern is worth calling a pattern.
#: Two mistakes that share a journal are a coincidence; four are a signal.
MIN_PATTERN_COUNT = 3

#: An error direction that dominates this share of a profile's errors is a
#: candidate for a boundary problem rather than scattered noise.
DOMINANT_SHARE = 0.75


@dataclass
class ErrorPattern:
    """A cluster of errors sharing one attribute, with the evidence behind it."""

    attribute: str
    value: str
    error_kind: str
    count: int
    of_total_errors: int
    paper_ids: tuple[int, ...]
    notes: tuple[str, ...]

    @property
    def share(self) -> float:
        return self.count / self.of_total_errors if self.of_total_errors else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute,
            "value": self.value,
            "error_kind": self.error_kind,
            "count": self.count,
            "share_of_that_error_kind": round(self.share, 4),
            "paper_ids": list(self.paper_ids),
            "notes": list(self.notes),
        }


@dataclass
class Diagnosis:
    """Read-only description of how a profile failed, pending human review.

    This deliberately produces *findings*, never fixes.  Every field here is a
    statement about observed labels; none of it is applied to a prompt, a
    threshold, or a profile file.  Acting on a finding is a separate, human step,
    and the report says so.
    """

    profile_id: str
    labelled: int
    false_positives: int
    false_negatives: int
    abstentions_from_human: int
    direction: str
    patterns: list[ErrorPattern]
    sample_adequate: bool
    conclusions_permitted: bool
    statement: str
    caveats: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "labelled": self.labelled,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "abstentions_from_human": self.abstentions_from_human,
            "direction": self.direction,
            "patterns": [pattern.as_dict() for pattern in self.patterns],
            "sample_adequate": self.sample_adequate,
            "conclusions_permitted": self.conclusions_permitted,
            "statement": self.statement,
            "caveats": list(self.caveats),
        }


def _cluster(
    items: list[dict[str, Any]],
    attribute: str,
    *,
    error_kind: str,
    total_errors: int,
) -> list[ErrorPattern]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        value = str(item.get(attribute) or "(unrecorded)").strip() or "(unrecorded)"
        buckets.setdefault(value, []).append(item)
    patterns: list[ErrorPattern] = []
    for value, group in buckets.items():
        if len(group) < MIN_PATTERN_COUNT:
            continue
        patterns.append(
            ErrorPattern(
                attribute=attribute,
                value=value,
                error_kind=error_kind,
                count=len(group),
                of_total_errors=total_errors,
                paper_ids=tuple(int(item["paper_id"]) for item in group),
                notes=tuple(
                    str(item.get("human_note") or "").strip()
                    for item in group
                    if str(item.get("human_note") or "").strip()
                ),
            )
        )
    return sorted(patterns, key=lambda pattern: (-pattern.count, pattern.value))


def diagnose(
    pairs: list[tuple[dict[str, str], AssessmentRow]],
    *,
    profile_id: str,
    errors: list[dict[str, Any]],
    total_errors: int,
    labelled: int,
) -> Diagnosis:
    """Describe a profile's errors without proposing any change.

    ``direction`` names what the evidence actually supports:

    * ``systematic_false_negative`` — the queue missed papers a human wants.  The
      costly direction for a reading radar.
    * ``systematic_false_positive`` — the queue surfaced papers a human does not
      want.  Costly in attention rather than in missed work.
    * ``mixed`` — both directions present, neither dominating.
    * ``insufficient`` — too few errors to say anything.

    A pattern is only reported when at least ``MIN_PATTERN_COUNT`` errors share
    an attribute; below that it is noise, and labelling noise a "systematic
    problem" is how a bad prompt edit gets justified.
    """
    positives = [item for item in errors if item.get("error_kind") == "false_positive"]
    negatives = [item for item in errors if item.get("error_kind") == "false_negative"]
    fp, fn = len(positives), len(negatives)

    caveats: list[str] = []
    if fp == 0 and fn == 0:
        direction = "none_observed" if labelled else "insufficient"
    elif total_errors == 0:
        direction = "insufficient"
    elif fn and fn / total_errors >= DOMINANT_SHARE:
        direction = "systematic_false_negative"
    elif fp and fp / total_errors >= DOMINANT_SHARE:
        direction = "systematic_false_positive"
    else:
        direction = "mixed"

    patterns: list[ErrorPattern] = []
    for attribute in ("auto_basis", "auto_confidence", "journal"):
        patterns.extend(
            _cluster(positives, attribute, error_kind="false_positive", total_errors=fp or 1)
        )
        patterns.extend(
            _cluster(negatives, attribute, error_kind="false_negative", total_errors=fn or 1)
        )
    patterns.sort(key=lambda pattern: (-pattern.count, pattern.error_kind, pattern.attribute))

    sample_adequate = labelled >= MIN_LABELS_FOR_RATE
    if not sample_adequate:
        caveats.append(
            f"only {labelled} labelled rows; below {MIN_LABELS_FOR_RATE} the rates are "
            "not reportable and no pattern here justifies a change"
        )
    if total_errors and total_errors < MIN_PATTERN_COUNT:
        caveats.append(
            f"only {total_errors} errors in total; no cluster can reach the "
            f"{MIN_PATTERN_COUNT}-error threshold"
        )
    if direction == "insufficient":
        caveats.append("the sample cannot distinguish a systematic problem from noise")

    if not labelled:
        statement = (
            "No labels yet: nothing can be said about this profile's accuracy or its "
            "error patterns."
        )
    elif not sample_adequate:
        statement = (
            f"{labelled} labelled rows, {fp} false positive(s) and {fn} false "
            "negative(s).  Enough to list cases, not enough to characterise the "
            "boundary."
        )
    elif direction == "systematic_false_negative":
        statement = (
            f"{fn} of {total_errors} errors are false negatives "
            f"({fn / total_errors:.0%}).  The queue is missing papers the human wants "
            "— the costly direction for a reading radar.  This is a description, not "
            "a diagnosis of cause."
        )
    elif direction == "systematic_false_positive":
        statement = (
            f"{fp} of {total_errors} errors are false positives "
            f"({fp / total_errors:.0%}).  The queue is surfacing papers the human "
            "filters out, which costs attention rather than missed work."
        )
    elif direction == "mixed":
        statement = (
            f"{fp} false positive(s) and {fn} false negative(s) across {labelled} "
            "labelled rows: both directions present, neither dominating."
        )
    else:
        statement = (
            f"No disagreement beyond abstentions across {labelled} labelled rows."
        )

    return Diagnosis(
        profile_id=profile_id,
        labelled=labelled,
        false_positives=fp,
        false_negatives=fn,
        abstentions_from_human=sum(
            1
            for row, _ in pairs
            if (row.get("human_decision") or "").strip().casefold() == "skip"
        ),
        direction=direction,
        patterns=patterns,
        sample_adequate=sample_adequate,
        conclusions_permitted=sample_adequate and total_errors > 0,
        statement=statement,
        caveats=caveats,
    )


def format_diagnosis(diagnosis: Diagnosis) -> str:
    lines = [
        f"Diagnosis — {diagnosis.profile_id}",
        f"  labelled rows        : {diagnosis.labelled}",
        f"  false positives      : {diagnosis.false_positives}",
        f"  false negatives      : {diagnosis.false_negatives}",
        f"  human abstentions    : {diagnosis.abstentions_from_human}",
        f"  direction            : {diagnosis.direction}",
        f"  sample adequate      : {diagnosis.sample_adequate}",
        f"  {diagnosis.statement}",
    ]
    if diagnosis.patterns:
        lines.append("  clusters (shared attribute, >= "
                     f"{MIN_PATTERN_COUNT} errors):")
        for pattern in diagnosis.patterns:
            lines.append(
                f"    {pattern.error_kind:15s} by {pattern.attribute:16s} "
                f"{pattern.value:22s} n={pattern.count} "
                f"({pattern.share:.0%} of that kind) ids={list(pattern.paper_ids)}"
            )
    else:
        lines.append("  clusters             : none reaching the threshold")
    for caveat in diagnosis.caveats:
        lines.append(f"  caveat: {caveat}")
    lines.append(
        "  NOTE: this is a description of the labels, not an instruction.  No prompt, "
        "threshold or profile file has been changed."
    )
    return "\n".join(lines)


def write_sheet(
    path: Path,
    rows: list[AssessmentRow],
    *,
    blind: bool = False,
    include_abstract: bool = True,
    stratum_of: dict[tuple[str, int], str] | None = None,
    design: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_sheet(
            rows,
            blind=blind,
            include_abstract=include_abstract,
            stratum_of=stratum_of,
        ),
        encoding="utf-8-sig",
    )
    strata = stratum_of or {}
    stratum_counts: dict[str, int] = {}
    for item in rows:
        key = strata.get((item.profile_id, item.paper_id), STRATUM_REPRESENTATIVE)
        stratum_counts[key] = stratum_counts.get(key, 0) + 1
    manifest = {
        "sheet": str(path),
        "created_at": datetime.now(UTC).isoformat(),
        "rows": len(rows),
        "blind": blind,
        "include_abstract": include_abstract,
        "profiles": sorted({item.profile_id for item in rows}),
        "paper_ids": [item.paper_id for item in rows],
        "decisions": dict(Counter(item.decision for item in rows)),
        "bases": dict(Counter(item.basis for item in rows)),
        "strata": stratum_counts,
        "design": dict(design or {}),
        "columns": list(SHEET_COLUMNS),
        "human_decisions_allowed": list(HUMAN_DECISIONS),
    }
    # The sidecar is what makes a blinded sheet decodable later; it is kept
    # beside the sheet rather than inside it so the sheet itself stays clean.
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


# --------------------------------------------------------------------------- #
# Screening basis: title-only and full screening are different claims
# --------------------------------------------------------------------------- #


@dataclass
class ScreeningBasisReport:
    """How much of a profile's judgement rests on the title alone.

    This matters more than it looks.  A title-only judgement is produced when no
    English abstract could be retrieved, and the screening contract then forbids
    a positive call — so those rows can essentially never be ``relevant``.  Left
    unmeasured, a batch with poor abstract coverage looks like a profile that
    *found nothing*, when what actually happened is that the queue never got to
    read most of what it was shown.
    """

    profile_id: str
    total: int
    by_basis: dict[str, int]
    by_basis_decision: dict[str, dict[str, int]]
    title_only_share: float
    papers_without_abstract: int
    papers_without_abstract_without_codes: int
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "total": self.total,
            "by_basis": self.by_basis,
            "by_basis_decision": self.by_basis_decision,
            "title_only_share": round(self.title_only_share, 4),
            "papers_without_abstract": self.papers_without_abstract,
            "papers_without_abstract_without_codes": self.papers_without_abstract_without_codes,
            "note": self.note,
        }


def measure_screening_basis(database: Path, profile_id: str) -> ScreeningBasisReport:
    """Read-only measurement of basis mix for one profile."""
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """SELECT COALESCE(NULLIF(a.screening_basis,''),'unrecorded') basis,
                      a.decision, COUNT(*) n,
                      SUM(CASE WHEN p.abstract IS NULL OR trim(p.abstract)='' THEN 1 ELSE 0 END)
                          no_abstract,
                      SUM(CASE WHEN (p.abstract IS NULL OR trim(p.abstract)='')
                                    AND p.jel_codes_json IN ('','[]','{}')
                               THEN 1 ELSE 0 END) no_abstract_no_codes
            FROM paper_profile_assessments a JOIN papers p ON p.id=a.paper_id
            WHERE a.profile_id=? GROUP BY basis, a.decision""",
            (profile_id,),
        ).fetchall()
    by_basis: dict[str, int] = {}
    by_basis_decision: dict[str, dict[str, int]] = {}
    total = 0
    no_abstract = 0
    no_abstract_no_codes = 0
    for row in rows:
        basis = str(row["basis"])
        count = int(row["n"])
        by_basis[basis] = by_basis.get(basis, 0) + count
        by_basis_decision.setdefault(basis, {})[str(row["decision"])] = count
        total += count
        no_abstract += int(row["no_abstract"] or 0)
        no_abstract_no_codes += int(row["no_abstract_no_codes"] or 0)
    title_only = by_basis.get("title_only", 0)
    note = ""
    if title_only:
        note = (
            "Title-only judgements cannot be positive under the shared screening "
            "contract, so they are not evidence about whether the queue's boundary "
            "is correct. Read them as missing data and separate them from full "
            "screenings before drawing any conclusion."
        )
    return ScreeningBasisReport(
        profile_id=profile_id,
        total=total,
        by_basis=by_basis,
        by_basis_decision=by_basis_decision,
        title_only_share=(title_only / total) if total else 0.0,
        papers_without_abstract=no_abstract,
        papers_without_abstract_without_codes=no_abstract_no_codes,
        note=note,
    )


def format_screening_basis(report: ScreeningBasisReport) -> str:
    lines = [
        f"Screening basis — {report.profile_id}",
        f"  judgements                 : {report.total}",
    ]
    for basis, count in sorted(report.by_basis.items(), key=lambda item: -item[1]):
        share = count / report.total if report.total else 0.0
        decisions = report.by_basis_decision.get(basis, {})
        detail = ", ".join(f"{key}={value}" for key, value in sorted(decisions.items()))
        lines.append(f"  {basis:26s}: {count:4} ({share:5.1%})  {detail}")
    lines.append(f"  title-only share           : {report.title_only_share:.1%}")
    lines.append(f"  judgements without abstract: {report.papers_without_abstract}")
    if report.note:
        lines.append(f"  NOTE: {report.note}")
    return "\n".join(lines)
