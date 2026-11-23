from __future__ import annotations

import csv
import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from eduecon_radar.services.archive import sha256_file
from eduecon_radar.services.doi import canonicalize_doi

DIFFICULTY_TAGS = {
    "clear_positive",
    "clear_negative",
    "education_false_friend",
    "hidden_education_topic",
    "no_abstract",
    "opaque_title",
    "labor_education_cross",
    "health_education_cross",
    "urban_education_cross",
    "domain_journal",
    "general_economics_journal",
    "rule_semantic_conflict_future",
    "metadata_sparse",
}

GOLD_SET_FIELDS = (
    "item_id",
    "paper_id",
    "doi",
    "original_title",
    "journal",
    "publication_date",
    "abstract_available",
    "gold_label",
    "difficulty_tags",
    "source_group",
    "reviewer",
    "reviewed_at",
    "reviewer_note",
    "source_database_sha256",
    "ontology_version_at_annotation",
    "annotation_version",
    "prior_ai_decision",
    "rule_evidence_tier",
    "review_priority_tier",
    "priority_score",
)


class GoldSetItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str
    paper_id: int | None = None
    doi: str | None = None
    original_title: str
    journal: str
    publication_date: str | None = None
    abstract_available: bool
    gold_label: Literal["include", "borderline", "exclude", "uncertain"] | None = None
    difficulty_tags: tuple[str, ...] = ()
    source_group: str
    reviewer: str | None = None
    reviewed_at: str | None = None
    reviewer_note: str | None = None
    source_database_sha256: str
    ontology_version_at_annotation: str
    annotation_version: str
    prior_ai_decision: str | None = None
    rule_evidence_tier: (
        Literal["strong_positive", "moderate_positive", "weak_or_mixed", "little_evidence"] | None
    ) = None
    review_priority_tier: Literal["urgent", "high", "normal", "low"] | None = None
    priority_score: float | None = None

    @model_validator(mode="after")
    def validate_annotation(self) -> GoldSetItem:
        if not self.item_id.strip():
            raise ValueError("item_id must not be empty")
        if not self.original_title.strip():
            raise ValueError("original_title must not be empty")
        for field_name in (
            "journal",
            "source_group",
            "source_database_sha256",
            "ontology_version_at_annotation",
            "annotation_version",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must not be empty")
        unknown_tags = set(self.difficulty_tags) - DIFFICULTY_TAGS
        if unknown_tags:
            raise ValueError(f"Unknown difficulty tags: {sorted(unknown_tags)}")
        if self.gold_label is not None and not (self.reviewer or "").strip():
            raise ValueError("reviewer is required for a labeled Gold Set item")
        return self


class GoldSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[GoldSetItem, ...]
    source_path: str

    @model_validator(mode="after")
    def unique_items(self) -> GoldSet:
        ids = [item.item_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("Gold Set item_id values must be unique")
        return self


class GoldSetLoader:
    def load(self, path: Path, *, require_labels: bool = False) -> GoldSet:
        if not path.is_file():
            raise FileNotFoundError(f"Gold Set file does not exist: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            missing = set(GOLD_SET_FIELDS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Gold Set is missing columns: {sorted(missing)}")
            rows = list(reader)
        items = tuple(self._parse_row(row, number + 2) for number, row in enumerate(rows))
        if require_labels and any(item.gold_label is None for item in items):
            raise ValueError("Formal evaluation requires every Gold Set row to have a human label")
        return GoldSet(items=items, source_path=str(path.resolve()))

    @staticmethod
    def _parse_row(row: dict[str, str], line: int) -> GoldSetItem:
        try:
            label = (row.get("gold_label") or "").strip() or None
            reviewer = (row.get("reviewer") or "").strip() or None
            paper_id = (row.get("paper_id") or "").strip()
            priority_score = (row.get("priority_score") or "").strip()
            return GoldSetItem(
                item_id=row.get("item_id", ""),
                paper_id=int(paper_id) if paper_id else None,
                doi=canonicalize_doi(row.get("doi")),
                original_title=row.get("original_title", ""),
                journal=row.get("journal", ""),
                publication_date=(row.get("publication_date") or "").strip() or None,
                abstract_available=(row.get("abstract_available") or "").strip().lower()
                in {"1", "true", "yes"},
                gold_label=label,
                difficulty_tags=tuple(
                    item.strip()
                    for item in (row.get("difficulty_tags") or "").split("|")
                    if item.strip()
                ),
                source_group=row.get("source_group", ""),
                reviewer=reviewer,
                reviewed_at=(row.get("reviewed_at") or "").strip() or None,
                reviewer_note=(row.get("reviewer_note") or "").strip() or None,
                source_database_sha256=row.get("source_database_sha256", ""),
                ontology_version_at_annotation=row.get("ontology_version_at_annotation", ""),
                annotation_version=row.get("annotation_version", ""),
                prior_ai_decision=(row.get("prior_ai_decision") or "").strip() or None,
                rule_evidence_tier=(row.get("rule_evidence_tier") or "").strip() or None,
                review_priority_tier=(row.get("review_priority_tier") or "").strip() or None,
                priority_score=float(priority_score) if priority_score else None,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Gold Set row at line {line}: {exc}") from exc


@dataclass(frozen=True)
class GoldSetTemplateManifest:
    source_database_sha256: str
    sample_seed: int
    sampling_strategy: str
    created_at: str
    row_count: int
    source_table_counts: dict[str, int]
    output_path: str


def create_gold_set_template(
    *,
    source_db: Path,
    output: Path,
    sample_size: int,
    random_seed: int,
    stratify: bool,
    include_existing_screening_metadata: bool,
    confirm_read_only_source: bool,
    overwrite: bool = False,
) -> GoldSetTemplateManifest:
    if not confirm_read_only_source:
        raise ValueError("Use --confirm-read-only-source for Gold Set sampling")
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    source_db = source_db.resolve()
    output = output.resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if not source_db.is_file():
        raise FileNotFoundError(f"Source database does not exist: {source_db}")
    if output == source_db or output.name.casefold() == ".env":
        raise ValueError("Gold Set output must be a separate non-production file")
    if output.suffix.casefold() != ".csv":
        raise ValueError("Gold Set output must use a .csv extension")
    if not overwrite and (output.exists() or manifest_path.exists()):
        raise FileExistsError(f"Gold Set output already exists: {output}")
    uri = source_db.as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Source database integrity_check failed")
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("papers", "screenings", "reviews")
        }
        rows = connection.execute(
            """SELECT p.id,p.doi,p.title,p.abstract,j.name journal,
            COALESCE(p.issue_date,p.online_date) publication_date,
            (SELECT s.ai_decision FROM screenings s WHERE s.paper_id=p.id
             ORDER BY s.screened_at DESC,s.id DESC LIMIT 1) prior_ai_decision
            FROM papers p JOIN journals j ON j.id=p.journal_id ORDER BY p.id"""
        ).fetchall()
    selected = _sample_rows(rows, sample_size, random_seed, stratify)
    source_hash = sha256_file(source_db)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with output.open(mode, encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=GOLD_SET_FIELDS)
        writer.writeheader()
        for index, row in enumerate(selected, start=1):
            writer.writerow(
                {
                    "item_id": f"sample-{index:04d}",
                    "paper_id": row["id"],
                    "doi": row["doi"] or "",
                    "original_title": row["title"],
                    "journal": row["journal"],
                    "publication_date": row["publication_date"] or "",
                    "abstract_available": bool(row["abstract"]),
                    "gold_label": "",
                    "difficulty_tags": "no_abstract" if not row["abstract"] else "",
                    "source_group": _source_group(row),
                    "reviewer": "",
                    "reviewed_at": "",
                    "reviewer_note": "",
                    "source_database_sha256": source_hash,
                    "ontology_version_at_annotation": "UNASSIGNED",
                    "annotation_version": "UNASSIGNED",
                    "prior_ai_decision": (
                        row["prior_ai_decision"] if include_existing_screening_metadata else ""
                    ),
                    "rule_evidence_tier": "",
                    "review_priority_tier": "",
                    "priority_score": "",
                }
            )
    manifest = GoldSetTemplateManifest(
        source_database_sha256=source_hash,
        sample_seed=random_seed,
        sampling_strategy="stratified" if stratify else "random",
        created_at=datetime.now(UTC).isoformat(),
        row_count=len(selected),
        source_table_counts=counts,
        output_path=str(output),
    )
    manifest_mode = "w" if overwrite else "x"
    with manifest_path.open(manifest_mode, encoding="utf-8") as target:
        json.dump(manifest.__dict__, target, ensure_ascii=False, indent=2, sort_keys=True)
        target.write("\n")
    return manifest


def _sample_rows(
    rows: list[sqlite3.Row], sample_size: int, seed: int, stratify: bool
) -> list[sqlite3.Row]:
    randomizer = random.Random(seed)
    if not stratify:
        return randomizer.sample(rows, min(sample_size, len(rows)))
    groups: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        groups.setdefault(_source_group(row), []).append(row)
    selected: list[sqlite3.Row] = []
    for name in sorted(groups):
        candidates = groups[name]
        selected.append(randomizer.choice(candidates))
        if len(selected) >= sample_size:
            return selected
    remaining = [row for row in rows if row not in selected]
    randomizer.shuffle(remaining)
    return selected + remaining[: max(0, sample_size - len(selected))]


def _source_group(row: sqlite3.Row) -> str:
    if not row["abstract"]:
        return "no_abstract"
    if row["prior_ai_decision"]:
        return f"ai_{row['prior_ai_decision']}"
    title = row["title"].casefold()
    return "title_education" if "education" in title else "random_unscreened"
