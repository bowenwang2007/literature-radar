# --- not part of the product core: see docs/PRODUCT_CORE.md ---
# RESEARCH (validation batches). CLI only; never on the product path.

from __future__ import annotations

import csv
import io
import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from eduecon_radar.db import Database, utcnow
from eduecon_radar.services.archive import sha256_file

VALIDATION_LABELS = ("include", "borderline", "exclude")
DIFFICULTIES = ("easy", "medium", "hard")
DEFAULT_STRATA = {"include": 0.3, "borderline": 0.3, "exclude": 0.4}


@dataclass(frozen=True)
class ValidationBatch:
    batch_id: int
    source_journals: tuple[str, ...]
    date_range: tuple[str | None, str | None]
    total_papers: int
    created_at: str
    ontology_version: str
    semantic_prompt_version: str
    composite_version: str
    status: str


@dataclass(frozen=True)
class ValidationSampleResult:
    batch: ValidationBatch
    annotation_csv: str
    stratum_counts: dict[str, int]


def apply_validation_migration(database: Database, migration_path: Path) -> None:
    sql = migration_path.read_text(encoding="utf-8")
    with database.connect() as connection:
        connection.executescript(sql)


class ValidationRepository:
    """Independent validation storage; it never writes the source paper database."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def initialize(self, migration_path: Path) -> None:
        apply_validation_migration(self.database, migration_path)

    def create_sample(
        self,
        *,
        source_database: Path,
        journals: tuple[str, ...] = (),
        year_from: int | None = None,
        year_to: int | None = None,
        size: int = 100,
        random_seed: int = 20260723,
    ) -> ValidationSampleResult:
        if not 1 <= size <= 5000:
            raise ValueError("Validation sample size must be between 1 and 5000")
        if year_from and year_to and year_from > year_to:
            raise ValueError("year_from cannot be later than year_to")
        source = source_database.resolve()
        if source == self.database.path.resolve():
            raise ValueError("Validation database must be separate from the source database")
        candidates = _read_source_candidates(
            source, journals=journals, year_from=year_from, year_to=year_to
        )
        if not candidates:
            raise ValueError("No completed AI decisions match the requested validation scope")
        selected = stratified_sample(
            candidates, size=min(size, len(candidates)), random_seed=random_seed
        )
        versions = {
            "ontology": _single_version(selected, "ontology_version"),
            "prompt": _single_version(selected, "semantic_prompt_version"),
            "composite": _single_version(selected, "composite_version"),
        }
        sampled_journals = tuple(sorted({str(item["journal_short_name"]) for item in selected}))
        now = utcnow()
        with self.database.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO validation_batches
                (source_database_path,source_database_sha256,source_journals_json,
                 date_from,date_to,total_papers,ontology_version,
                 semantic_prompt_version,composite_version,status,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,'annotation',?)""",
                (
                    str(source),
                    sha256_file(source),
                    json.dumps(sampled_journals, ensure_ascii=False),
                    f"{year_from:04d}-01-01" if year_from else None,
                    f"{year_to:04d}-12-31" if year_to else None,
                    len(selected),
                    versions["ontology"],
                    versions["prompt"],
                    versions["composite"],
                    now,
                ),
            )
            batch_id = int(cursor.lastrowid)
            for item in selected:
                connection.execute(
                    """INSERT INTO validation_items
                    (batch_id,source_paper_id,doi,original_title,chinese_title,
                     authors_json,original_abstract,journal,publication_date,
                     ai_decision,ai_reason,semantic_score,confidence,evidence_json,
                     semantic_json,sampled_stratum,created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        batch_id,
                        item["source_paper_id"],
                        item["doi"],
                        item["original_title"],
                        item["chinese_title"],
                        item["authors_json"],
                        item["original_abstract"],
                        item["journal"],
                        item["publication_date"],
                        item["ai_decision"],
                        item["ai_reason"],
                        item["semantic_score"],
                        item["confidence"],
                        json.dumps(item["evidence"], ensure_ascii=False, sort_keys=True),
                        json.dumps(item["semantic"], ensure_ascii=False, sort_keys=True),
                        item["ai_decision"],
                        now,
                    ),
                )
        batch = self.get_batch(batch_id)
        return ValidationSampleResult(
            batch=batch,
            annotation_csv=self.annotation_csv(batch_id),
            stratum_counts={
                label: sum(item["ai_decision"] == label for item in selected)
                for label in VALIDATION_LABELS
            },
        )

    def get_batch(self, batch_id: int) -> ValidationBatch:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM validation_batches WHERE id=?", (batch_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"Validation batch not found: {batch_id}")
        return _decode_batch(row)

    def list_batches(self) -> list[ValidationBatch]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM validation_batches ORDER BY created_at DESC,id DESC"
            ).fetchall()
        return [_decode_batch(row) for row in rows]

    def list_items(
        self,
        batch_id: int,
        *,
        page: int = 1,
        page_size: int = 20,
        annotated: bool | None = None,
    ) -> tuple[list[dict], int]:
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError("Invalid validation pagination")
        conditions = ["i.batch_id=?"]
        params: list[object] = [batch_id]
        if annotated is True:
            conditions.append("a.id IS NOT NULL")
        elif annotated is False:
            conditions.append("a.id IS NULL")
        where = " AND ".join(conditions)
        with self.database.connect() as connection:
            total = int(
                connection.execute(
                    f"""SELECT COUNT(*) FROM validation_items i
                    LEFT JOIN validation_annotations a ON a.validation_item_id=i.id
                    WHERE {where}""",  # noqa: S608 -- fixed conditions
                    params,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""SELECT i.*,a.label human_label,a.reviewer,a.notes,a.difficulty,
                a.updated_at annotated_at
                FROM validation_items i
                LEFT JOIN validation_annotations a ON a.validation_item_id=i.id
                WHERE {where}
                ORDER BY CASE i.sampled_stratum
                    WHEN 'exclude' THEN 0 WHEN 'borderline' THEN 1 ELSE 2 END,
                    i.semantic_score DESC,i.id
                LIMIT ? OFFSET ?""",  # noqa: S608 -- fixed conditions
                (*params, page_size, (page - 1) * page_size),
            ).fetchall()
        return [_decode_item(row) for row in rows], total

    def all_annotated_items(self, batch_id: int) -> list[dict]:
        items: list[dict] = []
        page = 1
        while True:
            chunk, total = self.list_items(
                batch_id, page=page, page_size=100, annotated=True
            )
            items.extend(chunk)
            if len(items) >= total:
                return items
            page += 1

    def save_annotation(
        self,
        item_id: int,
        *,
        label: str,
        reviewer: str,
        notes: str = "",
        difficulty: str,
    ) -> None:
        if label not in VALIDATION_LABELS:
            raise ValueError(f"Invalid validation label: {label}")
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"Invalid validation difficulty: {difficulty}")
        reviewer = reviewer.strip()
        if not reviewer:
            raise ValueError("Reviewer is required")
        now = utcnow()
        with self.database.connect() as connection:
            item = connection.execute(
                "SELECT batch_id FROM validation_items WHERE id=?", (item_id,)
            ).fetchone()
            if item is None:
                raise ValueError(f"Validation item not found: {item_id}")
            connection.execute(
                """INSERT INTO validation_annotations
                (validation_item_id,label,reviewer,notes,difficulty,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(validation_item_id) DO UPDATE SET
                  label=excluded.label,reviewer=excluded.reviewer,
                  notes=excluded.notes,difficulty=excluded.difficulty,
                  updated_at=excluded.updated_at""",
                (item_id, label, reviewer, notes.strip() or None, difficulty, now, now),
            )
            batch_id = int(item["batch_id"])
            remaining = int(
                connection.execute(
                    """SELECT COUNT(*) FROM validation_items i
                    WHERE i.batch_id=? AND NOT EXISTS(
                        SELECT 1 FROM validation_annotations a
                        WHERE a.validation_item_id=i.id)""",
                    (batch_id,),
                ).fetchone()[0]
            )
            connection.execute(
                "UPDATE validation_batches SET status=? WHERE id=?",
                ("completed" if remaining == 0 else "annotation", batch_id),
            )

    def annotation_csv(self, batch_id: int) -> str:
        items, total = self.list_items(batch_id, page=1, page_size=100)
        while len(items) < total:
            page, _ = self.list_items(
                batch_id, page=(len(items) // 100) + 1, page_size=100
            )
            items.extend(page)
        output = io.StringIO(newline="")
        fields = [
            "validation_item_id",
            "source_paper_id",
            "doi",
            "original_title",
            "chinese_title",
            "journal",
            "publication_date",
            "ai_decision",
            "semantic_score",
            "human_label",
            "reviewer",
            "difficulty",
            "notes",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow({field: item.get(field) for field in fields})
        return "\ufeff" + output.getvalue()


def stratified_sample(
    candidates: list[dict],
    *,
    size: int,
    random_seed: int,
    proportions: dict[str, float] | None = None,
) -> list[dict]:
    if size < 1:
        raise ValueError("Sample size must be positive")
    proportions = proportions or DEFAULT_STRATA
    if set(proportions) != set(VALIDATION_LABELS):
        raise ValueError("Sampling proportions must define include, borderline and exclude")
    if abs(sum(proportions.values()) - 1.0) > 1e-9:
        raise ValueError("Sampling proportions must sum to 1")
    unique = {int(item["source_paper_id"]): item for item in candidates}
    groups = {
        label: [item for item in unique.values() if item["ai_decision"] == label]
        for label in VALIDATION_LABELS
    }
    rng = random.Random(random_seed)
    for group in groups.values():
        rng.shuffle(group)
    requested = min(size, len(unique))
    targets = {
        "include": round(requested * proportions["include"]),
        "borderline": round(requested * proportions["borderline"]),
    }
    targets["exclude"] = requested - targets["include"] - targets["borderline"]
    selected: list[dict] = []
    selected_ids: set[int] = set()
    for label in VALIDATION_LABELS:
        for item in groups[label][: targets[label]]:
            selected.append(item)
            selected_ids.add(int(item["source_paper_id"]))
    if len(selected) < requested:
        remainder = [
            item
            for label in VALIDATION_LABELS
            for item in groups[label]
            if int(item["source_paper_id"]) not in selected_ids
        ]
        rng.shuffle(remainder)
        selected.extend(remainder[: requested - len(selected)])
    rng.shuffle(selected)
    return selected


def _read_source_candidates(
    source: Path,
    *,
    journals: tuple[str, ...],
    year_from: int | None,
    year_to: int | None,
) -> list[dict]:
    if not source.is_file():
        raise FileNotFoundError(f"Source database does not exist: {source}")
    uri = source.as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        semantic_ready = {
            "domain_composite_decisions",
            "domain_semantic_runs",
            "domain_evidence_runs",
        } <= tables
        translation_ready = "paper_translations" in tables
        conditions: list[str] = []
        params: list[object] = []
        if journals:
            placeholders = ",".join("?" for _ in journals)
            conditions.append(
                f"(j.short_name IN ({placeholders}) OR j.name IN ({placeholders}))"
            )
            params.extend(journals)
            params.extend(journals)
        publication = "COALESCE(p.issue_date,p.online_date)"
        if year_from:
            conditions.append(f"CAST(SUBSTR({publication},1,4) AS INTEGER)>=?")
            params.append(year_from)
        if year_to:
            conditions.append(f"CAST(SUBSTR({publication},1,4) AS INTEGER)<=?")
            params.append(year_to)
        where = " AND ".join(conditions) if conditions else "1=1"
        translation_join = (
            "LEFT JOIN paper_translations pt ON pt.paper_id=p.id"
            if translation_ready
            else ""
        )
        chinese_title = "pt.chinese_title" if translation_ready else "NULL"
        if semantic_ready:
            semantic_joins = """
            LEFT JOIN domain_composite_decisions dc ON dc.id=(
                SELECT x.id FROM domain_composite_decisions x
                WHERE x.paper_id=p.id ORDER BY x.created_at DESC,x.id DESC LIMIT 1)
            LEFT JOIN domain_semantic_runs fs ON fs.id=dc.fast_semantic_run_id
            LEFT JOIN domain_evidence_runs er ON er.id=dc.domain_evidence_run_id
            """
            semantic_fields = """dc.provisional_decision composite_decision,
                dc.policy_version composite_version,fs.result_json fast_result_json,
                fs.semantic_score,fs.confidence,fs.prompt_version semantic_prompt_version,
                er.evidence_json,er.ontology_version"""
        else:
            semantic_joins = ""
            semantic_fields = """NULL composite_decision,NULL composite_version,
                NULL fast_result_json,NULL semantic_score,NULL confidence,
                NULL semantic_prompt_version,NULL evidence_json,NULL ontology_version"""
        rows = connection.execute(
            f"""SELECT p.id source_paper_id,p.doi,p.title original_title,
            {chinese_title} chinese_title,p.authors_json,p.abstract original_abstract,
            j.name journal,j.short_name journal_short_name,{publication} publication_date,
            s.ai_decision legacy_decision,s.ai_reason legacy_reason,{semantic_fields}
            FROM papers p JOIN journals j ON j.id=p.journal_id
            LEFT JOIN screenings s ON s.id=(
                SELECT s2.id FROM screenings s2 WHERE s2.paper_id=p.id
                ORDER BY s2.screened_at DESC,s2.id DESC LIMIT 1)
            {translation_join} {semantic_joins}
            WHERE {where}""",  # noqa: S608 -- fragments are fixed; journal values parameterized
            params,
        ).fetchall()
    output: list[dict] = []
    for row in rows:
        item = dict(row)
        decision = item["composite_decision"] or item["legacy_decision"]
        if decision not in VALIDATION_LABELS:
            continue
        fast = _json_object(item.pop("fast_result_json"))
        evidence = _json_object(item.pop("evidence_json"))
        output.append(
            {
                **item,
                "ai_decision": decision,
                "ai_reason": fast.get("reason") or item["legacy_reason"],
                "semantic_score": item["semantic_score"],
                "confidence": item["confidence"],
                "semantic": fast,
                "evidence": evidence,
                "ontology_version": item["ontology_version"] or "unavailable",
                "semantic_prompt_version": item["semantic_prompt_version"] or "unavailable",
                "composite_version": item["composite_version"] or "legacy-screening",
            }
        )
    return output


def _json_object(raw: object) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(str(raw))
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _single_version(items: list[dict], field: str) -> str:
    versions = sorted({str(item[field]) for item in items})
    return versions[0] if len(versions) == 1 else "mixed:" + ",".join(versions)


def _decode_batch(row: sqlite3.Row) -> ValidationBatch:
    return ValidationBatch(
        batch_id=int(row["id"]),
        source_journals=tuple(json.loads(row["source_journals_json"])),
        date_range=(row["date_from"], row["date_to"]),
        total_papers=int(row["total_papers"]),
        created_at=str(row["created_at"]),
        ontology_version=str(row["ontology_version"]),
        semantic_prompt_version=str(row["semantic_prompt_version"]),
        composite_version=str(row["composite_version"]),
        status=str(row["status"]),
    )


def _decode_item(row: sqlite3.Row) -> dict:
    item = dict(row)
    item["validation_item_id"] = int(item["id"])
    item["authors"] = json.loads(item.pop("authors_json") or "[]")
    item["evidence"] = _json_object(item.pop("evidence_json"))
    item["semantic"] = _json_object(item.pop("semantic_json"))
    return item
