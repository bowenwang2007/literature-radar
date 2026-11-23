# --- not part of the product core: see docs/PRODUCT_CORE.md ---
# RESEARCH/GOVERNANCE (Gold Set). CLI only; never on the product path.

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from eduecon_radar.db import Database, utcnow
from eduecon_radar.services.archive import sha256_file
from eduecon_radar.services.deduplication import normalize_title
from eduecon_radar.services.doi import canonicalize_doi

LABELS = ("include", "borderline", "exclude")
LABEL_WEIGHT = {"exclude": 0, "borderline": 1, "include": 2}
PROTOCOL_VERSION = "gold-labeling-protocol-v1"
AI_FIELD_FRAGMENTS = (
    "screen",
    "semantic",
    "composite",
    "evidence",
    "confidence",
    "rule_score",
    "missed",
    "strong",
    "fast",
    "ai_",
)


@dataclass(frozen=True)
class CorpusPreparationResult:
    corpus_id: int | None
    corpus_version: str
    source_database_sha256: str
    sample_plan_sha256: str
    corpus_sha256: str
    population_size: int
    requested_size: int
    realized_size: int
    stratum_counts: dict[str, int]
    dry_run: bool
    source_database_unchanged: bool


@dataclass(frozen=True)
class AgreementMetrics:
    pair_count: int
    exact_agreement: float | None
    weighted_agreement: float | None
    cohen_kappa: float | None
    confusion_matrix: dict[str, dict[str, int]]
    include_exclude_extreme_flips: int
    confidence_disagreement: int
    median_elapsed_seconds: float | None
    warnings: tuple[str, ...]


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def load_sample_plan(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Gold sample plan must be a YAML object")
    if int(payload.get("sample_size", 0)) <= 0:
        raise ValueError("Gold sample plan requires a positive sample_size")
    if not isinstance(payload.get("journal_families"), dict):
        raise ValueError("Gold sample plan requires journal_families")
    return payload


def prepare_gold_corpus(
    *,
    source_database: Path,
    gold_database: Path,
    sample_plan_path: Path,
    output_manifest: Path | None,
    corpus_version: str | None = None,
    dry_run: bool = False,
    ontology_path: Path | None = None,
) -> CorpusPreparationResult:
    source = source_database.resolve()
    target = gold_database.resolve()
    if source == target:
        raise ValueError("Gold database must be independent from the source database")
    if not source.is_file():
        raise FileNotFoundError(f"Source database does not exist: {source}")
    if not dry_run and not target.is_file():
        raise FileNotFoundError(
            "Gold database does not exist; initialize the Gold profile before preparing a corpus"
        )
    plan = load_sample_plan(sample_plan_path)
    version = (corpus_version or str(plan.get("corpus_version") or "")).strip()
    if not version:
        raise ValueError("corpus_version is required")
    seed = int(plan.get("random_seed", 20260726))
    requested = int(plan["sample_size"])
    before = sha256_file(source)
    rows = _read_source_population(source)
    title_terms = _load_existing_ontology_terms(ontology_path) if ontology_path else ()
    classified = [_classify_source_row(row, plan, title_terms) for row in rows]
    selected = _sample_without_replacement(classified, requested, seed)
    population_counts = Counter(str(item["stratum"]) for item in classified)
    realized_counts = Counter(str(item["stratum"]) for item in selected)
    frozen = [
        _freeze_paper(
            item,
            source_sha256=before,
            corpus_version=version,
            population_size=population_counts[str(item["stratum"])],
            selected_size=realized_counts[str(item["stratum"])],
        )
        for item in selected
    ]
    after = sha256_file(source)
    if after != before:
        raise RuntimeError("Source database changed during read-only Gold corpus preparation")
    plan_body = {
        "plan_version": plan.get("plan_version", "gold-sample-plan-v1"),
        "corpus_version": version,
        "sample_size": requested,
        "random_seed": seed,
        "replacement": False,
        "journal_families": plan["journal_families"],
        "dimensions": plan.get("dimensions", []),
        "sampling_code_version": "phase-6.6d1-sampler-v1",
    }
    plan_sha = canonical_hash(plan_body)
    paper_hashes = [item["content_sha256"] for item in frozen]
    corpus_body = {
        "manifest_version": "gold-corpus-manifest-v1",
        "corpus_version": version,
        "source_database_sha256": before,
        "sample_plan_sha256": plan_sha,
        "population_size": len(classified),
        "requested_size": requested,
        "realized_size": len(frozen),
        "replacement": False,
        "paper_content_sha256": paper_hashes,
        "strata": {
            name: {
                "population_size": population_counts[name],
                "realized_size": realized_counts[name],
                "inclusion_probability": (
                    realized_counts[name] / population_counts[name]
                    if population_counts[name]
                    else 0.0
                ),
            }
            for name in sorted(population_counts)
        },
    }
    corpus_sha = canonical_hash(corpus_body)
    result = CorpusPreparationResult(
        corpus_id=None,
        corpus_version=version,
        source_database_sha256=before,
        sample_plan_sha256=plan_sha,
        corpus_sha256=corpus_sha,
        population_size=len(classified),
        requested_size=requested,
        realized_size=len(frozen),
        stratum_counts=dict(sorted(realized_counts.items())),
        dry_run=dry_run,
        source_database_unchanged=True,
    )
    if dry_run:
        _write_optional_manifest(
            output_manifest, {**asdict(result), "corpus_manifest": corpus_body}
        )
        return result
    repository = GoldGovernanceRepository(Database(target))
    corpus_id = repository.create_frozen_corpus(
        corpus_version=version,
        source_database_sha256=before,
        source_manifest={
            "database_sha256": before,
            "integrity_check": "ok",
            "paper_count": len(rows),
        },
        sample_plan=plan_body,
        sample_plan_sha256=plan_sha,
        corpus_manifest=corpus_body,
        corpus_sha256=corpus_sha,
        papers=frozen,
    )
    result = CorpusPreparationResult(**{**asdict(result), "corpus_id": corpus_id})
    _write_optional_manifest(output_manifest, {**asdict(result), "corpus_manifest": corpus_body})
    return result


def _write_optional_manifest(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    if path.exists():
        raise FileExistsError(f"Manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_source_population(path: Path) -> list[dict[str, Any]]:
    uri = path.as_uri() + "?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Source database integrity_check failed")
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        provenance = (
            """(SELECT json_group_array(json_object(
                    'field',pm.field_name,'source',pm.source_name,
                    'decision',pm.decision,'observed_at',pm.observed_at))
                FROM paper_metadata_provenance pm WHERE pm.paper_id=p.id)"""
            if "paper_metadata_provenance" in tables
            else "'[]'"
        )
        rows = connection.execute(
            f"""SELECT p.id source_paper_id,p.doi,p.fingerprint,p.title,p.abstract,
            p.authors_json,p.online_date,p.issue_date,p.source_url,p.open_access_url,
            p.last_seen_at,p.updated_at,p.keywords_json,p.jel_codes_json,
            j.name journal,j.short_name journal_short_name,j.issn,j.eissn,
            {provenance} provenance_json
            FROM papers p JOIN journals j ON j.id=p.journal_id ORDER BY p.id"""
        ).fetchall()
    return [dict(row) for row in rows]


def _load_existing_ontology_terms(path: Path | None) -> tuple[str, ...]:
    if path is None or not path.is_file():
        return ()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    terms: set[str] = set()

    def visit(value: object, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and key in {
            "term",
            "phrase",
            "canonical_label",
            "preferred_term",
            "surface_form",
        }:
            normalized = normalize_title(value)
            if len(normalized) >= 3:
                terms.add(normalized)

    visit(payload)
    return tuple(sorted(terms))


def _classify_source_row(
    row: dict[str, Any], plan: dict[str, Any], title_terms: tuple[str, ...]
) -> dict[str, Any]:
    abstract_status = _abstract_status(row.get("abstract"))
    title = normalize_title(str(row["title"]))
    title_term = any(term in title for term in title_terms)
    family = "other_or_low_prior"
    short_name = str(row["journal_short_name"])
    journal_name = str(row["journal"])
    for name, journals in plan["journal_families"].items():
        if short_name in journals or journal_name in journals:
            family = str(name)
            break
    publication_date = row.get("issue_date") or row.get("online_date")
    year = (
        int(str(publication_date)[:4])
        if publication_date and str(publication_date)[:4].isdigit()
        else None
    )
    year_band = f"{(year // 5) * 5}-{(year // 5) * 5 + 4}" if year else "unknown"
    stratum = f"{family}|{abstract_status}|title_term:{int(title_term)}|year:{year_band}"
    return {**row, "stratum": stratum, "abstract_status": abstract_status, "year": year}


def _sample_without_replacement(items: list[dict], size: int, seed: int) -> list[dict]:
    if size >= len(items):
        return sorted(items, key=lambda item: int(item["source_paper_id"]))
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[str(item["stratum"])].append(item)
    for name, members in groups.items():
        members.sort(
            key=lambda item: canonical_hash(
                {"seed": seed, "stratum": name, "paper_id": item["source_paper_id"]}
            )
        )
    selected: list[dict] = []
    positions = {name: 0 for name in groups}
    ordered_groups = sorted(groups)
    while len(selected) < size:
        progressed = False
        for name in ordered_groups:
            index = positions[name]
            if index < len(groups[name]):
                selected.append(groups[name][index])
                positions[name] += 1
                progressed = True
                if len(selected) == size:
                    break
        if not progressed:
            break
    return sorted(
        selected,
        key=lambda item: canonical_hash({"seed": seed, "paper_id": item["source_paper_id"]}),
    )


def _freeze_paper(
    item: dict[str, Any],
    *,
    source_sha256: str,
    corpus_version: str,
    population_size: int,
    selected_size: int,
) -> dict[str, Any]:
    doi = canonicalize_doi(item.get("doi"))
    abstract = str(item["abstract"]) if item.get("abstract") else None
    links = {
        "doi": f"https://doi.org/{doi}" if doi else None,
        "source": item.get("source_url"),
        "open_access": item.get("open_access_url"),
    }
    provenance_raw = json.loads(item.get("provenance_json") or "[]")
    provenance = [
        {
            "field": entry.get("field"),
            "source": entry.get("source"),
            "decision": entry.get("decision"),
            "observed_at": entry.get("observed_at"),
        }
        for entry in provenance_raw
        if isinstance(entry, dict)
    ]
    paper = {
        "gold_paper_id": f"{corpus_version}-paper-{int(item['source_paper_id']):06d}",
        "source_database_sha256": source_sha256,
        "source_paper_id": int(item["source_paper_id"]),
        "canonical_doi": doi,
        "fallback_fingerprint": str(item["fingerprint"]),
        "original_title": str(item["title"]),
        "normalized_title": normalize_title(str(item["title"])),
        "abstract": abstract,
        "abstract_length": len(abstract or ""),
        "abstract_quality_status": str(item["abstract_status"]),
        "journal": str(item["journal"]),
        "journal_short_name": str(item["journal_short_name"]),
        "issn": str(item.get("issn") or item.get("eissn") or "") or None,
        "publication_date": item.get("issue_date") or item.get("online_date"),
        "publication_year": item.get("year"),
        "authors": json.loads(item.get("authors_json") or "[]"),
        "metadata_provenance_summary": provenance,
        "source_links": links,
        "source_record_timestamp": item.get("last_seen_at") or item.get("updated_at"),
        "corpus_version": corpus_version,
        "sampling_stratum": str(item["stratum"]),
        "stratum_population_size": population_size,
        "inclusion_probability": selected_size / population_size,
        "duplicate_cluster_key": (
            f"doi:{doi}" if doi else f"fingerprint:{str(item['fingerprint']).lower()}"
        ),
    }
    _assert_no_ai_fields(paper)
    paper["content_sha256"] = canonical_hash(paper)
    return paper


def _abstract_status(value: object) -> str:
    if not value:
        return "missing"
    text = str(value).strip()
    if not text:
        return "missing"
    if len(text) < 200:
        return "short"
    if text.endswith(("…", "...")):
        return "likely_truncated"
    if "<" in text and ">" in text:
        return "html_or_xml"
    return "complete"


def _assert_no_ai_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in AI_FIELD_FRAGMENTS):
                raise ValueError(f"AI-derived field is forbidden in Gold corpus: {key}")
            _assert_no_ai_fields(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_ai_fields(child)


class GoldGovernanceRepository:
    def __init__(self, database: Database) -> None:
        self.database = database
        if not self.schema_ready(database):
            raise RuntimeError("Phase 6.6D1 Gold governance schema is not installed")

    @staticmethod
    def schema_ready(database: Database) -> bool:
        if not database.path.is_file():
            return False
        uri = database.path.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            names = {
                str(row[0])
                for row in connection.execute(
                    """SELECT name FROM sqlite_master WHERE type='table' AND name IN
                    ('gold_corpora','gold_corpus_papers','gold_presented_records',
                     'gold_annotation_revisions','gold_governance_snapshots',
                     'gold_split_manifests','gold_pilot_runs_v2')"""
                )
            }
        return len(names) == 7

    def create_frozen_corpus(
        self,
        *,
        corpus_version: str,
        source_database_sha256: str,
        source_manifest: dict,
        sample_plan: dict,
        sample_plan_sha256: str,
        corpus_manifest: dict,
        corpus_sha256: str,
        papers: list[dict],
    ) -> int:
        with self.database.connect() as connection:
            existing = connection.execute(
                "SELECT id,corpus_sha256 FROM gold_corpora WHERE corpus_version=?",
                (corpus_version,),
            ).fetchone()
            if existing:
                if str(existing["corpus_sha256"]) != corpus_sha256:
                    raise ValueError("The same corpus version cannot contain different content")
                return int(existing["id"])
            now = utcnow()
            cursor = connection.execute(
                """INSERT INTO gold_corpora
                (corpus_version,status,source_database_sha256,source_manifest_json,
                 sample_plan_json,sample_plan_sha256,corpus_manifest_json,corpus_sha256,
                 paper_count,created_at)
                VALUES (?,'building',?,?,?,?,?,?,?,?)""",
                (
                    corpus_version,
                    source_database_sha256,
                    canonical_json(source_manifest),
                    canonical_json(sample_plan),
                    sample_plan_sha256,
                    canonical_json(corpus_manifest),
                    corpus_sha256,
                    len(papers),
                    now,
                ),
            )
            corpus_id = int(cursor.lastrowid)
            for paper in papers:
                _assert_no_ai_fields(paper)
                connection.execute(
                    """INSERT INTO gold_corpus_papers
                    (corpus_id,gold_paper_id,source_paper_id,canonical_doi,
                     fallback_fingerprint,original_title,normalized_title,abstract,
                     abstract_length,abstract_quality_status,journal,journal_short_name,
                     issn,publication_date,publication_year,authors_json,
                     metadata_provenance_summary_json,source_links_json,
                     source_record_timestamp,sampling_stratum,stratum_population_size,
                     inclusion_probability,duplicate_cluster_key,content_sha256,created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        corpus_id,
                        paper["gold_paper_id"],
                        paper["source_paper_id"],
                        paper["canonical_doi"],
                        paper["fallback_fingerprint"],
                        paper["original_title"],
                        paper["normalized_title"],
                        paper["abstract"],
                        paper["abstract_length"],
                        paper["abstract_quality_status"],
                        paper["journal"],
                        paper["journal_short_name"],
                        paper["issn"],
                        paper["publication_date"],
                        paper["publication_year"],
                        canonical_json(paper["authors"]),
                        canonical_json(paper["metadata_provenance_summary"]),
                        canonical_json(paper["source_links"]),
                        paper["source_record_timestamp"],
                        paper["sampling_stratum"],
                        paper["stratum_population_size"],
                        paper["inclusion_probability"],
                        paper["duplicate_cluster_key"],
                        paper["content_sha256"],
                        now,
                    ),
                )
            connection.execute(
                "UPDATE gold_corpora SET status='frozen',frozen_at=? WHERE id=?",
                (utcnow(), corpus_id),
            )
        return corpus_id

    def list_corpora(self) -> list[dict]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM gold_corpora ORDER BY created_at DESC,id DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def inspect_corpus(self, corpus_version: str) -> dict:
        with self.database.connect() as connection:
            corpus = connection.execute(
                "SELECT * FROM gold_corpora WHERE corpus_version=?", (corpus_version,)
            ).fetchone()
            if corpus is None:
                raise ValueError("Gold corpus not found")
            strata = connection.execute(
                """SELECT sampling_stratum,COUNT(*) count,MIN(stratum_population_size) population,
                MIN(inclusion_probability) probability FROM gold_corpus_papers
                WHERE corpus_id=? GROUP BY sampling_stratum ORDER BY sampling_stratum""",
                (int(corpus["id"]),),
            ).fetchall()
        return {
            "corpus": dict(corpus),
            "strata": [dict(row) for row in strata],
            "ai_fields_exposed": False,
            "annotation_query_fields": (
                "original_title",
                "abstract",
                "abstract_quality_status",
                "publication_year",
            ),
        }

    def verify_corpus(self, corpus_version: str) -> dict:
        inspected = self.inspect_corpus(corpus_version)
        corpus = inspected["corpus"]
        with self.database.connect() as connection:
            papers = connection.execute(
                "SELECT * FROM gold_corpus_papers WHERE corpus_id=? ORDER BY id",
                (int(corpus["id"]),),
            ).fetchall()
        hashes = []
        errors: list[str] = []
        for row in papers:
            content = _paper_content_from_row(
                row, str(corpus["source_database_sha256"]), corpus_version
            )
            actual = canonical_hash(content)
            hashes.append(actual)
            if actual != str(row["content_sha256"]):
                errors.append(f"paper_hash_mismatch:{row['gold_paper_id']}")
        manifest = json.loads(corpus["corpus_manifest_json"])
        if hashes != manifest.get("paper_content_sha256"):
            errors.append("manifest_paper_hashes_mismatch")
        if canonical_hash(manifest) != str(corpus["corpus_sha256"]):
            errors.append("corpus_sha256_mismatch")
        return {
            "corpus_version": corpus_version,
            "paper_count": len(papers),
            "corpus_sha256": corpus["corpus_sha256"],
            "valid": not errors,
            "errors": errors,
        }

    def ensure_presentations(
        self,
        *,
        corpus_version: str,
        reviewer_id: str,
        annotation_round: int = 1,
        repeat_fraction: float = 0.10,
        random_seed: int = 20260726,
    ) -> int:
        reviewer = reviewer_id.strip()
        if not reviewer:
            raise ValueError("reviewer identity is required")
        if annotation_round not in {1, 2}:
            raise ValueError("annotation_round must be 1 or 2")
        if not 0 <= repeat_fraction <= 0.25:
            raise ValueError("repeat_fraction must be between 0 and 0.25")
        with self.database.connect() as connection:
            corpus = connection.execute(
                "SELECT id FROM gold_corpora WHERE corpus_version=?", (corpus_version,)
            ).fetchone()
            if corpus is None:
                raise ValueError("Gold corpus not found")
            corpus_id = int(corpus["id"])
            count = int(
                connection.execute(
                    """SELECT COUNT(*) FROM gold_presented_records
                    WHERE corpus_id=? AND reviewer_id=? AND annotation_round=?""",
                    (corpus_id, reviewer, annotation_round),
                ).fetchone()[0]
            )
            if count:
                return count
            papers = [
                dict(row)
                for row in connection.execute(
                    "SELECT id,gold_paper_id FROM gold_corpus_papers WHERE corpus_id=? ORDER BY id",
                    (corpus_id,),
                ).fetchall()
            ]
            schedule = _presentation_schedule(
                papers,
                repeat_fraction=repeat_fraction if annotation_round == 1 else 0.0,
                seed=random_seed,
                reviewer=reviewer,
                annotation_round=annotation_round,
            )
            for order, record in enumerate(schedule, start=1):
                connection.execute(
                    """INSERT INTO gold_presented_records
                    (corpus_id,corpus_paper_id,reviewer_id,annotation_round,
                     presented_record_id,presentation_order,hidden_repeat,
                     repeat_of_presented_record_id,created_at)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        corpus_id,
                        record["paper_id"],
                        reviewer,
                        annotation_round,
                        record["presented_record_id"],
                        order,
                        int(record["hidden_repeat"]),
                        record.get("repeat_of_presented_record_id"),
                        utcnow(),
                    ),
                )
        return len(schedule)

    def next_blind_record(
        self,
        *,
        corpus_version: str,
        reviewer_id: str,
        annotation_round: int = 1,
    ) -> dict | None:
        with self.database.connect() as connection:
            row = connection.execute(
                """SELECT pr.presented_record_id,pr.presentation_order,p.original_title,
                p.abstract,p.abstract_quality_status,p.publication_year
                FROM gold_presented_records pr
                JOIN gold_corpora c ON c.id=pr.corpus_id
                JOIN gold_corpus_papers p ON p.id=pr.corpus_paper_id
                WHERE c.corpus_version=? AND pr.reviewer_id=? AND pr.annotation_round=?
                AND NOT EXISTS (
                    SELECT 1 FROM gold_annotation_revisions a
                    WHERE a.presented_record_id=pr.presented_record_id
                    AND a.reviewer_id=pr.reviewer_id
                    AND a.annotation_round=pr.annotation_round AND a.active=1
                )
                ORDER BY pr.presentation_order LIMIT 1""",
                (corpus_version, reviewer_id.strip(), annotation_round),
            ).fetchone()
        if row is None:
            return None
        return {
            **dict(row),
            "ai_visible": False,
            "journal_visible": False,
            "source_visible": False,
            "repeat_visible": False,
        }

    def ensure_review_presentations(
        self,
        *,
        corpus_version: str,
        reviewer_id: str,
        random_seed: int = 20260726,
    ) -> int:
        reviewer = reviewer_id.strip()
        if not reviewer:
            raise ValueError("reviewer identity is required")
        self._enqueue_repeat_disagreements(corpus_version)
        with self.database.connect() as connection:
            corpus = connection.execute(
                "SELECT id FROM gold_corpora WHERE corpus_version=?", (corpus_version,)
            ).fetchone()
            if corpus is None:
                raise ValueError("Gold corpus not found")
            corpus_id = int(corpus["id"])
            existing = int(
                connection.execute(
                    """SELECT COUNT(*) FROM gold_presented_records
                    WHERE corpus_id=? AND reviewer_id=? AND annotation_round=2""",
                    (corpus_id, reviewer),
                ).fetchone()[0]
            )
            if existing:
                return existing
            papers = [
                dict(row)
                for row in connection.execute(
                    """SELECT DISTINCT p.id,p.gold_paper_id FROM gold_review_queue q
                    JOIN gold_corpus_papers p ON p.id=q.corpus_paper_id
                    WHERE q.corpus_id=? AND q.status='open' ORDER BY p.id""",
                    (corpus_id,),
                ).fetchall()
            ]
            ordered = sorted(
                papers,
                key=lambda item: canonical_hash(
                    {
                        "seed": random_seed,
                        "reviewer": reviewer,
                        "gold_paper_id": item["gold_paper_id"],
                    }
                ),
            )
            for order, paper in enumerate(ordered, start=1):
                connection.execute(
                    """INSERT INTO gold_presented_records
                    (corpus_id,corpus_paper_id,reviewer_id,annotation_round,
                     presented_record_id,presentation_order,hidden_repeat,
                     repeat_of_presented_record_id,created_at)
                    VALUES (?,?,?,2,?,?,0,NULL,?)""",
                    (
                        corpus_id,
                        int(paper["id"]),
                        reviewer,
                        canonical_hash(
                            {
                                "seed": random_seed,
                                "reviewer": reviewer,
                                "round": 2,
                                "gold_paper_id": paper["gold_paper_id"],
                            }
                        )[:24],
                        order,
                        utcnow(),
                    ),
                )
        return len(ordered)

    def next_review_record(
        self, *, corpus_version: str, reviewer_id: str
    ) -> dict | None:
        with self.database.connect() as connection:
            row = connection.execute(
                """SELECT pr.presented_record_id,pr.presentation_order,p.id paper_id,
                p.gold_paper_id,p.original_title,p.abstract,p.abstract_quality_status,
                p.publication_year,p.journal,p.canonical_doi,
                p.metadata_provenance_summary_json
                FROM gold_presented_records pr
                JOIN gold_corpora c ON c.id=pr.corpus_id
                JOIN gold_corpus_papers p ON p.id=pr.corpus_paper_id
                WHERE c.corpus_version=? AND pr.reviewer_id=? AND pr.annotation_round=2
                AND NOT EXISTS (
                    SELECT 1 FROM gold_annotation_revisions a
                    WHERE a.presented_record_id=pr.presented_record_id
                    AND a.reviewer_id=pr.reviewer_id AND a.annotation_round=2 AND a.active=1
                ) ORDER BY pr.presentation_order LIMIT 1""",
                (corpus_version, reviewer_id.strip()),
            ).fetchone()
            if row is None:
                return None
            prior = connection.execute(
                """SELECT reviewer_id,annotation_round,label,reviewer_confidence,
                evidence_sufficiency,reason_codes_json,optional_notes
                FROM gold_annotation_revisions
                WHERE corpus_paper_id=? AND annotation_round=1 AND active=1
                ORDER BY submitted_at,id""",
                (int(row["paper_id"]),),
            ).fetchall()
        return {
            **dict(row),
            "metadata_provenance": json.loads(row["metadata_provenance_summary_json"]),
            "prior_human_annotations": [
                {
                    **dict(annotation),
                    "reason_codes": json.loads(annotation["reason_codes_json"]),
                }
                for annotation in prior
            ],
            "ai_visible": False,
        }

    def submit_annotation(
        self,
        *,
        presented_record_id: str,
        reviewer_id: str,
        annotation_round: int,
        label: str | None,
        education_centrality: str | None,
        economics_relevance: str | None,
        evidence_sufficiency: str | None,
        reviewer_confidence: str | None,
        reason_codes: list[str] | tuple[str, ...],
        optional_notes: str | None,
        started_at: str,
        marked_for_review: bool = False,
        deferred: bool = False,
        protocol_version: str,
        protocol_sha256: str,
    ) -> int:
        if not reviewer_id.strip():
            raise ValueError("reviewer identity is required")
        if not deferred and label not in LABELS:
            raise ValueError("A submitted annotation requires include/borderline/exclude")
        if deferred:
            label = education_centrality = economics_relevance = None
            evidence_sufficiency = reviewer_confidence = None
        submitted = utcnow()
        elapsed = _elapsed_seconds(started_at, submitted)
        with self.database.connect() as connection:
            presentation = connection.execute(
                """SELECT pr.*,p.id paper_id FROM gold_presented_records pr
                JOIN gold_corpus_papers p ON p.id=pr.corpus_paper_id
                WHERE pr.presented_record_id=? AND pr.reviewer_id=?
                AND pr.annotation_round=?""",
                (presented_record_id, reviewer_id.strip(), annotation_round),
            ).fetchone()
            if presentation is None:
                raise ValueError("Presented record is not available to this reviewer/round")
            previous = connection.execute(
                """SELECT id,revision_number FROM gold_annotation_revisions
                WHERE presented_record_id=? AND reviewer_id=? AND annotation_round=?
                AND active=1""",
                (presented_record_id, reviewer_id.strip(), annotation_round),
            ).fetchone()
            revision = int(previous["revision_number"]) + 1 if previous else 1
            if previous:
                connection.execute(
                    "UPDATE gold_annotation_revisions SET active=0 WHERE id=?",
                    (int(previous["id"]),),
                )
            cursor = connection.execute(
                """INSERT INTO gold_annotation_revisions
                (corpus_id,corpus_paper_id,presented_record_id,annotation_round,
                 reviewer_id,revision_number,label,education_centrality,
                 economics_relevance,evidence_sufficiency,reviewer_confidence,
                 reason_codes_json,optional_notes,disposition,marked_for_review,
                 source_revealed,reveal_timestamp,presentation_order,started_at,
                 submitted_at,elapsed_seconds,protocol_version,protocol_sha256,
                 supersedes_annotation_id,active,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,NULL,?,?,?,?,?,?,?,1,?)""",
                (
                    int(presentation["corpus_id"]),
                    int(presentation["paper_id"]),
                    presented_record_id,
                    annotation_round,
                    reviewer_id.strip(),
                    revision,
                    label,
                    education_centrality,
                    economics_relevance,
                    evidence_sufficiency,
                    reviewer_confidence,
                    canonical_json(sorted(set(reason_codes))),
                    (optional_notes or "").strip() or None,
                    "deferred" if deferred else "submitted",
                    int(marked_for_review),
                    int(presentation["presentation_order"]),
                    started_at,
                    submitted,
                    elapsed,
                    protocol_version,
                    protocol_sha256.upper(),
                    int(previous["id"]) if previous else None,
                    utcnow(),
                ),
            )
            annotation_id = int(cursor.lastrowid)
            reasons = _review_reasons(
                label=label,
                centrality=education_centrality,
                sufficiency=evidence_sufficiency,
                confidence=reviewer_confidence,
                reason_codes=reason_codes,
                marked=marked_for_review,
                deferred=deferred,
            )
            for reason in reasons:
                connection.execute(
                    """INSERT OR IGNORE INTO gold_review_queue
                    (corpus_id,corpus_paper_id,queue_reason,source_annotation_id,
                     status,created_at) VALUES (?,?,?,?, 'open',?)""",
                    (
                        int(presentation["corpus_id"]),
                        int(presentation["paper_id"]),
                        reason,
                        annotation_id,
                        utcnow(),
                    ),
                )
            if annotation_round == 2 and not deferred:
                labels = connection.execute(
                    """SELECT DISTINCT label FROM gold_annotation_revisions
                    WHERE corpus_paper_id=? AND active=1 AND disposition='submitted'""",
                    (int(presentation["paper_id"]),),
                ).fetchall()
                if len(labels) == 1 and not reasons:
                    connection.execute(
                        """UPDATE gold_review_queue SET status='resolved',resolved_at=?
                        WHERE corpus_id=? AND corpus_paper_id=? AND status='open'""",
                        (
                            utcnow(),
                            int(presentation["corpus_id"]),
                            int(presentation["paper_id"]),
                        ),
                    )
                elif len(labels) > 1:
                    connection.execute(
                        """INSERT OR IGNORE INTO gold_review_queue
                        (corpus_id,corpus_paper_id,queue_reason,source_annotation_id,
                         status,created_at) VALUES (?,?,?,?, 'open',?)""",
                        (
                            int(presentation["corpus_id"]),
                            int(presentation["paper_id"]),
                            "round_disagreement",
                            annotation_id,
                            utcnow(),
                        ),
                    )
        return annotation_id

    def reveal_source(self, *, annotation_id: int, reviewer_id: str) -> dict:
        with self.database.connect() as connection:
            row = connection.execute(
                """SELECT a.id,a.reviewer_id,p.journal,p.journal_short_name,p.issn,
                p.canonical_doi,p.source_links_json,p.metadata_provenance_summary_json
                FROM gold_annotation_revisions a
                JOIN gold_corpus_papers p ON p.id=a.corpus_paper_id
                WHERE a.id=? AND a.disposition='submitted' AND a.active=1""",
                (annotation_id,),
            ).fetchone()
            if row is None or str(row["reviewer_id"]) != reviewer_id.strip():
                raise ValueError("Source information is available only after your submitted label")
            connection.execute(
                """INSERT OR IGNORE INTO gold_source_reveals
                (annotation_id,reviewer_id,revealed_at) VALUES (?,?,?)""",
                (annotation_id, reviewer_id.strip(), utcnow()),
            )
        return {
            "journal": row["journal"],
            "journal_short_name": row["journal_short_name"],
            "issn": row["issn"],
            "doi": row["canonical_doi"],
            "source_links": json.loads(row["source_links_json"]),
            "metadata_provenance": json.loads(row["metadata_provenance_summary_json"]),
        }

    def annotation_status(self, corpus_version: str) -> dict:
        self._enqueue_repeat_disagreements(corpus_version)
        with self.database.connect() as connection:
            corpus = connection.execute(
                "SELECT id,paper_count FROM gold_corpora WHERE corpus_version=?",
                (corpus_version,),
            ).fetchone()
            if corpus is None:
                raise ValueError("Gold corpus not found")
            corpus_id = int(corpus["id"])
            presented = int(
                connection.execute(
                    "SELECT COUNT(*) FROM gold_presented_records WHERE corpus_id=?",
                    (corpus_id,),
                ).fetchone()[0]
            )
            active = connection.execute(
                """SELECT a.*,EXISTS(
                    SELECT 1 FROM gold_source_reveals r WHERE r.annotation_id=a.id
                ) revealed FROM gold_annotation_revisions a
                WHERE a.corpus_id=? AND a.active=1""",
                (corpus_id,),
            ).fetchall()
            open_queue = connection.execute(
                """SELECT queue_reason,COUNT(*) count FROM gold_review_queue
                WHERE corpus_id=? AND status='open' GROUP BY queue_reason""",
                (corpus_id,),
            ).fetchall()
            arbitrations = int(
                connection.execute(
                    """SELECT COUNT(*) FROM gold_arbitration_revisions
                    WHERE corpus_id=? AND active=1""",
                    (corpus_id,),
                ).fetchone()[0]
            )
            first_round_complete = int(
                connection.execute(
                    """SELECT COUNT(DISTINCT pr.corpus_paper_id)
                    FROM gold_presented_records pr
                    JOIN gold_annotation_revisions a
                      ON a.presented_record_id=pr.presented_record_id AND a.active=1
                    WHERE pr.corpus_id=? AND pr.annotation_round=1
                    AND pr.hidden_repeat=0 AND a.disposition='submitted'""",
                    (corpus_id,),
                ).fetchone()[0]
            )
        labels = Counter(row["label"] for row in active if row["label"])
        confidences = Counter(
            row["reviewer_confidence"] for row in active if row["reviewer_confidence"]
        )
        elapsed = [int(row["elapsed_seconds"]) for row in active]
        logical = sum(
            bool(
                _logical_inconsistencies(
                    row["label"],
                    row["education_centrality"],
                    row["evidence_sufficiency"],
                    row["reviewer_confidence"],
                )
            )
            for row in active
        )
        return {
            "corpus_size": int(corpus["paper_count"]),
            "unique_papers": int(corpus["paper_count"]),
            "presented_records": presented,
            "first_round_complete": first_round_complete,
            "repeated_records_complete": self._completed_repeat_count(corpus_id),
            "review_queue": sum(int(row["count"]) for row in open_queue),
            "review_queue_reasons": {
                str(row["queue_reason"]): int(row["count"]) for row in open_queue
            },
            "arbitration_queue": len(self.arbitration_queue(corpus_version)),
            "unresolved": self.unresolved_count(corpus_version),
            "label_distribution": dict(labels),
            "confidence_distribution": dict(confidences),
            "average_annotation_time": statistics.fmean(elapsed) if elapsed else None,
            "source_reveal_rate": (
                sum(int(row["revealed"]) for row in active) / len(active) if active else None
            ),
            "duplicate_agreement": asdict(self.calculate_agreement(corpus_version)),
            "missing_fields": sum(
                row["disposition"] == "submitted"
                and any(
                    row[name] is None
                    for name in (
                        "label",
                        "education_centrality",
                        "economics_relevance",
                        "evidence_sufficiency",
                        "reviewer_confidence",
                    )
                )
                for row in active
            ),
            "logical_inconsistencies": logical,
            "adjudicated": arbitrations,
        }

    def calculate_agreement(self, corpus_version: str) -> AgreementMetrics:
        pairs = self._repeat_pairs(corpus_version)
        observed = [
            (left["label"], right["label"])
            for left, right in pairs
            if left["label"] and right["label"]
        ]
        matrix = {left: {right: 0 for right in LABELS} for left in LABELS}
        for left, right in observed:
            matrix[left][right] += 1
        exact = _ratio(sum(left == right for left, right in observed), len(observed))
        weighted = _ratio(
            sum(1 - abs(LABEL_WEIGHT[left] - LABEL_WEIGHT[right]) / 2 for left, right in observed),
            len(observed),
        )
        kappa = _cohen_kappa(observed)
        extreme = sum({left, right} == {"include", "exclude"} for left, right in observed)
        confidence_disagreement = sum(
            left["reviewer_confidence"] != right["reviewer_confidence"] for left, right in pairs
        )
        elapsed = [
            int(item["elapsed_seconds"])
            for pair in pairs
            for item in pair
            if item["elapsed_seconds"] is not None
        ]
        warnings = []
        if len(observed) < 20:
            warnings.append("small_repeat_sample_no_precision_claim")
        if not observed:
            warnings.append("no_completed_repeat_pairs")
        return AgreementMetrics(
            pair_count=len(observed),
            exact_agreement=exact,
            weighted_agreement=weighted,
            cohen_kappa=kappa,
            confusion_matrix=matrix,
            include_exclude_extreme_flips=extreme,
            confidence_disagreement=confidence_disagreement,
            median_elapsed_seconds=statistics.median(elapsed) if elapsed else None,
            warnings=tuple(warnings),
        )

    def review_queue(self, corpus_version: str) -> list[dict]:
        self._enqueue_repeat_disagreements(corpus_version)
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT q.id,q.queue_reason,p.gold_paper_id,p.original_title,p.abstract,
                p.journal,p.canonical_doi FROM gold_review_queue q
                JOIN gold_corpora c ON c.id=q.corpus_id
                JOIN gold_corpus_papers p ON p.id=q.corpus_paper_id
                WHERE c.corpus_version=? AND q.status='open'
                ORDER BY q.created_at,q.id""",
                (corpus_version,),
            ).fetchall()
        return [dict(row) for row in rows]

    def arbitration_queue(self, corpus_version: str) -> list[dict]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT p.id,p.gold_paper_id,p.original_title,
                json_group_array(json_object(
                    'annotation_id',a.id,'round',a.annotation_round,
                    'reviewer',a.reviewer_id,'label',a.label,
                    'confidence',a.reviewer_confidence)) annotations_json
                FROM gold_corpus_papers p JOIN gold_corpora c ON c.id=p.corpus_id
                JOIN gold_annotation_revisions a ON a.corpus_paper_id=p.id AND a.active=1
                LEFT JOIN gold_arbitration_revisions ar
                  ON ar.corpus_paper_id=p.id AND ar.active=1
                WHERE c.corpus_version=? AND a.disposition='submitted' AND ar.id IS NULL
                GROUP BY p.id HAVING COUNT(DISTINCT a.label)>1
                ORDER BY p.id""",
                (corpus_version,),
            ).fetchall()
        return [{**dict(row), "annotations": json.loads(row["annotations_json"])} for row in rows]

    def arbitrate(
        self,
        *,
        corpus_version: str,
        gold_paper_id: str,
        final_gold_label: str,
        basis: str,
        arbitrator_id: str,
        protocol_version: str,
        protocol_sha256: str,
    ) -> int:
        if final_gold_label not in LABELS:
            raise ValueError("Invalid final Gold label")
        if not basis.strip() or not arbitrator_id.strip():
            raise ValueError("Arbitration basis and identity are required")
        with self.database.connect() as connection:
            paper = connection.execute(
                """SELECT p.id,p.corpus_id FROM gold_corpus_papers p
                JOIN gold_corpora c ON c.id=p.corpus_id
                WHERE c.corpus_version=? AND p.gold_paper_id=?""",
                (corpus_version, gold_paper_id),
            ).fetchone()
            if paper is None:
                raise ValueError("Gold paper not found")
            previous = connection.execute(
                """SELECT id FROM gold_arbitration_revisions
                WHERE corpus_paper_id=? AND active=1""",
                (int(paper["id"]),),
            ).fetchone()
            if previous:
                connection.execute(
                    "UPDATE gold_arbitration_revisions SET active=0 WHERE id=?",
                    (int(previous["id"]),),
                )
            cursor = connection.execute(
                """INSERT INTO gold_arbitration_revisions
                (corpus_id,corpus_paper_id,final_gold_label,basis,arbitrator_id,
                 protocol_version,protocol_sha256,supersedes_arbitration_id,
                 active,created_at) VALUES (?,?,?,?,?,?,?,?,1,?)""",
                (
                    int(paper["corpus_id"]),
                    int(paper["id"]),
                    final_gold_label,
                    basis.strip(),
                    arbitrator_id.strip(),
                    protocol_version,
                    protocol_sha256.upper(),
                    int(previous["id"]) if previous else None,
                    utcnow(),
                ),
            )
            connection.execute(
                """UPDATE gold_review_queue SET status='resolved',resolved_at=?
                WHERE corpus_id=? AND corpus_paper_id=? AND status='open'""",
                (utcnow(), int(paper["corpus_id"]), int(paper["id"])),
            )
        return int(cursor.lastrowid)

    def build_duplicate_clusters(self, corpus_version: str) -> dict:
        with self.database.connect() as connection:
            corpus = connection.execute(
                "SELECT id FROM gold_corpora WHERE corpus_version=?", (corpus_version,)
            ).fetchone()
            if corpus is None:
                raise ValueError("Gold corpus not found")
            corpus_id = int(corpus["id"])
            papers = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM gold_corpus_papers WHERE corpus_id=? ORDER BY id",
                    (corpus_id,),
                )
            ]
            groups: dict[str, list[dict]] = defaultdict(list)
            for paper in papers:
                groups[str(paper["duplicate_cluster_key"])].append(paper)
            candidate_groups = _title_duplicate_candidates(papers)
            created = 0
            for key, members in sorted(groups.items()):
                status = "confirmed" if len(members) > 1 else "not_duplicate"
                created += self._insert_cluster(
                    connection, corpus_id, key, status, {"basis": key.split(":", 1)[0]}, members
                )
            for key, members in candidate_groups.items():
                created += self._insert_cluster(
                    connection,
                    corpus_id,
                    key,
                    "candidate",
                    {"basis": "normalized_title_author_overlap_year_proximity"},
                    members,
                )
        return {
            "corpus_version": corpus_version,
            "clusters_created": created,
            "candidate_clusters": len(candidate_groups),
        }

    def duplicate_candidates(self, corpus_version: str) -> list[dict]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT c.cluster_uid,c.status,c.basis_json,
                json_group_array(json_object(
                    'gold_paper_id',p.gold_paper_id,'title',p.original_title,
                    'doi',p.canonical_doi,'year',p.publication_year)) members_json
                FROM gold_duplicate_clusters c
                JOIN gold_corpora corpus ON corpus.id=c.corpus_id
                JOIN gold_duplicate_cluster_members m ON m.cluster_id=c.id
                JOIN gold_corpus_papers p ON p.id=m.corpus_paper_id
                WHERE corpus.corpus_version=? AND c.status='candidate'
                GROUP BY c.id ORDER BY c.cluster_uid""",
                (corpus_version,),
            ).fetchall()
        return [
            {
                "cluster_uid": row["cluster_uid"],
                "status": row["status"],
                "basis": json.loads(row["basis_json"]),
                "members": json.loads(row["members_json"]),
            }
            for row in rows
        ]

    def _insert_cluster(
        self,
        connection: sqlite3.Connection,
        corpus_id: int,
        uid: str,
        status: str,
        basis: dict,
        members: list[dict],
    ) -> int:
        cursor = connection.execute(
            """INSERT OR IGNORE INTO gold_duplicate_clusters
            (corpus_id,cluster_uid,status,basis_json,created_at) VALUES (?,?,?,?,?)""",
            (corpus_id, uid, status, canonical_json(basis), utcnow()),
        )
        row = connection.execute(
            "SELECT id FROM gold_duplicate_clusters WHERE corpus_id=? AND cluster_uid=?",
            (corpus_id, uid),
        ).fetchone()
        for member in members:
            connection.execute(
                """INSERT OR IGNORE INTO gold_duplicate_cluster_members
                (cluster_id,corpus_paper_id) VALUES (?,?)""",
                (int(row["id"]), int(member["id"])),
            )
        return int(cursor.rowcount > 0)

    def create_split(
        self,
        *,
        corpus_version: str,
        split_version: str,
        random_seed: int,
        development: float = 0.20,
        calibration: float = 0.50,
        holdout: float = 0.30,
    ) -> dict:
        if not math.isclose(development + calibration + holdout, 1.0):
            raise ValueError("Split proportions must sum to 1")
        self.build_duplicate_clusters(corpus_version)
        final_labels = self._final_labels(corpus_version)
        if not final_labels or any(item["final_label"] is None for item in final_labels):
            raise ValueError("All Gold papers must have a resolved final label before split")
        if self.unresolved_count(corpus_version):
            raise ValueError("Unresolved review/arbitration items block split creation")
        groups: dict[str, list[dict]] = defaultdict(list)
        for item in final_labels:
            groups[str(item["cluster_uid"])].append(item)
        by_label: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
        for uid, members in groups.items():
            labels = {str(item["final_label"]) for item in members}
            if len(labels) != 1:
                raise ValueError(f"Duplicate cluster has conflicting final labels: {uid}")
            by_label[next(iter(labels))].append((uid, members))
        assignments: list[dict] = []
        for label, clusters in sorted(by_label.items()):
            ordered = sorted(
                clusters,
                key=lambda pair: canonical_hash(
                    {"seed": random_seed, "label": label, "cluster": pair[0]}
                ),
            )
            for index, (uid, members) in enumerate(ordered):
                position = (index + 0.5) / len(ordered)
                split = (
                    "development"
                    if position <= development
                    else "calibration"
                    if position <= development + calibration
                    else "holdout"
                )
                assignments.extend(
                    {
                        "gold_paper_id": member["gold_paper_id"],
                        "paper_id": member["id"],
                        "duplicate_cluster_uid": uid,
                        "split_name": split,
                        "gold_label": label,
                    }
                    for member in members
                )
        core = {
            "manifest_version": "gold-split-manifest-v1",
            "corpus_version": corpus_version,
            "split_version": split_version,
            "random_seed": random_seed,
            "strategy": {
                "group_aware": True,
                "stratified_by_gold_label": True,
                "development": development,
                "calibration": calibration,
                "holdout": holdout,
            },
            "assignments": [
                {key: value for key, value in item.items() if key != "paper_id"}
                for item in sorted(assignments, key=lambda row: row["gold_paper_id"])
            ],
        }
        digest = canonical_hash(core)
        with self.database.connect() as connection:
            corpus_id = int(
                connection.execute(
                    "SELECT id FROM gold_corpora WHERE corpus_version=?", (corpus_version,)
                ).fetchone()["id"]
            )
            cursor = connection.execute(
                """INSERT INTO gold_split_manifests
                (corpus_id,split_version,random_seed,strategy_json,manifest_json,
                 manifest_sha256,frozen,created_at) VALUES (?,?,?,?,?,?,1,?)""",
                (
                    corpus_id,
                    split_version,
                    random_seed,
                    canonical_json(core["strategy"]),
                    canonical_json(core),
                    digest,
                    utcnow(),
                ),
            )
            split_id = int(cursor.lastrowid)
            for item in assignments:
                connection.execute(
                    """INSERT INTO gold_split_assignments
                    (split_manifest_id,corpus_paper_id,duplicate_cluster_uid,
                     split_name,gold_label) VALUES (?,?,?,?,?)""",
                    (
                        split_id,
                        item["paper_id"],
                        item["duplicate_cluster_uid"],
                        item["split_name"],
                        item["gold_label"],
                    ),
                )
        return {**core, "manifest_sha256": digest}

    def verify_split(self, manifest_sha256: str) -> dict:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT id,manifest_json FROM gold_split_manifests WHERE manifest_sha256=?",
                (manifest_sha256.upper(),),
            ).fetchone()
            if row is None:
                raise ValueError("Gold split manifest not found")
            assignments = connection.execute(
                """SELECT duplicate_cluster_uid,COUNT(DISTINCT split_name) split_count
                FROM gold_split_assignments WHERE split_manifest_id=?
                GROUP BY duplicate_cluster_uid HAVING COUNT(DISTINCT split_name)>1""",
                (int(row["id"]),),
            ).fetchall()
        manifest = json.loads(row["manifest_json"])
        valid_hash = canonical_hash(manifest) == manifest_sha256.upper()
        return {
            "manifest_sha256": manifest_sha256.upper(),
            "hash_valid": valid_hash,
            "cross_split_duplicate_clusters": len(assignments),
            "valid": valid_hash and not assignments,
        }

    def freeze_snapshot(
        self,
        *,
        corpus_version: str,
        snapshot_version: str,
        split_manifest_sha256: str,
        protocol_path: Path,
        output: Path,
        git_commit: str | None,
        known_limitations: list[str],
    ) -> dict:
        status = self.annotation_status(corpus_version)
        if status["unresolved"] or status["logical_inconsistencies"] or status["missing_fields"]:
            raise ValueError("Gold quality gates are unresolved")
        split_check = self.verify_split(split_manifest_sha256)
        if not split_check["valid"]:
            raise ValueError("Gold split verification failed")
        labels = self._final_labels(corpus_version)
        if any(item["final_label"] is None for item in labels):
            raise ValueError("Every Gold paper requires a resolved final label")
        if output.exists():
            raise FileExistsError(f"Gold snapshot already exists: {output}")
        protocol_sha = sha256_file(protocol_path)
        with self.database.connect() as connection:
            corpus = connection.execute(
                "SELECT * FROM gold_corpora WHERE corpus_version=?", (corpus_version,)
            ).fetchone()
            split = connection.execute(
                """SELECT manifest_json FROM gold_split_manifests
                WHERE manifest_sha256=?""",
                (split_manifest_sha256.upper(),),
            ).fetchone()
        payload = {
            "snapshot_schema_version": "gold-governance-snapshot-v1",
            "snapshot_status": "IMMUTABLE HUMAN GOLD TRUTH",
            "status": "frozen",
            "snapshot_version": snapshot_version,
            "corpus_version": corpus_version,
            "corpus_sha256": corpus["corpus_sha256"],
            "corpus_manifest": json.loads(corpus["corpus_manifest_json"]),
            "annotation_protocol": {
                "version": PROTOCOL_VERSION,
                "sha256": protocol_sha,
            },
            "final_gold_labels": [
                {
                    "gold_paper_id": item["gold_paper_id"],
                    "final_label": item["final_label"],
                    "education_centrality": item["education_centrality"],
                    "economics_relevance": item["economics_relevance"],
                    "evidence_sufficiency": item["evidence_sufficiency"],
                    "reviewer_confidence": item["reviewer_confidence"],
                    "reason_codes": item["reason_codes"],
                }
                for item in labels
            ],
            "reviewer_manifest": self._reviewer_manifest(int(corpus["id"])),
            "consistency_metrics": asdict(self.calculate_agreement(corpus_version)),
            "duplicate_cluster_manifest": self._cluster_manifest(int(corpus["id"])),
            "split_manifest": json.loads(split["manifest_json"]),
            "split_manifest_sha256": split_manifest_sha256.upper(),
            "source_database_sha256": corpus["source_database_sha256"],
            "git_commit": git_commit,
            "known_limitations": known_limitations,
            "created_at": utcnow(),
        }
        digest = canonical_hash(payload)
        snapshot = {**payload, "snapshot_sha256": digest}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            with self.database.connect() as connection:
                cursor = connection.execute(
                    """INSERT INTO gold_governance_snapshots
                    (corpus_id,snapshot_version,snapshot_path,snapshot_sha256,
                     corpus_sha256,protocol_version,protocol_sha256,
                     split_manifest_sha256,manifest_json,status,created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,'frozen',?)""",
                    (
                        int(corpus["id"]),
                        snapshot_version,
                        str(output.resolve()),
                        digest,
                        corpus["corpus_sha256"],
                        PROTOCOL_VERSION,
                        protocol_sha,
                        split_manifest_sha256.upper(),
                        canonical_json(snapshot),
                        snapshot["created_at"],
                    ),
                )
        except Exception:
            output.unlink(missing_ok=True)
            raise
        return {"snapshot_id": int(cursor.lastrowid), **snapshot}

    def inspect_snapshot(self, snapshot_sha256: str) -> dict:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM gold_governance_snapshots WHERE snapshot_sha256=?",
                (snapshot_sha256.upper(),),
            ).fetchone()
        if row is None:
            raise ValueError("Gold snapshot not found")
        payload = json.loads(row["manifest_json"])
        return {
            "snapshot_version": row["snapshot_version"],
            "snapshot_sha256": row["snapshot_sha256"],
            "status": row["status"],
            "hash_valid": canonical_hash(
                {key: value for key, value in payload.items() if key != "snapshot_sha256"}
            )
            == row["snapshot_sha256"],
            "corpus_sha256": row["corpus_sha256"],
            "split_manifest_sha256": row["split_manifest_sha256"],
        }

    def compare_snapshots(self, left_sha256: str, right_sha256: str) -> dict:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT snapshot_sha256,manifest_json FROM gold_governance_snapshots
                WHERE snapshot_sha256 IN (?,?)""",
                (left_sha256.upper(), right_sha256.upper()),
            ).fetchall()
        values = {str(row["snapshot_sha256"]): json.loads(row["manifest_json"]) for row in rows}
        if len(values) != 2:
            raise ValueError("Both Gold snapshots must exist")
        left = values[left_sha256.upper()]
        right = values[right_sha256.upper()]
        left_labels = {x["gold_paper_id"]: x["final_label"] for x in left["final_gold_labels"]}
        right_labels = {x["gold_paper_id"]: x["final_label"] for x in right["final_gold_labels"]}
        shared = set(left_labels) & set(right_labels)
        return {
            "same_snapshot": left_sha256.upper() == right_sha256.upper(),
            "same_corpus": left["corpus_sha256"] == right["corpus_sha256"],
            "shared_papers": len(shared),
            "label_changes": sum(left_labels[item] != right_labels[item] for item in shared),
            "added_papers": len(set(right_labels) - set(left_labels)),
            "removed_papers": len(set(left_labels) - set(right_labels)),
        }

    def unresolved_count(self, corpus_version: str) -> int:
        with self.database.connect() as connection:
            row = connection.execute(
                """SELECT COUNT(DISTINCT q.corpus_paper_id) FROM gold_review_queue q
                JOIN gold_corpora c ON c.id=q.corpus_id
                WHERE c.corpus_version=? AND q.status='open'""",
                (corpus_version,),
            ).fetchone()
        return int(row[0])

    def _completed_repeat_count(self, corpus_id: int) -> int:
        with self.database.connect() as connection:
            return int(
                connection.execute(
                    """SELECT COUNT(*) FROM gold_presented_records pr
                    WHERE pr.corpus_id=? AND pr.hidden_repeat=1 AND EXISTS(
                        SELECT 1 FROM gold_annotation_revisions a
                        WHERE a.presented_record_id=pr.presented_record_id AND a.active=1
                        AND a.disposition='submitted')""",
                    (corpus_id,),
                ).fetchone()[0]
            )

    def _repeat_pairs(self, corpus_version: str) -> list[tuple[dict, dict]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT pr.corpus_paper_id,pr.hidden_repeat,a.*
                FROM gold_presented_records pr
                JOIN gold_corpora c ON c.id=pr.corpus_id
                JOIN gold_annotation_revisions a
                  ON a.presented_record_id=pr.presented_record_id AND a.active=1
                WHERE c.corpus_version=? AND pr.annotation_round=1
                AND a.disposition='submitted'
                ORDER BY pr.reviewer_id,pr.corpus_paper_id,pr.hidden_repeat""",
                (corpus_version,),
            ).fetchall()
        grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for row in rows:
            grouped[(str(row["reviewer_id"]), int(row["corpus_paper_id"]))].append(dict(row))
        return [(items[0], items[1]) for items in grouped.values() if len(items) >= 2]

    def _enqueue_repeat_disagreements(self, corpus_version: str) -> None:
        pairs = self._repeat_pairs(corpus_version)
        with self.database.connect() as connection:
            for left, right in pairs:
                if left["label"] == right["label"]:
                    continue
                connection.execute(
                    """INSERT OR IGNORE INTO gold_review_queue
                    (corpus_id,corpus_paper_id,queue_reason,source_annotation_id,
                     status,created_at) VALUES (?,?,?,?, 'open',?)""",
                    (
                        int(left["corpus_id"]),
                        int(left["corpus_paper_id"]),
                        "hidden_repeat_disagreement",
                        int(right["id"]),
                        utcnow(),
                    ),
                )

    def _final_labels(self, corpus_version: str) -> list[dict]:
        self._enqueue_repeat_disagreements(corpus_version)
        with self.database.connect() as connection:
            papers = connection.execute(
                """SELECT p.* FROM gold_corpus_papers p JOIN gold_corpora c
                ON c.id=p.corpus_id WHERE c.corpus_version=? ORDER BY p.id""",
                (corpus_version,),
            ).fetchall()
            result = []
            for paper in papers:
                arbitration = connection.execute(
                    """SELECT * FROM gold_arbitration_revisions
                    WHERE corpus_paper_id=? AND active=1 ORDER BY id DESC LIMIT 1""",
                    (int(paper["id"]),),
                ).fetchone()
                annotations = connection.execute(
                    """SELECT * FROM gold_annotation_revisions
                    WHERE corpus_paper_id=? AND active=1 AND disposition='submitted'
                    ORDER BY annotation_round DESC,submitted_at DESC,id DESC""",
                    (int(paper["id"]),),
                ).fetchall()
                queue_open = connection.execute(
                    """SELECT 1 FROM gold_review_queue
                    WHERE corpus_paper_id=? AND status='open' LIMIT 1""",
                    (int(paper["id"]),),
                ).fetchone()
                chosen = dict(annotations[0]) if annotations else None
                final = str(arbitration["final_gold_label"]) if arbitration else (
                    str(chosen["label"]) if chosen and not queue_open else None
                )
                result.append(
                    {
                        "id": int(paper["id"]),
                        "gold_paper_id": paper["gold_paper_id"],
                        "cluster_uid": self._effective_cluster_uid(
                            connection, int(paper["id"]), str(paper["duplicate_cluster_key"])
                        ),
                        "final_label": final,
                        "education_centrality": (
                            chosen.get("education_centrality") if chosen else None
                        ),
                        "economics_relevance": (
                            chosen.get("economics_relevance") if chosen else None
                        ),
                        "evidence_sufficiency": (
                            chosen.get("evidence_sufficiency") if chosen else None
                        ),
                        "reviewer_confidence": (
                            chosen.get("reviewer_confidence") if chosen else None
                        ),
                        "reason_codes": json.loads(chosen["reason_codes_json"]) if chosen else [],
                    }
                )
        return result

    @staticmethod
    def _effective_cluster_uid(
        connection: sqlite3.Connection, paper_id: int, fallback: str
    ) -> str:
        row = connection.execute(
            """SELECT c.cluster_uid
            FROM gold_duplicate_clusters c
            JOIN gold_duplicate_cluster_members m ON m.cluster_id=c.id
            WHERE m.corpus_paper_id=? AND c.status IN ('confirmed','candidate')
            AND (SELECT COUNT(*) FROM gold_duplicate_cluster_members members
                 WHERE members.cluster_id=c.id)>1
            ORDER BY CASE c.status WHEN 'confirmed' THEN 0 ELSE 1 END,c.cluster_uid
            LIMIT 1""",
            (paper_id,),
        ).fetchone()
        return str(row["cluster_uid"]) if row else fallback

    def _reviewer_manifest(self, corpus_id: int) -> list[dict]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT reviewer_id,annotation_round,COUNT(*) annotations,
                MIN(submitted_at) first_submitted_at,MAX(submitted_at) last_submitted_at
                FROM gold_annotation_revisions WHERE corpus_id=? AND active=1
                GROUP BY reviewer_id,annotation_round ORDER BY reviewer_id,annotation_round""",
                (corpus_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _cluster_manifest(self, corpus_id: int) -> list[dict]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT c.cluster_uid,c.status,
                json_group_array(p.gold_paper_id) members
                FROM gold_duplicate_clusters c
                JOIN gold_duplicate_cluster_members m ON m.cluster_id=c.id
                JOIN gold_corpus_papers p ON p.id=m.corpus_paper_id
                WHERE c.corpus_id=? GROUP BY c.id ORDER BY c.cluster_uid""",
                (corpus_id,),
            ).fetchall()
        return [
            {
                "cluster_uid": row["cluster_uid"],
                "status": row["status"],
                "members": json.loads(row["members"]),
            }
            for row in rows
        ]


def _paper_content_from_row(
    row: sqlite3.Row, source_sha256: str, corpus_version: str
) -> dict[str, Any]:
    return {
        "gold_paper_id": row["gold_paper_id"],
        "source_database_sha256": source_sha256,
        "source_paper_id": row["source_paper_id"],
        "canonical_doi": row["canonical_doi"],
        "fallback_fingerprint": row["fallback_fingerprint"],
        "original_title": row["original_title"],
        "normalized_title": row["normalized_title"],
        "abstract": row["abstract"],
        "abstract_length": row["abstract_length"],
        "abstract_quality_status": row["abstract_quality_status"],
        "journal": row["journal"],
        "journal_short_name": row["journal_short_name"],
        "issn": row["issn"],
        "publication_date": row["publication_date"],
        "publication_year": row["publication_year"],
        "authors": json.loads(row["authors_json"]),
        "metadata_provenance_summary": json.loads(row["metadata_provenance_summary_json"]),
        "source_links": json.loads(row["source_links_json"]),
        "source_record_timestamp": row["source_record_timestamp"],
        "corpus_version": corpus_version,
        "sampling_stratum": row["sampling_stratum"],
        "stratum_population_size": row["stratum_population_size"],
        "inclusion_probability": row["inclusion_probability"],
        "duplicate_cluster_key": row["duplicate_cluster_key"],
    }


def _presentation_schedule(
    papers: list[dict],
    *,
    repeat_fraction: float,
    seed: int,
    reviewer: str,
    annotation_round: int,
) -> list[dict]:
    base = [
        {
            "paper_id": int(item["id"]),
            "presented_record_id": canonical_hash(
                {
                    "seed": seed,
                    "reviewer": reviewer,
                    "round": annotation_round,
                    "gold_paper_id": item["gold_paper_id"],
                    "copy": 0,
                }
            )[:24],
            "hidden_repeat": False,
        }
        for item in papers
    ]
    repeat_count = min(len(papers), round(len(papers) * repeat_fraction))
    repeat_source = sorted(
        base,
        key=lambda item: canonical_hash({"seed": seed, "repeat": item["presented_record_id"]}),
    )[:repeat_count]
    repeats = [
        {
            "paper_id": item["paper_id"],
            "presented_record_id": canonical_hash(
                {"seed": seed, "repeat_of": item["presented_record_id"]}
            )[:24],
            "hidden_repeat": True,
            "repeat_of_presented_record_id": item["presented_record_id"],
        }
        for item in repeat_source
    ]
    records = base + repeats
    for attempt in range(100):
        ordered = sorted(
            records,
            key=lambda item: canonical_hash(
                {
                    "seed": seed + attempt,
                    "presented_record_id": item["presented_record_id"],
                }
            ),
        )
        positions = {item["presented_record_id"]: index for index, item in enumerate(ordered)}
        if all(
            abs(
                positions[item["presented_record_id"]]
                - positions[item["repeat_of_presented_record_id"]]
            )
            > 1
            for item in ordered
            if item["hidden_repeat"]
        ):
            return ordered
    raise RuntimeError("Could not create a non-adjacent hidden-repeat schedule")


def _review_reasons(
    *,
    label: str | None,
    centrality: str | None,
    sufficiency: str | None,
    confidence: str | None,
    reason_codes: list[str] | tuple[str, ...],
    marked: bool,
    deferred: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if deferred:
        reasons.append("deferred")
    if confidence == "low":
        reasons.append("low_confidence")
    if sufficiency == "insufficient":
        reasons.append("evidence_insufficient")
    if marked:
        reasons.append("reviewer_marked")
    if any(str(code).startswith(("boundary", "ambiguous")) for code in reason_codes):
        reasons.append("boundary_reason")
    reasons.extend(_logical_inconsistencies(label, centrality, sufficiency, confidence))
    return tuple(dict.fromkeys(reasons))


def _logical_inconsistencies(
    label: str | None,
    centrality: str | None,
    sufficiency: str | None,
    confidence: str | None,
) -> tuple[str, ...]:
    reasons = []
    if label == "include" and centrality == "none":
        reasons.append("logic_include_centrality_none")
    if label == "exclude" and centrality == "core":
        reasons.append("logic_exclude_centrality_core")
    if confidence == "high" and sufficiency == "insufficient":
        reasons.append("logic_high_confidence_insufficient_evidence")
    return tuple(reasons)


def _title_duplicate_candidates(papers: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for paper in papers:
        authors = json.loads(paper["authors_json"] or "[]")
        first_author = normalize_title(str(authors[0])) if authors else ""
        year = paper.get("publication_year")
        key = f"title:{paper['normalized_title']}|author:{first_author}|year:{year}"
        groups[key].append(paper)
    return {
        key: members
        for key, members in groups.items()
        if len(members) > 1
        and len({member["duplicate_cluster_key"] for member in members}) > 1
    }


def _elapsed_seconds(started_at: str, submitted_at: str) -> int:
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(submitted_at)
    except ValueError as exc:
        raise ValueError("started_at must be an ISO-8601 timestamp") from exc
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    return max(0, round((end - start).total_seconds()))


def _ratio(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left = Counter(item[0] for item in pairs)
    right = Counter(item[1] for item in pairs)
    expected = sum(left[label] * right[label] for label in LABELS) / (len(pairs) ** 2)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0
