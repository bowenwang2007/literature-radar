from __future__ import annotations

import csv
import json
import random
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from eduecon_radar import __version__
from eduecon_radar.domain.evidence import EvidenceExtractor, PaperEvidenceInput
from eduecon_radar.domain.models import DomainConfigBundle
from eduecon_radar.domain.scoring import RuleEvidenceScorer
from eduecon_radar.services.archive import sha256_file


@dataclass(frozen=True)
class RealGoldSamplingManifest:
    gold_set_id: str
    source_db_path: str
    source_db_sha256: str
    sampling_seed: int
    sampling_strategy: str
    strata_targets: dict[str, int]
    strata_actual_counts: dict[str, int]
    total_rows: int
    created_at: str
    application_version: str
    ontology_version: str
    scoring_version: str
    original_abstract_included: bool
    gold_labels_blank: bool
    old_ai_used_only_for_sampling: bool
    output_files: dict[str, str]


@dataclass(frozen=True)
class GoldSampleRecord:
    item_id: str
    source_paper_id: int
    doi: str | None
    original_title: str
    formal_chinese_title: str | None
    original_abstract: str | None
    authors_json: str
    journal: str
    journal_group: str
    publication_date: str | None
    keywords_json: str
    subject_codes_json: str
    source_metadata_json: str
    stratum: str
    difficulty_tags_json: str
    display_order_seed: int
    sampling_prior_ai_decision: str | None
    sampling_rule_evidence_tier: str
    gold_label: str = ""


STRATA = (
    "rules_only_exclude",
    "deepseek_exclude",
    "ai_include",
    "no_abstract",
    "explicit_education_title",
    "hidden_topic_candidate",
    "opaque_title",
    "general_economics_journal",
    "applied_labor_public_journal",
    "domain_journal_candidate",
    "false_friend_candidate",
    "adjacent_domain_candidate",
    "cross_domain_intersection_candidate",
    "metadata_sparse",
    "random_control",
)


def create_real_gold_set_sample(
    *,
    source_db: Path,
    output_dir: Path,
    gold_set_id: str,
    bundle: DomainConfigBundle,
    sample_size: int = 300,
    random_seed: int = 20260719,
    confirm_read_only_source: bool = False,
) -> RealGoldSamplingManifest:
    if not confirm_read_only_source:
        raise ValueError("Use --confirm-read-only-source for real Gold Set sampling")
    if not 200 <= sample_size <= 400:
        raise ValueError("Real Gold Set sample_size must be between 200 and 400")
    if not gold_set_id.strip():
        raise ValueError("gold_set_id is required")
    source_db = source_db.resolve()
    output_dir = output_dir.resolve()
    if not source_db.is_file():
        raise FileNotFoundError(f"Source database does not exist: {source_db}")
    outputs = {
        "items": output_dir / "gold_set_items.csv",
        "manifest": output_dir / "sampling_manifest.json",
        "source_snapshot": output_dir / "source_snapshot.json",
        "instructions": output_dir / "annotation_instructions.md",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Gold Set output files already exist: {existing}")
    source_hash_before = sha256_file(source_db)
    rows, counts = _load_source_rows(source_db)
    classified = [_classify_row(row, bundle) for row in rows]
    selected, targets = _stratified_select(classified, sample_size, random_seed)
    source_hash_after = sha256_file(source_db)
    if source_hash_after != source_hash_before:
        raise RuntimeError("Source database changed during read-only sampling")
    records = tuple(
        _to_record(item, index, random_seed) for index, item in enumerate(selected, start=1)
    )
    actual = dict(sorted(Counter(item.stratum for item in records).items()))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_items(outputs["items"], records)
    source_snapshot = {
        "source_db_path": str(source_db),
        "source_db_sha256": source_hash_before,
        "integrity_check": "ok",
        "table_counts": counts,
        "captured_at": datetime.now(UTC).isoformat(),
    }
    outputs["source_snapshot"].write_text(
        json.dumps(source_snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["instructions"].write_text(_annotation_instructions(), encoding="utf-8")
    manifest = RealGoldSamplingManifest(
        gold_set_id=gold_set_id,
        source_db_path=str(source_db),
        source_db_sha256=source_hash_before,
        sampling_seed=random_seed,
        sampling_strategy="deterministic_multi_stratum_round_robin_v1",
        strata_targets=targets,
        strata_actual_counts=actual,
        total_rows=len(records),
        created_at=datetime.now(UTC).isoformat(),
        application_version=__version__,
        ontology_version=bundle.ontology.version,
        scoring_version=bundle.scoring.scoring_version,
        original_abstract_included=True,
        gold_labels_blank=all(not item.gold_label for item in records),
        old_ai_used_only_for_sampling=True,
        output_files={key: str(path) for key, path in outputs.items()},
    )
    outputs["manifest"].write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_source_rows(path: Path) -> tuple[list[sqlite3.Row], dict[str, int]]:
    uri = path.as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"Source integrity_check failed: {integrity}")
        counts = {
            name: int(connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
            for name in ("papers", "screenings", "reviews")
        }
        available = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        chinese_title = (
            "(SELECT pt.chinese_title FROM paper_translations pt WHERE pt.paper_id=p.id)"
            if "paper_translations" in available
            else "NULL"
        )
        rows = connection.execute(
            f"""SELECT p.id,p.doi,p.title,{chinese_title} formal_chinese_title,
            p.abstract,p.authors_json,p.keywords_json,
            p.jel_codes_json,p.source_metadata_json,j.name journal,j.short_name,
            COALESCE(p.issue_date,p.online_date) publication_date,
            (SELECT s.ai_decision FROM screenings s WHERE s.paper_id=p.id
             ORDER BY s.screened_at DESC,s.id DESC LIMIT 1) prior_ai_decision,
            EXISTS(SELECT 1 FROM screenings s WHERE s.paper_id=p.id
                   AND s.model_name='rules-only' AND s.ai_decision='exclude') rules_only_exclude,
            EXISTS(SELECT 1 FROM screenings s WHERE s.paper_id=p.id
                   AND s.model_name!='rules-only' AND s.ai_decision='exclude') deepseek_exclude,
            EXISTS(SELECT 1 FROM screenings s WHERE s.paper_id=p.id
                   AND s.model_name!='rules-only' AND s.ai_decision='include') ai_include
            FROM papers p JOIN journals j ON j.id=p.journal_id ORDER BY p.id"""
        ).fetchall()
    return rows, counts


def _classify_row(row: sqlite3.Row, bundle: DomainConfigBundle) -> dict:
    evidence = EvidenceExtractor(bundle).extract(
        PaperEvidenceInput(
            paper_id=int(row["id"]),
            doi=row["doi"],
            title=row["title"],
            abstract=row["abstract"],
            authors=_json_strings(row["authors_json"]),
            journal=row["journal"],
            publication_date=row["publication_date"],
            keywords=row["keywords_json"],
            jel_codes=row["jel_codes_json"],
            source_metadata=row["source_metadata_json"],
        )
    )
    scoring = RuleEvidenceScorer(bundle.scoring).score(evidence)
    title_concepts = [item for item in evidence.matched_concepts if item.source_field == "title"]
    abstract_concepts = [
        item for item in evidence.matched_concepts if item.source_field == "abstract"
    ]
    tags: list[str] = []
    candidates: list[str] = []
    if row["rules_only_exclude"]:
        candidates.append("rules_only_exclude")
    if row["deepseek_exclude"]:
        candidates.append("deepseek_exclude")
    if row["ai_include"]:
        candidates.append("ai_include")
    if not row["abstract"]:
        candidates.append("no_abstract")
        tags.append("no_abstract")
    if title_concepts:
        candidates.append("explicit_education_title")
    if abstract_concepts and not title_concepts:
        candidates.append("hidden_topic_candidate")
        tags.append("hidden_topic")
    if evidence.opaque_title_features.opaque_title:
        candidates.append("opaque_title")
        tags.append("opaque_title")
    journal_group = _journal_group(row["short_name"], evidence.journal_prior_signal.status)
    candidates.append(journal_group)
    if evidence.matched_exclusion_contexts:
        candidates.append("false_friend_candidate")
        tags.append("false_friend")
    if evidence.concept_family_coverage and scoring.rule_evidence_tier in {
        "weak_or_mixed",
        "moderate_positive",
    }:
        candidates.append("adjacent_domain_candidate")
        tags.append("adjacent_domain")
    if (
        any(
            signal.matched and signal.signal_type in {"adjacent", "context"}
            for signal in evidence.subject_code_signals
        )
        and evidence.concept_family_coverage
    ):
        candidates.append("cross_domain_intersection_candidate")
        tags.append("cross_domain_intersection")
    if evidence.metadata_evidence.overall_metadata_confidence < 0.5:
        candidates.append("metadata_sparse")
        tags.append("metadata_sparse")
    candidates.append("random_control")
    return {
        "row": row,
        "candidate_strata": tuple(dict.fromkeys(candidates)),
        "difficulty_tags": tuple(sorted(set(tags))),
        "journal_group": journal_group,
        "rule_evidence_tier": scoring.rule_evidence_tier,
    }


def _journal_group(short_name: str, prior_status: str) -> str:
    if prior_status in {"candidate", "draft", "released"}:
        return "domain_journal_candidate"
    if short_name in {"AEJApplied", "JPubE", "JLaborEcon"}:
        return "applied_labor_public_journal"
    return "general_economics_journal"


def _stratified_select(
    items: list[dict], sample_size: int, seed: int
) -> tuple[list[dict], dict[str, int]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        for stratum in item["candidate_strata"]:
            groups[stratum].append(item)
    for members in groups.values():
        rng.shuffle(members)
    base = sample_size // len(STRATA)
    targets = {name: base for name in STRATA}
    for name in STRATA[: sample_size % len(STRATA)]:
        targets[name] += 1
    selected: list[dict] = []
    selected_ids: set[int] = set()
    for name in STRATA:
        added = 0
        for item in groups.get(name, []):
            paper_id = int(item["row"]["id"])
            if paper_id in selected_ids:
                continue
            chosen = dict(item)
            chosen["primary_stratum"] = name
            selected.append(chosen)
            selected_ids.add(paper_id)
            added += 1
            if added >= targets[name] or len(selected) >= sample_size:
                break
    remaining = [item for item in items if int(item["row"]["id"]) not in selected_ids]
    rng.shuffle(remaining)
    for item in remaining[: max(0, sample_size - len(selected))]:
        chosen = dict(item)
        chosen["primary_stratum"] = "random_control"
        selected.append(chosen)
    rng.shuffle(selected)
    return selected, targets


def _to_record(item: dict, index: int, seed: int) -> GoldSampleRecord:
    row = item["row"]
    return GoldSampleRecord(
        item_id=f"gold-{index:04d}",
        source_paper_id=int(row["id"]),
        doi=row["doi"],
        original_title=row["title"],
        formal_chinese_title=row["formal_chinese_title"],
        original_abstract=row["abstract"],
        authors_json=row["authors_json"] or "[]",
        journal=row["journal"],
        journal_group=item["journal_group"],
        publication_date=row["publication_date"],
        keywords_json=row["keywords_json"] or "[]",
        subject_codes_json=row["jel_codes_json"] or "[]",
        source_metadata_json=row["source_metadata_json"] or "{}",
        stratum=item["primary_stratum"],
        difficulty_tags_json=json.dumps(item["difficulty_tags"], ensure_ascii=False),
        display_order_seed=seed + int(row["id"]),
        sampling_prior_ai_decision=row["prior_ai_decision"],
        sampling_rule_evidence_tier=item["rule_evidence_tier"],
    )


def _write_items(path: Path, records: tuple[GoldSampleRecord, ...]) -> None:
    fields = list(GoldSampleRecord.__dataclass_fields__)
    with path.open("x", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for item in records:
            writer.writerow(asdict(item))


def _json_strings(raw: str | None) -> tuple[str, ...]:
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return ()
    return tuple(item for item in value if isinstance(item, str)) if isinstance(value, list) else ()


def _annotation_instructions() -> str:
    return """# Gold Set annotation instructions

Round 1 is blind: do not inspect old AI, rule, Fast, Strong or Composite decisions.
Use only source metadata. Labels are include, borderline, exclude and uncertain.
Uncertain means the available metadata is insufficient and must not be forced to exclude.
Conflicts and uncertain labels require adjudication before snapshot freezing.
"""
