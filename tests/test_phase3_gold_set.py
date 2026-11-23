from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eduecon_radar import cli
from eduecon_radar.db import Database
from eduecon_radar.domain.evaluation import DomainEvaluator
from eduecon_radar.domain.gold_set import (
    DIFFICULTY_TAGS,
    GOLD_SET_FIELDS,
    GoldSetLoader,
    create_gold_set_template,
)
from eduecon_radar.schemas import JournalConfig, PaperMetadata
from eduecon_radar.services.archive import sha256_file

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "config"
    / "domains"
    / "education_economics"
    / "gold_sets"
    / "fixtures"
    / "test_gold_set.csv"
)


def sample_database(tmp_path):
    path = tmp_path / "sampling.db"
    database = Database(path)
    database.init_schema()
    journal_id = database.upsert_journal(
        JournalConfig(name="General Economics Journal", short_name="GEJ", issn="0000-0000")
    )
    for index in range(12):
        paper_id, _ = database.upsert_paper(
            PaperMetadata(
                title=(
                    f"Education title {index}"
                    if index % 2
                    else f"General title {index}"
                ),
                abstract=None if index % 3 == 0 else f"Abstract {index}",
                doi=f"10.0000/sample{index}",
            ),
            journal_id,
        )
        if index % 4 == 0:
            with database.connect() as connection:
                connection.execute(
                    """INSERT INTO screenings
                    (paper_id,ai_decision,model_name,prompt_version,screened_at)
                    VALUES (?,'exclude','fixture','fixture','2026-01-01T00:00:00Z')""",
                    (paper_id,),
                )
    return path


def test_synthetic_gold_set_loads_and_preserves_human_provenance():
    gold_set = GoldSetLoader().load(FIXTURE, require_labels=True)
    assert len(gold_set.items) == 12
    assert all(item.reviewer == "Fixture Reviewer" for item in gold_set.items)
    assert all(
        item.source_database_sha256 == "TEST-FIXTURE-NOT-PRODUCTION"
        for item in gold_set.items
    )
    assert any(item.doi is None for item in gold_set.items)
    assert all(item.original_title for item in gold_set.items)


def test_empty_template_is_valid_for_annotation_but_not_formal_evaluation():
    template = FIXTURE.parents[1] / "gold_set_template.csv"
    result = GoldSetLoader().load(template)
    assert result.items == ()
    assert GoldSetLoader().load(template, require_labels=True).items == ()


def test_unlabeled_row_is_allowed_for_template_and_rejected_for_evaluation(tmp_path):
    path = tmp_path / "unlabeled.csv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=GOLD_SET_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "item_id": "pending-1",
                "original_title": "Human annotation pending",
                "journal": "Journal",
                "abstract_available": "false",
                "source_group": "random",
                "source_database_sha256": "ABC",
                "ontology_version_at_annotation": "draft",
                "annotation_version": "draft",
            }
        )
    assert GoldSetLoader().load(path).items[0].gold_label is None
    with pytest.raises(ValueError, match="human label"):
        GoldSetLoader().load(path, require_labels=True)


def test_labeled_row_requires_reviewer_and_ai_never_fills_gold_label(tmp_path):
    path = tmp_path / "invalid.csv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=GOLD_SET_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "item_id": "x",
                "original_title": "Title",
                "journal": "Journal",
                "abstract_available": "true",
                "gold_label": "include",
                "source_group": "ai_include",
                "source_database_sha256": "ABC",
                "ontology_version_at_annotation": "draft",
                "annotation_version": "draft",
                "prior_ai_decision": "include",
            }
        )
    with pytest.raises(ValueError, match="reviewer"):
        GoldSetLoader().load(path)

    rows = list(csv.DictReader(FIXTURE.open(encoding="utf-8-sig")))
    assert all(
        row["gold_label"] != row["prior_ai_decision"]
        or not row["prior_ai_decision"]
        for row in rows
    )


def test_duplicate_item_and_unknown_difficulty_tag_are_rejected(tmp_path):
    rows = list(csv.DictReader(FIXTURE.open(encoding="utf-8-sig")))
    duplicate = tmp_path / "duplicate.csv"
    with duplicate.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=GOLD_SET_FIELDS)
        writer.writeheader()
        writer.writerows([rows[0], rows[0]])
    with pytest.raises(ValueError, match="unique"):
        GoldSetLoader().load(duplicate)

    rows[0]["difficulty_tags"] = "invented_tag"
    invalid = tmp_path / "invalid-tag.csv"
    with invalid.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=GOLD_SET_FIELDS)
        writer.writeheader()
        writer.writerow(rows[0])
    with pytest.raises(ValueError, match="difficulty"):
        GoldSetLoader().load(invalid)
    assert "opaque_title" in DIFFICULTY_TAGS


def test_sampling_is_read_only_seeded_stratified_and_manifested(tmp_path):
    source = sample_database(tmp_path)
    before = (sha256_file(source), source.stat().st_mtime_ns)
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first_manifest = create_gold_set_template(
        source_db=source,
        output=first,
        sample_size=8,
        random_seed=42,
        stratify=True,
        include_existing_screening_metadata=True,
        confirm_read_only_source=True,
    )
    create_gold_set_template(
        source_db=source,
        output=second,
        sample_size=8,
        random_seed=42,
        stratify=True,
        include_existing_screening_metadata=True,
        confirm_read_only_source=True,
    )
    assert (sha256_file(source), source.stat().st_mtime_ns) == before
    first_rows = list(csv.DictReader(first.open(encoding="utf-8-sig")))
    second_rows = list(csv.DictReader(second.open(encoding="utf-8-sig")))
    assert [row["paper_id"] for row in first_rows] == [row["paper_id"] for row in second_rows]
    assert {row["source_group"] for row in first_rows} >= {"no_abstract", "ai_exclude"}
    assert all(not row["gold_label"] for row in first_rows)
    assert len(GoldSetLoader().load(first).items) == 8
    assert first_manifest.row_count == 8
    payload = json.loads(
        first.with_suffix(".csv.manifest.json").read_text(encoding="utf-8")
    )
    assert payload["source_database_sha256"] == before[0]
    assert payload["sample_seed"] == 42
    assert payload["source_table_counts"]["papers"] == 12


def test_sampling_requires_confirmation_and_refuses_overwrite(tmp_path):
    source = sample_database(tmp_path)
    output = tmp_path / "gold.csv"
    options = {
        "source_db": source,
        "output": output,
        "sample_size": 2,
        "random_seed": 1,
        "stratify": False,
        "include_existing_screening_metadata": False,
    }
    with pytest.raises(ValueError, match="confirm-read-only"):
        create_gold_set_template(**options, confirm_read_only_source=False)
    create_gold_set_template(**options, confirm_read_only_source=True)
    with pytest.raises(FileExistsError):
        create_gold_set_template(**options, confirm_read_only_source=True)
    with pytest.raises(ValueError, match="separate non-production"):
        create_gold_set_template(
            **{**options, "output": source}, confirm_read_only_source=True
        )


def test_evaluation_metrics_uncertain_subgroups_and_top_k_are_correct():
    report = DomainEvaluator().evaluate(GoldSetLoader().load(FIXTURE, require_labels=True))
    assert report.report_status == "UNCALIBRATED - SIMULATION ONLY"
    assert report.sample_count == 12
    assert report.valid_label_count == 11
    assert report.uncertain_count == 1
    assert report.candidate_recall == pytest.approx(4 / 7)
    assert report.candidate_precision == 1.0
    assert report.false_negative_count == 3
    assert report.false_positive_count == 0
    assert report.false_negative_rate == pytest.approx(3 / 7)
    assert report.evidence_coverage == 1.0
    assert report.subgroup_reports["opaque_title"]["count"] == 2
    assert report.subgroup_reports["education_false_friend"]["count"] == 3
    assert report.top_k_discovery["5"]["relevant_found"] == 5
    assert report.top_k_discovery["10"]["reviewed"] == 10
    assert report.top_k_discovery["20"]["reviewed"] == 11
    assert report.top_k_discovery["50"]["reviewed"] == 11
    assert report.top_k_discovery["5"]["papers_per_relevant"] == 1.0
    assert json.loads(report.model_dump_json())["candidate_recall"] == pytest.approx(4 / 7)


def test_empty_evaluation_does_not_invent_metrics(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text(",".join(GOLD_SET_FIELDS) + "\n", encoding="utf-8")
    report = DomainEvaluator().evaluate(GoldSetLoader().load(empty))
    assert report.candidate_recall is None
    assert report.candidate_precision is None
    assert report.false_negative_rate is None
    assert report.evidence_coverage is None
    assert "no_core_human_labels" in report.warnings


def test_phase3_cli_parsers_require_explicit_paths_and_safe_flags():
    template = cli.build_parser().parse_args(
        [
            "create-gold-set-template",
            "--source-db",
            "source.db",
            "--output",
            "gold.csv",
        ]
    )
    assert not template.confirm_read_only_source
    assert not template.overwrite
    validate = cli.build_parser().parse_args(["validate-gold-set", "--input", "gold.csv"])
    assert validate.command == "validate-gold-set"
    migrate = cli.build_parser().parse_args(
        ["migrate-domain-evidence", "--database", "test.db"]
    )
    assert not migrate.backup_confirmed


def test_validation_and_evaluation_cli_emit_json_without_database_creation(
    tmp_path, capsys
):
    settings = SimpleNamespace(project_root=ROOT)
    validate = cli.build_parser().parse_args(
        ["validate-gold-set", "--input", str(FIXTURE), "--require-labels"]
    )
    assert cli.run(validate, settings) == 0
    assert json.loads(capsys.readouterr().out)["labeled"] == 12

    output = tmp_path / "evaluation.json"
    evaluate = cli.build_parser().parse_args(
        [
            "evaluate-domain-evidence",
            "--gold-set",
            str(FIXTURE),
            "--output",
            str(output),
        ]
    )
    assert cli.run(evaluate, settings) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["report_status"] == "UNCALIBRATED - SIMULATION ONLY"
    with pytest.raises(FileExistsError):
        cli.run(evaluate, settings)
