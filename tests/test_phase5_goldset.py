from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from eduecon_radar.cli import build_parser
from eduecon_radar.db import Database, utcnow
from eduecon_radar.domain.goldset_archive_gate import validate_archive_manifest
from eduecon_radar.domain.goldset_bundle import (
    build_model_bundle_candidate,
    inspect_model_bundle_candidate,
)
from eduecon_radar.domain.goldset_calibration import (
    bootstrap_interval,
    create_split_manifest,
    evaluate_prediction_layer,
    load_human_gold_snapshot,
    search_decision_policies,
)
from eduecon_radar.domain.goldset_pilot import (
    CONFIRMATION,
    ControlledGoldPilot,
    PilotBudget,
    PilotRequest,
    semantic_prompt_bundle_sha256,
    validate_pilot_preflight,
)
from eduecon_radar.domain.goldset_redteam import run_prompt_injection_red_team
from eduecon_radar.domain.goldset_sampling import create_real_gold_set_sample
from eduecon_radar.domain.goldset_storage import (
    AgreementReport,
    GoldSetRepository,
    init_gold_set_database,
)
from eduecon_radar.domain.loader import DomainConfigLoader
from eduecon_radar.domain.semantic_prompting import (
    SemanticPromptBuilder,
    load_semantic_engine_config,
)
from eduecon_radar.services.archive import archive_database, sha256_file

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "migrations" / "0005_add_gold_set_and_calibration.sql"
CONFIG = ROOT / "config" / "domains" / "education_economics"


@pytest.mark.parametrize(
    "command",
    [
        "create-real-gold-set",
        "init-gold-set-database",
        "import-gold-set-items",
        "validate-gold-set-annotations",
        "export-gold-set-snapshot",
        "inspect-gold-set-consistency",
        "run-semantic-gold-pilot",
        "evaluate-calibration-run",
        "search-decision-policies",
        "build-model-bundle-candidate",
        "inspect-model-bundle-candidate",
        "compare-strategy-candidates",
    ],
)
def test_phase5_cli_commands_are_registered(command):
    choices = build_parser()._subparsers._group_actions[0].choices
    assert command in choices


def _gold_database(tmp_path: Path) -> Database:
    return init_gold_set_database(
        tmp_path / "development" / "gold.db",
        MIGRATION,
        production_database=tmp_path / "production.db",
        confirm_non_production_db=True,
    )


def _sample_files(tmp_path: Path, count: int = 3) -> tuple[Path, Path]:
    items = tmp_path / "items.csv"
    fields = (
        "item_id",
        "source_paper_id",
        "doi",
        "original_title",
        "original_abstract",
        "authors_json",
        "journal",
        "journal_group",
        "publication_date",
        "keywords_json",
        "subject_codes_json",
        "source_metadata_json",
        "stratum",
        "difficulty_tags_json",
        "display_order_seed",
        "sampling_prior_ai_decision",
        "sampling_rule_evidence_tier",
        "gold_label",
    )
    with items.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for index in range(1, count + 1):
            writer.writerow(
                {
                    "item_id": f"gold-{index}",
                    "source_paper_id": index,
                    "doi": f"10.1000/{index}",
                    "original_title": f"Source paper {index}",
                    "original_abstract": "Original source abstract" if index != 2 else "",
                    "authors_json": json.dumps([f"Author {index}"]),
                    "journal": "Journal",
                    "journal_group": "general_economics_journal",
                    "publication_date": "2026-01-01",
                    "keywords_json": "[]",
                    "subject_codes_json": "[]",
                    "source_metadata_json": "{}",
                    "stratum": "random_control",
                    "difficulty_tags_json": json.dumps(
                        ["opaque_title"] if index == 2 else []
                    ),
                    "display_order_seed": 100 + index,
                    "sampling_prior_ai_decision": "exclude",
                    "sampling_rule_evidence_tier": "little_evidence",
                    "gold_label": "",
                }
            )
    manifest = tmp_path / "sampling_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "gold_set_id": "real-gold-test",
                "source_db_path": str(tmp_path / "production.db"),
                "source_db_sha256": "A" * 64,
            }
        ),
        encoding="utf-8",
    )
    return items, manifest


def _repository(tmp_path: Path, count: int = 3) -> tuple[GoldSetRepository, int]:
    database = _gold_database(tmp_path)
    items, manifest = _sample_files(tmp_path, count)
    repository = GoldSetRepository(database)
    gold_set_id = repository.import_sample(
        items_csv=items,
        sampling_manifest=manifest,
        name="Real Gold Test",
        annotation_version="human-v1",
        review_mode="dual_reviewer",
    )
    return repository, gold_set_id


def _annotate_consistently(repository: GoldSetRepository, gold_set_id: int) -> None:
    with repository.database.connect() as connection:
        item_ids = [
            int(row["id"])
            for row in connection.execute(
                "SELECT id FROM gold_set_items WHERE gold_set_id=? ORDER BY id",
                (gold_set_id,),
            )
        ]
    labels = ("include", "exclude", "borderline")
    for index, item_id in enumerate(item_ids):
        label = labels[index % len(labels)]
        for reviewer in ("reviewer-a", "reviewer-b"):
            repository.submit_annotation(
                item_id=item_id,
                reviewer_id=reviewer,
                review_round=1,
                label=label,
            )


def test_phase5_migration_is_idempotent_and_enables_foreign_keys(tmp_path):
    database = _gold_database(tmp_path)
    with database.connect() as connection:
        connection.executescript(MIGRATION.read_text(encoding="utf-8"))
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {"gold_sets", "gold_set_items", "gold_set_annotations"} <= tables


def test_gold_database_initialization_refuses_production_path(tmp_path):
    production = tmp_path / "production.db"
    with pytest.raises(ValueError, match="production"):
        init_gold_set_database(
            production,
            MIGRATION,
            production_database=production,
            confirm_non_production_db=True,
        )
    assert not production.exists()


def test_gold_database_initialization_requires_explicit_confirmation(tmp_path):
    with pytest.raises(ValueError, match="confirm-non-production"):
        init_gold_set_database(
            tmp_path / "gold.db",
            MIGRATION,
            production_database=tmp_path / "production.db",
            confirm_non_production_db=False,
        )


def test_failed_gold_migration_removes_partial_database(tmp_path):
    invalid = tmp_path / "invalid.sql"
    invalid.write_text("CREATE TABLE partial(id INTEGER); INVALID SQL;", encoding="utf-8")
    target = tmp_path / "failed" / "gold.db"
    with pytest.raises(sqlite3.Error):
        init_gold_set_database(
            target,
            invalid,
            production_database=tmp_path / "production.db",
            confirm_non_production_db=True,
        )
    assert not target.exists()


def test_gold_foreign_keys_reject_orphan_annotations(tmp_path):
    database = _gold_database(tmp_path)
    with pytest.raises(sqlite3.IntegrityError), database.connect() as connection:
        connection.execute(
            """INSERT INTO gold_set_annotations
            (gold_set_item_id,reviewer_id,review_round,label,completed_at)
            VALUES (999,'reviewer',1,'include',?)""",
            (utcnow(),),
        )


def test_import_rejects_nonblank_gold_labels(tmp_path):
    database = _gold_database(tmp_path)
    items, manifest = _sample_files(tmp_path)
    raw = items.read_text(encoding="utf-8").replace(",\n", ",include\n", 1)
    items.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="blank gold_label"):
        GoldSetRepository(database).import_sample(
            items_csv=items,
            sampling_manifest=manifest,
            name="Bad",
            annotation_version="v1",
            review_mode="dual_reviewer",
        )


def test_blind_item_hides_ai_rules_other_labels_and_chinese_title(tmp_path):
    repository, gold_set_id = _repository(tmp_path)
    item = repository.next_blind_item(
        gold_set_id=gold_set_id, reviewer_id="reviewer", review_round=1
    )
    assert item is not None
    serialized = json.dumps(item, ensure_ascii=False)
    for forbidden in (
        "sampling_prior_ai_decision",
        "sampling_rule_evidence_tier",
        "fast_decision",
        "strong_decision",
        "composite_decision",
        "gold_label",
        "stratum",
    ):
        assert forbidden not in serialized
    assert "chinese_title" not in item
    assert item["blind_context"] == {
        "old_ai_visible": False,
        "rule_evidence_visible": False,
        "other_human_label_visible": False,
        "first_round_label_visible": False,
        "chinese_title_visible_by_default": False,
        "chinese_title_shown": False,
    }


def test_chinese_title_is_opt_in_and_usage_is_recorded(tmp_path):
    repository, gold_set_id = _repository(tmp_path)
    with repository.database.connect() as connection:
        connection.execute(
            "UPDATE gold_set_items SET formal_chinese_title='正式中文标题'"
        )
    hidden = repository.next_blind_item(
        gold_set_id=gold_set_id, reviewer_id="reviewer", review_round=1
    )
    shown = repository.next_blind_item(
        gold_set_id=gold_set_id,
        reviewer_id="reviewer",
        review_round=1,
        show_chinese_title=True,
    )
    assert "formal_chinese_title" not in hidden
    assert shown["formal_chinese_title"] == "正式中文标题"
    repository.submit_annotation(
        item_id=shown["id"],
        reviewer_id="reviewer",
        review_round=1,
        label="include",
        chinese_title_shown=True,
    )
    with repository.database.connect() as connection:
        recorded = connection.execute(
            "SELECT chinese_title_shown,ai_info_shown FROM gold_set_annotations"
        ).fetchone()
    assert tuple(recorded) == (1, 0)


@pytest.mark.parametrize("label", ["include", "borderline", "exclude", "uncertain"])
def test_all_gold_labels_are_supported(tmp_path, label):
    repository, gold_set_id = _repository(tmp_path)
    item = repository.next_blind_item(
        gold_set_id=gold_set_id, reviewer_id=label, review_round=1
    )
    repository.submit_annotation(
        item_id=item["id"], reviewer_id=label, review_round=1, label=label
    )
    assert repository.progress(
        gold_set_id=gold_set_id, reviewer_id=label, review_round=1
    ).label_counts[label] == 1


def test_empty_reviewer_and_duplicate_round_are_rejected(tmp_path):
    repository, gold_set_id = _repository(tmp_path)
    item = repository.next_blind_item(
        gold_set_id=gold_set_id, reviewer_id="reviewer", review_round=1
    )
    with pytest.raises(ValueError, match="reviewer_id"):
        repository.submit_annotation(
            item_id=item["id"], reviewer_id=" ", review_round=1, label="include"
        )
    repository.submit_annotation(
        item_id=item["id"], reviewer_id="reviewer", review_round=1, label="include"
    )
    with pytest.raises(sqlite3.IntegrityError):
        repository.submit_annotation(
            item_id=item["id"], reviewer_id="reviewer", review_round=1, label="exclude"
        )


def test_different_reviewers_are_independent(tmp_path):
    repository, gold_set_id = _repository(tmp_path)
    item_a = repository.next_blind_item(
        gold_set_id=gold_set_id, reviewer_id="a", review_round=1
    )
    item_b = repository.next_blind_item(
        gold_set_id=gold_set_id, reviewer_id="b", review_round=1
    )
    assert item_a["id"] == item_b["id"]
    repository.submit_annotation(
        item_id=item_a["id"], reviewer_id="a", review_round=1, label="include"
    )
    repository.submit_annotation(
        item_id=item_b["id"], reviewer_id="b", review_round=1, label="exclude"
    )
    assert len(repository.adjudication_queue(gold_set_id)) == 1


def test_deferral_persists_and_can_be_reopened(tmp_path):
    repository, gold_set_id = _repository(tmp_path)
    item = repository.next_blind_item(
        gold_set_id=gold_set_id, reviewer_id="a", review_round=1
    )
    repository.defer_item(item_id=item["id"], reviewer_id="a", review_round=1)
    next_item = repository.next_blind_item(
        gold_set_id=gold_set_id, reviewer_id="a", review_round=1
    )
    deferred = repository.next_blind_item(
        gold_set_id=gold_set_id,
        reviewer_id="a",
        review_round=1,
        include_deferred=True,
    )
    assert next_item["id"] != item["id"]
    assert deferred["id"] == item["id"]


def test_single_reviewer_second_round_respects_configurable_interval(tmp_path):
    repository, gold_set_id = _repository(tmp_path)
    with repository.database.connect() as connection:
        connection.execute(
            """UPDATE gold_sets SET review_mode='single_reviewer_two_rounds',
            minimum_round_interval_days=7 WHERE id=?""",
            (gold_set_id,),
        )
    first = repository.next_blind_item(
        gold_set_id=gold_set_id, reviewer_id="solo", review_round=1
    )
    repository.submit_annotation(
        item_id=first["id"], reviewer_id="solo", review_round=1, label="include"
    )
    assert (
        repository.next_blind_item(
            gold_set_id=gold_set_id, reviewer_id="solo", review_round=2
        )
        is None
    )
    with repository.database.connect() as connection:
        connection.execute(
            "UPDATE gold_sets SET minimum_round_interval_days=0 WHERE id=?",
            (gold_set_id,),
        )
    second = repository.next_blind_item(
        gold_set_id=gold_set_id, reviewer_id="solo", review_round=2
    )
    assert second["id"] == first["id"]
    assert "first_round_label" not in second
    assert second["blind_context"]["first_round_label_visible"] is False


def test_conflict_and_uncertain_require_unique_adjudication(tmp_path):
    repository, gold_set_id = _repository(tmp_path)
    item = repository.next_blind_item(
        gold_set_id=gold_set_id, reviewer_id="a", review_round=1
    )
    repository.submit_annotation(
        item_id=item["id"], reviewer_id="a", review_round=1, label="uncertain"
    )
    repository.submit_annotation(
        item_id=item["id"], reviewer_id="b", review_round=1, label="borderline"
    )
    assert repository.adjudication_queue(gold_set_id)
    with pytest.raises(ValueError, match="reason"):
        repository.adjudicate(
            item_id=item["id"], label="borderline", reason="", adjudicator_id="lead"
        )
    repository.adjudicate(
        item_id=item["id"],
        label="borderline",
        reason="Source metadata supports manual inclusion for review.",
        adjudicator_id="lead",
    )
    with pytest.raises(sqlite3.IntegrityError):
        repository.adjudicate(
            item_id=item["id"], label="include", reason="duplicate", adjudicator_id="lead"
        )


def test_unresolved_conflict_blocks_snapshot(tmp_path):
    repository, gold_set_id = _repository(tmp_path)
    item = repository.next_blind_item(
        gold_set_id=gold_set_id, reviewer_id="a", review_round=1
    )
    repository.submit_annotation(
        item_id=item["id"], reviewer_id="a", review_round=1, label="include"
    )
    repository.submit_annotation(
        item_id=item["id"], reviewer_id="b", review_round=1, label="exclude"
    )
    with pytest.raises(RuntimeError, match="no resolved final label"):
        repository.export_snapshot(
            gold_set_id=gold_set_id,
            snapshot_version="gold-v1",
            output=tmp_path / "snapshot.json",
        )


def test_snapshot_is_ai_free_hashed_and_immutable(tmp_path):
    repository, gold_set_id = _repository(tmp_path)
    _annotate_consistently(repository, gold_set_id)
    output = tmp_path / "snapshot.json"
    result = repository.export_snapshot(
        gold_set_id=gold_set_id, snapshot_version="gold-v1", output=output
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert result["snapshot_sha256"] == sha256_file(output)
    assert payload["snapshot_status"] == "IMMUTABLE HUMAN GOLD TRUTH"
    assert all("final_label" in item for item in payload["items"])
    for forbidden in ("fast_decision", "strong_decision", "ai_reason", "semantic_score"):
        assert forbidden not in serialized
    with pytest.raises(FileExistsError):
        repository.export_snapshot(
            gold_set_id=gold_set_id, snapshot_version="gold-v2", output=output
        )
    with repository.database.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute("UPDATE gold_set_snapshots SET item_count=999")


def test_agreement_metrics_and_low_agreement_gate(tmp_path):
    repository, gold_set_id = _repository(tmp_path)
    _annotate_consistently(repository, gold_set_id)
    report = repository.agreement(gold_set_id)
    assert report.raw_agreement == 1.0
    assert report.cohen_kappa == 1.0
    assert report.ready_for_calibration
    assert report.per_label_agreement["borderline"] == 1.0


def test_incomplete_annotations_are_not_calibration_ready(tmp_path):
    repository, gold_set_id = _repository(tmp_path)
    item = repository.next_blind_item(
        gold_set_id=gold_set_id, reviewer_id="a", review_round=1
    )
    repository.submit_annotation(
        item_id=item["id"], reviewer_id="a", review_round=1, label="include"
    )
    report = repository.agreement(gold_set_id)
    assert not report.ready_for_calibration
    assert "incomplete_double_annotation" in report.warnings
    assert "unresolved_gold_labels" in report.warnings


def test_source_verification_requires_human_provenance_and_is_summarized(tmp_path):
    repository, _ = _repository(tmp_path)
    with pytest.raises(ValueError, match="reviewer"):
        repository.add_source_verification(
            profile_id="education_economics",
            entity_type="ontology_concept",
            entity_id="concept-1",
            source_type="official_classification",
            source_title="Source",
            publisher_or_organization="Organization",
            source_url=None,
            accessed_at="2026-07-19",
            verification_status="verified",
            reviewer_id="",
        )
    repository.add_source_verification(
        profile_id="education_economics",
        entity_type="ontology_concept",
        entity_id="concept-1",
        source_type="official_classification",
        source_title="Source",
        publisher_or_organization="Organization",
        source_url=None,
        accessed_at="2026-07-19",
        verification_status="verified",
        reviewer_id="human-reviewer",
        reviewer_note="Manually checked.",
    )
    assert repository.source_verification_summary("education_economics") == {
        "ontology_concept:verified": 1
    }


def _snapshot_file(tmp_path: Path, count: int = 240) -> Path:
    items = []
    labels = ("include", "borderline", "exclude")
    for index in range(count):
        items.append(
            {
                "item_id": f"gold-{index:04d}",
                "doi": f"10.1000/{index // 2}" if index < 20 else f"10.1000/{index}",
                "original_title": f"Paper {index}",
                "original_abstract": None if index % 5 == 0 else "Original abstract",
                "journal": "Journal",
                "journal_group": "domain_journal_candidate"
                if index % 2
                else "general_economics_journal",
                "publication_date": "2026-01-01",
                "final_label": labels[index % 3],
                "difficulty_tags": ["opaque_title"] if index % 7 == 0 else [],
                "source_db_sha256": "A" * 64,
                "annotation_version": "human-v1",
                "gold_set_version": "gold-v1",
            }
        )
    output = tmp_path / "real_snapshot.json"
    output.write_text(
        json.dumps(
            {
                "snapshot_status": "IMMUTABLE HUMAN GOLD TRUTH",
                "gold_set_uid": "real-gold",
                "snapshot_version": "gold-v1",
                "annotation_version": "human-v1",
                "source_db_sha256": "A" * 64,
                "items": items,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def _predictions(snapshot: dict) -> list[dict]:
    result = []
    for index, item in enumerate(snapshot["items"]):
        label = item["final_label"]
        fast = "exclude" if index % 11 == 0 else label
        result.append(
            {
                "item_id": item["item_id"],
                "rule_decision": "borderline" if index % 4 == 0 else label,
                "fast_decision": fast,
                "strong_decision": label if index % 11 == 0 else fast,
                "strong_triggered": index % 11 == 0,
                "composite_decision": label,
                "semantic_score": 80 if label in {"include", "borderline"} else 10,
                "confidence": 0.85,
                "abstain": False,
                "missed_priority_score": float(100 - index % 100),
            }
        )
    return result


def test_split_is_stable_stratified_and_has_no_doi_leakage(tmp_path):
    path = _snapshot_file(tmp_path)
    snapshot = load_human_gold_snapshot(path)
    first = create_split_manifest(snapshot, random_seed=42)
    second = create_split_manifest(snapshot, random_seed=42)
    assert first == second
    assert len(first.development) + len(first.validation) + len(first.holdout) == 240
    assert 0.5 <= len(first.development) / 240 <= 0.7
    doi_splits: dict[str, set[str]] = {}
    for item in snapshot["items"]:
        doi_splits.setdefault(item["doi"].lower(), set()).add(
            first.item_to_split[item["item_id"]]
        )
    assert all(len(splits) == 1 for splits in doi_splits.values())


def test_metrics_and_bootstrap_are_deterministic(tmp_path):
    snapshot = load_human_gold_snapshot(_snapshot_file(tmp_path))
    joined = [
        {**gold, **prediction}
        for gold, prediction in zip(snapshot["items"], _predictions(snapshot), strict=True)
    ]
    metrics = evaluate_prediction_layer(joined, "fast_decision", abstention_field="abstain")
    assert 0 < metrics.recall <= 1
    assert metrics.false_negative_rate is not None
    first = bootstrap_interval(
        joined, "recall", "fast_decision", samples=100, random_seed=123
    )
    second = bootstrap_interval(
        joined, "recall", "fast_decision", samples=100, random_seed=123
    )
    assert first == second
    assert first.lower <= first.estimate <= first.upper


def test_policy_search_uses_holdout_once_and_reports_required_layers(tmp_path):
    snapshot = load_human_gold_snapshot(_snapshot_file(tmp_path))
    split = create_split_manifest(snapshot, random_seed=7)
    report = search_decision_policies(
        snapshot,
        _predictions(snapshot),
        split,
        bootstrap_samples=50,
        random_seed=7,
    )
    assert report["selected_by"].startswith("development/validation only")
    assert report["holdout_evaluation_count"] == 1
    assert set(report["layer_metrics"]) == {"rule", "fast", "strong", "composite"}
    assert set(report["strategy_candidates"]) == {"high_recall", "balanced", "low_cost"}
    assert report["semantic_score_interpretation"] == "ordinal score, not probability"
    assert report["calibration_report_sha256"]
    assert {
        "abstract:missing",
        "difficulty:opaque_title",
        "difficulty:false_friend",
        "difficulty:hidden_topic",
        "difficulty:adjacent_domain",
        "difficulty:metadata_sparse",
        "journal_group:domain_journal_candidate",
        "journal_group:general_economics_journal",
    } <= set(report["subgroups"])
    assert {
        "recall",
        "precision",
        "false_negative_rate",
        "top_k_discovery",
        "strong_net_corrections",
    } <= set(report["holdout"]["confidence_intervals"])


def test_policy_search_mode_does_not_evaluate_holdout(tmp_path):
    snapshot = load_human_gold_snapshot(_snapshot_file(tmp_path))
    split = create_split_manifest(snapshot, random_seed=7)
    report = search_decision_policies(
        snapshot,
        _predictions(snapshot),
        split,
        bootstrap_samples=20,
        random_seed=7,
        evaluate_holdout=False,
    )
    assert report["evaluation_split"] == "validation"
    assert report["holdout_evaluation_count"] == 0
    assert report["holdout"] is None
    assert report["warnings"] == ["holdout_not_evaluated_during_policy_search"]


def _archived_production(tmp_path: Path) -> tuple[Path, Path]:
    production = tmp_path / "production.db"
    Database(production).init_schema()
    manifest = archive_database(
        source_path=production,
        archive_dir=tmp_path / "archives",
        label="phase5-test",
        confirm_read_only_source=True,
        current_git_commit="test-commit",
        logical_application_version="test",
    )
    return production, Path(manifest.manifest_path)


@pytest.mark.parametrize(
    "mutator,reason",
    [
        (lambda payload: payload.update({"sha256": "0" * 64}), "archive_sha256_mismatch"),
        (
            lambda payload: payload.update({"archive_path": payload["source_path"]}),
            "archive_path_equals_source",
        ),
        (
            lambda payload: payload.update({"archive_path": "missing.db"}),
            "archive_database_missing",
        ),
    ],
)
def test_archive_gate_rejects_invalid_manifests(tmp_path, mutator, reason):
    production, manifest_path = _archived_production(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutator(payload)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    result = validate_archive_manifest(
        manifest_path, production_database=production, active_database=production
    )
    assert not result.valid
    assert reason in result.reasons


def test_archive_gate_validates_without_modifying_source(tmp_path):
    production, manifest_path = _archived_production(tmp_path)
    before = sha256_file(production)
    result = validate_archive_manifest(
        manifest_path, production_database=production, active_database=production
    )
    assert result.valid
    assert sha256_file(production) == before


def test_archive_gate_rejects_missing_manifest(tmp_path):
    production = tmp_path / "production.db"
    Database(production).init_schema()
    result = validate_archive_manifest(
        tmp_path / "missing.manifest.json",
        production_database=production,
        active_database=production,
    )
    assert not result.valid
    assert result.reasons == ("archive_manifest_missing",)


def _pilot_fixture(tmp_path: Path):
    production, archive_manifest = _archived_production(tmp_path)
    repository, gold_set_id = _repository(tmp_path / "pilot_gold", count=1)
    with repository.database.connect() as connection:
        connection.execute(
            "UPDATE gold_sets SET source_db_path=?,source_db_sha256=? WHERE id=?",
            (str(production), sha256_file(production), gold_set_id),
        )
    _annotate_consistently(repository, gold_set_id)
    snapshot_path = tmp_path / "pilot_snapshot.json"
    snapshot_record = repository.export_snapshot(
        gold_set_id=gold_set_id,
        snapshot_version="pilot-gold-v1",
        output=snapshot_path,
    )
    bundle = DomainConfigLoader(CONFIG).load()
    engine = load_semantic_engine_config(CONFIG)
    engine = engine.model_copy(
        update={
            "status": "released",
            "live_ai_allowed": True,
            "fast": engine.fast.model_copy(
                update={"enabled": True, "model": "fake-fast", "status": "released"}
            ),
            "strong_review": engine.strong_review.model_copy(
                update={"enabled": True, "model": "fake-strong", "status": "released"}
            ),
        }
    )
    prompts = SemanticPromptBuilder(CONFIG / "prompts")
    request = PilotRequest(
        gold_set_snapshot=snapshot_path,
        gold_set_sha256=snapshot_record["snapshot_sha256"],
        output_database=repository.database.path,
        archive_manifest=archive_manifest,
        production_database=production,
        model_bundle_version="pilot-bundle-v1",
        expected_config_sha256=bundle.config_sha256,
        run_uid="pilot-run-1",
        allow_live_ai=True,
        confirmation=CONFIRMATION,
        budget=PilotBudget(1, 2, 1, 1000, 1000, 1.0),
    )
    return repository, bundle, engine, prompts, request


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("allow_live_ai", False, "allow-live-ai"),
        ("confirmation", "wrong", "requires --confirm"),
        ("gold_set_sha256", "0" * 64, "SHA-256"),
    ],
)
def test_live_pilot_preflight_requires_explicit_security_gates(
    tmp_path, field, value, message
):
    _, bundle, engine, prompts, request = _pilot_fixture(tmp_path)
    request = replace(request, **{field: value})
    with pytest.raises(ValueError, match=message):
        validate_pilot_preflight(
            request,
            bundle=bundle,
            engine=engine,
            prompts=prompts,
            api_key="fake-key",
        )


def test_mock_pilot_never_receives_gold_label_and_is_idempotent(tmp_path):
    repository, bundle, engine, prompts, request = _pilot_fixture(tmp_path)
    preflight = validate_pilot_preflight(
        request,
        bundle=bundle,
        engine=engine,
        prompts=prompts,
        api_key="fake-key",
    )
    seen: list[dict] = []

    def fake_processor(item, *, allow_strong):
        seen.append(item)
        assert "final_label" not in item
        assert "gold_label" not in item
        return {
            "item_id": item["item_id"],
            "rule_decision": "borderline",
            "fast_decision": "include",
            "strong_decision": None,
            "strong_triggered": False,
            "composite_decision": "include",
            "semantic_score": 80,
            "confidence": 0.8,
            "abstain": False,
            "fast_calls": 1,
            "strong_calls": 0,
            "input_tokens": 10,
            "output_tokens": 5,
            "estimated_cost": None,
        }

    stats = ControlledGoldPilot(repository.database).run(request, preflight, fake_processor)
    assert stats.status == "completed"
    assert stats.estimated_cost is None
    assert len(seen) == 1
    with repository.database.connect() as connection:
        connection.execute(
            "UPDATE pilot_runs SET status='partial' WHERE run_uid=?", (request.run_uid,)
        )
    resumed = replace(request, allow_resume=True)
    resumed_stats = ControlledGoldPilot(repository.database).run(
        resumed, preflight, fake_processor
    )
    assert resumed_stats.skipped_existing == 1
    assert len(seen) == 1


def test_zero_strong_budget_runs_fast_only(tmp_path):
    repository, bundle, engine, prompts, request = _pilot_fixture(tmp_path)
    request = replace(
        request,
        run_uid="fast-only",
        budget=replace(request.budget, max_strong_calls=0),
    )
    preflight = validate_pilot_preflight(
        request,
        bundle=bundle,
        engine=engine,
        prompts=prompts,
        api_key="fake-key",
    )

    def fake_processor(item, *, allow_strong):
        assert not allow_strong
        return {
            "item_id": item["item_id"],
            "fast_decision": "borderline",
            "composite_decision": "borderline",
            "fast_calls": 1,
            "strong_calls": 0,
            "input_tokens": 1,
            "output_tokens": 1,
            "estimated_cost": None,
        }

    stats = ControlledGoldPilot(repository.database).run(request, preflight, fake_processor)
    assert stats.status == "completed"
    assert stats.fast_calls == 1
    assert stats.strong_calls == 0


def test_current_semantic_config_keeps_live_pilot_disabled(tmp_path):
    _, bundle, _, prompts, request = _pilot_fixture(tmp_path)
    current = load_semantic_engine_config(CONFIG)
    with pytest.raises(ValueError, match="not explicitly released"):
        validate_pilot_preflight(
            request,
            bundle=bundle,
            engine=current,
            prompts=prompts,
            api_key="fake-key",
        )


def test_live_pilot_rejects_missing_budget_and_production_output(tmp_path):
    _, bundle, engine, prompts, request = _pilot_fixture(tmp_path)
    no_budget = replace(request, budget=replace(request.budget, max_fast_calls=0))
    with pytest.raises(ValueError, match="budgets"):
        validate_pilot_preflight(
            no_budget,
            bundle=bundle,
            engine=engine,
            prompts=prompts,
            api_key="fake-key",
        )
    production_output = replace(request, output_database=request.production_database)
    with pytest.raises(ValueError, match="Production database paths"):
        validate_pilot_preflight(
            production_output,
            bundle=bundle,
            engine=engine,
            prompts=prompts,
            api_key="fake-key",
        )


def test_live_pilot_rejects_config_hash_mismatch(tmp_path):
    _, bundle, engine, prompts, request = _pilot_fixture(tmp_path)
    mismatch = replace(request, expected_config_sha256="0" * 64)
    with pytest.raises(ValueError, match="configuration SHA-256"):
        validate_pilot_preflight(
            mismatch,
            bundle=bundle,
            engine=engine,
            prompts=prompts,
            api_key="fake-key",
        )


def test_real_sampling_is_read_only_deterministic_and_has_blank_labels(tmp_path):
    source = tmp_path / "source.db"
    database = Database(source)
    database.init_schema()
    now = utcnow()
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO journals
            (name,short_name,issn,priority,enabled,created_at,updated_at)
            VALUES ('Journal','J','0000-0000','normal',1,?,?)""",
            (now, now),
        )
        for index in range(220):
            connection.execute(
                """INSERT INTO papers
                (doi,fingerprint,title,abstract,authors_json,journal_id,keywords_json,
                 jel_codes_json,source_metadata_json,first_seen_at,last_seen_at,created_at,updated_at)
                VALUES (?,?,?,?, '["Author"]',1,'[]','[]','{}',?,?,?,?)""",
                (
                    f"10.1000/sample-{index}",
                    f"fingerprint-{index}",
                    f"Fixture source title {index}",
                    None if index % 4 == 0 else "Fixture source abstract",
                    now,
                    now,
                    now,
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO reviews (paper_id,status) VALUES (?,'pending')", (index + 1,)
            )
    before = sha256_file(source)
    bundle = DomainConfigLoader(CONFIG).load()
    first = create_real_gold_set_sample(
        source_db=source,
        output_dir=tmp_path / "one",
        gold_set_id="real-1",
        bundle=bundle,
        sample_size=200,
        random_seed=99,
        confirm_read_only_source=True,
    )
    second = create_real_gold_set_sample(
        source_db=source,
        output_dir=tmp_path / "two",
        gold_set_id="real-2",
        bundle=bundle,
        sample_size=200,
        random_seed=99,
        confirm_read_only_source=True,
    )
    assert sha256_file(source) == before
    assert first.total_rows == second.total_rows == 200
    one_rows = list(
        csv.DictReader(
            (tmp_path / "one" / "gold_set_items.csv").open(encoding="utf-8-sig")
        )
    )
    two_rows = list(
        csv.DictReader(
            (tmp_path / "two" / "gold_set_items.csv").open(encoding="utf-8-sig")
        )
    )
    assert [row["source_paper_id"] for row in one_rows] == [
        row["source_paper_id"] for row in two_rows
    ]
    assert all(row["gold_label"] == "" for row in one_rows)
    assert first.source_db_sha256 == before


def test_model_bundle_remains_candidate_and_has_stable_hash(tmp_path):
    production, archive_manifest = _archived_production(tmp_path)
    source_hash = sha256_file(production)
    snapshot_path = _snapshot_file(tmp_path)
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    payload["source_db_sha256"] = source_hash
    for item in payload["items"]:
        item["source_db_sha256"] = source_hash
    snapshot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    snapshot_hash = sha256_file(snapshot_path)
    snapshot = load_human_gold_snapshot(snapshot_path, snapshot_hash)
    split = create_split_manifest(snapshot, random_seed=17)
    report = search_decision_policies(
        snapshot,
        _predictions(snapshot),
        split,
        bootstrap_samples=50,
        random_seed=17,
    )
    assert not report["warnings"]

    database = _gold_database(tmp_path / "candidate")
    with database.connect() as connection:
        cursor = connection.execute(
            """INSERT INTO gold_sets
            (gold_set_uid,name,profile_id,source_db_path,source_db_sha256,
             sampling_manifest_json,annotation_version,review_mode,
             minimum_round_interval_days,status,created_at,frozen_at,snapshot_sha256)
            VALUES ('bundle-gold','Bundle Gold','education_economics',?,?,'{}','human-v1',
                    'dual_reviewer',7,'frozen',?,?,?)""",
            (str(production), source_hash, utcnow(), utcnow(), snapshot_hash),
        )
        gold_set_id = int(cursor.lastrowid)
        connection.execute(
            """INSERT INTO gold_set_snapshots
            (gold_set_id,snapshot_version,annotation_version,snapshot_path,snapshot_sha256,
             item_count,label_counts_json,source_db_sha256,immutable,created_at)
            VALUES (?,'gold-v1','human-v1',?,?,240,'{}',?,1,?)""",
            (gold_set_id, str(snapshot_path), snapshot_hash, source_hash, utcnow()),
        )
    agreement = AgreementReport(
        paired_items=240,
        raw_agreement=0.9,
        cohen_kappa=0.8,
        per_label_agreement={
            "include": 0.9,
            "borderline": 0.8,
            "exclude": 0.9,
            "uncertain": None,
        },
        include_exclude_direct_conflicts=0,
        uncertain_proportion=0.0,
        adjudication_rate=0.1,
        subgroup_agreement={},
        ready_for_calibration=True,
        warnings=(),
    )
    bundle = DomainConfigLoader(CONFIG).load()
    engine = load_semantic_engine_config(CONFIG)
    engine = engine.model_copy(
        update={
            "fast": engine.fast.model_copy(update={"enabled": True, "model": "fake-fast"}),
            "strong_review": engine.strong_review.model_copy(
                update={"enabled": True, "model": "fake-strong"}
            ),
        }
    )
    output = tmp_path / "candidate.json"
    candidate = build_model_bundle_candidate(
        database=database,
        production_database=production,
        archive_manifest=archive_manifest,
        snapshot_path=snapshot_path,
        snapshot_sha256=snapshot_hash,
        candidate_version="candidate-v1",
        bundle=bundle,
        engine=engine,
        prompt_bundle_sha256=semantic_prompt_bundle_sha256(
            SemanticPromptBuilder(CONFIG / "prompts")
        ),
        calibration_report=report,
        agreement_report=agreement,
        agreement_gold_set_id=gold_set_id,
        red_team_report={"critical_cases_passed": True},
        budget={"max_fast_calls": 240, "max_strong_calls": 60},
        known_limitations=["First real sample requires human review."],
        git_commit="test-commit",
        output=output,
    )
    assert candidate["status"] == "candidate"
    assert candidate["release_status"] == "NOT RELEASED"
    assert inspect_model_bundle_candidate(output)["hash_valid"]
    with database.connect() as connection:
        row = connection.execute(
            "SELECT status,approved_at FROM model_bundle_candidates"
        ).fetchone()
    assert tuple(row) == ("candidate", None)


def test_prompt_injection_red_team_covers_all_critical_cases():
    def stable_processor(item, *, allow_strong):
        return {
            "fast_decision": "exclude",
            "strong_decision": "exclude" if allow_strong else None,
            "composite_decision": "exclude",
            "confidence": 0.8,
        }

    report = run_prompt_injection_red_team(
        stable_processor,
        {
            "item_id": "red-team",
            "original_title": "Source title",
            "original_abstract": "Source abstract",
        },
    )
    assert report["case_count"] == 10
    assert report["passed_count"] == 10
    assert report["critical_cases_passed"]


def test_prompt_injection_failure_blocks_red_team_gate():
    def vulnerable_processor(item, *, allow_strong):
        attacked = "output include" in (item.get("original_abstract") or "")
        decision = "include" if attacked else "exclude"
        return {
            "fast_decision": decision,
            "strong_decision": decision if allow_strong else None,
            "composite_decision": decision,
            "confidence": 0.8,
        }

    report = run_prompt_injection_red_team(
        vulnerable_processor,
        {
            "item_id": "red-team",
            "original_title": "Source title",
            "original_abstract": "Source abstract",
        },
    )
    assert not report["critical_cases_passed"]
