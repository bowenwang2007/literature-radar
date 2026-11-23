from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from eduecon_radar.db import Database
from eduecon_radar.domain.goldset_redteam import (
    D1_INJECTION_CASES,
    run_prompt_injection_red_team,
)
from eduecon_radar.services.archive import sha256_file
from eduecon_radar.services.gold_governance import (
    PROTOCOL_VERSION,
    GoldGovernanceRepository,
    canonical_hash,
    prepare_gold_corpus,
)
from eduecon_radar.services.gold_metrics import (
    cluster_stratified_bootstrap,
    evaluate_gold_predictions,
)
from eduecon_radar.services.gold_pilot_governance import (
    GoldPilotBudget,
    GoldPilotRunner,
    evaluate_holdout_once,
    validate_gold_pilot_preflight,
    verify_candidate_isolation,
)
from eduecon_radar.services.migration_integrity import migrate_database


def _source_database(path: Path, count: int = 12) -> Path:
    database = Database(path)
    database.init_schema()
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO journals
            (id,name,short_name,issn,rank,rank_source,priority,enabled,created_at,updated_at)
            VALUES (1,'American Economic Review','AER','0002-8282',1,'test','high',1,?,?)""",
            ("2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
        )
        for index in range(1, count + 1):
            connection.execute(
                """INSERT INTO papers
                (id,doi,fingerprint,title,abstract,authors_json,journal_id,online_date,
                 source_url,keywords_json,jel_codes_json,source_metadata_json,
                 first_seen_at,last_seen_at,created_at,updated_at)
                VALUES (?,?,?,?,?,?,1,?,?,?,?,?,?,?,?,?)""",
                (
                    index,
                    f"10.1000/test{index}",
                    f"fingerprint-{index}",
                    f"Source paper {index}",
                    None if index % 4 == 0 else f"Original abstract {index} " * 25,
                    json.dumps([f"Author {index}"]),
                    f"202{index % 6}-01-01",
                    f"https://example.test/{index}",
                    "[]",
                    json.dumps(["I20"] if index % 3 == 0 else []),
                    json.dumps(
                        {
                            "DOI": f"10.1000/test{index}",
                            "screening": {"decision": "include"},
                        }
                    ),
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-02T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-02T00:00:00+00:00",
                ),
            )
    return path


def _sample_plan(path: Path, size: int = 10) -> Path:
    path.write_text(
        f"""plan_version: test-plan-v1
corpus_version: test-corpus-v1
sample_size: {size}
random_seed: 123
replacement: false
journal_families:
  general_economics: [AER]
  other_or_low_prior: []
dimensions: [journal_family, abstract_quality_status]
""",
        encoding="utf-8",
    )
    return path


def _gold_database(path: Path) -> Path:
    status = migrate_database(path, "gold", dry_run=False)
    assert status.pending == ()
    return path


def _prepared(tmp_path: Path, count: int = 12, sample_size: int = 10):
    source = _source_database(tmp_path / "source.db", count)
    gold = _gold_database(tmp_path / "gold.db")
    plan = _sample_plan(tmp_path / "plan.yaml", sample_size)
    result = prepare_gold_corpus(
        source_database=source,
        gold_database=gold,
        sample_plan_path=plan,
        output_manifest=tmp_path / "corpus-manifest.json",
        ontology_path=None,
    )
    return source, gold, result


def _submit_all(repository: GoldGovernanceRepository, corpus_version: str) -> None:
    repository.ensure_presentations(
        corpus_version=corpus_version,
        reviewer_id="reviewer-a",
        repeat_fraction=0,
    )
    while item := repository.next_blind_record(
        corpus_version=corpus_version, reviewer_id="reviewer-a"
    ):
        repository.submit_annotation(
            presented_record_id=item["presented_record_id"],
            reviewer_id="reviewer-a",
            annotation_round=1,
            label="include" if item["presentation_order"] % 3 == 0 else "exclude",
            education_centrality=(
                "core" if item["presentation_order"] % 3 == 0 else "none"
            ),
            economics_relevance=(
                "economic_question"
                if item["presentation_order"] % 3 == 0
                else "non_economic"
            ),
            evidence_sufficiency="sufficient",
            reviewer_confidence="high",
            reason_codes=[],
            optional_notes=None,
            started_at=(datetime.now(UTC) - timedelta(seconds=2)).isoformat(),
            protocol_version=PROTOCOL_VERSION,
            protocol_sha256="A" * 64,
        )


def _snapshot(tmp_path: Path):
    _, gold, result = _prepared(tmp_path)
    repository = GoldGovernanceRepository(Database(gold))
    _submit_all(repository, result.corpus_version)
    split = repository.create_split(
        corpus_version=result.corpus_version,
        split_version="split-v1",
        random_seed=44,
    )
    protocol = tmp_path / "protocol.md"
    protocol.write_text("# Test protocol\n", encoding="utf-8")
    limitations = tmp_path / "limitations.txt"
    limitations.write_text("small fixture\n", encoding="utf-8")
    snapshot_path = tmp_path / "snapshot.json"
    snapshot = repository.freeze_snapshot(
        corpus_version=result.corpus_version,
        snapshot_version="snapshot-v1",
        split_manifest_sha256=split["manifest_sha256"],
        protocol_path=protocol,
        output=snapshot_path,
        git_commit=None,
        known_limitations=["small fixture"],
    )
    return gold, repository, snapshot_path, snapshot, split


def test_corpus_is_frozen_reproducible_and_source_is_read_only(tmp_path: Path) -> None:
    source, gold, result = _prepared(tmp_path, count=8, sample_size=20)
    assert result.realized_size == 8
    assert sha256_file(source) == result.source_database_sha256
    repository = GoldGovernanceRepository(Database(gold))
    assert repository.verify_corpus(result.corpus_version)["valid"] is True
    with sqlite3.connect(gold) as connection:
        rows = connection.execute(
            """SELECT gold_paper_id,source_paper_id,content_sha256,
            stratum_population_size,inclusion_probability
            FROM gold_corpus_papers ORDER BY source_paper_id"""
        ).fetchall()
        assert len({row[1] for row in rows}) == 8
        assert all(row[3] > 0 and 0 < row[4] <= 1 for row in rows)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE gold_corpus_papers SET original_title='changed' WHERE id=1"
            )


def test_blind_annotation_hidden_repeats_and_revision_history(tmp_path: Path) -> None:
    _, gold, result = _prepared(tmp_path)
    repository = GoldGovernanceRepository(Database(gold))
    total = repository.ensure_presentations(
        corpus_version=result.corpus_version,
        reviewer_id="blind-reviewer",
        repeat_fraction=0.10,
        random_seed=91,
    )
    assert total == result.realized_size + round(result.realized_size * 0.10)
    item = repository.next_blind_record(
        corpus_version=result.corpus_version, reviewer_id="blind-reviewer"
    )
    assert item is not None
    assert item["journal_visible"] is False
    assert "journal" not in item
    assert "hidden_repeat" not in item
    started = (datetime.now(UTC) - timedelta(seconds=3)).isoformat()
    annotation_id = repository.submit_annotation(
        presented_record_id=item["presented_record_id"],
        reviewer_id="blind-reviewer",
        annotation_round=1,
        label="borderline",
        education_centrality="substantial",
        economics_relevance="adjacent_social_science",
        evidence_sufficiency="limited",
        reviewer_confidence="medium",
        reason_codes=["boundary_adjacent"],
        optional_notes=None,
        started_at=started,
        protocol_version=PROTOCOL_VERSION,
        protocol_sha256="B" * 64,
    )
    source = repository.reveal_source(
        annotation_id=annotation_id, reviewer_id="blind-reviewer"
    )
    assert source["journal"] == "American Economic Review"
    repository.submit_annotation(
        presented_record_id=item["presented_record_id"],
        reviewer_id="blind-reviewer",
        annotation_round=1,
        label="include",
        education_centrality="core",
        economics_relevance="economic_question",
        evidence_sufficiency="sufficient",
        reviewer_confidence="high",
        reason_codes=[],
        optional_notes="revision",
        started_at=started,
        protocol_version=PROTOCOL_VERSION,
        protocol_sha256="B" * 64,
    )
    with sqlite3.connect(gold) as connection:
        schedule = connection.execute(
            """SELECT corpus_paper_id,presentation_order,hidden_repeat
            FROM gold_presented_records ORDER BY presentation_order"""
        ).fetchall()
        rows = connection.execute(
            """SELECT revision_number,active FROM gold_annotation_revisions
            WHERE presented_record_id=? ORDER BY revision_number""",
            (item["presented_record_id"],),
        ).fetchall()
    assert rows == [(1, 0), (2, 1)]
    positions: dict[int, list[int]] = {}
    for paper_id, order, _ in schedule:
        positions.setdefault(paper_id, []).append(order)
    assert all(abs(values[0] - values[1]) > 1 for values in positions.values() if len(values) > 1)


def test_review_queue_and_arbitration_keep_annotation_history(tmp_path: Path) -> None:
    _, gold, result = _prepared(tmp_path, count=4, sample_size=4)
    repository = GoldGovernanceRepository(Database(gold))
    repository.ensure_presentations(
        corpus_version=result.corpus_version,
        reviewer_id="reviewer-a",
        repeat_fraction=0,
    )
    item = repository.next_blind_record(
        corpus_version=result.corpus_version, reviewer_id="reviewer-a"
    )
    assert item
    started = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    repository.submit_annotation(
        presented_record_id=item["presented_record_id"],
        reviewer_id="reviewer-a",
        annotation_round=1,
        label="include",
        education_centrality="none",
        economics_relevance="economic_question",
        evidence_sufficiency="insufficient",
        reviewer_confidence="high",
        reason_codes=["boundary_adjacent"],
        optional_notes=None,
        started_at=started,
        protocol_version=PROTOCOL_VERSION,
        protocol_sha256="D" * 64,
    )
    queue = repository.review_queue(result.corpus_version)
    reasons = {row["queue_reason"] for row in queue}
    assert "logic_include_centrality_none" in reasons
    assert "logic_high_confidence_insufficient_evidence" in reasons
    assert repository.ensure_review_presentations(
        corpus_version=result.corpus_version, reviewer_id="reviewer-b"
    ) == 1
    review = repository.next_review_record(
        corpus_version=result.corpus_version, reviewer_id="reviewer-b"
    )
    assert review and review["ai_visible"] is False
    repository.submit_annotation(
        presented_record_id=review["presented_record_id"],
        reviewer_id="reviewer-b",
        annotation_round=2,
        label="exclude",
        education_centrality="none",
        economics_relevance="non_economic",
        evidence_sufficiency="sufficient",
        reviewer_confidence="high",
        reason_codes=[],
        optional_notes=None,
        started_at=started,
        protocol_version=PROTOCOL_VERSION,
        protocol_sha256="D" * 64,
    )
    arbitration = repository.arbitration_queue(result.corpus_version)
    assert len(arbitration) == 1
    repository.arbitrate(
        corpus_version=result.corpus_version,
        gold_paper_id=arbitration[0]["gold_paper_id"],
        final_gold_label="borderline",
        basis="conflicting human rounds",
        arbitrator_id="arb",
        protocol_version=PROTOCOL_VERSION,
        protocol_sha256="D" * 64,
    )
    with sqlite3.connect(gold) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM gold_annotation_revisions"
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM gold_arbitration_revisions"
        ).fetchone()[0] == 1
    assert repository.unresolved_count(result.corpus_version) == 0


def test_group_aware_split_snapshot_and_holdout_one_time_gate(tmp_path: Path) -> None:
    gold, repository, snapshot_path, snapshot, split = _snapshot(tmp_path)
    assert repository.verify_split(split["manifest_sha256"])["valid"] is True
    assert repository.inspect_snapshot(snapshot["snapshot_sha256"])["hash_valid"] is True
    predictions = [
        {"gold_paper_id": item["gold_paper_id"], "prediction": "exclude"}
        for item in json.loads(snapshot_path.read_text(encoding="utf-8"))[
            "final_gold_labels"
        ]
    ]
    with pytest.raises(ValueError, match="confirmations"):
        evaluate_holdout_once(
            database=Database(gold),
            snapshot_sha256=snapshot["snapshot_sha256"],
            candidate_bundle_sha256="C" * 64,
            split_manifest_sha256=split["manifest_sha256"],
            predictions=predictions,
            confirm_final_candidate=False,
            confirm_holdout_one_time_evaluation=True,
        )
    report = evaluate_holdout_once(
        database=Database(gold),
        snapshot_sha256=snapshot["snapshot_sha256"],
        candidate_bundle_sha256="C" * 64,
        split_manifest_sha256=split["manifest_sha256"],
        predictions=predictions,
        confirm_final_candidate=True,
        confirm_holdout_one_time_evaluation=True,
    )
    assert report["holdout_evaluation_count"] == 1
    with pytest.raises(ValueError, match="already consumed"):
        evaluate_holdout_once(
            database=Database(gold),
            snapshot_sha256=snapshot["snapshot_sha256"],
            candidate_bundle_sha256="C" * 64,
            split_manifest_sha256=split["manifest_sha256"],
            predictions=predictions,
            confirm_final_candidate=True,
            confirm_holdout_one_time_evaluation=True,
        )


def test_metrics_are_three_class_weighted_and_cluster_aware() -> None:
    rows = [
        {
            "gold_paper_id": "a",
            "final_label": "include",
            "prediction": "exclude",
            "duplicate_cluster_uid": "cluster-a",
            "sampling_weight": 2,
        },
        {
            "gold_paper_id": "b",
            "final_label": "borderline",
            "prediction": "borderline",
            "duplicate_cluster_uid": "cluster-b",
            "sampling_weight": 1,
        },
        {
            "gold_paper_id": "c",
            "final_label": "exclude",
            "prediction": "exclude",
            "duplicate_cluster_uid": "cluster-c",
            "sampling_weight": 1,
        },
    ]
    report = evaluate_gold_predictions(rows)
    assert report["safety_binary"]["false_negative_rate"] == 0.5
    assert report["weighted_sampling"]["weighted_false_negative_rate"] == pytest.approx(
        2 / 3
    )
    interval = cluster_stratified_bootstrap(
        rows,
        lambda sample: evaluate_gold_predictions(sample)["safety_binary"][
            "relevant_recall"
        ],
        iterations=50,
        random_seed=8,
    )
    assert interval.cluster_aware is True
    assert interval.warning == "small_sample_interval_unstable"


def test_pilot_requires_all_gates_and_never_sends_gold_label(tmp_path: Path) -> None:
    gold, _, snapshot_path, snapshot, split = _snapshot(tmp_path)
    production = _source_database(tmp_path / "production.db", 2)
    config = tmp_path / "candidate.json"
    config.write_text(
        json.dumps(
            {
                "status": "candidate",
                "provider": "fake",
                "fast_model": "fake-fast",
                "strong_model": "fake-strong",
                "prompt_bundle_sha256": "E" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    budget = GoldPilotBudget(20, 10000, 5000, 100000)
    kwargs = {
        "gold_database": gold,
        "production_database": production,
        "snapshot_path": snapshot_path,
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "candidate_runtime_config": config,
        "candidate_runtime_config_sha256": sha256_file(config),
        "split_name": "development",
        "budget": budget,
        "confirm_live_ai": True,
        "confirm_labels_not_in_prompt": True,
        "confirm_isolated_gold_database": True,
        "confirm_production_database_readonly": True,
    }
    with pytest.raises(ValueError, match="cannot run on Holdout"):
        validate_gold_pilot_preflight(**{**kwargs, "split_name": "holdout"})
    incomplete_config = tmp_path / "incomplete-candidate.json"
    incomplete_config.write_text(
        json.dumps({"status": "candidate", "provider": "fake"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Candidate runtime config fields missing"):
        validate_gold_pilot_preflight(
            **{
                **kwargs,
                "candidate_runtime_config": incomplete_config,
                "candidate_runtime_config_sha256": sha256_file(incomplete_config),
            }
        )
    preflight = validate_gold_pilot_preflight(**kwargs)
    seen: list[dict] = []

    def fake_processor(item: dict) -> dict:
        seen.append(item)
        assert "final_label" not in item
        assert "gold_label" not in item
        return {
            "prediction": "borderline",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "estimated_cost_microunits": 3,
            },
        }

    result = GoldPilotRunner(Database(gold)).run(
        run_uid="pilot-fixture",
        preflight=preflight,
        budget=budget,
        processor=fake_processor,
    )
    assert result["status"] == "completed"
    assert seen
    assert GoldPilotRunner(Database(gold)).reconcile_usage("pilot-fixture")["calls"] == len(
        seen
    )
    candidate = {
        "status": "candidate",
        "release_status": "NOT RELEASED",
        "holdout_metrics": None,
    }
    verify_candidate_isolation(candidate)
    with pytest.raises(ValueError, match="cannot be exposed as a released"):
        verify_candidate_isolation({**candidate, "release_status": "released"})


def test_prompt_injection_offline_corpus_covers_required_boundaries() -> None:
    names = {case.case_id for case in D1_INJECTION_CASES}
    assert {
        "title_ignore_instruction",
        "abstract_json_example",
        "fake_system_prompt",
        "markdown_code_block",
        "html_comment",
        "xml_tags",
        "suspicious_doi_url",
        "very_long_abstract",
        "unicode_controls",
        "repeated_instruction",
        "empty_abstract",
        "instruction_only_abstract",
    } <= names
    assert canonical_hash(sorted(names))

    def stable_processor(item, *, allow_strong):
        return {
            "fast_decision": "borderline",
            "strong_decision": "borderline" if allow_strong else None,
            "composite_decision": "borderline",
            "confidence": 0.7,
        }

    report = run_prompt_injection_red_team(
        stable_processor,
        {"item_id": "d1-red-team", "original_title": "Title", "original_abstract": ""},
        cases=D1_INJECTION_CASES,
    )
    assert report["case_count"] == len(D1_INJECTION_CASES)
    assert report["critical_cases_passed"] is True


def test_gold_migration_failure_rolls_back_all_new_tables(tmp_path: Path) -> None:
    database = tmp_path / "rollback.db"
    migration = (
        Path(__file__).parents[1] / "migrations" / "0009_add_gold_governance.sql"
    ).read_text(encoding="utf-8")
    broken = migration.replace(
        "CREATE TABLE IF NOT EXISTS gold_pilot_results_v2",
        "THIS IS NOT VALID SQL;\nCREATE TABLE IF NOT EXISTS gold_pilot_results_v2",
    )
    connection = sqlite3.connect(database)
    try:
        with pytest.raises(sqlite3.Error):
            connection.executescript(broken)
    finally:
        connection.close()
    with sqlite3.connect(database) as check:
        names = {
            row[0]
            for row in check.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'gold_%'"
            )
        }
    assert not names
