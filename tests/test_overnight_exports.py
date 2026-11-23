from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from eduecon_radar.db import Database
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.migrations import (
    apply_overnight_migration,
    apply_p0_hardening_migration,
    apply_profile_migration,
    apply_translation_migration,
    apply_workflow_migration,
)
from eduecon_radar.services.overnight_exports import (
    collect_run_papers,
    overnight_stats,
    write_overnight_exports,
)
from eduecon_radar.services.profile_registry import default_profile_registry

ROOT = Path(__file__).resolve().parents[1]
ABSTRACT = (
    "This paper estimates how school finance reform affects student achievement "
    "and economic outcomes using administrative data and a difference-in-differences design."
)
QUOTE = "school finance reform affects student achievement"


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "exports.db")
    database.init_schema()
    apply_translation_migration(database, ROOT / "migrations/0001_add_paper_translations.sql")
    apply_workflow_migration(database, ROOT / "migrations/0002_add_workflow_batches.sql")
    apply_p0_hardening_migration(
        database, ROOT / "migrations/0007_add_runtime_leases_and_task_reviews.sql"
    )
    apply_overnight_migration(
        database, ROOT / "migrations/0010_add_overnight_pipeline.sql"
    )
    registry = default_profile_registry()
    journal = registry.by_key("economics_of_education_review")
    journal_id = database.upsert_journal(journal.as_journal_config())
    ids = []
    for index, (abstract, with_abstract) in enumerate(
        ((ABSTRACT, True), (None, False)), start=1
    ):
        paper_id, _ = database.upsert_paper(
            PaperMetadata(
                doi=f"10.1000/export-{index}",
                title=f"Education finance paper {index}",
                abstract=abstract,
                authors=["Researcher"],
                issue_date=date(2026, 7, index),
                source_metadata={
                    "DOI": f"10.1000/export-{index}",
                    "abstract_source": "openalex" if with_abstract else None,
                    "abstract_enrichment": {
                        "status": "completed" if with_abstract else "failed",
                        "final_reason": None if with_abstract else "strategy_exhausted_no_match",
                        "attempts": (
                            [
                                {
                                    "provider": "openalex",
                                    "query_mode": "doi",
                                    "status": "success",
                                    "abstract_text": ABSTRACT if with_abstract else None,
                                }
                            ]
                            if with_abstract
                            else [
                                {
                                    "provider": "crossref",
                                    "query_mode": "doi",
                                    "status": "not_found",
                                    "error_category": "not_found",
                                }
                            ]
                        ),
                    },
                },
            ),
            journal_id,
        )
        ids.append(paper_id)
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO workflow_batches
            (id,name,source_type,status,source_config_json,created_at,updated_at)
            VALUES (1,'overnight-fixture','sync','ready','{}',
                    '2026-07-31T00:00:00+00:00','2026-07-31T00:00:00+00:00')"""
        )
        connection.execute(
            """INSERT INTO paper_translations
            (paper_id,chinese_title,chinese_abstract,title_translation_status,
             abstract_translation_status,title_translation_source,
             abstract_translation_source,model_name,prompt_version,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                ids[0],
                "学校财政改革论文一",
                "学校财政改革论文一的中文摘要。",
                "ai_generated",
                "ai_generated",
                "ai_generated",
                "ai_generated",
                "fixture",
                "v1",
                "2026-07-31T00:00:00+00:00",
            ),
        )
        connection.execute(
            """INSERT INTO paper_screenings
            (paper_id,input_hash,model_decision,education_centrality,
             economic_relevance,evidence_sufficiency,confidence,evidence_json,
             reason,topic_tags_json,review_priority,model_name,prompt_version,
             ontology_version,screened_at,raw_response_json,parser_version,
             evidence_validation_version,validation_status,validation_errors_json,
             review_decision,final_model_decision,disagreement_reason,
             validated_evidence_json,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ids[0],
                "hash-1",
                "relevant",
                "central",
                "substantive",
                "sufficient",
                "high",
                json.dumps([{"quote": QUOTE, "supports": "final_decision"}]),
                "clear evidence",
                json.dumps(["finance"]),
                "high",
                "fixture",
                "v1",
                "ontology-v1",
                "2026-07-31T00:00:00+00:00",
                "{}",
                "parser-v1",
                "evidence-v1",
                "valid",
                "[]",
                "not_required",
                "relevant",
                "",
                json.dumps([{"quote": QUOTE, "supports": "final_decision"}]),
                "2026-07-31T00:00:00+00:00",
                "2026-07-31T00:00:00+00:00",
            ),
        )
        connection.execute(
            """INSERT INTO workflow_task_reviews
            (task_id,paper_id,review_status,final_decision,reviewer_notes,
             source,provenance_json,created_at,updated_at)
            VALUES (1,?,?,?,?,?,?,?,?)""",
            (
                ids[1],
                "pending",
                None,
                "",
                "v05_human",
                "{}",
                "2026-07-31T00:00:00+00:00",
                "2026-07-31T00:00:00+00:00",
            ),
        )
    apply_profile_migration(
        database, ROOT / "migrations/0013_add_profile_scoped_assessment.sql"
    )
    return database


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_overnight_exports_write_seven_scoped_files(tmp_path) -> None:
    database = _database(tmp_path)
    with database.connect() as connection:
        ids = tuple(
            int(row[0])
            for row in connection.execute("SELECT id FROM papers ORDER BY id").fetchall()
        )
    export_dir = tmp_path / "out"
    paths = write_overnight_exports(
        database, ids, task_id=1, export_dir=export_dir
    )
    expected = [
        paths.mentor_csv,
        paths.research_csv,
        paths.digest_md,
        paths.reading_queue_md,
        paths.manual_abstract_queue_csv,
        paths.provider_audit_csv,
        paths.screening_audit_csv,
    ]
    for path in expected:
        assert path.is_file(), path

    mentor = _read_csv(paths.mentor_csv)
    assert len(mentor) == 2
    assert set(mentor[0]) >= {
        "paper_id", "中文标题", "英文标题", "作者", "期刊", "年份", "DOI",
        "中文摘要", "自动判断", "判断依据", "人工决定", "判断理由",
    }
    research = _read_csv(paths.research_csv)
    assert set(research[0]) >= {
        "英文摘要", "摘要来源", "Provider轨迹", "education_centrality",
        "economic_relevance", "evidence_sufficiency", "confidence",
        "evidence", "topic_tags", "model", "prompt_version", "用户备注",
    }
    assert research[0]["education_centrality"] == "central"

    manual = _read_csv(paths.manual_abstract_queue_csv)
    assert len(manual) == 1
    assert manual[0]["DOI链接"] == "https://doi.org/10.1000/export-2"

    provider = _read_csv(paths.provider_audit_csv)
    assert len(provider) == 2
    statuses = {row["status"] for row in provider}
    assert statuses == {"success", "not_found"}

    screening = _read_csv(paths.screening_audit_csv)
    assert len(screening) == 1
    assert screening[0]["final_model_decision"] == "relevant"
    assert screening[0]["model_decision"] == "relevant"

    digest = paths.digest_md.read_text(encoding="utf-8")
    assert "## 自动筛选" in digest
    assert "学校财政改革论文一" in digest
    reading = paths.reading_queue_md.read_text(encoding="utf-8")
    assert "立即精读（人工相关）" in reading
    assert "待补摘要" in reading


def test_export_declares_judgment_basis_and_never_overstates_it(tmp_path) -> None:
    """The export must say how much evidence a decision had.

    Production shipped exports that listed a relevance decision while
    ``paper_screenings.raw_response_json`` carried no ``screening_basis`` for
    most rows, so 172 recovered papers were mislabelled as "basis not recorded".
    The authoritative ``paper_processing_status.classification_json`` must be
    consulted, and an unrecorded basis must never be reported as a full
    judgment.
    """
    from eduecon_radar.services.overnight_exports import _judgment_kind

    # Authoritative classification wins over a raw payload that lacks the key.
    assert (
        _judgment_kind(
            {"final_model_decision": "relevant", "raw_response_json": "{}"},
            {"decision": "relevant", "screening_basis": "title_and_abstract"},
        )
        == "完整判断"
    )
    assert (
        _judgment_kind(
            {"final_model_decision": "irrelevant", "raw_response_json": "{}"},
            {"decision": "irrelevant", "screening_basis": "title_only"},
        )
        == "标题初判"
    )
    assert (
        _judgment_kind(
            {"final_model_decision": "irrelevant", "raw_response_json": "{}"},
            {"decision": "irrelevant", "screening_basis": "deterministic_metadata_rule"},
        )
        == "非研究内容判断"
    )
    # No basis anywhere -> explicit historical label, never a complete judgment.
    assert (
        _judgment_kind(
            {"final_model_decision": "relevant", "raw_response_json": "{}"},
            {"decision": "relevant"},
        )
        == "历史判断（依据未记录）"
    )
    # Fallback still works when only the screening record carries the basis.
    assert (
        _judgment_kind(
            {
                "final_model_decision": "irrelevant",
                "raw_response_json": json.dumps({"screening_basis": "title_only"}),
            },
            None,
        )
        == "标题初判"
    )


def test_overnight_stats_and_collect_run_papers(tmp_path) -> None:
    database = _database(tmp_path)
    with database.connect() as connection:
        ids = tuple(
            int(row[0])
            for row in connection.execute("SELECT id FROM papers ORDER BY id").fetchall()
        )
    papers = collect_run_papers(database, ids, task_id=1)
    assert len(papers) == 2
    stats = overnight_stats(papers)
    assert stats["with_abstract"] == 1
    assert stats["chinese_title"] == 1
    assert stats["screened"] == 1
    assert stats["relevant"] == 1
    assert stats["provider_success"] == 1
    assert stats["provider_attempts"] == 2
