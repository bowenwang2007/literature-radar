from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from eduecon_radar.db import Database, utcnow
from eduecon_radar.domain.semantic_storage import apply_domain_semantic_migration
from eduecon_radar.domain.storage import apply_domain_evidence_migration
from eduecon_radar.journals import load_journals
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.exporting import export_mentor_csv_text, mentor_records
from eduecon_radar.services.migrations import (
    apply_data_integrity_migration,
    apply_p0_hardening_migration,
    apply_translation_migration,
    apply_workflow_migration,
)
from eduecon_radar.services.prototype_workflow import (
    FinalReviewInput,
    LiteratureSearchScope,
    V05LiteratureOrchestrator,
    _decision_detail,
    create_prototype_literature_task,
    list_task_papers,
    prototype_task_state,
    save_final_reviews,
    save_manual_abstract,
    task_processing_counts,
)
from eduecon_radar.services.workflow import batch_paper_ids
from eduecon_radar.services.workflow_resume import WorkflowCheckpointStore

ROOT = Path(__file__).resolve().parents[1]


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "v05.db")
    database.init_schema()
    apply_translation_migration(database, ROOT / "migrations/0001_add_paper_translations.sql")
    apply_workflow_migration(database, ROOT / "migrations/0002_add_workflow_batches.sql")
    apply_domain_evidence_migration(
        database, ROOT / "migrations/0003_add_domain_evidence_runs.sql"
    )
    apply_domain_semantic_migration(
        database, ROOT / "migrations/0004_add_domain_semantic_runs.sql"
    )
    apply_p0_hardening_migration(
        database, ROOT / "migrations/0007_add_runtime_leases_and_task_reviews.sql"
    )
    apply_data_integrity_migration(
        database,
        ROOT / "migrations/0008_add_metadata_provenance_and_ai_budgets.sql",
    )
    return database


class FakeSyncService:
    def __init__(self, database: Database, calls: list[tuple]) -> None:
        self.database = database
        self.calls = calls

    def sync(self, journal_file, *, since, until, limit):
        self.calls.append(("sync", since, until, limit))
        journal_id = self.database.upsert_journal(journal_file.journals[0])
        papers = (
            PaperMetadata(
                doi="10.1000/include",
                title="Include paper",
                abstract="Original include abstract.",
                authors=["A"],
                issue_date=date(2026, 1, 5),
            ),
            PaperMetadata(
                doi="10.1000/recover",
                title="Recoverable exclude",
                abstract="Original recovered abstract.",
                authors=["B"],
                issue_date=date(2026, 1, 6),
            ),
            PaperMetadata(
                doi="10.1000/no-abstract",
                title="No abstract paper",
                abstract=None,
                authors=["C"],
                issue_date=date(2026, 1, 7),
            ),
            PaperMetadata(
                doi="10.1000/exclude",
                title="Ordinary exclude",
                abstract="This abstract must not be translated.",
                authors=["D"],
                issue_date=date(2026, 1, 8),
            ),
        )
        ids = [self.database.upsert_paper(item, journal_id)[0] for item in papers]
        return {
            "journals_checked": 1,
            "records_fetched": len(ids),
            "records_inserted": len(ids),
            "records_updated": 0,
            "records_failed": 0,
            "affected_paper_ids": ids,
        }


class FakeTranslationService:
    def __init__(self, database: Database, calls: list[tuple]) -> None:
        self.database = database
        self.calls = calls

    def run(self, parameters):
        failed = unavailable = 0
        for paper_id in parameters.paper_ids:
            with self.database.connect() as connection:
                row = connection.execute(
                    """SELECT p.title,p.abstract,pt.chinese_title,pt.chinese_abstract
                    FROM papers p LEFT JOIN paper_translations pt ON pt.paper_id=p.id
                    WHERE p.id=?""",
                    (paper_id,),
                ).fetchone()
                title = row["chinese_title"]
                abstract = row["chinese_abstract"]
                title_status = "ai_generated" if title else "pending"
                abstract_status = "ai_generated" if abstract else "pending"
                if parameters.translate_titles and not title:
                    self.calls.append(("title", paper_id))
                    title = f"中文：{row['title']}"
                    title_status = "ai_generated"
                if parameters.translate_abstracts:
                    if not row["abstract"]:
                        unavailable += 1
                        abstract_status = "unavailable"
                    elif not abstract:
                        self.calls.append(("abstract", paper_id))
                        abstract = f"中文：{row['abstract']}"
                        abstract_status = "ai_generated"
                connection.execute(
                    """INSERT INTO paper_translations
                    (paper_id,chinese_title,chinese_abstract,title_translation_status,
                     abstract_translation_status,title_translation_source,
                     abstract_translation_source,model_name,prompt_version,translated_at,updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(paper_id) DO UPDATE SET
                      chinese_title=excluded.chinese_title,
                      chinese_abstract=excluded.chinese_abstract,
                      title_translation_status=excluded.title_translation_status,
                      abstract_translation_status=excluded.abstract_translation_status,
                      title_translation_source=excluded.title_translation_source,
                      abstract_translation_source=excluded.abstract_translation_source,
                      updated_at=excluded.updated_at""",
                    (
                        paper_id,
                        title,
                        abstract,
                        title_status,
                        abstract_status,
                        "ai_generated" if title else None,
                        "ai_generated" if abstract else None,
                        "mock",
                        "mock",
                        utcnow(),
                        utcnow(),
                    ),
                )
        return SimpleNamespace(failed=failed, unavailable=unavailable)


class FakeEvidenceService:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls
        self.completed = {}

    def analyze_paper(self, paper_id, *, workflow_batch_id):
        if paper_id not in self.completed:
            self.calls.append(("evidence", paper_id))
            self.completed[paper_id] = SimpleNamespace(
                scoring=SimpleNamespace(rule_evidence_tier="little_evidence"),
                priority=SimpleNamespace(priority_tier="high", priority_score=80.0),
            )
        return self.completed[paper_id]


class FakeSemanticService:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls
        self.completed = {}

    def analyze_many(self, paper_ids, *, workflow_batch_id):
        assert workflow_batch_id > 0
        for index, paper_id in enumerate(paper_ids):
            if paper_id in self.completed:
                continue
            decision = "include" if index == 0 else "exclude"
            strong = index == 1
            self.calls.append(("fast", paper_id))
            if strong:
                self.calls.append(("strong", paper_id))
            self.calls.append(("composite", paper_id))
            self.completed[paper_id] = SimpleNamespace(
                fast=SimpleNamespace(
                    decision=decision,
                    information_sufficiency="medium",
                    confidence=0.9,
                    semantic_score=75.0,
                    reason=f"mock {decision}",
                ),
                strong=(
                    SimpleNamespace(
                        recommended_final_decision="exclude",
                        decision_sufficiency="medium",
                    )
                    if strong
                    else None
                ),
                escalation=SimpleNamespace(
                    conflict_types=(), escalate=strong
                ),
                composite=SimpleNamespace(conflict_types=(), uncertainty_level="medium"),
                missed_priority=SimpleNamespace(
                    missed_review_priority_tier="high",
                    provisional_score=80.0,
                    missed_review_priority_reasons=("mock-risk",),
                ),
                strong_failed=False,
            )
        return SimpleNamespace(completed=dict(self.completed), failures={})


def _prepared_task(tmp_path: Path):
    database = _database(tmp_path)
    calls: list[tuple] = []
    journals = load_journals(ROOT / "journals.yaml")
    selected = journals.model_copy(update={"journals": [journals.journals[0]]})
    task_id = create_prototype_literature_task(
        database,
        name="integration",
        scope=LiteratureSearchScope(
            (selected.journals[0].short_name,),
            date(2026, 1, 1),
            date(2026, 1, 31),
            4,
        ),
    )
    orchestrator = V05LiteratureOrchestrator(
        database=database,
        journal_file=selected,
        sync_service=FakeSyncService(database, calls),
        translation_service=FakeTranslationService(database, calls),
        evidence_service=FakeEvidenceService(calls),
        semantic_service=FakeSemanticService(calls),
    )
    return database, orchestrator, task_id, calls


def test_v05_search_to_composite_is_exact_ordered_and_resumable(tmp_path):
    database, orchestrator, task_id, calls = _prepared_task(tmp_path)

    state = orchestrator.run(task_id)
    ids = batch_paper_ids(database, task_id)

    assert state["current_step"] == "missed_recovery"
    assert len(ids) == 4
    assert state["result"]["total"] == 4
    assert {item[1] for item in calls if item[0] == "title"} == set(ids)
    for paper_id in ids:
        stages = [item[0] for item in calls if len(item) > 1 and item[1] == paper_id]
        assert stages.index("title") < stages.index("evidence") < stages.index("fast")
        assert stages[-1] == "composite"
    assert sum(item[0] == "strong" for item in calls) == 1
    no_abstract = next(
        item for item in list_task_papers(database, task_id) if item["title"] == "No abstract paper"
    )
    assert no_abstract["semantic_decision"] is None
    assert no_abstract["execution_state"] == "needs_abstract"

    completed_calls = list(calls)
    restored = prototype_task_state(Database(database.path), task_id)
    assert restored["result"]["total"] == 4
    orchestrator.run(task_id)
    assert calls == completed_calls


def test_recovery_candidate_translation_review_and_safe_export(tmp_path):
    database, orchestrator, task_id, calls = _prepared_task(tmp_path)
    orchestrator.run(task_id)
    papers = list_task_papers(database, task_id)
    recover_id = next(int(item["id"]) for item in papers if item["title"] == "Recoverable exclude")
    ordinary_exclude = next(
        int(item["id"]) for item in papers if item["title"] == "Ordinary exclude"
    )
    no_abstract = next(int(item["id"]) for item in papers if item["title"] == "No abstract paper")

    state = orchestrator.confirm_recovery(task_id, (recover_id,))
    candidates = set(state["candidate_ids"])
    abstract_calls = {item[1] for item in calls if item[0] == "abstract"}
    assert recover_id in candidates
    assert ordinary_exclude not in candidates
    assert ordinary_exclude not in abstract_calls
    assert no_abstract in candidates and no_abstract not in abstract_calls
    no_abstract_row = next(
        item for item in list_task_papers(database, task_id) if item["id"] == no_abstract
    )
    assert no_abstract_row["chinese_abstract"] is None
    assert no_abstract_row["abstract_translation_status"] == "unavailable"

    candidate_rows = [
        item for item in list_task_papers(database, task_id) if item["candidate"]
    ]
    reviews = {
        int(item["id"]): FinalReviewInput(
            keep=int(item["id"]) != no_abstract,
            chinese_title=str(item["chinese_title"] or ""),
            chinese_abstract=str(item["chinese_abstract"] or ""),
            note="human decision",
        )
        for item in candidate_rows
    }
    assert save_final_reviews(database, task_id, reviews) == {}
    assert prototype_task_state(Database(database.path), task_id)["current_step"] == "export"
    assert WorkflowCheckpointStore(database).get(task_id, "export").status == "completed"
    assert next(
        item for item in list_task_papers(database, task_id) if item["id"] == recover_id
    )["decision"] == "exclude"

    mentor = mentor_records(database, task_id=task_id)
    assert {item.id for item in mentor} == candidates - {no_abstract}
    csv_text = export_mentor_csv_text(database, task_id=task_id)
    header = csv_text.splitlines()[0]
    assert "original_abstract" in header
    assert "ai_" not in header
    assert "screening" not in header


def test_undecided_human_review_is_not_written_or_counted_complete(tmp_path):
    database, orchestrator, task_id, _ = _prepared_task(tmp_path)
    orchestrator.run(task_id)
    papers = list_task_papers(database, task_id)
    recover_id = next(int(item["id"]) for item in papers if item["title"] == "Recoverable exclude")
    orchestrator.confirm_recovery(task_id, (recover_id,))
    with database.connect() as connection:
        before = connection.execute(
            """SELECT paper_id,review_status,final_decision,reviewer_notes
            FROM workflow_task_reviews WHERE task_id=? ORDER BY paper_id""",
            (task_id,),
        ).fetchall()

    assert save_final_reviews(
        database,
        task_id,
        {recover_id: FinalReviewInput()},
    ) == {}

    with database.connect() as connection:
        after = connection.execute(
            """SELECT paper_id,review_status,final_decision,reviewer_notes
            FROM workflow_task_reviews WHERE task_id=? ORDER BY paper_id""",
            (task_id,),
        ).fetchall()
    assert [tuple(item) for item in after] == [tuple(item) for item in before]
    state = prototype_task_state(database, task_id)
    assert state["current_step"] == "final_review"
    assert recover_id not in state["final_reviewed_ids"]


def test_text_only_review_edit_is_saved_without_creating_human_decision(tmp_path):
    database, orchestrator, task_id, _ = _prepared_task(tmp_path)
    orchestrator.run(task_id)
    recover_id = next(
        int(item["id"])
        for item in list_task_papers(database, task_id)
        if item["title"] == "Recoverable exclude"
    )
    orchestrator.confirm_recovery(task_id, (recover_id,))

    assert save_final_reviews(
        database,
        task_id,
        {recover_id: FinalReviewInput(note="human correction only")},
    ) == {}

    with database.connect() as connection:
        row = connection.execute(
            """SELECT review_status,final_decision,reviewer_notes
            FROM workflow_task_reviews WHERE task_id=? AND paper_id=?""",
            (task_id, recover_id),
        ).fetchone()
    assert tuple(row) == ("pending", None, "human correction only")
    assert recover_id not in prototype_task_state(database, task_id)["final_reviewed_ids"]


def test_final_review_writes_only_real_changes_and_repeated_payload_is_noop(tmp_path):
    database, orchestrator, task_id, _ = _prepared_task(tmp_path)
    orchestrator.run(task_id)
    recover_id = next(
        int(item["id"])
        for item in list_task_papers(database, task_id)
        if item["title"] == "Recoverable exclude"
    )
    orchestrator.confirm_recovery(task_id, (recover_id,))

    assert save_final_reviews(
        database,
        task_id,
        {recover_id: FinalReviewInput(decision="include")},
    ) == {}
    with database.connect() as connection:
        included = connection.execute(
            """SELECT review_status,final_decision FROM workflow_task_reviews
            WHERE task_id=? AND paper_id=?""",
            (task_id, recover_id),
        ).fetchone()
    assert tuple(included) == ("approved", "include")

    exclude = {recover_id: FinalReviewInput(decision="exclude")}
    assert save_final_reviews(database, task_id, exclude) == {}
    with database.connect() as connection:
        excluded = connection.execute(
            """SELECT review_status,final_decision FROM workflow_task_reviews
            WHERE task_id=? AND paper_id=?""",
            (task_id, recover_id),
        ).fetchone()
    assert tuple(excluded) == ("rejected", "exclude")

    before_repeat = hashlib.sha256(database.path.read_bytes()).hexdigest()
    assert save_final_reviews(database, task_id, exclude) == {}
    assert hashlib.sha256(database.path.read_bytes()).hexdigest() == before_repeat

    assert save_final_reviews(
        database,
        task_id,
        {recover_id: FinalReviewInput(decision="undecided")},
    ) == {}
    with database.connect() as connection:
        cleared = connection.execute(
            """SELECT review_status,final_decision FROM workflow_task_reviews
            WHERE task_id=? AND paper_id=?""",
            (task_id, recover_id),
        ).fetchone()
    assert tuple(cleared) == ("pending", None)
    assert recover_id not in prototype_task_state(database, task_id)["final_reviewed_ids"]


def test_manual_abstract_reprocesses_only_one_paper_and_updates_counts(tmp_path):
    database, orchestrator, task_id, calls = _prepared_task(tmp_path)
    orchestrator.run(task_id)
    papers = list_task_papers(database, task_id)
    paper_id = next(int(item["id"]) for item in papers if item["title"] == "No abstract paper")
    before_other_calls = {
        item
        for item in calls
        if len(item) > 1 and item[1] != paper_id and item[0] in {"evidence", "fast", "strong"}
    }
    orchestrator.evidence_service.completed.pop(paper_id)
    orchestrator.semantic_service.completed.pop(paper_id)
    before_reprocess = len(calls)

    result = orchestrator.save_and_reprocess_manual_abstract(
        task_id,
        paper_id,
        "Human supplied original English abstract about schooling outcomes. " * 5,
        "https://example.test/manual-source",
    )

    assert result["paper_id"] == paper_id
    assert result["status"] in {"completed", "retryable_error"}
    after_other_calls = {
        item
        for item in calls
        if len(item) > 1 and item[1] != paper_id and item[0] in {"evidence", "fast", "strong"}
    }
    assert after_other_calls == before_other_calls
    target_stages = [
        item[0]
        for item in calls[before_reprocess:]
        if len(item) > 1 and item[1] == paper_id
    ]
    assert target_stages == ["abstract", "evidence", "fast", "composite"]
    counts = task_processing_counts(database, task_id)
    assert counts["english_abstracts"] == 4
    assert counts["chinese_abstracts"] == 1
    with database.connect() as connection:
        row = connection.execute(
            "SELECT abstract,source_metadata_json FROM papers WHERE id=?",
            (paper_id,),
        ).fetchone()
    assert row["abstract"].startswith("Human supplied original English abstract")
    assert json.loads(row["source_metadata_json"])["abstract_source"] == "manual"

    calls_before_repeat = len(calls)
    repeated = orchestrator.save_and_reprocess_manual_abstract(
        task_id,
        paper_id,
        "Human supplied original English abstract about schooling outcomes. " * 5,
        "https://example.test/manual-source",
    )
    assert repeated["status"] == "completed"
    assert "未重复运行" in repeated["skipped"]
    assert calls[calls_before_repeat:] == []


def test_manual_save_only_does_not_call_translation_or_semantic_services(tmp_path):
    database, orchestrator, task_id, calls = _prepared_task(tmp_path)
    orchestrator.run(task_id)
    paper_id = next(
        int(item["id"])
        for item in list_task_papers(database, task_id)
        if item["title"] == "No abstract paper"
    )
    before = list(calls)

    result = save_manual_abstract(
        database,
        paper_id,
        "Human supplied original English abstract for later processing. " * 5,
        "https://example.test/manual-source",
    )

    assert result["success"] is True
    assert calls == before
    with database.connect() as connection:
        row = connection.execute(
            """SELECT p.abstract,p.source_metadata_json,pt.chinese_abstract
            FROM papers p LEFT JOIN paper_translations pt ON pt.paper_id=p.id
            WHERE p.id=?""",
            (paper_id,),
        ).fetchone()
    assert row["abstract"].startswith("Human supplied original English abstract")
    assert json.loads(row["source_metadata_json"])["abstract_source"] == "manual"
    assert row["chinese_abstract"] is None


def test_semantic_execution_failure_is_not_relabelled_borderline(tmp_path):
    database, orchestrator, task_id, _ = _prepared_task(tmp_path)
    orchestrator.run(task_id)
    paper = next(
        item
        for item in list_task_papers(database, task_id)
        if item["title"] == "Include paper"
    )
    evidence = orchestrator.evidence_service.completed[int(paper["id"])]

    for failure in ("schema_validation_failed", "api_timeout"):
        detail = _decision_detail(
            database,
            int(paper["id"]),
            evidence,
            None,
            failure,
        )
        assert detail["semantic_decision"] is None
        assert detail["decision"] is None
        assert detail["execution_state"] == "retryable_error"
        assert detail["human_final_decision"] == "undecided"
