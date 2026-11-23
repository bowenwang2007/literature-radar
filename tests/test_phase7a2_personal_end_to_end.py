from __future__ import annotations

import hashlib
import json
import sqlite3
import statistics
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from eduecon_radar.db import Database, utcnow
from eduecon_radar.schemas import JournalConfig, PaperMetadata, PersonalStrongDecision
from eduecon_radar.services.deepseek import DeepSeekError, DeepSeekResponse
from eduecon_radar.services.migrations import (
    apply_p0_hardening_migration,
    apply_translation_migration,
    apply_workflow_migration,
)
from eduecon_radar.services.personal_mode import (
    JournalSyncResult,
    PersonalReviewDraft,
    PersonalWorkflowService,
    create_personal_task,
    enrich_personal_task_abstracts,
    export_personal_mentor_csv,
    export_personal_mentor_markdown,
    export_personal_task_csv,
    export_personal_task_markdown,
    personal_review_changes,
    personal_review_save_enabled,
    personal_review_widget_key,
    personal_snapshot_counts,
    personal_task_snapshot,
    save_and_process_manual_abstract,
    save_personal_human_decisions,
    translate_personal_task_titles,
)
from eduecon_radar.services.profile_registry import default_profile_registry
from eduecon_radar.ui.views import personal_mode as personal_view

ROOT = Path(__file__).resolve().parents[1]


def _database(tmp_path: Path, name: str = "phase7a2.db") -> Database:
    database = Database(tmp_path / name)
    database.init_schema()
    apply_translation_migration(
        database, ROOT / "migrations/0001_add_paper_translations.sql"
    )
    apply_workflow_migration(
        database, ROOT / "migrations/0002_add_workflow_batches.sql"
    )
    apply_p0_hardening_migration(
        database, ROOT / "migrations/0007_add_runtime_leases_and_task_reviews.sql"
    )
    return database


def _papers(database: Database, count: int = 12) -> list[int]:
    journal_id = database.upsert_journal(
        JournalConfig(
            name="Economics of Education Review",
            short_name="EER",
            issn="0272-7757",
            eissn="1873-7382",
        )
    )
    ids = []
    existing = {8, 9, 11, 12}
    for number in range(1, count + 1):
        abstract = (
            f"Paper {number} studies education outcomes and economic mechanisms."
            if number in existing or count > 12
            else None
        )
        ids.append(
            database.upsert_paper(
                PaperMetadata(
                    doi=f"10.7000/personal-{number}",
                    title=f"Personal paper {number}",
                    abstract=abstract,
                    authors=[f"Author {number}"],
                    issue_date=date(2026, 7, min(number, 28)),
                    source_metadata={"published": {"date-parts": [["2026"]]}},
                ),
                journal_id,
            )[0]
        )
    return ids


def _task(database: Database, paper_ids: list[int]) -> int:
    registry = default_profile_registry()
    journal = registry.by_key("economics_of_education_review")
    result = JournalSyncResult(
        journal_key=journal.journal_key,
        canonical_name=journal.canonical_name,
        found=len(paper_ids),
        inserted=len(paper_ids),
        paper_ids=paper_ids,
    )
    return create_personal_task(
        database,
        journal_results=(result,),
        selected_journals=(journal,),
        since=date(2026, 7, 1),
        until=date(2026, 7, 31),
        per_journal_limit=20,
        max_task_papers=20,
    ).task_id


class ScenarioEnrichment:
    def __init__(self, database: Database, paper_ids: list[int]) -> None:
        self.database = database
        self.paper_ids = paper_ids
        self.calls: dict[int, int] = {}

    def enrich_papers(
        self, paper_ids, *, workflow_batch_id=None, force_refresh=False
    ):
        for paper_id in paper_ids:
            attempt = self.calls.get(paper_id, 0) + 1
            self.calls[paper_id] = attempt
            index = self.paper_ids.index(paper_id) + 1
            abstract = None
            retryable = False
            reason = "no_match"
            attempts = [{"provider": "crossref", "state": "not_found"}]
            if index == 1:
                abstract = "DOI lookup found education and economic evidence."
                reason = "doi_success"
            elif index == 2:
                abstract = "Second provider found an education economics abstract."
                reason = "provider_fallback_success"
                attempts = [
                    {"provider": "crossref", "state": "error"},
                    {"provider": "openalex", "state": "success"},
                ]
            elif index == 3:
                abstract = "Verified title fallback found the original abstract."
                reason = "title_fallback_success"
                attempts = [{"provider": "openalex", "state": "title_match"}]
            elif index == 5:
                reason = "matched_record_has_no_abstract"
                attempts = [{"provider": "crossref", "state": "matched_no_abstract"}]
            elif index in {6, 7} and attempt == 1:
                retryable = True
                reason = "rate_limited" if index == 6 else "timeout"
                attempts = [
                    {
                        "provider": "semantic_scholar",
                        "state": reason,
                        "next_retry_at": "2026-07-31T12:00:00+00:00",
                    }
                ]
            elif index in {6, 7}:
                abstract = "Retry succeeded with a verified original abstract."
                reason = "retry_success"
            metadata = {
                "abstract_enrichment": {
                    "final_reason": reason,
                    "retryable": retryable,
                    "execution_state": "pending" if retryable else "completed",
                    "next_retry_at": (
                        "2026-07-31T12:00:00+00:00" if retryable else None
                    ),
                    "attempts": attempts,
                }
            }
            with self.database.connect() as connection:
                if abstract:
                    connection.execute(
                        "UPDATE papers SET abstract=?,source_metadata_json=? WHERE id=?",
                        (abstract, json.dumps(metadata), paper_id),
                    )
                else:
                    connection.execute(
                        "UPDATE papers SET source_metadata_json=? WHERE id=?",
                        (json.dumps(metadata), paper_id),
                    )
        with self.database.connect() as connection:
            available = sum(
                bool(
                    connection.execute(
                        "SELECT abstract FROM papers WHERE id=?", (paper_id,)
                    ).fetchone()[0]
                )
                for paper_id in paper_ids
            )
        return SimpleNamespace(
            total=len(paper_ids),
            enriched=available,
            failed=len(paper_ids) - available,
            provider_queries=len(paper_ids),
            cache_hits=0,
            by_source={},
            by_result_state={},
            papers_without_abstract=len(paper_ids) - available,
        )


class ScenarioTranslation:
    def __init__(self, database: Database, unexpected_paper_id: int) -> None:
        self.database = database
        self.unexpected_paper_id = unexpected_paper_id
        self.failed_once = False
        self.title_calls: list[int] = []
        self.abstract_calls: list[int] = []

    def run(self, parameters):
        failed = 0
        for paper_id in parameters.paper_ids:
            if (
                paper_id == self.unexpected_paper_id
                and parameters.translate_titles
                and not self.failed_once
            ):
                self.failed_once = True
                failed += 1
                continue
            with self.database.connect() as connection:
                paper = connection.execute(
                    "SELECT title,abstract FROM papers WHERE id=?", (paper_id,)
                ).fetchone()
                existing = connection.execute(
                    "SELECT * FROM paper_translations WHERE paper_id=?", (paper_id,)
                ).fetchone()
                chinese_title = existing["chinese_title"] if existing else None
                chinese_abstract = existing["chinese_abstract"] if existing else None
                if parameters.translate_titles and not chinese_title:
                    chinese_title = f"中译：{paper['title']}"
                    self.title_calls.append(paper_id)
                if parameters.translate_abstracts and paper["abstract"] and not chinese_abstract:
                    chinese_abstract = f"中译：{paper['abstract']}"
                    self.abstract_calls.append(paper_id)
                connection.execute(
                    """INSERT INTO paper_translations
                    (paper_id,chinese_title,chinese_abstract,title_translation_status,
                     abstract_translation_status,title_translation_source,
                     abstract_translation_source,updated_at)
                    VALUES (?,?,?,?,?,?,?,?)
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
                        chinese_title,
                        chinese_abstract,
                        "ai_generated" if chinese_title else "pending",
                        "ai_generated" if chinese_abstract else "unavailable",
                        "ai_generated" if chinese_title else None,
                        "ai_generated" if chinese_abstract else None,
                        utcnow(),
                    ),
                )
        return SimpleNamespace(
            failed=failed,
            failure_reasons=(
                ["isolated fixture translation error"] if failed else []
            ),
            ai_calls=len(parameters.paper_ids) - failed,
            input_tokens=10,
            output_tokens=5,
        )


class ScenarioClassifier:
    model_name = "fixture-strong"

    def __init__(self, paper_ids: list[int]) -> None:
        self.paper_ids = paper_ids
        self.calls: dict[int, int] = {}

    def classify(self, paper):
        paper_id = int(paper["id"])
        index = self.paper_ids.index(paper_id) + 1
        attempt = self.calls.get(paper_id, 0) + 1
        self.calls[paper_id] = attempt
        if index == 11 and attempt == 1:
            raise DeepSeekError(
                "invalid structured output",
                raw_response='{"decision":"maybe"}',
                validation_error="schema_validation_failed",
            )
        confidence = "low" if index == 9 else "high"
        decision = "uncertain" if index == 9 else "relevant"
        return DeepSeekResponse(
            data=PersonalStrongDecision(
                decision=decision,
                education_centrality="central",
                economic_relevance="substantive",
                confidence=confidence,
                evidence=["education"],
                reason="Fixture decision based on the supplied original abstract.",
            ),
            model_name=self.model_name,
            input_tokens=20,
            output_tokens=10,
        )


def _workflow(database: Database, paper_ids: list[int]):
    enrichment = ScenarioEnrichment(database, paper_ids)
    translation = ScenarioTranslation(database, paper_ids[11])
    classifier = ScenarioClassifier(paper_ids)
    return (
        PersonalWorkflowService(
            database,
            enrichment_service=enrichment,
            translation_service=translation,
            classifier=classifier,
        ),
        enrichment,
        translation,
        classifier,
    )


def test_twelve_paper_personal_flow_survives_failures_refresh_review_and_export(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    paper_ids = _papers(database)
    task_id = _task(database, paper_ids)
    workflow, enrichment, translation, classifier = _workflow(database, paper_ids)

    translate_personal_task_titles(database, task_id, translation)
    enrich_personal_task_abstracts(database, task_id, enrichment)
    enrich_personal_task_abstracts(
        database,
        task_id,
        enrichment,
        paper_ids=(paper_ids[5], paper_ids[6]),
    )

    for _ in range(12):
        result = workflow.process_next_batch(task_id, batch_size=3)
        if not result.remaining:
            break
    first_snapshot = personal_task_snapshot(database, task_id)
    status_by_id = {
        int(paper["id"]): paper["processing"]["status"]
        for paper in first_snapshot.papers
    }
    for paper_id in (paper_ids[3], paper_ids[4], paper_ids[9]):
        retrieval = next(
            paper["processing"]["abstract_retrieval"]
            for paper in first_snapshot.papers
            if int(paper["id"]) == paper_id
        )
        assert retrieval["status"] == "needs_manual_abstract"
    assert enrichment.calls[paper_ids[5]] == 2
    assert enrichment.calls[paper_ids[6]] == 2
    assert classifier.calls[paper_ids[10]] == 2
    assert status_by_id[paper_ids[11]] == "classified"

    for paper_id in (paper_ids[3], paper_ids[4], paper_ids[9]):
        outcome = save_and_process_manual_abstract(
            workflow,
            task_id,
            paper_id,
            "Manually supplied original abstract about education economics.",
            "https://publisher.example/manual",
        )
        assert outcome["status"] == "classified"

    completed = personal_task_snapshot(database, task_id)
    assert completed.counts["retrieved"] == 12
    assert completed.counts["english_abstracts"] == 12
    assert completed.counts["chinese_abstracts"] == 12
    assert completed.counts["automatic_judgments"] == 12
    assert completed.counts["review_pending"] == 12
    calls_before_refresh = (
        dict(enrichment.calls),
        list(translation.title_calls),
        list(translation.abstract_calls),
        dict(classifier.calls),
    )
    refreshed_workflow = PersonalWorkflowService(
        database,
        enrichment_service=enrichment,
        translation_service=translation,
        classifier=classifier,
    )
    assert not refreshed_workflow.process_next_batch(task_id, batch_size=3).selected
    assert calls_before_refresh == (
        enrichment.calls,
        translation.title_calls,
        translation.abstract_calls,
        classifier.calls,
    )

    drafts = {
        paper_id: PersonalReviewDraft(
            decision="related" if offset < 6 else "irrelevant",
            note=f"Human note {offset}",
        )
        for offset, paper_id in enumerate(paper_ids[:10])
    }
    changes = personal_review_changes(completed.papers, drafts)
    assert len(changes) == 10
    assert save_personal_human_decisions(
        database, task_id, changes, snapshot=completed
    ) == tuple(paper_ids[:10])
    reviewed = personal_task_snapshot(database, task_id)
    assert reviewed.counts["human_decisions"] == 10
    assert reviewed.counts["completed"] == 10
    before_noop = hashlib.sha256(database.path.read_bytes()).hexdigest()
    assert not save_personal_human_decisions(
        database, task_id, drafts, snapshot=reviewed
    )
    assert hashlib.sha256(database.path.read_bytes()).hexdigest() == before_noop

    mentor_csv = export_personal_mentor_csv(reviewed.papers)
    mentor_md = export_personal_mentor_markdown(reviewed.papers)
    personal_csv = export_personal_task_csv(
        database, task_id, papers=reviewed.papers
    )
    personal_md = export_personal_task_markdown(
        database, task_id, papers=reviewed.papers
    )
    assert "automatic_decision" not in mentor_csv
    assert "AI reason" not in mentor_md
    assert "original_abstract" in mentor_csv
    assert "automatic_decision" in personal_csv
    assert "Automatic decision" in personal_md
    assert "NOT RELEASED" not in personal_md


def test_strong_schema_normalizes_enums_and_rejects_malformed_payloads() -> None:
    value = PersonalStrongDecision.model_validate(
        {
            "decision": " Relevant ",
            "education_centrality": " CENTRAL ",
            "economic_relevance": " Substantive ",
            "confidence": " HIGH ",
            "evidence": [" direct evidence "],
            "reason": " valid reason ",
        }
    )
    assert value.decision == "relevant"
    assert value.reason == "valid reason"
    with pytest.raises(ValidationError):
        PersonalStrongDecision.model_validate(
            {
                "decision": "relevant",
                "education_centrality": "central",
                "economic_relevance": "substantive",
                "confidence": "high",
                "evidence": "not-a-json-array",
                "reason": "x",
            }
        )
    with pytest.raises(ValidationError):
        PersonalStrongDecision.model_validate(
            {
                "decision": "relevant",
                "education_centrality": "central",
                "economic_relevance": "substantive",
                "confidence": "high",
                "evidence": [],
                "reason": " ",
                "unexpected": True,
            }
        )


def test_review_widget_keys_and_confirmation_gate_are_stable() -> None:
    assert personal_review_widget_key(4, 9, "decision") == (
        "personal_review_4_9_decision"
    )
    assert personal_review_widget_key(4, 9, "decision") != (
        personal_review_widget_key(4, 10, "decision")
    )
    assert not personal_review_save_enabled(1, confirmed=False, saving=False)
    assert not personal_review_save_enabled(0, confirmed=True, saving=False)
    assert not personal_review_save_enabled(1, confirmed=True, saving=True)
    assert personal_review_save_enabled(1, confirmed=True, saving=False)


def test_snapshot_and_ten_review_save_have_fixed_query_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = _database(tmp_path)
    paper_ids = _papers(database, count=20)
    task_id = _task(database, paper_ids)
    statements: list[str] = []
    original_connect = database.connect

    @contextmanager
    def counted_connect():
        with original_connect() as connection:
            connection.set_trace_callback(statements.append)
            yield connection

    monkeypatch.setattr(database, "connect", counted_connect)
    timings = []
    for _ in range(12):
        statements.clear()
        started = time.perf_counter()
        snapshot = personal_task_snapshot(database, task_id)
        timings.append(time.perf_counter() - started)
        selects = [item for item in statements if item.lstrip().upper().startswith("SELECT")]
        assert len(selects) == 2
    assert statistics.median(timings) < 0.5
    assert sorted(timings)[-2] < 1.0

    drafts = {
        paper_id: PersonalReviewDraft(decision="related", note=f"note {paper_id}")
        for paper_id in paper_ids[:10]
    }
    statements.clear()
    saved = save_personal_human_decisions(
        database, task_id, drafts, snapshot=snapshot
    )
    assert len(saved) == 10
    selects = [item for item in statements if item.lstrip().upper().startswith("SELECT")]
    writes = [
        item
        for item in statements
        if item.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE"))
    ]
    assert len(selects) <= 3
    assert len(writes) == 2


class _ReadOnlyUI:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __getattr__(self, name):
        if name in {
            "warning",
            "info",
            "title",
            "caption",
            "header",
            "subheader",
            "metric",
            "write",
            "markdown",
            "json",
            "dataframe",
            "success",
            "error",
        }:
            return lambda *args, **kwargs: None
        if name == "form_submit_button":
            return self.button
        if name in {"form", "container", "expander", "spinner"}:
            return lambda *args, **kwargs: self
        raise AttributeError(name)

    def columns(self, count, **_kwargs):
        size = count if isinstance(count, int) else len(count)
        return [self for _ in range(size)]

    def tabs(self, *args, **kwargs):
        labels = args[0] if args else []
        return [self for _ in range(len(labels))]

    def segmented_control(self, _label, options, *, default=None, **_kwargs):
        return default or options[0]

    def toggle(self, *_args, **_kwargs):
        return False

    def selectbox(self, _label, options, **_kwargs):
        return options[0]

    def multiselect(self, _label, _options, *, default=(), **_kwargs):
        return default

    def text_input(self, _label, *, key=None, **_kwargs):
        return str(self.session_state.get(key, ""))

    def text_area(self, _label, *, key=None, **_kwargs):
        return str(self.session_state.get(key, ""))

    def date_input(self, _label, *, value, **_kwargs):
        return value

    def number_input(self, _label, *, value, **_kwargs):
        return value

    def radio(self, _label, options, *, index=0, key=None, **_kwargs):
        value = self.session_state.get(key, options[index])
        self.session_state[key] = value
        return value

    def checkbox(self, _label, *, key=None, **_kwargs):
        return bool(self.session_state.get(key, False))

    def button(self, *_args, **_kwargs):
        return False

    def download_button(self, *_args, **_kwargs):
        return False

    def rerun(self):
        raise AssertionError("Read-only render must not rerun")


def test_twenty_paper_personal_render_is_database_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = _database(tmp_path)
    task_id = _task(database, _papers(database, count=20))
    fake = _ReadOnlyUI()
    fake.session_state["personal_current_task_id"] = task_id
    monkeypatch.setattr(personal_view, "st", fake)
    settings = SimpleNamespace(
        journal_registry_path=ROOT / "config" / "journals" / "catalog.json",
        journals_path=ROOT / "journals.yaml",
        max_per_journal_limit=20,
        project_root=ROOT,
    )
    before = hashlib.sha256(database.path.read_bytes()).hexdigest()
    personal_view.render(database, settings)
    assert hashlib.sha256(database.path.read_bytes()).hexdigest() == before


def test_stats_invariants_for_mixed_snapshot() -> None:
    papers = [
        {
            "id": 1,
            "abstract": None,
            "chinese_abstract": None,
            "human_decision": "undecided",
            "processing": {"status": "needs_retry"},
        },
        {
            "id": 2,
            "abstract": "A",
            "chinese_abstract": "中",
            "human_decision": "related",
            "processing": {
                "status": "classified",
                "classification": {"decision": "relevant"},
            },
        },
    ]
    counts = personal_snapshot_counts(papers)
    assert counts == {
        "retrieved": 2,
        "pending_processing": 1,
        "missing_abstract": 1,
        "needs_retry": 1,
        "review_pending": 0,
        "completed": 1,
        "english_abstracts": 1,
        "chinese_abstracts": 1,
        "automatic_judgments": 1,
        "human_decisions": 1,
    }


def test_production_database_path_is_never_used(tmp_path: Path) -> None:
    database = _database(tmp_path)
    assert database.path.parent == tmp_path
    assert database.path.name == "phase7a2.db"
    assert Path("data/eduecon_radar.db").resolve() != database.path.resolve()
    assert sqlite3.connect(database.path).execute("PRAGMA integrity_check").fetchone()[0] == "ok"
