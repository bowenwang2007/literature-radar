from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from eduecon_radar.db import Database, utcnow
from eduecon_radar.schemas import (
    JournalConfig,
    PaperMetadata,
    PersonalStrongDecision,
)
from eduecon_radar.services.deepseek import DeepSeekError, DeepSeekResponse
from eduecon_radar.services.exporting import export_mentor_csv_text
from eduecon_radar.services.journal_registry import (
    ADJACENT_GROUP,
    CORE_GROUP,
    DEFAULT_PERSONAL_JOURNALS,
    make_custom_journal,
    normalize_journal_name,
)
from eduecon_radar.services.migrations import (
    apply_p0_hardening_migration,
    apply_translation_migration,
    apply_workflow_migration,
)
from eduecon_radar.services.personal_mode import (
    PersonalJournalSyncService,
    PersonalWorkflowService,
    create_personal_task,
    export_personal_task_csv,
    list_personal_task_papers,
    load_personal_state,
    personal_mode_allowed,
    save_and_process_manual_abstract,
    save_personal_human_decision,
)
from eduecon_radar.services.profile_registry import default_profile_registry
from eduecon_radar.ui.views import personal_mode as personal_view

ROOT = Path(__file__).resolve().parents[1]


def _registry():
    return default_profile_registry()


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "phase7a.db")
    database.init_schema()
    apply_translation_migration(database, ROOT / "migrations/0001_add_paper_translations.sql")
    apply_workflow_migration(database, ROOT / "migrations/0002_add_workflow_batches.sql")
    apply_p0_hardening_migration(
        database, ROOT / "migrations/0007_add_runtime_leases_and_task_reviews.sql"
    )
    return database


def _insert_paper(
    database: Database, *, doi: str, title: str, abstract: str | None
) -> int:
    journal_id = database.upsert_journal(
        JournalConfig(
            name="Economics of Education Review",
            short_name="EER",
            issn="0272-7757",
            eissn="1873-7382",
        )
    )
    return database.upsert_paper(
        PaperMetadata(
            doi=doi,
            title=title,
            abstract=abstract,
            authors=["Researcher"],
            issue_date=date(2026, 7, 1),
        ),
        journal_id,
    )[0]


class FakeCrossref:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def fetch(self, issn, since, limit, *, until=None):
        self.calls.append(issn)
        value = self.responses[issn]
        if isinstance(value, Exception):
            raise value
        return list(value)[:limit]


class FakeTranslation:
    def __init__(self, database: Database) -> None:
        self.database = database
        self.calls: list[tuple[int, bool, bool]] = []

    def run(self, parameters):
        for paper_id in parameters.paper_ids:
            self.calls.append(
                (paper_id, parameters.translate_titles, parameters.translate_abstracts)
            )
            with self.database.connect() as connection:
                paper = connection.execute(
                    "SELECT title,abstract FROM papers WHERE id=?", (paper_id,)
                ).fetchone()
                existing = connection.execute(
                    "SELECT * FROM paper_translations WHERE paper_id=?", (paper_id,)
                ).fetchone()
                title = existing["chinese_title"] if existing else None
                abstract = existing["chinese_abstract"] if existing else None
                if parameters.translate_titles and not title:
                    title = f"中译：{paper['title']}"
                if parameters.translate_abstracts and paper["abstract"] and not abstract:
                    abstract = f"中译：{paper['abstract']}"
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
                        title,
                        abstract,
                        "ai_generated" if title else "pending",
                        "ai_generated" if abstract else "unavailable",
                        "ai_generated" if title else None,
                        "ai_generated" if abstract else None,
                        utcnow(),
                    ),
                )
        return SimpleNamespace(
            failed=0,
            failure_reasons=[],
            ai_calls=len(parameters.paper_ids),
            input_tokens=10 * len(parameters.paper_ids),
            output_tokens=5 * len(parameters.paper_ids),
        )


class NoopEnrichment:
    def enrich_papers(
        self, paper_ids, *, workflow_batch_id=None, force_refresh=False
    ):
        return SimpleNamespace(failed=len(paper_ids))


class FakeClassifier:
    model_name = "fake-strong"

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[int] = []
        self.fail = fail

    def classify(self, paper):
        self.calls.append(int(paper["id"]))
        if self.fail:
            raise DeepSeekError(
                "invalid response",
                raw_response="{broken",
                validation_error="json_parse_failed",
            )
        return DeepSeekResponse(
            data=PersonalStrongDecision(
                decision="relevant",
                education_centrality="central",
                economic_relevance="substantive",
                confidence="high",
                evidence=["education outcomes"],
                reason="Direct education and economic evidence.",
            ),
            model_name=self.model_name,
            input_tokens=20,
            output_tokens=10,
        )


def _task_for(database: Database, paper_ids: list[int]) -> int:
    registry = _registry()
    journal = registry.by_key("economics_of_education_review")
    from eduecon_radar.services.personal_mode import JournalSyncResult

    result = JournalSyncResult(
        journal.journal_key,
        journal.canonical_name,
        found=len(paper_ids),
        inserted=len(paper_ids),
        with_abstract=len(paper_ids),
        paper_ids=paper_ids,
    )
    return create_personal_task(
        database,
        journal_results=(result,),
        selected_journals=(journal,),
        since=date(2026, 7, 1),
        until=date(2026, 7, 31),
        per_journal_limit=5,
    ).task_id


def test_registry_version_groups_and_core_defaults() -> None:
    """The default profile's core tier holds the education journals it names.

    The catalog has grown into a global library, so the core tier is no longer
    exactly three journals; what must stay true is that the education-economics
    journals are in it, that the default selection is its leading group, and
    that the tier carries no discipline-specific wording that would misdescribe
    another research direction.
    """
    registry = _registry()
    assert registry.version
    core = tuple(item.journal_key for item in registry.expand_group(CORE_GROUP))
    for key in DEFAULT_PERSONAL_JOURNALS:
        assert key in core, key
    # The default selection is still the education-economics journals, first.
    assert core[: len(DEFAULT_PERSONAL_JOURNALS)] == DEFAULT_PERSONAL_JOURNALS
    assert len(registry.expand_group(ADJACENT_GROUP)) >= 5
    assert "专刊" not in registry.groups[ADJACENT_GROUP].display_name


@pytest.mark.parametrize(
    ("query", "journal_key"),
    [
        ("EER", "economics_of_education_review"),
        ("0272-7757", "economics_of_education_review"),
        ("1873-7382", "economics_of_education_review"),
        ("0964-5292", "education_economics"),
        ("1469-5782", "education_economics"),
        ("1557-3060", "education_finance_and_policy"),
        ("1557-3079", "education_finance_and_policy"),
    ],
)
def test_core_alias_and_dual_issn_resolution(query: str, journal_key: str) -> None:
    assert _registry().resolve(query).journal_key == journal_key


def test_alias_normalization() -> None:
    assert normalize_journal_name("Economics & Education.") == normalize_journal_name(
        "economics and education"
    )


def test_journal_group_is_not_a_relevance_input() -> None:
    system_prompt = (ROOT / "prompts/personal_relevance_system.txt").read_text(
        encoding="utf-8"
    )
    user_prompt = (ROOT / "prompts/personal_relevance_user.txt").read_text(
        encoding="utf-8"
    )
    assert "journal_group" not in system_prompt
    assert "journal_group" not in user_prompt


def test_custom_journal_issn_conflict_is_rejected() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        make_custom_journal(
            _registry(),
            canonical_name="Not EER",
            short_name="NE",
            print_issn="0272-7757",
        )


def test_personal_mode_requires_exact_env_intent_and_localhost() -> None:
    assert personal_mode_allowed(True, "personal", "localhost:8501")
    assert personal_mode_allowed(True, "personal", "127.0.0.1:8501")
    assert not personal_mode_allowed(False, "personal", "localhost")
    assert not personal_mode_allowed(True, "personal", "example.com")
    assert personal_mode_allowed(True, None, "localhost")


def test_issn_first_dual_query_deduplicates_same_paper(tmp_path: Path) -> None:
    database = _database(tmp_path)
    registry = _registry()
    eer = registry.by_key("economics_of_education_review")
    item = PaperMetadata(
        doi="10.1000/same",
        title="Same paper",
        abstract="Original abstract.",
        source_metadata={
            "ISSN": ["0272-7757", "1873-7382"],
            "container-title": ["Economics of Education Review"],
        },
    )
    client = FakeCrossref({eer.print_issn: [item], eer.online_issn: [item]})
    result = PersonalJournalSyncService(database, client).sync(
        (eer,),
        since=date(2026, 1, 1),
        until=date(2026, 1, 31),
        per_journal_limit=5,
    )[0]
    assert client.calls == ["0272-7757", "1873-7382"]
    assert result.inserted == 1
    assert result.duplicates == 1
    with database.connect() as connection:
        metadata = json.loads(
            connection.execute(
                "SELECT source_metadata_json FROM papers"
            ).fetchone()[0]
        )
    assert metadata["eduecon_journal_match"]["queried_issn"] == "0272-7757"


def test_one_journal_failure_does_not_block_other_journal(tmp_path: Path) -> None:
    database = _database(tmp_path)
    registry = _registry()
    eer, efp = (
        registry.by_key("economics_of_education_review"),
        registry.by_key("education_finance_and_policy"),
    )
    paper = PaperMetadata(
        doi="10.1000/efp",
        title="Finance paper",
        source_metadata={
            "ISSN": ["1557-3060"],
            "container-title": ["Education Finance and Policy"],
        },
    )
    responses = {
        eer.print_issn: RuntimeError("provider down"),
        eer.online_issn: RuntimeError("provider down"),
        efp.print_issn: [paper],
        efp.online_issn: [],
    }
    results = PersonalJournalSyncService(database, FakeCrossref(responses)).sync(
        (eer, efp),
        since=date(2026, 1, 1),
        until=date(2026, 1, 31),
        per_journal_limit=5,
    )
    assert results[0].query_errors
    assert results[1].inserted == 1


def test_same_doi_returned_by_two_journals_is_deduplicated(tmp_path: Path) -> None:
    database = _database(tmp_path)
    registry = _registry()
    eer, efp = (
        registry.by_key("economics_of_education_review"),
        registry.by_key("education_finance_and_policy"),
    )
    eer_record = PaperMetadata(
        doi="10.1000/cross-journal",
        title="Shared record",
        source_metadata={"ISSN": [eer.print_issn]},
    )
    efp_record = PaperMetadata(
        doi="10.1000/cross-journal",
        title="Shared record",
        source_metadata={"ISSN": [efp.print_issn]},
    )
    client = FakeCrossref(
        {
            eer.print_issn: [eer_record],
            eer.online_issn: [],
            efp.print_issn: [efp_record],
            efp.online_issn: [],
        }
    )
    results = PersonalJournalSyncService(database, client).sync(
        (eer, efp),
        since=date(2026, 1, 1),
        until=date(2026, 1, 31),
        per_journal_limit=5,
    )
    assert results[0].inserted == 1
    assert results[1].duplicates == 1
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1


def test_personal_processing_is_per_paper_and_missing_abstract_never_classified(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    ready = _insert_paper(
        database,
        doi="10.1000/ready",
        title="Education outcomes",
        abstract="We estimate education outcomes using an economic model.",
    )
    missing = _insert_paper(
        database,
        doi="10.1000/missing",
        title="Title only",
        abstract=None,
    )
    task_id = _task_for(database, [ready, missing])
    translation = FakeTranslation(database)
    classifier = FakeClassifier()
    service = PersonalWorkflowService(
        database,
        enrichment_service=NoopEnrichment(),
        translation_service=translation,
        classifier=classifier,
    )
    result = service.process_next_batch(task_id, batch_size=3)
    state = load_personal_state(database, task_id)
    assert result.completed == (ready,)
    assert state["items"][str(ready)]["status"] == "classified"
    assert state["items"][str(missing)]["status"] == "pending"
    assert missing not in result.selected
    assert classifier.calls == [ready]
    with database.connect() as connection:
        source = json.loads(
            connection.execute(
                "SELECT source_config_json FROM workflow_batches WHERE id=?", (task_id,)
            ).fetchone()[0]
        )
    assert source["runtime_mode"] == "personal_local"
    assert "release_status" not in source
    assert "pilot_authorization_id" not in source


def test_one_processing_exception_does_not_block_next_paper(tmp_path: Path) -> None:
    database = _database(tmp_path)
    first = _insert_paper(
        database,
        doi="10.1000/fails",
        title="First paper",
        abstract="First abstract.",
    )
    second = _insert_paper(
        database,
        doi="10.1000/continues",
        title="Education outcomes",
        abstract="We estimate education outcomes.",
    )
    task_id = _task_for(database, [first, second])

    class FirstTranslationFails(FakeTranslation):
        def run(self, parameters):
            if parameters.paper_ids == (first,):
                raise RuntimeError("isolated translation failure")
            return super().run(parameters)

    service = PersonalWorkflowService(
        database,
        enrichment_service=NoopEnrichment(),
        translation_service=FirstTranslationFails(database),
        classifier=FakeClassifier(),
    )
    result = service.process_next_batch(task_id, batch_size=3)
    assert result.failed == (first,)
    assert result.completed == (second,)
    assert load_personal_state(database, task_id)["items"][str(first)]["status"] == "failed"


def test_structured_parse_failure_preserves_raw_response_and_does_not_pollute_decision(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    paper_id = _insert_paper(
        database,
        doi="10.1000/bad-json",
        title="Education outcomes",
        abstract="We study education outcomes.",
    )
    task_id = _task_for(database, [paper_id])
    service = PersonalWorkflowService(
        database,
        enrichment_service=NoopEnrichment(),
        translation_service=FakeTranslation(database),
        classifier=FakeClassifier(fail=True),
    )
    service.process_next_batch(task_id, batch_size=1)
    item = load_personal_state(database, task_id)["items"][str(paper_id)]
    assert item["status"] == "needs_retry"
    assert item["raw_ai_response"] == "{broken"
    assert "classification" not in item


def test_manual_abstract_save_then_process_and_manual_source_is_preserved(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    paper_id = _insert_paper(
        database,
        doi="10.1000/manual",
        title="Education outcomes",
        abstract=None,
    )
    task_id = _task_for(database, [paper_id])
    service = PersonalWorkflowService(
        database,
        enrichment_service=NoopEnrichment(),
        translation_service=FakeTranslation(database),
        classifier=FakeClassifier(),
    )
    result = save_and_process_manual_abstract(
        service,
        task_id,
        paper_id,
        "We estimate education outcomes using an economic model.",
        "https://publisher.example/paper",
    )
    assert result["status"] == "classified"
    with database.connect() as connection:
        row = connection.execute(
            "SELECT abstract,source_metadata_json FROM papers WHERE id=?", (paper_id,)
        ).fetchone()
    assert row["abstract"].startswith("We estimate")
    assert json.loads(row["source_metadata_json"])["abstract_source"] == "manual"


def test_existing_manual_abstract_skips_enrichment(tmp_path: Path) -> None:
    database = _database(tmp_path)
    paper_id = _insert_paper(
        database,
        doi="10.1000/manual-existing",
        title="Education outcomes",
        abstract="Human supplied education outcomes abstract.",
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE papers SET source_metadata_json=? WHERE id=?",
            (json.dumps({"abstract_source": "manual"}), paper_id),
        )
    task_id = _task_for(database, [paper_id])

    class ForbiddenEnrichment:
        def enrich_papers(self, *args, **kwargs):
            raise AssertionError("Manual abstract must not be overwritten or re-enriched")

    service = PersonalWorkflowService(
        database,
        enrichment_service=ForbiddenEnrichment(),
        translation_service=FakeTranslation(database),
        classifier=FakeClassifier(),
    )
    assert service.process_next_batch(task_id, batch_size=1).completed == (paper_id,)


def test_batch_size_and_task_budget_are_bounded(tmp_path: Path) -> None:
    database = _database(tmp_path)
    paper_id = _insert_paper(
        database,
        doi="10.1000/bounds",
        title="Education outcomes",
        abstract="We estimate education outcomes.",
    )
    task_id = _task_for(database, [paper_id])
    service = PersonalWorkflowService(
        database,
        enrichment_service=NoopEnrichment(),
        translation_service=FakeTranslation(database),
        classifier=FakeClassifier(),
    )
    with pytest.raises(ValueError, match="1, 3, or 5"):
        service.process_next_batch(task_id, batch_size=2)
    registry = _registry()
    result = load_personal_state(database, task_id)
    assert "release_status" not in result
    from eduecon_radar.services.personal_mode import JournalSyncResult

    with pytest.raises(ValueError, match="between 1 and 500"):
        create_personal_task(
            database,
            journal_results=(
                JournalSyncResult(
                    "economics_of_education_review",
                    "Economics of Education Review",
                    paper_ids=[paper_id],
                ),
            ),
            selected_journals=(
                registry.by_key("economics_of_education_review"),
            ),
            since=date(2026, 1, 1),
            until=date(2026, 1, 2),
            per_journal_limit=1,
            max_task_papers=501,
        )


def test_interrupted_processing_item_is_resumable(tmp_path: Path) -> None:
    database = _database(tmp_path)
    paper_id = _insert_paper(
        database,
        doi="10.1000/resume",
        title="Education outcomes",
        abstract="We estimate education outcomes.",
    )
    task_id = _task_for(database, [paper_id])
    with database.connect() as connection:
        row = connection.execute(
            "SELECT result_summary_json FROM workflow_batches WHERE id=?", (task_id,)
        ).fetchone()
        payload = json.loads(row[0])
        payload["personal_local"]["items"][str(paper_id)]["status"] = "processing"
        payload["personal_local"]["items"][str(paper_id)]["stage"] = "abstract_translation"
        connection.execute(
            "UPDATE workflow_batches SET result_summary_json=? WHERE id=?",
            (json.dumps(payload), task_id),
        )
    service = PersonalWorkflowService(
        database,
        enrichment_service=NoopEnrichment(),
        translation_service=FakeTranslation(database),
        classifier=FakeClassifier(),
    )
    assert service.process_next_batch(task_id, batch_size=1).completed == (paper_id,)


def test_human_default_is_undecided_and_save_accumulates_gold_candidate(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    paper_id = _insert_paper(
        database,
        doi="10.1000/human",
        title="Education outcomes",
        abstract="We estimate education outcomes.",
    )
    task_id = _task_for(database, [paper_id])
    assert list_personal_task_papers(database, task_id)[0]["human_decision"] == "undecided"
    assert not save_personal_human_decision(
        database, task_id, paper_id, decision="undecided"
    )
    assert save_personal_human_decision(
        database, task_id, paper_id, decision="related", note="Gold candidate"
    )
    state = load_personal_state(database, task_id)
    assert state["gold_candidates"][-1]["user_label"] == "related"
    with database.connect() as connection:
        row = connection.execute(
            """SELECT final_decision,review_status FROM workflow_task_reviews
            WHERE task_id=? AND paper_id=?""",
            (task_id, paper_id),
        ).fetchone()
    assert (row["final_decision"], row["review_status"]) == ("include", "approved")
    assert not save_personal_human_decision(
        database, task_id, paper_id, decision="related", note="Gold candidate"
    )


def test_mentor_export_excludes_ai_fields_and_personal_export_labels_them(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    paper_id = _insert_paper(
        database,
        doi="10.1000/export",
        title="Education outcomes",
        abstract="We estimate education outcomes using an economic model.",
    )
    task_id = _task_for(database, [paper_id])
    service = PersonalWorkflowService(
        database,
        enrichment_service=NoopEnrichment(),
        translation_service=FakeTranslation(database),
        classifier=FakeClassifier(),
    )
    service.process_next_batch(task_id, batch_size=1)
    save_personal_human_decision(
        database, task_id, paper_id, decision="related", note="Human note"
    )
    mentor = export_mentor_csv_text(
        database, task_id=task_id, review_status="approved"
    )
    personal = export_personal_task_csv(database, task_id)
    assert "original_abstract" in mentor
    assert "automatic_decision" not in mentor
    assert "ai_reason" not in mentor
    assert "automatic_decision" in personal
    assert "direct_evidence" in personal
    assert "NOT_RELEASED" not in personal


class _ReadOnlyStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}

    def warning(self, *args, **kwargs):
        return None

    def title(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def header(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def selectbox(self, _label, options, **kwargs):
        return options[0]

    def text_input(self, *args, **kwargs):
        return ""

    def button(self, *args, **kwargs):
        return False

    def multiselect(self, _label, _options, *, default=(), **kwargs):
        return default

    def columns(self, count, **_kwargs):
        size = count if isinstance(count, int) else len(count)
        return [self for _ in range(size)]

    def date_input(self, _label, *, value, **kwargs):
        return value

    def number_input(self, _label, *, value, **kwargs):
        return value

    def metric(self, *args, **kwargs):
        return None

    def tabs(self, *args, **kwargs):
        labels = args[0] if args else []
        return [self for _ in range(len(labels))]

    def segmented_control(self, _label, options, *, default=None, **_kwargs):
        return default or options[0]

    def toggle(self, *_args, **_kwargs):
        return False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __getattr__(self, name):
        if name in {
            "subheader",
            "markdown",
            "json",
            "write",
            "dataframe",
            "success",
            "error",
            "text_area",
            "radio",
            "checkbox",
            "download_button",
            "link_button",
            "toast",
        }:
            return lambda *args, **kwargs: None
        if name == "form_submit_button":
            return self.button
        if name in {"form", "container", "expander", "spinner", "status"}:
            return lambda *args, **kwargs: self
        raise AttributeError(name)


def test_personal_page_render_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = _database(tmp_path)
    before = hashlib.sha256(database.path.read_bytes()).hexdigest()
    fake = _ReadOnlyStreamlit()
    monkeypatch.setattr(personal_view, "st", fake)
    settings = SimpleNamespace(
        journal_registry_path=ROOT / "config" / "journals" / "catalog.json",
        journals_path=ROOT / "journals.yaml",
        max_per_journal_limit=20,
        project_root=ROOT,
    )
    personal_view.render(database, settings)
    after = hashlib.sha256(database.path.read_bytes()).hexdigest()
    assert before == after
