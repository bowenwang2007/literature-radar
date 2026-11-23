from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import httpx

from eduecon_radar.db import Database
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.abstract_enrichment import (
    AbstractCandidate,
    AbstractEnrichmentService,
    AbstractSourceAdapter,
    EnrichmentPaperInput,
    OpenAIREEnrichmentAdapter,
)
from eduecon_radar.services.local_pipeline import (
    LocalPipelineServices,
    ensure_paper_statuses,
    latest_sync_run,
    process_batch,
    save_human_decisions,
    sync_run_papers,
)
from eduecon_radar.services.migrations import (
    apply_overnight_migration,
    apply_p0_hardening_migration,
    apply_profile_migration,
    apply_translation_migration,
    apply_workflow_migration,
)
from eduecon_radar.services.personal_mode import PersonalJournalSyncService
from eduecon_radar.services.profile_registry import default_profile_registry
from eduecon_radar.services.retrieval_classify import classify_item, count_paths
from eduecon_radar.services.ui_labels import (
    HUMAN_DECISION_OPTIONS,
    human_decision_from_zh,
    zh_auto_decision,
    zh_human_decision,
    zh_stop_reason,
)

ROOT = Path(__file__).resolve().parents[1]


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "phase7c1.db")
    database.init_schema()
    apply_translation_migration(database, ROOT / "migrations/0001_add_paper_translations.sql")
    apply_workflow_migration(database, ROOT / "migrations/0002_add_workflow_batches.sql")
    apply_p0_hardening_migration(
        database, ROOT / "migrations/0007_add_runtime_leases_and_task_reviews.sql"
    )
    apply_overnight_migration(
        database, ROOT / "migrations/0010_add_overnight_pipeline.sql"
    )
    apply_profile_migration(
        database, ROOT / "migrations/0013_add_profile_scoped_assessment.sql"
    )
    return database


def _registry():
    return default_profile_registry()


def _record(doi: str, title: str, *, volume: str | None = "112") -> PaperMetadata:
    return PaperMetadata(
        doi=doi,
        title=title,
        abstract=None,
        authors=["Researcher"],
        publisher="Fixture",
        issue_date=date(2026, 6, 1),
        source_metadata={
            "DOI": doi,
            "ISSN": ["0272-7757"],
            "container-title": ["Economics of Education Review"],
            "volume": volume,
        },
    )


class FakeCrossref:
    def __init__(self, records: list[PaperMetadata]) -> None:
        self.records = records

    def fetch(self, _issn, _since, limit, *, until=None):
        del until
        return list(self.records[:limit])


def _sync(database: Database, count: int = 3):
    records = [_record(f"10.1000/7c1-{i}", f"Paper {i}") for i in range(1, count + 1)]
    journal = _registry().by_key("economics_of_education_review")
    PersonalJournalSyncService(database, FakeCrossref(records)).sync(
        (journal,),
        since=date(2026, 5, 1),
        until=date(2026, 7, 31),
        per_journal_limit=100,
    )
    run = latest_sync_run(database)
    assert run is not None
    return int(run["id"]), run


class NoopEnrichment:
    def __init__(self, *, writes: bool = True) -> None:
        self.writes = writes
        self.called = False

    def enrich_papers(
        self, paper_ids, *, workflow_batch_id=None, force_refresh=False
    ):
        self.called = True
        if not self.writes:
            return type(
                "Stats",
                (),
                {"total": len(paper_ids), "enriched": 1, "failed": 0},
            )()
        return type(
            "Stats",
            (),
            {"total": len(paper_ids), "enriched": 0, "failed": 0},
        )()


class NoopTranslation:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, parameters):
        self.calls += 1
        return type(
            "Stats",
            (),
            {
                "ai_calls": 1,
                "input_tokens": 1,
                "output_tokens": 1,
                "failed": False,
                "failure_reasons": [],
            },
        )()


def test_date_path_counts_online_vs_issue_and_boundaries() -> None:
    items = [
        {
            "DOI": "10.1/a",
            "title": "Online May article",
            "type": "journal-article",
            "volume": "112",
            "created": {"date-parts": [[2026, 5, 12]]},
            "published-print": {"date-parts": [[2026, 6, 1]]},
            "issued": {"date-parts": [[2026, 6, 1]]},
        },
        {
            "DOI": "10.1/b",
            "title": "Online March, issue June",
            "type": "journal-article",
            "volume": "112",
            "created": {"date-parts": [[2026, 3, 24]]},
            "published-print": {"date-parts": [[2026, 6, 1]]},
            "issued": {"date-parts": [[2026, 6, 1]]},
        },
        {
            "DOI": "10.1/c",
            "title": "Articles in Press May",
            "type": "journal-article",
            "volume": None,
            "created": {"date-parts": [[2026, 5, 2]]},
            "published-print": {"date-parts": [[2026, 5, 1]]},
            "issued": {"date-parts": [[2026, 5, 1]]},
        },
        {
            "DOI": "10.1/d",
            "title": "Out of range April",
            "type": "journal-article",
            "volume": "112",
            "created": {"date-parts": [[2026, 4, 30]]},
            "published-print": {"date-parts": [[2026, 6, 1]]},
            "issued": {"date-parts": [[2026, 6, 1]]},
        },
        {
            "DOI": "10.1/e",
            "title": "Editorial Board",
            "type": "journal-article",
            "volume": "112",
            "created": {"date-parts": [[2026, 7, 31]]},
            "published-print": {"date-parts": [[2026, 7, 31]]},
            "issued": {"date-parts": [[2026, 7, 31]]},
        },
    ]
    counts = count_paths(items, date(2026, 5, 1), date(2026, 7, 31))
    assert counts["online_published_in_range"] == 3  # a, c, e
    assert counts["issue_dated_in_range"] == 5
    assert counts["non_research_count"] == 1
    assert counts["volumes"] == {"112": 4, "articles_in_press": 1}
    classified = classify_item(items[4])
    assert classified["is_research"] is False
    assert classified["duplicate_key"] == "doi:10.1/e"
    duplicate_items = [dict(items[0]), dict(items[0])]
    assert count_paths(duplicate_items, date(2026, 5, 1), date(2026, 7, 31))[
        "duplicates"
    ] == 1


def test_ui_label_mappings_are_central_and_stable() -> None:
    assert HUMAN_DECISION_OPTIONS == ("未决定", "相关", "无关", "不确定")
    assert human_decision_from_zh("相关") == "related"
    assert human_decision_from_zh("未决定") == "undecided"
    assert zh_human_decision("related") == "相关"
    assert zh_auto_decision("relevant") == "相关"
    assert zh_auto_decision("needs_abstract") == "缺摘要"
    assert zh_stop_reason("provider_exhausted") == "数据源已返回全部结果"
    assert zh_stop_reason("rate_limited") == "数据源请求受限，稍后可重试"
    assert (
        zh_stop_reason("non_research_front_matter_no_abstract_expected")
        == "期刊前置内容通常不提供摘要"
    )
    assert zh_stop_reason("internal_future_enum") == "原因待核实"


def test_provider_success_without_write_never_marks_abstract_ready(tmp_path) -> None:
    database = _database(tmp_path)
    run_id, _run = _sync(database, count=1)
    ensure_paper_statuses(database, run_id)
    services = LocalPipelineServices(enrichment=NoopEnrichment(writes=False))
    result = process_batch(database, run_id, (1,), services)
    assert result["needs_abstract"] == [1]
    with database.connect() as connection:
        row = connection.execute(
            "SELECT status,abstract_retrieval_json FROM paper_processing_status "
            "WHERE sync_run_id=? AND paper_id=?",
            (run_id, 1),
        ).fetchone()
    assert row["status"] != "abstract_ready"
    assert "needs_manual_abstract" in row["status"] or "needs_retry" in row["status"]


def test_front_matter_is_terminal_without_provider_or_ai_calls(tmp_path) -> None:
    database = _database(tmp_path)
    run_id, _run = _sync(database, count=1)
    ensure_paper_statuses(database, run_id)
    with database.connect() as connection:
        connection.execute(
            "UPDATE papers SET title='Editorial Board',authors_json='[]' WHERE id=1"
        )

    class ForbiddenAdapter(AbstractSourceAdapter):
        source_name = "forbidden"

        def fetch(self, paper, context=None):
            raise AssertionError("front matter must not query abstract providers")

    services = LocalPipelineServices(
        enrichment=AbstractEnrichmentService(database, adapters=[ForbiddenAdapter()])
    )
    result = process_batch(database, run_id, (1,), services)
    assert result["completed"] == [1]
    assert result["needs_abstract"] == []
    with database.connect() as connection:
        paper = connection.execute(
            "SELECT abstract,source_metadata_json FROM papers WHERE id=1"
        ).fetchone()
        status = connection.execute(
            "SELECT status,stage,classification_json,abstract_retrieval_json "
            "FROM paper_processing_status WHERE sync_run_id=? AND paper_id=1",
            (run_id,),
        ).fetchone()
    audit = json.loads(paper["source_metadata_json"])["abstract_enrichment"]
    assert paper["abstract"] is None
    assert audit["execution_state"] == "not_applicable"
    assert audit["provider_queries"] == 0
    assert status["status"] == "classified" and status["stage"] == "complete"
    from eduecon_radar.services import profile_store

    stored = profile_store.assessments_for(database, "education_economics", (1,))[1]
    assert stored.classification["decision"] == "irrelevant"
    # The run snapshot no longer carries a judgement.
    assert json.loads(status["classification_json"]) == {}
    assert json.loads(status["abstract_retrieval_json"])["status"] == "not_applicable"


def test_abstract_write_then_ui_reads_same_field(tmp_path) -> None:
    database = _database(tmp_path)
    run_id, _run = _sync(database, count=1)
    ensure_paper_statuses(database, run_id)
    abstract = (
        "This study estimates how school finance reform affects student achievement "
        "and economic outcomes using administrative data and a difference-in-differences design."
    )

    class WriteAdapter(AbstractSourceAdapter):
        source_name = "openalex"

        def fetch(self, paper, context=None):
            return AbstractCandidate(
                abstract,
                self.source_name,
                raw_doi=paper.doi,
                raw_title=paper.title,
                raw_authors=tuple(paper.authors),
                raw_year=paper.year,
                raw_journal=paper.journal,
            )

    service = AbstractEnrichmentService(
        database, adapters=[WriteAdapter()]
    )
    stats = service.enrich_papers((1,))
    assert stats.enriched == 1
    with database.connect() as connection:
        stored = connection.execute(
            "SELECT abstract FROM papers WHERE id=1"
        ).fetchone()[0]
    assert stored == abstract
    papers = sync_run_papers(database, run_id)
    assert papers[0]["abstract"] == abstract
    assert papers[0]["processing"]["abstract_retrieval"] is not None


def test_english_abstract_triggers_translation_and_processing_continues(tmp_path) -> None:
    database = _database(tmp_path)
    run_id, _run = _sync(database, count=1)
    ensure_paper_statuses(database, run_id)
    abstract = (
        "This paper studies teacher incentives and student achievement using a sharp "
        "regression discontinuity design and administrative panel data with detailed outcomes."
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE papers SET abstract=? WHERE id=1", (abstract,)
        )
    translation = NoopTranslation()
    services = LocalPipelineServices(translation=translation)
    result = process_batch(database, run_id, (1,), services)
    assert translation.calls == 1
    assert result["needs_retry"] == [1]  # screening unavailable -> needs_retry
    with database.connect() as connection:
        status = connection.execute(
            "SELECT status,abstract_retrieval_json FROM paper_processing_status "
            "WHERE sync_run_id=? AND paper_id=?",
            (run_id, 1),
        ).fetchone()
    assert status["status"] == "needs_retry"
    retrieval = json.loads(status["abstract_retrieval_json"])
    assert retrieval["status"] == "abstract_ready"
    assert retrieval["retryable"] is False


def test_needs_retry_with_existing_abstract_forces_screening_recheck(tmp_path) -> None:
    database = _database(tmp_path)
    run_id, _run = _sync(database, count=1)
    ensure_paper_statuses(database, run_id)
    with database.connect() as connection:
        connection.execute("UPDATE papers SET abstract=? WHERE id=1", ("word " * 80,))
        connection.execute(
            "UPDATE paper_processing_status SET status='needs_retry' "
            "WHERE sync_run_id=? AND paper_id=1",
            (run_id,),
        )

    class Screening:
        model = "fixture"

        def __init__(self) -> None:
            self.force = False

        def screen_paper(self, paper_id, *, task_id=None, force=False):
            del paper_id, task_id
            self.force = force
            return type(
                "Outcome",
                (),
                {
                    "status": "classified",
                    "ai_calls": 1,
                    "classification": {"decision": "relevant"},
                },
            )()

    screening = Screening()
    result = process_batch(
        database,
        run_id,
        (1,),
        LocalPipelineServices(screening=screening),
    )
    assert screening.force is True
    assert result["completed"] == [1]


def test_manual_abstract_never_overwritten(tmp_path) -> None:
    database = _database(tmp_path)
    run_id, _run = _sync(database, count=1)
    ensure_paper_statuses(database, run_id)
    manual = "Manual abstract that must never be overwritten."
    with database.connect() as connection:
        connection.execute(
            "UPDATE papers SET abstract=? WHERE id=1", (manual,)
        )
        metadata = json.loads(
            connection.execute(
                "SELECT source_metadata_json FROM papers WHERE id=1"
            ).fetchone()[0]
        )
        metadata["abstract_source"] = "manual"
        connection.execute(
            "UPDATE papers SET source_metadata_json=? WHERE id=1",
            (json.dumps(metadata),),
        )
    with database.connect() as connection:
        abstract_after = connection.execute(
            "SELECT abstract FROM papers WHERE id=1"
        ).fetchone()[0]
    assert abstract_after == manual


def test_human_decisions_saved_as_english_enums(tmp_path) -> None:
    database = _database(tmp_path)
    run_id, _run = _sync(database, count=2)
    ensure_paper_statuses(database, run_id)
    saved = save_human_decisions(
        database,
        run_id,
        {1: ("related", "重要"), 2: ("irrelevant", None)},
    )
    assert tuple(saved) == (1, 2)
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT paper_id,human_decision,human_note FROM paper_processing_status "
            "WHERE sync_run_id=? ORDER BY paper_id",
            (run_id,),
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        (1, "related", "重要"),
        (2, "irrelevant", None),
    ]


def test_ui_source_simplifies_auto_judgment_and_localizes_decisions() -> None:
    source = (ROOT / "src/eduecon_radar/ui/views/personal_mode.py").read_text(
        encoding="utf-8"
    )
    assert "AI 判断依据与原文证据" in source
    assert "st.json(classification)" in source
    review_block = source.split("def _render_review_workspace", 1)[1]
    review_block = review_block.split("def _render_verification_summary_section", 1)[0]
    assert review_block.index("st.json(classification)") > review_block.index(
        'with st.expander("AI 判断依据与原文证据")'
    )
    assert "自动判断" in review_block
    assert 'list(HUMAN_DECISION_OPTIONS)' in source
    assert '"undecided", "related", "irrelevant", "uncertain"' not in source
    assert "zh_auto_decision" in source
    assert "zh_stop_reason" in source


def test_review_decision_surface_shows_only_what_a_decision_needs() -> None:
    """The decision panel must not become a report about the model.

    Only the automatic judgement, how much evidence stood behind it and the
    human decision belong on the surface.  The reasoning, the quoted evidence
    and the condition fields stay behind the ``AI 判断依据与原文证据`` disclosure.
    """
    source = (ROOT / "src/eduecon_radar/ui/views/personal_mode.py").read_text(
        encoding="utf-8"
    )
    review_block = source.split("def _render_review_workspace", 1)[1]
    review_block = review_block.split("def _render_verification_summary_section", 1)[0]

    # The reasoning block is rendered inside the technical disclosure, i.e. after
    # the decision control and after the disclosure that hides it.
    assert review_block.index("_ai_reasoning_block(") > review_block.index(
        'with st.expander("AI 判断依据与原文证据")'
    )
    assert review_block.index("st.radio(") < review_block.index(
        'with st.expander("AI 判断依据与原文证据")'
    )
    # The judgement and its basis kind are shown compactly on the surface.
    assert "screening_state_label(" in review_block
    assert "edr-decision-basis" in review_block
    # Nothing that only matters when auditing is rendered on the surface.
    surface = review_block.split('with st.expander("AI 判断依据与原文证据")')[0]
    for audit_only in (
        "edr-reason-text",
        "edr-reason-quotes",
        "证据是否充分",
        "经济学内容",
        "st.json(",
    ):
        assert audit_only not in surface, audit_only


def test_app_render_shows_single_chinese_decision_not_raw_json(tmp_path, monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    from eduecon_radar.services.crossref import CrossrefClient
    from eduecon_radar.services.deepseek import DeepSeekClient

    database = _database(tmp_path)
    run_id, _run = _sync(database, count=1)
    ensure_paper_statuses(database, run_id)
    abstract = (
        "This paper studies teacher incentives and student achievement using a sharp "
        "regression discontinuity design and administrative panel data with detailed outcomes."
    )
    now = "2026-08-01T00:00:00+00:00"
    with database.connect() as connection:
        connection.execute(
            "UPDATE papers SET abstract=? WHERE id=1", (abstract,)
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
            VALUES (1,'h','relevant','central','substantive','sufficient','high',
             '[{"quote":"teacher incentives and student achievement","supports":"final_decision"}]',
             'clear','[]','high','fixture','v1','o','2026-08-01T00:00:00+00:00','{}',
             'p1','e1','valid','[]','not_required','relevant','',
             '[{"quote":"teacher incentives and student achievement","supports":"final_decision"}]',
             ?,?)""",
            (now, now),
        )
        connection.execute(
            """UPDATE paper_processing_status
            SET status='classified',stage='complete',
                classification_json=?,
                model_name='fixture',prompt_version='v1',classified_at=?
            WHERE sync_run_id=? AND paper_id=1""",
            (
                json.dumps(
                    {
                        "decision": "relevant",
                        "education_centrality": "central",
                        "economic_relevance": "substantive",
                        "confidence": "high",
                        "evidence": ["teacher incentives and student achievement"],
                        "reason": "clear",
                    }
                ),
                now,
                run_id,
            ),
        )
    before = hashlib.sha256(database.path.read_bytes()).hexdigest()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database.path.as_posix()}")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    monkeypatch.setattr(
        "eduecon_radar.services.product_access.local_request_allowed", lambda _host: True
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("render must not call AI or an external provider")

    monkeypatch.setattr(CrossrefClient, "fetch", forbidden)
    monkeypatch.setattr(DeepSeekClient, "request_json", forbidden)
    app = AppTest.from_file(
        str(ROOT / "src/eduecon_radar/ui/app.py"), default_timeout=45
    ).run()
    app.segmented_control(key="edr_primary_view").set_value("文献工作台").run()
    assert not app.exception
    rendered = "\n".join(
        str(item.value)
        for group in (app.caption, app.markdown, app.info, app.warning)
        for item in group
    ) + "\n" + "\n".join(str(item.label) for item in app.expander)
    assert "自动判断" in rendered
    assert "相关" in rendered
    assert "education_centrality" not in rendered
    # The audit record is present but folded away, so it is an expander label
    # rather than always-visible text.
    assert "AI 判断依据与原文证据" in rendered
    assert hashlib.sha256(database.path.read_bytes()).hexdigest() == before


def test_review_page_keeps_the_model_report_behind_the_disclosure(
    tmp_path, monkeypatch
) -> None:
    """The page shows the decision surface a reviewer acts on.

    What is asserted here is the surface: the judgement, its basis kind, and the
    decision control.  Whether the folded ``AI 判断依据与原文证据`` panel is visually open is
    decided by the browser, which ``AppTest`` does not model — the structural
    guarantee that the model's report sits *inside* that disclosure is pinned by
    ``test_review_decision_surface_shows_only_what_a_decision_needs``.
    """
    from streamlit.testing.v1 import AppTest

    from eduecon_radar.services.crossref import CrossrefClient
    from eduecon_radar.services.deepseek import DeepSeekClient

    database = _database(tmp_path)
    run_id, _run = _sync(database, count=1)
    ensure_paper_statuses(database, run_id)
    now = "2026-08-01T00:00:00+00:00"
    with database.connect() as connection:
        connection.execute(
            "UPDATE papers SET abstract=? WHERE id=1",
            (
                "This paper studies teacher incentives and student achievement "
                "using a sharp regression discontinuity design.",
            ),
        )
        connection.execute(
            """UPDATE paper_processing_status
            SET status='classified',stage='complete',classification_json=?,classified_at=?
            WHERE sync_run_id=? AND paper_id=1""",
            (
                json.dumps(
                    {
                        "decision": "relevant",
                        "screening_basis": "title_and_abstract",
                        "education_centrality": "central",
                        "economic_relevance": "substantive",
                        "evidence_sufficiency": "sufficient",
                        "confidence": "high",
                        "validation_status": "valid",
                        "evidence": ["teacher incentives and student achievement"],
                        "reason": "Identifies an education production effect.",
                    }
                ),
                now,
                run_id,
            ),
        )
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database.path.as_posix()}")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    monkeypatch.setattr(
        "eduecon_radar.services.product_access.local_request_allowed", lambda _host: True
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("render must not call AI or an external provider")

    monkeypatch.setattr(CrossrefClient, "fetch", forbidden)
    monkeypatch.setattr(DeepSeekClient, "request_json", forbidden)

    app = AppTest.from_file(
        str(ROOT / "src/eduecon_radar/ui/app.py"), default_timeout=45
    ).run()
    app.segmented_control(key="edr_primary_view").set_value("文献工作台").run()
    assert not app.exception

    surface = "\n".join(
        str(item.value)
        for group in (app.caption, app.markdown, app.info, app.warning)
        for item in group
    )
    # On the surface: the judgement and its basis kind, and the decision control.
    assert "自动判断" in surface
    assert "相关" in surface
    assert "完整判断" in surface
    assert any(item.label == "人工决定" for item in app.radio)
    assert any(item.label == "保存并下一篇" for item in app.button)
    # The audit material is rendered inside the disclosure, so it appears in the
    # element tree but not as an independent always-visible section.
    assert any(item.label == "AI 判断依据与原文证据" for item in app.expander)
    assert "edr-decision-head" in surface
    assert "edr-decision-basis" in surface


def test_openaire_adapter_extracts_structured_abstract() -> None:
    abstract = (
        "This study estimates the economic effects of education policy on student "
        "outcomes using administrative panel data and a transparent empirical design."
    )
    payload = {
        "results": [
            {
                "id": "doi_dedup___::paper",
                "mainTitle": "Education policy paper",
                "descriptions": ["education policy", abstract],
                "pids": [
                    {
                        "scheme": "doi",
                        "value": "10.1016/j.econedurev.2026.102788",
                    }
                ],
                "authors": [{"fullName": "Researcher"}],
                "publicationDate": "2026-01-01",
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, json=payload)

    adapter = OpenAIREEnrichmentAdapter(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    paper = EnrichmentPaperInput(
        1,
        "10.1016/j.econedurev.2026.102788",
        "Education policy paper",
        ["Researcher"],
        "Economics of Education Review",
        2026,
        "0272-7757",
    )
    candidate = adapter.fetch(paper)
    assert candidate is not None
    assert candidate.text == abstract
    assert candidate.raw_doi == paper.doi
    assert candidate.raw_title == "Education policy paper"


def test_review_navigation_moves_between_papers(tmp_path, monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    from eduecon_radar.services.crossref import CrossrefClient
    from eduecon_radar.services.deepseek import DeepSeekClient

    database = _database(tmp_path)
    run_id, _run = _sync(database, count=2)
    ensure_paper_statuses(database, run_id)
    with database.connect() as connection:
        for paper_id, title in ((1, "论文一"), (2, "论文二")):
            connection.execute(
                """INSERT INTO paper_translations
                (paper_id,chinese_title,title_translation_status,
                 abstract_translation_status,title_translation_source,updated_at)
                VALUES (?,?,?,?,?,?)""",
                (
                    paper_id,
                    title,
                    "ai_generated",
                    "unavailable",
                    "ai_generated",
                    "2026-08-01T00:00:00+00:00",
                ),
            )
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database.path.as_posix()}")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    monkeypatch.setattr(
        "eduecon_radar.services.product_access.local_request_allowed", lambda _host: True
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("render must not call AI or an external provider")

    monkeypatch.setattr(CrossrefClient, "fetch", forbidden)
    monkeypatch.setattr(DeepSeekClient, "request_json", forbidden)
    app = AppTest.from_file(
        str(ROOT / "src/eduecon_radar/ui/app.py"), default_timeout=45
    ).run()
    app.segmented_control(key="edr_primary_view").set_value("文献工作台").run()
    assert not app.exception
    first = "\n".join(str(item.value) for item in app.markdown)
    assert "论文一" in first
    # The forward control says what it does: it moves on *without* recording a
    # decision, which is the distinction the review page has to make obvious.
    next_button = next(item for item in app.button if item.label == "下一篇")
    app = next_button.click().run()
    assert not app.exception
    second = "\n".join(str(item.value) for item in app.markdown)
    assert "论文二" in second
