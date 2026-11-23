from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from eduecon_radar.db import Database
from eduecon_radar.domain.semantic_storage import apply_domain_semantic_migration
from eduecon_radar.domain.storage import apply_domain_evidence_migration
from eduecon_radar.schemas import JournalConfig, PaperMetadata
from eduecon_radar.services.crossref import CrossrefClient
from eduecon_radar.services.deepseek import DeepSeekClient
from eduecon_radar.services.migrations import (
    apply_p0_hardening_migration,
    apply_translation_migration,
    apply_workflow_migration,
)
from eduecon_radar.ui import theme
from eduecon_radar.ui.views import paper_library

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src/eduecon_radar/ui/app.py"
PRODUCT_NAME = "Literature Radar"
TAGLINE = "多研究方向文献发现与人工审核工作台"


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "misty-blue-ui.db")
    database.init_schema()
    apply_translation_migration(
        database, ROOT / "migrations/0001_add_paper_translations.sql"
    )
    apply_workflow_migration(
        database, ROOT / "migrations/0002_add_workflow_batches.sql"
    )
    apply_domain_evidence_migration(
        database, ROOT / "migrations/0003_add_domain_evidence_runs.sql"
    )
    apply_domain_semantic_migration(
        database, ROOT / "migrations/0004_add_domain_semantic_runs.sql"
    )
    apply_p0_hardening_migration(
        database, ROOT / "migrations/0007_add_runtime_leases_and_task_reviews.sql"
    )
    journal_id = database.upsert_journal(
        JournalConfig(name="Visual Test Journal", short_name="VTJ", issn="1234-5678")
    )
    database.upsert_paper(
        PaperMetadata(
            title="A visual test paper about education",
            abstract="Original abstract for visual rendering.",
            authors=["Test Author"],
            doi="10.1000/misty-blue",
        ),
        journal_id,
    )
    return database


def test_theme_palette_is_centralized_and_matches_streamlit_config() -> None:
    css = theme.MISTY_BLUE_CSS
    required_colors = {
        "--sea-mist-500": "#7E97AC",
        "--sea-mist-700": "#48647D",
        "--sea-mist-900": "#24394C",
        "--surface-solid": "#FBFDFE",
        "--text-primary": "#24394C",
        "--accent": "#4F6E88",
        "--success": "#4E6C80",
        "--warning": "#8A7049",
        "--error": "#955D68",
    }
    for variable, value in required_colors.items():
        assert f"{variable}: {value}" in css
    lowered = css.lower()
    assert "radial-gradient" in lowered
    assert "linear-gradient" in lowered
    assert "data:image/svg+xml" not in lowered
    assert "0 0 90px 30px" not in lowered
    assert "backdrop-filter" not in lowered
    assert "neon" not in lowered
    for old_dashboard_class in (".edr-overview", ".edr-secondary-strip", ".edr-runtime"):
        assert old_dashboard_class not in css
    config = tomllib.loads((ROOT / ".streamlit/config.toml").read_text("utf-8"))
    assert config["theme"] == {
        "base": "light",
        "primaryColor": "#4F6E88",
        "backgroundColor": "#EAF1F6",
        "secondaryBackgroundColor": "#F6F9FB",
        "textColor": "#24394C",
        "font": "sans serif",
    }


def test_apply_theme_is_idempotent_and_has_no_business_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    class FakeStreamlit:
        @staticmethod
        def markdown(value: str, *, unsafe_allow_html: bool) -> None:
            calls.append((value, unsafe_allow_html))

    monkeypatch.setattr(theme, "st", FakeStreamlit())
    theme.apply_app_theme()
    theme.apply_app_theme()
    assert calls == [
        (theme.MISTY_BLUE_CSS, True),
        (theme.MISTY_BLUE_CSS, True),
    ]


def test_three_local_pages_render_with_theme_without_writes_or_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _database(tmp_path)
    before = hashlib.sha256(database.path.read_bytes()).hexdigest()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database.path.as_posix()}")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    monkeypatch.setattr(
        "eduecon_radar.services.product_access.local_request_allowed", lambda _host: True
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ordinary render must not call AI or external providers")

    monkeypatch.setattr(CrossrefClient, "fetch", forbidden)
    monkeypatch.setattr(DeepSeekClient, "request_json", forbidden)
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    assert not app.exception
    assert app.title[0].value == PRODUCT_NAME
    assert any(item.value == TAGLINE for item in app.caption)
    assert len(app.segmented_control) == 1
    assert app.segmented_control[0].value == "文献工作台"
    rendered_summary = "\n".join(str(item.value) for item in app.markdown)
    for label in ("本次获取", "待补摘要", "待审核", "已完成"):
        assert label in rendered_summary
    rendered_markdown = "\n".join(str(item.value) for item in app.markdown)
    assert "eduecon-misty-blue-theme" in rendered_markdown
    assert 'class="edr-flow"' in rendered_markdown
    assert "edr-hero-eyebrow" in rendered_markdown
    assert PRODUCT_NAME in [item.value for item in app.title]
    app.segmented_control[0].set_value("论文库").run()
    assert not app.exception
    rendered_markdown = "\n".join(str(item.value) for item in app.markdown)
    assert "edr-library-card" in rendered_markdown
    assert "Original abstract for visual rendering." not in rendered_markdown
    next(item for item in app.toggle if item.label == "查看摘要与详情").set_value(
        True
    ).run()
    rendered_markdown = "\n".join(str(item.value) for item in app.markdown)
    assert "Original abstract for visual rendering." in rendered_markdown

    assert any(item.value == "论文库" for item in app.header)
    assert any(item.label == "搜索" for item in app.text_input)

    visible = "\n".join(
        str(item.value)
        for group in (app.title, app.header, app.caption, app.info, app.warning, app.error)
        for item in group
    )
    forbidden_words = ("受控试运行", "NOT RELEASED", "task #16", "Runtime bundle")
    assert not [word for word in forbidden_words if word.casefold() in visible.casefold()]
    assert hashlib.sha256(database.path.read_bytes()).hexdigest() == before


def test_review_page_shows_ai_reason_and_basis_without_raw_json() -> None:
    """The daily review decision needs the *why*, not the raw payload.

    A reviewer accepting or overruling a suggestion has to see the reason, the
    words it was based on, and how much of the paper the model actually read.
    The raw classification JSON stays behind a disclosure: it is the audit
    record, not the reading material.
    """
    from eduecon_radar.ui.views.personal_mode import _ai_reasoning_block

    classification = {
        "decision": "relevant",
        "screening_basis": "title_and_abstract",
        "education_centrality": "central",
        "economic_relevance": "substantive",
        "evidence_sufficiency": "sufficient",
        "confidence": "high",
        "validation_status": "valid",
        "evidence": ["teacher incentives and student achievement"],
        "reason": "The paper identifies an education production effect.",
    }
    block = _ai_reasoning_block(
        classification, list(classification["evidence"]), has_abstract=True
    )

    assert "AI 判断依据" in block
    assert "The paper identifies an education production effect." in block
    assert "teacher incentives and student achievement" in block
    assert "依据标题与英文摘要" in block
    assert "核心议题" in block
    assert "有实质经济学内容" in block
    assert "证据充分" in block
    assert ">高<" in block

    # Stored field names are internal vocabulary and must never reach the reader.
    for raw_key in ("screening_basis", "education_centrality", "validation_status"):
        assert raw_key not in block

    # A judgement whose basis was never recorded says so instead of guessing.
    historical = _ai_reasoning_block(
        {"decision": "irrelevant", "evidence": [], "reason": ""},
        [],
        has_abstract=False,
    )
    assert "未记录" in historical
    assert "本次判断没有记录理由" in historical

    # The review workspace wires this block in (guards against it being dropped),
    # and it is rendered inside the technical disclosure rather than on the
    # decision surface.
    source = (
        ROOT / "src/eduecon_radar/ui/views/personal_mode.py"
    ).read_text(encoding="utf-8")
    assert "_ai_reasoning_block(" in source
    assert 'with st.expander("AI 判断依据与原文证据")' in source
    # rindex: the *call site*, not the function definition.
    assert source.rindex("_ai_reasoning_block(") > source.index(
        'with st.expander("AI 判断依据与原文证据")'
    )


def test_review_opens_on_open_work_and_library_titles_are_readable() -> None:
    """Two small things that decide whether the tool feels usable daily.

    Opening the review queue on an already-decided paper while the counter says
    one paper is waiting makes the page contradict itself; publisher titles
    carrying JATS markup and hard wraps read as corrupt data in a card.
    """
    from eduecon_radar.ui.views.paper_library import _clean_title
    from eduecon_radar.ui.views.personal_mode import _first_open_review_index

    decided = {"human_decision": "related", "processing": {"classification": {}}}
    undecided_without_judgment = {
        "human_decision": "undecided",
        "processing": {"classification": {}},
    }
    undecided_with_judgment = {
        "human_decision": "undecided",
        "processing": {"classification": {"decision": "relevant"}},
    }
    papers = [decided, undecided_without_judgment, undecided_with_judgment]
    assert _first_open_review_index(papers) == 2
    # Nothing left to decide: still show a paper rather than an empty page.
    assert _first_open_review_index([decided]) == 0

    assert (
        _clean_title("Comment on\n                    <i>‘Fisher–Schultz Lecture’</i>")
        == "Comment on ‘Fisher–Schultz Lecture’"
    )
    assert _clean_title("  Spaced   out  title ") == "Spaced out title"
    assert _clean_title(None) == ""


def test_responsive_theme_avoids_fixed_horizontal_layout() -> None:
    css = theme.MISTY_BLUE_CSS
    assert "@media (max-width: 1180px)" in css
    assert "@media (max-width: 860px)" in css
    assert "flex-wrap: wrap" in css
    assert "max-width: 100%" in css
    assert "overflow-x: hidden" not in css
    assert "--font-body: 16px" in css
    assert "--font-secondary-metric" not in css
    assert ".edr-system-dots" not in css
    assert ".edr-system-dot" not in css
    assert ".edr-flow" in css
    assert ".edr-overview" not in css
    assert ".edr-secondary-strip" not in css
    assert "--workbench-max: 1280px" in css
    assert "max-width: var(--workbench-max)" in css
    assert "max-width: 1620px" not in css
    assert ".edr-verification-stats" in css
    assert ".edr-status-bar" in css
    assert ".edr-next-action" in css
    assert ".edr-audit-line" in css
    assert ".edr-reason-grid" in css


def test_library_uses_twenty_item_reading_pages_without_render_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _database(tmp_path)
    with database.connect() as connection:
        journal_id = int(
            connection.execute(
                "SELECT id FROM journals WHERE name='Visual Test Journal'"
            ).fetchone()[0]
        )
    for index in range(2, 22):
        database.upsert_paper(
            PaperMetadata(
                title=f"Visual library paper {index}",
                abstract=f"Original abstract {index}.",
                authors=["Test Author"],
                doi=f"10.1000/visual-library-{index}",
            ),
            journal_id,
        )
    before = hashlib.sha256(database.path.read_bytes()).hexdigest()
    page_calls: list[dict[str, object]] = []
    real_library_papers = paper_library._library_papers

    def tracked_library_papers(*args, **kwargs):
        page_calls.append(dict(kwargs))
        return real_library_papers(*args, **kwargs)

    monkeypatch.setattr(paper_library, "_library_papers", tracked_library_papers)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database.path.as_posix()}")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    monkeypatch.setattr(
        "eduecon_radar.services.product_access.local_request_allowed", lambda _host: True
    )
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    assert not app.exception
    app.segmented_control[0].set_value("论文库").run()
    assert not app.exception
    rendered = "\n".join(str(item.value) for item in app.markdown)
    assert "共 21 篇 · 第 1/2 页 · 当前显示 20 篇" in rendered
    assert rendered.count('<div class="edr-library-card">') == 20
    assert len(page_calls) == 1
    assert page_calls[0]["limit"] == 20
    assert page_calls[0]["offset"] == 0
    assert hashlib.sha256(database.path.read_bytes()).hexdigest() == before


def test_visual_tests_never_target_production_database(tmp_path: Path) -> None:
    database = _database(tmp_path)
    assert database.path.parent == tmp_path
    assert database.path.resolve() != (ROOT / "data/eduecon_radar.db").resolve()
