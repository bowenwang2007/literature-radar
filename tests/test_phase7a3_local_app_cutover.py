from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from eduecon_radar.db import Database
from eduecon_radar.services.crossref import CrossrefClient
from eduecon_radar.services.deepseek import DeepSeekClient
from eduecon_radar.services.migrations import (
    apply_p0_hardening_migration,
    apply_translation_migration,
    apply_workflow_migration,
)

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src/eduecon_radar/ui/app.py"


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "local-app.db")
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


def test_run_web_starts_only_the_shipped_local_app() -> None:
    script = (ROOT / "run_web.ps1").read_text(encoding="utf-8")
    assert '"src\\eduecon_radar\\ui\\app.py"' in script
    assert "legacy_app.py" not in script
    assert "literature_workflow" not in script
    assert "pilot" not in script.casefold()
    assert "released" not in script.casefold()
    assert "Literature Radar Local App" in script
    assert "http://localhost:8501/" in script
    assert 'Get-Content' not in script  # The application reads its own credentials.
    assert '$env:LITERATURE_RADAR_MODE = $PreviousMode' in script


def test_shipped_app_has_no_active_pilot_or_legacy_imports() -> None:
    source = APP.read_text(encoding="utf-8")
    forbidden = (
        "literature_workflow",
        "PilotAuthorizationService",
        "TaskLeaseManager",
        "RuntimeBundleResolver",
        "pilot_runtime",
        "released/current.json",
        "EDUERADAR_PERSONAL_MODE",
        "_latest_task",
    )
    assert not [value for value in forbidden if value in source]
    assert 'LOCAL_PAGES = ["文献工作台"]' in source


def test_local_main_never_touches_retired_guards_or_latest_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eduecon_radar.services.pilot_runtime import PilotAuthorizationService
    from eduecon_radar.services.runtime_bundle import RuntimeBundleResolver
    from eduecon_radar.services.task_lock import TaskLeaseManager
    from eduecon_radar.ui import app as app_module

    database = _database(tmp_path)

    class Sidebar:
        def radio(self, _label, options, **_kwargs):
            return options[0]

    class FakeStreamlit:
        sidebar = Sidebar()

        def __init__(self) -> None:
            self.errors: list[str] = []

        def error(self, value) -> None:
            self.errors.append(str(value))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("retired runtime guard must not be called")

    settings = SimpleNamespace(
        database_path=database.path,
        project_root=ROOT,
        domain_config_dir=ROOT / "config/domains/education_economics",
    )
    fake = FakeStreamlit()
    calls: list[str] = []
    monkeypatch.setattr(
        app_module.Settings,
        "load",
        classmethod(lambda cls, project_root=None: settings),
    )
    monkeypatch.setattr(app_module, "st", fake)
    monkeypatch.setattr(app_module, "_request_host", lambda: "localhost:8501")
    monkeypatch.setattr(
        app_module.personal_mode,
        "render",
        lambda *_args: calls.append("personal"),
    )
    monkeypatch.setattr(PilotAuthorizationService, "__init__", forbidden)
    monkeypatch.setattr(TaskLeaseManager, "__init__", forbidden)
    monkeypatch.setattr(RuntimeBundleResolver, "resolve", forbidden)
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    app_module.main()
    assert calls == ["personal"]
    assert not fake.errors


def test_real_root_entry_is_local_personal_and_render_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _database(tmp_path)
    before = hashlib.sha256(database.path.read_bytes()).hexdigest()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database.path.as_posix()}")
    monkeypatch.delenv("EDUERADAR_PERSONAL_MODE", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    # AppTest has no HTTP Host header; host policy itself is covered separately.
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    monkeypatch.setattr(
        "eduecon_radar.services.product_access.local_request_allowed", lambda _host: True
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("render must not call AI or an external provider")

    monkeypatch.setattr(CrossrefClient, "fetch", forbidden)
    monkeypatch.setattr(DeepSeekClient, "request_json", forbidden)
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    assert not app.exception
    # The product wordmark and the scope line are one identity: a profile-neutral
    # tool name, with the research direction shown as its own field.
    assert [item.value for item in app.title] == ["Literature Radar"]
    assert any(
        item.value == "多研究方向文献发现与人工审核工作台" for item in app.caption
    )
    assert len(app.segmented_control) == 1
    assert app.segmented_control[0].value == "文献工作台"
    rendered_summary = "\n".join(str(item.value) for item in app.markdown)
    for label in ("本次获取", "待补摘要", "待审核", "已完成"):
        assert label in rendered_summary
    assert any("尚未配置 DeepSeek API Key" in item.value for item in app.info)
    assert any(item.label == "获取论文" for item in app.button)
    assert not any(item.label == "只选 EER" for item in app.button)
    rendered = "\n".join(
        str(item.value)
        for group in (app.title, app.caption, app.info, app.warning, app.error, app.markdown)
        for item in group
    )
    forbidden_text = (
        "受控试运行",
        "NOT RELEASED",
        "正式AI模式不可用",
        "authorization",
        "released_bundle_missing",
        "task #16",
        "lease",
        "runtime bundle",
    )
    assert not [value for value in forbidden_text if value.casefold() in rendered.casefold()]
    assert hashlib.sha256(database.path.read_bytes()).hexdigest() == before


def test_cutover_tests_never_use_production_database(tmp_path: Path) -> None:
    database = _database(tmp_path)
    assert database.path.parent == tmp_path
    assert database.path.resolve() != (ROOT / "data/eduecon_radar.db").resolve()
