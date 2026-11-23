from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from eduecon_radar.db import Database, utcnow
from eduecon_radar.services.migrations import (
    apply_p0_hardening_migration,
    apply_translation_migration,
    apply_workflow_migration,
)
from eduecon_radar.services.personal_mode import (
    list_personal_tasks,
    local_request_allowed,
    personal_mode_allowed,
    personal_session_key,
)
from eduecon_radar.ui import app as app_module
from eduecon_radar.ui.views import personal_mode as personal_view

ROOT = Path(__file__).resolve().parents[1]


class _Sidebar:
    def __init__(self) -> None:
        self.keys: list[str] = []

    def radio(self, _label, options, *, key):
        self.keys.append(key)
        return options[0]


class _RouteStreamlit:
    def __init__(self, query_params: dict[str, str] | None = None) -> None:
        self.query_params = query_params or {}
        self.session_state: dict[str, object] = {"v05_current_task_id": 16}
        self.sidebar = _Sidebar()
        self.errors: list[str] = []

    def error(self, value):
        self.errors.append(str(value))


def _settings(tmp_path: Path) -> SimpleNamespace:
    database_path = tmp_path / "route.db"
    database_path.touch()
    return SimpleNamespace(
        database_path=database_path,
        project_root=ROOT,
        journals_path=ROOT / "journals.yaml",
        domain_config_dir=ROOT / "config/domains/education_economics",
        ui_version="v05",
    )


def _patch_settings(
    monkeypatch: pytest.MonkeyPatch, settings: SimpleNamespace
) -> None:
    monkeypatch.setattr(
        app_module.Settings,
        "load",
        classmethod(lambda cls, project_root=None: settings),
    )


def test_localhost_defaults_root_to_personal_without_mode_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    fake_st = _RouteStreamlit()
    calls: list[str] = []
    _patch_settings(monkeypatch, settings)
    monkeypatch.setattr(app_module, "st", fake_st)
    monkeypatch.setattr(app_module, "_request_host", lambda: "localhost:8501")
    monkeypatch.setattr(
        app_module.personal_mode,
        "render",
        lambda database, loaded: calls.append("personal"),
    )
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    app_module.main()
    assert calls == ["personal"]
    assert fake_st.sidebar.keys == []
    assert fake_st.session_state["v05_current_task_id"] == 16
    assert "personal_current_task_id" not in fake_st.session_state


def test_explicit_personal_route_works_on_localhost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    fake_st = _RouteStreamlit({"mode": "personal"})
    calls: list[str] = []
    _patch_settings(monkeypatch, settings)
    monkeypatch.setattr(app_module, "st", fake_st)
    monkeypatch.setattr(app_module, "_request_host", lambda: "127.0.0.1:8501")
    monkeypatch.setattr(
        app_module.personal_mode,
        "render",
        lambda database, loaded: calls.append("personal"),
    )
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    app_module.main()
    assert calls == ["personal"]


def test_old_query_parameters_do_not_restore_legacy_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    fake_st = _RouteStreamlit({"developer": "pilot", "mode": "production"})
    calls: list[str] = []
    _patch_settings(monkeypatch, settings)
    monkeypatch.setattr(app_module, "st", fake_st)
    monkeypatch.setattr(app_module, "_request_host", lambda: "localhost")
    monkeypatch.setattr(
        app_module.personal_mode,
        "render",
        lambda *args: calls.append("personal"),
    )
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    app_module.main()
    assert calls == ["personal"]
    assert not fake_st.errors


def test_nonlocalhost_rejects_default_personal_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    fake_st = _RouteStreamlit()
    _patch_settings(monkeypatch, settings)
    monkeypatch.setattr(app_module, "st", fake_st)
    monkeypatch.setattr(app_module, "_request_host", lambda: "research.example")
    monkeypatch.setattr(
        app_module.personal_mode,
        "render",
        lambda *args: pytest.fail("remote personal mode must not render"),
    )
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    app_module.main()
    assert any("localhost" in item for item in fake_st.errors)


def test_personal_route_never_calls_latest_workflow_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    fake_st = _RouteStreamlit()
    _patch_settings(monkeypatch, settings)
    monkeypatch.setattr(app_module, "st", fake_st)
    monkeypatch.setattr(app_module, "_request_host", lambda: "localhost")
    monkeypatch.setattr(app_module.personal_mode, "render", lambda *args: None)
    monkeypatch.setenv("LITERATURE_RADAR_MODE", "local")
    app_module.main()
    assert not fake_st.errors


def test_task_16_is_not_a_personal_task_or_auto_loaded(tmp_path: Path) -> None:
    database = Database(tmp_path / "task16.db")
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
    now = utcnow()
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO workflow_batches
            (id,name,source_type,status,source_config_json,screening_policy,
             translation_policy,created_at,updated_at,result_summary_json)
            VALUES (16,'Historical controlled pilot','sync','paused',
                    '{"runtime_mode":"controlled_pilot"}','{}','{}',?,?, '{}')""",
            (now, now),
        )
    before = hashlib.sha256(database.path.read_bytes()).hexdigest()
    assert list_personal_tasks(database) == []
    after = hashlib.sha256(database.path.read_bytes()).hexdigest()
    assert before == after
    with database.connect() as connection:
        row = connection.execute(
            "SELECT status,source_config_json,updated_at FROM workflow_batches WHERE id=16"
        ).fetchone()
    assert row["status"] == "paused"
    assert row["source_config_json"] == '{"runtime_mode":"controlled_pilot"}'
    assert row["updated_at"] == now


def test_personal_session_keys_are_isolated() -> None:
    assert personal_session_key("current_task_id") == "personal_current_task_id"
    assert personal_session_key("current_task_id") != "v05_current_task_id"
    assert personal_mode_allowed(True, None, "localhost")
    assert not personal_mode_allowed(False, None, "localhost")
    assert not personal_mode_allowed(True, None, "remote.example")
    assert local_request_allowed("localhost:8501")
    assert local_request_allowed("127.0.0.1:8501")
    assert not local_request_allowed("remote.example")


def test_personal_default_render_is_read_only_with_legacy_task_16(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = Database(tmp_path / "render.db")
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
    now = utcnow()
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO workflow_batches
            (id,name,source_type,status,source_config_json,screening_policy,
             translation_policy,created_at,updated_at,result_summary_json)
            VALUES (16,'Historical controlled pilot','sync','paused',
                    '{"runtime_mode":"controlled_pilot"}','{}','{}',?,?, '{}')""",
            (now, now),
        )
    before = hashlib.sha256(database.path.read_bytes()).hexdigest()
    fake = _PersonalRenderStreamlit()
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
    source = (ROOT / "src/eduecon_radar/ui/views/personal_mode.py").read_text(
        encoding="utf-8"
    )
    for retired_header_text in (
        "本地运行",
        "数据库已连接",
        "AI已配置",
        "数据仅在本机数据库中处理",
    ):
        assert retired_header_text not in source
    assert fake.metrics == []


class _PersonalRenderStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {"v05_current_task_id": 16}
        self.infos: list[str] = []
        self.markdowns: list[str] = []
        self.metrics: list[tuple[str, int]] = []

    def warning(self, *args, **kwargs):
        return None

    def info(self, value, *args, **kwargs):
        self.infos.append(str(value))

    def title(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def header(self, *args, **kwargs):
        return None

    def markdown(self, value, *args, **kwargs):
        self.markdowns.append(str(value))

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

    def metric(self, label, value, *args, **kwargs):
        self.metrics.append((str(label), int(value)))

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
