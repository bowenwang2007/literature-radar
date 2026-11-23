import pytest

from eduecon_radar.config import Settings


def test_default_sqlite_url_is_relative_to_project(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    settings = Settings.load(tmp_path)
    assert settings.database_path == (tmp_path / "data" / "eduecon_radar.db").resolve()


def test_windows_style_sqlite_path_is_supported(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///data/custom.db")
    settings = Settings.load(tmp_path)
    assert settings.database_path == (tmp_path / "data" / "custom.db").resolve()


def test_windows_dotenv_first_line_with_bom(tmp_path, monkeypatch):
    monkeypatch.setattr("os.environ", {})
    (tmp_path / ".env").write_text("DATABASE_URL=sqlite:///notes.db\n", encoding="utf-8-sig")
    assert Settings.load(tmp_path).database_path == (tmp_path / "notes.db").resolve()


def test_default_ui_version_is_v05(tmp_path, monkeypatch):
    monkeypatch.delenv("EDUECON_UI_VERSION", raising=False)
    assert Settings.load(tmp_path).ui_version == "v05"

    monkeypatch.setenv("EDUECON_UI_VERSION", "v04")
    assert Settings.load(tmp_path).ui_version == "v04"


def test_process_environment_wins_over_dotenv_and_semantic_limits_are_valid(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "DEEPSEEK_MODEL_FAST=dotenv-model\nDEEPSEEK_TIMEOUT_SECONDS=30\n", encoding="utf-8"
    )
    monkeypatch.setenv("DEEPSEEK_MODEL_FAST", "process-model")
    monkeypatch.setenv("DEEPSEEK_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("DEEPSEEK_MAX_RETRIES", "3")
    settings = Settings.load(tmp_path)
    assert settings.deepseek_model_fast == "process-model"
    assert settings.deepseek_timeout_seconds == 12
    assert settings.deepseek_max_retries == 3


def test_invalid_semantic_timeout_and_retries_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="TIMEOUT"):
        Settings.load(tmp_path)
    monkeypatch.setenv("DEEPSEEK_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("DEEPSEEK_MAX_RETRIES", "0")
    with pytest.raises(ValueError, match="MAX_RETRIES"):
        Settings.load(tmp_path)


def test_domain_profile_defaults_to_education_economics(tmp_path, monkeypatch):
    monkeypatch.delenv("EDUERADAR_DOMAIN_PROFILE", raising=False)
    settings = Settings.load(tmp_path)

    assert settings.domain_profile == "education_economics"
    # With no project-local config the packaged resource must be used, so the
    # default still resolves to the shipped domain package.
    assert settings.domain_config_dir.name == "education_economics"
    assert settings.journal_registry_path is not None
    # Journals are global: the profile selector no longer picks a second
    # per-domain journal file.
    assert settings.journal_registry_path.name == "catalog.json"
    assert settings.screening_prompts_dir.name == "prompts"


def test_domain_profile_selects_project_local_config(tmp_path, monkeypatch):
    (tmp_path / "config" / "domains" / "labor_economics").mkdir(parents=True)
    (tmp_path / "config" / "journals").mkdir(parents=True)
    (tmp_path / "config" / "journals" / "catalog.json").write_text(
        '{"catalog_version": "test", "journals": []}', encoding="utf-8"
    )
    monkeypatch.setenv("EDUERADAR_DOMAIN_PROFILE", "labor_economics")

    settings = Settings.load(tmp_path)

    assert settings.domain_profile == "labor_economics"
    assert settings.domain_config_dir == (
        tmp_path / "config" / "domains" / "labor_economics"
    )
    # One catalog for every profile; the profile selects from it instead of
    # owning a copy of it.
    assert settings.journal_registry_path == (
        tmp_path / "config" / "journals" / "catalog.json"
    )


@pytest.mark.parametrize("value", ["../evil", "a/b", "a\\b", "..", "", "   "])
def test_domain_profile_rejects_path_traversal_and_blanks(tmp_path, monkeypatch, value):
    monkeypatch.setenv("EDUERADAR_DOMAIN_PROFILE", value)

    with pytest.raises(ValueError, match="EDUERADAR_DOMAIN_PROFILE"):
        Settings.load(tmp_path)
