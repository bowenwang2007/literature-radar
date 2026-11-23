import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from eduecon_radar import cli
from eduecon_radar.services.deepseek import DeepSeekDiagnostic, DeepSeekResponse
from eduecon_radar.services.semantic_smoke import (
    run_fast_semantic_smoke,
    semantic_settings_diagnostic,
)


def _settings(tmp_path):
    return SimpleNamespace(
        project_root=tmp_path,
        deepseek_api_key="secret-value",
        deepseek_base_url="https://api.deepseek.example/v1",
        deepseek_model_fast="fast-model",
        deepseek_model_strong="strong-model",
        deepseek_timeout_seconds=12.0,
        deepseek_max_retries=3,
    )


def test_diagnostic_reports_configuration_without_key():
    result = semantic_settings_diagnostic(_settings(Path(".")))
    assert result == {
        "key_configured": True,
        "base_url_host": "api.deepseek.example",
        "fast_model": "fast-model",
        "strong_model": "strong-model",
        "timeout_seconds": 12.0,
        "max_retries": 3,
    }
    assert "secret" not in str(result)


def test_cli_diagnose_is_sanitized_and_never_calls_live_client(monkeypatch, capsys, tmp_path):
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        cli,
        "diagnose_semantic_runs",
        lambda *args, **kwargs: {
            "sample_count": 1,
            "failed_run_samples": [{"error_type": "schema_validation_failed"}],
        },
    )
    args = Namespace(command="diagnose-semantic", database=tmp_path / "db.sqlite", paper_limit=1)
    assert cli.run(args, settings) == 0
    output = capsys.readouterr().out
    assert "schema_validation_failed" in output
    assert "secret-value" not in output


def test_cli_smoke_requires_explicit_confirmation(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    args = Namespace(
        command="semantic-smoke",
        database=tmp_path / "db.sqlite",
        paper_id=1,
        confirm_live_call=False,
    )
    with pytest.raises(ValueError, match="confirm-live-call"):
        cli.run(args, settings)
    monkeypatch.setattr(
        cli,
        "run_fast_semantic_smoke",
        lambda **kwargs: {"status": "succeeded", "paper_id": 1, "run_id": 9},
    )
    args.confirm_live_call = True
    assert cli.run(args, settings) == 0


def test_smoke_parser_has_single_paper_and_confirmation_gate():
    args = cli.build_parser().parse_args(
        ["semantic-smoke", "--database", "data.db", "--paper-id", "7", "--confirm-live-call"]
    )
    assert args.command == "semantic-smoke" and args.paper_id == 7 and args.confirm_live_call
    assert args.debug is False


def test_valid_fast_result_is_persisted_by_full_smoke_path(monkeypatch, tmp_path):
    from test_phase4_semantic import CONFIG, _fast, _prepared_database

    from eduecon_radar.services import semantic_smoke

    database, paper_id = _prepared_database(tmp_path)

    class FakeClient:
        def __init__(self, **kwargs):
            self.last_diagnostic = DeepSeekDiagnostic(200, "req-test", 12, True, True)

        def request_json(self, **kwargs):
            return DeepSeekResponse(
                _fast().model_copy(update={"paper_id": paper_id, "model_name": "fast-model"}),
                "fast-model",
                4,
                2,
            )

    monkeypatch.setattr(semantic_smoke, "DeepSeekClient", FakeClient)
    settings = _settings(tmp_path)
    settings.domain_config_dir = CONFIG
    result = run_fast_semantic_smoke(
        database=database,
        settings=settings,
        paper_id=paper_id,
        allow_unreleased_bundle=True,
    )
    assert result["status"] == "succeeded"
    assert result["diagnostic"]["http_status"] == 200
    with database.connect() as connection:
        row = connection.execute(
            "SELECT result_json,decision,succeeded,failure_type,failure_message "
            "FROM domain_semantic_runs WHERE id=?",
            (result["run_id"],),
        ).fetchone()
    assert row["result_json"] and row["decision"] == "include" and row["succeeded"] == 1
    assert row["failure_type"] is None and row["failure_message"] is None


def test_post_schema_failure_has_stage_and_never_becomes_unexpected_error(monkeypatch, tmp_path):
    from test_phase4_semantic import CONFIG, _fast, _prepared_database

    from eduecon_radar.services import semantic_smoke

    database, paper_id = _prepared_database(tmp_path)
    with database.connect() as connection:
        connection.execute("UPDATE papers SET abstract=NULL WHERE id=?", (paper_id,))

    class FakeClient:
        def __init__(self, **kwargs):
            self.last_diagnostic = DeepSeekDiagnostic(200, None, 12, True, True)

        def request_json(self, **kwargs):
            # Valid Pydantic result, but intentionally violates the no-abstract safety contract.
            return DeepSeekResponse(
                _fast().model_copy(update={"paper_id": paper_id}), "fast-model", 4, 2
            )

    monkeypatch.setattr(semantic_smoke, "DeepSeekClient", FakeClient)
    settings = _settings(tmp_path)
    settings.domain_config_dir = CONFIG
    result = run_fast_semantic_smoke(
        database=database,
        settings=settings,
        paper_id=paper_id,
        debug=True,
        allow_unreleased_bundle=True,
    )
    assert result["status"] == "failed"
    assert result["diagnostic"]["schema_valid"] is True
    assert result["diagnostic"]["error_type"] == "ValueError"
    assert result["diagnostic"]["finalization_stage"] == "schema_validated"
    assert result["error_type"] == "ValueError"
    assert result["debug_traceback"]["exception_type"] == "ValueError"
    with database.connect() as connection:
        row = connection.execute(
            "SELECT succeeded,failure_type,failure_message,result_json FROM domain_semantic_runs "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert tuple(row[:3]) == (
        0,
        "SemanticFinalizationError",
        "semantic_finalization_failed:schema_validated:ValueError",
    )
    diagnostic = json.loads(row["result_json"])
    assert diagnostic["execution_state"] == "retryable_error"
    assert diagnostic["failure_type"] == "SemanticFinalizationError"
    assert diagnostic["model_identity"]["model"] == "fast-model"
    assert diagnostic["runtime_fingerprint"]
    assert "debug_traceback" in result
