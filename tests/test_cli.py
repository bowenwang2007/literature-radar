import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from eduecon_radar import cli


class FakeDatabase:
    def __init__(self, path):
        self.path = path

    def init_schema(self):
        pass


class FakeSyncService:
    def __init__(self, database, crossref, unpaywall):
        pass

    def sync(self, journal_file, **kwargs):
        return {
            "journals_checked": 2,
            "records_fetched": 5,
            "records_inserted": 3,
            "records_updated": 2,
            "records_failed": 0,
        }


def test_sync_only_prints_metadata_statistics(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "Database", FakeDatabase)
    monkeypatch.setattr(cli, "SyncService", FakeSyncService)
    monkeypatch.setattr(cli, "CrossrefClient", lambda: object())
    monkeypatch.setattr(cli, "load_journals", lambda path: object())
    monkeypatch.setattr(
        cli,
        "_screening_service",
        lambda *args, **kwargs: pytest.fail("sync must not create a screening service"),
    )
    settings = SimpleNamespace(
        database_path=tmp_path / "radar.db",
        unpaywall_email=None,
        journals_path=tmp_path / "journals.yaml",
    )
    args = Namespace(command="sync", since=None, journal=None, limit=None, dry_run=False)

    assert cli.run(args, settings) == 0
    output = capsys.readouterr().out
    assert "journals_checked" in output
    assert "records_fetched" in output
    assert "records_inserted" in output
    assert "records_updated" in output
    assert "records_failed" in output
    assert "screened" not in output
    assert "ai_calls" not in output


def test_sync_parser_rejects_removed_no_ai_option():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["sync", "--no-ai"])


def test_screen_command_remains_available():
    args = cli.build_parser().parse_args(["screen", "--no-ai"])
    assert args.command == "screen"
    assert args.no_ai is True


def test_export_defaults_to_safe_mentor_mode():
    args = cli.build_parser().parse_args(["export", "--format", "csv"])

    assert args.mode == "mentor"
    assert args.status == "approved"
    assert args.include_human_note is False


def test_personal_export_requires_explicit_mode():
    args = cli.build_parser().parse_args(
        ["export", "--format", "markdown", "--mode", "personal"]
    )

    assert args.mode == "personal"


def test_translate_defaults_to_approved_small_missing_only_batch():
    args = cli.build_parser().parse_args(["translate"])

    assert args.status == "approved"
    assert args.limit == 10
    assert args.missing_only is True
    assert args.dry_run is False
    assert args.confirm_ai_cost is False


def test_workflow_migration_requires_explicit_backup_confirmation():
    args = cli.build_parser().parse_args(["migrate-workflows"])

    assert args.command == "migrate-workflows"
    assert args.backup_confirmed is False


def test_archive_command_requires_destination_and_explicit_confirmation():
    args = cli.build_parser().parse_args(
        ["archive-production-database", "--archive-dir", "archives"]
    )

    assert args.command == "archive-production-database"
    assert args.label == "v0.4.1-frozen"
    assert args.confirm_read_only_source is False


def test_stats_prints_explicit_screening_and_review_fields(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "Database", FakeDatabase)
    expected = {
        "awaiting_screening": 4,
        "screening_failed": 1,
        "ai_excluded": 3,
        "ai_included": 2,
        "ai_borderline": 1,
        "review_pending": 3,
        "approved": 1,
        "rejected": 1,
        "needs_revision": 1,
        "total": 11,
        "records_inserted": 5,
        "ai_calls": 3,
        "input_tokens": 100,
        "output_tokens": 50,
    }
    monkeypatch.setattr(cli, "dashboard_stats", lambda database: expected)
    settings = SimpleNamespace(database_path=tmp_path / "radar.db")

    assert cli.run(Namespace(command="stats"), settings) == 0

    output = json.loads(capsys.readouterr().out)
    assert output == expected
    assert "pending" not in output
