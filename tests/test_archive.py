from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from eduecon_radar.db import Database
from eduecon_radar.services import archive as archive_module
from eduecon_radar.services.archive import archive_database, sha256_file

FIXED_TIME = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)


def _source_database(tmp_path):
    path = tmp_path / "production.db"
    Database(path).init_schema()
    return path


def _archive(source, archive_dir, **overrides):
    options = {
        "source_path": source,
        "archive_dir": archive_dir,
        "label": "v0.4.1-frozen",
        "confirm_read_only_source": True,
        "current_git_commit": "abc123",
        "logical_application_version": "v0.4.1",
        "now": FIXED_TIME,
    }
    options.update(overrides)
    return archive_database(**options)


def test_archive_is_verified_and_source_remains_byte_identical(tmp_path):
    source = _source_database(tmp_path)
    source_hash = sha256_file(source)
    source_stat = source.stat()

    manifest = _archive(source, tmp_path / "archives")

    assert sha256_file(source) == source_hash == manifest.sha256
    assert sha256_file(type(source)(manifest.archive_path)) == source_hash
    assert source.stat().st_size == source_stat.st_size
    assert source.stat().st_mtime_ns == source_stat.st_mtime_ns
    assert manifest.integrity_result == "ok"


def test_archive_manifest_contains_complete_traceability_fields(tmp_path):
    source = _source_database(tmp_path)
    manifest = _archive(source, tmp_path / "archives")
    payload = json.loads(type(source)(manifest.manifest_path).read_text(encoding="utf-8"))

    assert set(payload) == {
        "source_path",
        "archive_path",
        "manifest_path",
        "created_at",
        "file_size",
        "sha256",
        "integrity_result",
        "table_counts",
        "current_git_commit",
        "logical_application_version",
    }
    assert payload["current_git_commit"] == "abc123"
    assert payload["logical_application_version"] == "v0.4.1"
    assert payload["table_counts"]["papers"] == 0


def test_archive_refuses_overwrite(tmp_path):
    source = _source_database(tmp_path)
    archive_dir = tmp_path / "archives"
    _archive(source, archive_dir)

    with pytest.raises(FileExistsError, match="already exists"):
        _archive(source, archive_dir)


def test_archive_requires_explicit_read_only_confirmation(tmp_path):
    source = _source_database(tmp_path)

    with pytest.raises(ValueError, match="confirm-read-only-source"):
        _archive(source, tmp_path / "archives", confirm_read_only_source=False)


def test_copy_failure_does_not_modify_source_or_leave_partial_archive(
    tmp_path, monkeypatch
):
    source = _source_database(tmp_path)
    before = (sha256_file(source), source.stat().st_mtime_ns)

    def fail_copy(*args, **kwargs):
        raise OSError("simulated copy failure")

    monkeypatch.setattr(archive_module.shutil, "copyfileobj", fail_copy)
    archive_dir = tmp_path / "archives"
    with pytest.raises(OSError, match="simulated"):
        _archive(source, archive_dir)

    assert (sha256_file(source), source.stat().st_mtime_ns) == before
    assert list(archive_dir.iterdir()) == []


def test_archive_does_not_read_or_modify_dotenv(tmp_path):
    source = _source_database(tmp_path)
    env = tmp_path / ".env"
    env.write_text("DEEPSEEK_API_KEY=untouched-test-value\n", encoding="utf-8")
    before = (sha256_file(env), env.stat().st_mtime_ns)

    _archive(source, tmp_path / "archives")

    assert (sha256_file(env), env.stat().st_mtime_ns) == before
