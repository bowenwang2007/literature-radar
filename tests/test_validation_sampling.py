from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from eduecon_radar import cli
from eduecon_radar.db import Database, utcnow
from eduecon_radar.schemas import JournalConfig, PaperMetadata
from eduecon_radar.services.archive import sha256_file
from eduecon_radar.services.validation_batch import (
    ValidationRepository,
    apply_validation_migration,
    stratified_sample,
)

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "migrations" / "0006_add_validation_batches.sql"


def _candidates() -> list[dict]:
    labels = ["include"] * 30 + ["borderline"] * 30 + ["exclude"] * 40
    return [
        {"source_paper_id": index + 1, "ai_decision": label}
        for index, label in enumerate(labels)
    ]


def _source_database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "production.db")
    database.init_schema()
    journal_id = database.upsert_journal(
        JournalConfig(name="Validation Journal", short_name="VJ", issn="0000-0000")
    )
    labels = ["include"] * 3 + ["borderline"] * 3 + ["exclude"] * 4
    for index, label in enumerate(labels):
        paper_id, _ = database.upsert_paper(
            PaperMetadata(
                doi=f"10.1000/validation-{index}",
                title=f"Validation paper {index}",
                abstract=f"Original abstract {index}",
                authors=["Researcher"],
                issue_date=date(2025, 1, index + 1),
            ),
            journal_id,
        )
        with database.connect() as connection:
            connection.execute(
                """INSERT INTO screenings
                (paper_id,ai_decision,ai_confidence,ai_reason,
                 model_name,prompt_version,screened_at)
                VALUES (?,?,?,?,?,?,?)""",
                (paper_id, label, 0.8, f"reason-{index}", "fixture", "v1", utcnow()),
            )
    return database


def test_validation_sampling_is_stratified_and_has_no_duplicates():
    selected = stratified_sample(_candidates(), size=100, random_seed=42)

    assert len(selected) == 100
    assert len({item["source_paper_id"] for item in selected}) == 100
    assert sum(item["ai_decision"] == "include" for item in selected) == 30
    assert sum(item["ai_decision"] == "borderline" for item in selected) == 30
    assert sum(item["ai_decision"] == "exclude" for item in selected) == 40


def test_real_sample_uses_read_only_source_and_independent_validation_db(tmp_path):
    source = _source_database(tmp_path)
    before = (sha256_file(source.path), source.path.stat().st_mtime_ns)
    validation = Database(tmp_path / "validation.db")
    repository = ValidationRepository(validation)
    repository.initialize(MIGRATION)

    result = repository.create_sample(
        source_database=source.path,
        journals=("VJ",),
        year_from=2025,
        year_to=2025,
        size=10,
        random_seed=7,
    )

    assert result.stratum_counts == {"include": 3, "borderline": 3, "exclude": 4}
    assert len(repository.list_items(result.batch.batch_id, page_size=20)[0]) == 10
    assert (sha256_file(source.path), source.path.stat().st_mtime_ns) == before
    with source.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 10
        assert connection.execute("SELECT COUNT(*) FROM screenings").fetchone()[0] == 10


def test_validation_annotation_is_recoverable_and_does_not_change_ai_snapshot(tmp_path):
    source = _source_database(tmp_path)
    repository = ValidationRepository(Database(tmp_path / "validation.db"))
    repository.initialize(MIGRATION)
    batch = repository.create_sample(source_database=source.path, size=3).batch
    item = repository.list_items(batch.batch_id, page_size=20)[0][0]

    repository.save_annotation(
        item["id"],
        label="include",
        reviewer="Human",
        notes="first pass",
        difficulty="hard",
    )
    restored = repository.list_items(
        batch.batch_id, page_size=20, annotated=True
    )[0][0]

    assert restored["human_label"] == "include"
    assert restored["reviewer"] == "Human"
    assert restored["ai_decision"] == item["ai_decision"]


def test_validation_migration_failure_rolls_back(tmp_path):
    database = Database(tmp_path / "validation.db")
    broken = tmp_path / "broken.sql"
    broken.write_text(
        "BEGIN IMMEDIATE; CREATE TABLE rollback_probe(id INTEGER); INVALID SQL; COMMIT;",
        encoding="utf-8",
    )

    with pytest.raises(sqlite3.OperationalError):
        apply_validation_migration(database, broken)

    with database.connect() as connection:
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='rollback_probe'"
        ).fetchone() is None


def test_validation_cli_commands_and_safe_defaults():
    sample = cli.build_parser().parse_args(
        [
            "create-validation-sample",
            "--journal",
            "Validation Journal",
            "--year-from",
            "2024",
            "--year-to",
            "2025",
            "--size",
            "100",
        ]
    )
    report = cli.build_parser().parse_args(
        ["validate-screening", "--validation-batch-id", "7"]
    )

    assert sample.validation_db == Path("data/validation.db")
    assert sample.journal == ["Validation Journal"]
    assert report.validation_batch_id == 7
    assert report.validation_db == Path("data/validation.db")


def test_validation_cli_rejects_source_database_as_validation_database(tmp_path):
    source = tmp_path / "source.db"
    source.touch()
    args = cli.build_parser().parse_args(
        ["create-validation-sample", "--validation-db", str(source)]
    )
    settings = SimpleNamespace(project_root=tmp_path, database_path=source)

    with pytest.raises(ValueError, match="must be separate"):
        cli.run(args, settings)


def test_validation_cli_creates_template_and_json_markdown_reports(
    tmp_path, capsys
):
    source = _source_database(tmp_path)
    validation_path = tmp_path / "validation.db"
    template_path = tmp_path / "annotations.csv"
    settings = SimpleNamespace(project_root=ROOT, database_path=source.path)
    create_args = cli.build_parser().parse_args(
        [
            "create-validation-sample",
            "--size",
            "3",
            "--validation-db",
            str(validation_path),
            "--output",
            str(template_path),
        ]
    )

    assert cli.run(create_args, settings) == 0
    created = json.loads(capsys.readouterr().out)
    batch_id = int(created["batch"]["batch_id"])
    assert template_path.is_file()
    assert created["ai_calls"] == 0
    assert created["source_database_mode"] == "read-only"

    repository = ValidationRepository(Database(validation_path))
    items, _ = repository.list_items(batch_id, page_size=20)
    for item in items:
        repository.save_annotation(
            int(item["id"]),
            label=item["ai_decision"],
            reviewer="Test reviewer",
            difficulty="easy",
        )

    report_args = cli.build_parser().parse_args(
        [
            "validate-screening",
            "--validation-batch-id",
            str(batch_id),
            "--validation-db",
            str(validation_path),
            "--output-dir",
            str(tmp_path / "reports"),
        ]
    )
    assert cli.run(report_args, settings) == 0
    reported = json.loads(capsys.readouterr().out)

    assert Path(reported["outputs"]["json"]).is_file()
    assert Path(reported["outputs"]["markdown"]).is_file()
    assert reported["performance"]["false_negative"] == 0
