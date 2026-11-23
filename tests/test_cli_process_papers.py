"""CLI continuation entry point for the product processing chain.

An interrupted batch previously had no scriptable resume path: the workbench
button acts only on the currently selected retrieval run, and the legacy
``screen`` command writes to a different table.  These tests pin the new
command's safety properties: preview by default, an explicit gate before any
network/AI work, and no second implementation of batch membership.
"""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from eduecon_radar import cli
from eduecon_radar.db import Database
from eduecon_radar.schemas import PaperMetadata
from eduecon_radar.services.local_pipeline import (
    ensure_paper_statuses,
    latest_sync_run,
)
from eduecon_radar.services.migrations import (
    apply_overnight_migration,
    apply_product_safety_migration,
    apply_profile_migration,
    apply_translation_migration,
    apply_verification_intake_migration,
    apply_workflow_migration,
)
from eduecon_radar.services.personal_mode import PersonalJournalSyncService

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- helpers
def _database(tmp_path: Path) -> Database:
    path = tmp_path / "cli_process.db"
    if path.exists():
        # A test that seeds twice must reuse the same database, otherwise the
        # later seed silently writes nowhere and assertions get confusing.
        return Database(path)
    database = Database(path)
    database.init_schema()
    apply_translation_migration(database, ROOT / "migrations/0001_add_paper_translations.sql")
    apply_workflow_migration(database, ROOT / "migrations/0002_add_workflow_batches.sql")
    apply_overnight_migration(database, ROOT / "migrations/0010_add_overnight_pipeline.sql")
    apply_product_safety_migration(
        database, ROOT / "migrations/0011_add_product_safety_and_verification.sql"
    )
    apply_verification_intake_migration(
        database, ROOT / "migrations/0012_add_verification_intake.sql"
    )
    apply_profile_migration(
        database, ROOT / "migrations/0013_add_profile_scoped_assessment.sql"
    )
    return database


class _Journal:
    journal_key = "economics_of_education_review"
    canonical_name = "Economics of Education Review"
    print_issn = "0272-7757"
    online_issn = "1873-7382"
    issns = (print_issn, online_issn)

    def as_journal_config(self):
        from eduecon_radar.schemas import JournalConfig

        return JournalConfig(
            name=self.canonical_name,
            short_name="EER",
            issn=self.print_issn,
            eissn=self.online_issn,
            priority="high",
        )


class _Crossref:
    def __init__(self, records):
        self.records = list(records)
        self.last_fetch_stats = {
            "provider_returned_count": len(self.records),
            "pages_requested": 1,
            "raw_candidate_count": len(self.records),
            "stopped_reason": "provider_exhausted",
        }

    def fetch(self, _issn, _since, limit, *, until=None):
        del until
        return list(self.records[:limit])


def _paper(doi: str, title: str, abstract: str | None = None) -> PaperMetadata:
    return PaperMetadata(
        doi=doi,
        title=title,
        abstract=abstract,
        authors=["Researcher"],
        issue_date=date(2026, 6, 1),
        source_metadata={"DOI": doi, "ISSN": ["0272-7757"]},
    )


def _seed(database: Database, papers: list[PaperMetadata]) -> int:
    """Create one sync run whose papers are all still unprocessed."""
    PersonalJournalSyncService(database, _Crossref(papers)).sync(
        (_Journal(),),
        since=date(2026, 5, 1),
        until=date(2026, 8, 1),
        per_journal_limit=100,
    )
    run = latest_sync_run(database)
    assert run is not None
    run_id = int(run["id"])
    ensure_paper_statuses(database, run_id)
    return run_id


def _sync_one_and_process(database: Database) -> tuple[int, int]:
    """Seed one paper and drive it to a judged state without any provider call."""
    from eduecon_radar.services.local_pipeline import (
        sync_run_papers,
    )

    run_id = _seed(database, [_paper("10.1000/done", "Done paper", abstract="Abstract.")])
    papers = sync_run_papers(database, run_id)
    paper_id = int(papers[0]["id"])
    with database.connect() as connection:
        connection.execute(
            """UPDATE paper_processing_status
            SET status='classified', stage='complete',
                model_name='deepseek-v4-pro', prompt_version='test',
                classification_json=?
            WHERE sync_run_id=? AND paper_id=?""",
            (
                json.dumps(
                    {"decision": "relevant", "screening_basis": "title_and_abstract"}
                ),
                run_id,
                paper_id,
            ),
        )
        connection.execute(
            """INSERT INTO paper_translations
            (paper_id,chinese_abstract,abstract_translation_status,
             abstract_translation_source,updated_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(paper_id) DO UPDATE SET
              chinese_abstract=excluded.chinese_abstract,
              abstract_translation_status=excluded.abstract_translation_status""",
            (paper_id, "中文摘要", "ai_generated", "ai_generated", "2026-08-03T18:00:00+00:00"),
        )
    return run_id, paper_id


def _settings(tmp_path: Path, *, api_key: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        database_path=tmp_path / "cli_process.db",
        deepseek_api_key=api_key,
    )


# ----------------------------------------------------------------------- tests
def test_parser_defaults_to_preview_and_requires_explicit_yes() -> None:
    args = cli.build_parser().parse_args(["process-papers"])

    assert args.command == "process-papers"
    assert args.yes is False
    assert args.sync_run_id is None
    assert args.limit is None


def test_preview_lists_papers_without_any_ai_or_provider_call(
    monkeypatch, capsys, tmp_path
) -> None:
    database = _database(tmp_path)
    run_id = _seed(
        database,
        [_paper("10.1000/cli-a", "Paper A"), _paper("10.1000/cli-b", "Paper B")],
    )

    # Any attempt to build the live pipeline is a hard failure in preview mode.
    def forbidden(*_args, **_kwargs):
        raise AssertionError("preview must not build live services")

    monkeypatch.setattr(
        "eduecon_radar.services.pipeline_builder.build_local_pipeline_services", forbidden
    )

    args = Namespace(command="process-papers", sync_run_id=None, limit=None, yes=False)
    assert cli.run(args, _settings(tmp_path, api_key="test-key")) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "preview"
    assert payload["sync_run_id"] == run_id
    assert payload["papers"] == 2
    assert len(payload["paper_ids"]) == 2


def test_limit_truncates_the_selected_papers(monkeypatch, capsys, tmp_path) -> None:
    database = _database(tmp_path)
    _seed(
        database,
        [
            _paper("10.1000/cli-a", "Paper A"),
            _paper("10.1000/cli-b", "Paper B"),
            _paper("10.1000/cli-c", "Paper C"),
        ],
    )

    args = Namespace(command="process-papers", sync_run_id=None, limit=2, yes=False)
    assert cli.run(args, _settings(tmp_path, api_key="test-key")) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["papers"] == 2


def test_already_processed_run_reports_nothing_to_do(monkeypatch, capsys, tmp_path) -> None:
    database = _database(tmp_path)
    run_id = _seed(database, [_paper("10.1000/cli-a", "Paper A")])
    with database.connect() as connection:
        connection.execute(
            "UPDATE paper_processing_status SET status='classified' WHERE sync_run_id=?",
            (run_id,),
        )

    args = Namespace(command="process-papers", sync_run_id=None, limit=None, yes=False)
    assert cli.run(args, _settings(tmp_path, api_key="test-key")) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "nothing_to_do"
    assert payload["papers"] == 0


def test_execution_refuses_without_api_key(monkeypatch, tmp_path) -> None:
    database = _database(tmp_path)
    _seed(database, [_paper("10.1000/cli-a", "Paper A")])

    args = Namespace(command="process-papers", sync_run_id=None, limit=None, yes=True)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        cli.run(args, _settings(tmp_path, api_key=None))


def test_unknown_sync_run_is_rejected(monkeypatch, tmp_path) -> None:
    _database(tmp_path)

    args = Namespace(command="process-papers", sync_run_id=4242, limit=None, yes=False)
    with pytest.raises(ValueError, match="检索批次不存在"):
        cli.run(args, _settings(tmp_path, api_key="test-key"))


def test_intermediate_stages_are_still_resumable(tmp_path) -> None:
    """A paper stranded mid-chain must stay reachable by continuation.

    abstract_ready / classification_ready are set *before* translation and
    screening.  If the process stops in between, the paper is not terminal, and
    a continuation that ignored these stages would strand it forever -- which
    is what happened to 6 papers in production.
    """
    from eduecon_radar.services.local_pipeline import (
        RESUMABLE_PROCESSING_STATUSES,
        resumable_paper_ids,
    )

    assert {"pending", "processing", "needs_retry"} <= RESUMABLE_PROCESSING_STATUSES
    assert "abstract_ready" in RESUMABLE_PROCESSING_STATUSES
    assert "classification_ready" in RESUMABLE_PROCESSING_STATUSES
    # Terminal states must never be re-sent for processing.
    assert "classified" not in RESUMABLE_PROCESSING_STATUSES
    assert "needs_manual_abstract" not in RESUMABLE_PROCESSING_STATUSES

    database = _database(tmp_path)
    run_id = _seed(
        database,
        [
            _paper("10.1000/cli-a", "Paper A"),
            _paper("10.1000/cli-b", "Paper B"),
            _paper("10.1000/cli-c", "Paper C"),
            _paper("10.1000/cli-d", "Paper D"),
            _paper("10.1000/cli-e", "Paper E"),
        ],
    )
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT paper_id FROM paper_processing_status WHERE sync_run_id=? ORDER BY paper_id",
            (run_id,),
        ).fetchall()
    ids = [int(row["paper_id"]) for row in rows]
    stranded_states = [
        "abstract_ready",
        "classification_ready",
        "classified",
        "needs_manual_abstract",
        "pending",
    ]
    for paper_id, status in zip(ids, stranded_states, strict=True):
        with database.connect() as connection:
            connection.execute(
                "UPDATE paper_processing_status SET status=? WHERE sync_run_id=? AND paper_id=?",
                (status, run_id, paper_id),
            )

    resumable = resumable_paper_ids(database, run_id)

    assert set(resumable) == {ids[0], ids[1], ids[4]}
    assert ids[2] not in resumable  # classified
    assert ids[3] not in resumable  # needs_manual_abstract

    # And the CLI must select exactly that set.
    assert cli._selectable_paper_ids(database, run_id) == resumable


def test_human_decision_is_paper_scoped_across_runs(tmp_path) -> None:
    """Human review belongs to the *paper*, not the retrieval batch.

    This pins deliberate behaviour, not an accident: production notes say
    "Human review belongs to the paper, even when retrieval creates a new run",
    and ``ensure_paper_statuses_for_paper_ids`` copies saved review data into a
    new run.  An earlier attempt to scope the UPDATE to one sync run broke this
    on purpose-built test coverage and was reverted.
    """
    from eduecon_radar.services.local_pipeline import (
        ensure_paper_statuses_for_paper_ids,
        save_human_decisions,
        sync_run_papers,
    )

    database = _database(tmp_path)
    run_a = _seed(database, [_paper("10.1000/shared", "Shared paper")])
    with database.connect() as connection:
        paper_id = int(
            connection.execute(
                "SELECT paper_id FROM paper_processing_status WHERE sync_run_id=?",
                (run_a,),
            ).fetchone()["paper_id"]
        )
    _seed(database, [_paper("10.1000/other", "Other paper")])
    with database.connect() as connection:
        run_b = int(connection.execute("SELECT MAX(id) FROM sync_runs").fetchone()[0])
    ensure_paper_statuses_for_paper_ids(database, run_b, (paper_id,))

    saved = save_human_decisions(database, run_a, {paper_id: ("related", "keep")})
    assert saved == (paper_id,)

    # Every processing row for this paper carries the decision, so the library
    # and the newest run cannot disagree.
    assert sync_run_papers(database, run_a)[0]["human_decision"] == "related"
    run_b_decision = {
        int(p["id"]): p["human_decision"] for p in sync_run_papers(database, run_b)
    }
    assert run_b_decision.get(paper_id) == "related"


def test_lifecycle_check_reports_abstract_without_judgment(tmp_path) -> None:
    """The 6 stranded production papers must be reported, not silently ignored."""
    from eduecon_radar.services.local_pipeline import lifecycle_invariant_report

    database = _database(tmp_path)
    run_id = _seed(
        database,
        [_paper("10.1000/stranded", "Stranded paper", abstract="Has an abstract.")],
    )
    with database.connect() as connection:
        paper_id = int(
            connection.execute(
                "SELECT paper_id FROM paper_processing_status WHERE sync_run_id=?",
                (run_id,),
            ).fetchone()["paper_id"]
        )
        # Reproduce the production shape: abstract present, translated, but the
        # status sits in an intermediate stage and no judgment was written.
        connection.execute(
            "UPDATE paper_processing_status SET status='abstract_ready',"
            " stage='abstract_translation' WHERE sync_run_id=? AND paper_id=?",
            (run_id, paper_id),
        )
        connection.execute(
            """INSERT INTO paper_translations
            (paper_id,chinese_abstract,abstract_translation_status,
             abstract_translation_source,updated_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(paper_id) DO UPDATE SET
              chinese_abstract=excluded.chinese_abstract,
              abstract_translation_status=excluded.abstract_translation_status""",
            (paper_id, "中文摘要", "ai_generated", "ai_generated", "2026-08-03T17:39:00+00:00"),
        )

    report = lifecycle_invariant_report(database, run_id)

    kinds = {v["kind"] for v in report["violations"]}
    assert "abstract_without_judgment" in kinds
    assert "finishable_status_stalled" in kinds
    assert report["violation_count"] >= 2

    # And it must be selectable for continuation rather than stranded.
    assert paper_id in cli._selectable_paper_ids(database, run_id)


def test_lifecycle_check_is_clean_once_judged(tmp_path) -> None:
    from eduecon_radar.services.local_pipeline import lifecycle_invariant_report

    database = _database(tmp_path)
    run_id, _paper_id = _sync_one_and_process(database)

    report = lifecycle_invariant_report(database, run_id)

    assert report["violation_count"] == 0


def test_execution_targets_only_selectable_papers(monkeypatch, capsys, tmp_path) -> None:
    """--yes must hand the pipeline exactly the unfinished papers of one run."""
    database = _database(tmp_path)
    run_id = _seed(
        database,
        [
            _paper("10.1000/cli-a", "Paper A"),
            _paper("10.1000/cli-b", "Paper B"),
            _paper("10.1000/cli-c", "Paper C"),
        ],
    )
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT paper_id FROM paper_processing_status WHERE sync_run_id=? ORDER BY paper_id",
            (run_id,),
        ).fetchall()
    ids = [int(row["paper_id"]) for row in rows]
    # Mark the middle paper as already complete: it must not be re-sent.
    with database.connect() as connection:
        connection.execute(
            "UPDATE paper_processing_status SET status='classified' "
            "WHERE sync_run_id=? AND paper_id=?",
            (run_id, ids[1]),
        )

    captured: dict[str, object] = {}

    def fake_process_batch(database, run_id_arg, paper_ids, services, **_kwargs):
        captured["run_id"] = run_id_arg
        captured["paper_ids"] = tuple(paper_ids)
        with database.connect() as connection:
            for paper_id in paper_ids:
                connection.execute(
                    "UPDATE paper_processing_status SET status='classified' "
                    "WHERE sync_run_id=? AND paper_id=?",
                    (run_id_arg, paper_id),
                )
        return {
            "completed": list(paper_ids),
            "needs_abstract": [],
            "needs_retry": [],
            "ai_calls": 2,
            "input_tokens": 10,
            "output_tokens": 5,
        }

    monkeypatch.setattr(
        "eduecon_radar.services.local_pipeline.process_batch", fake_process_batch
    )
    monkeypatch.setattr(
        "eduecon_radar.services.pipeline_builder.build_local_pipeline_services",
        lambda database, settings, profile=None: SimpleNamespace(enrichment=object()),
    )

    args = Namespace(command="process-papers", sync_run_id=run_id, limit=None, yes=True)
    assert cli.run(args, _settings(tmp_path, api_key="test-key")) == 0

    assert captured["run_id"] == run_id
    assert captured["paper_ids"] == (ids[0], ids[2])

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "processed"
    assert payload["requested"] == 2
    assert payload["completed"] == 2
    assert payload["ai_calls"] == 2
    # status_after covers only the two papers this run was asked to process.
    assert payload["status_after"] == {"classified": 2}
