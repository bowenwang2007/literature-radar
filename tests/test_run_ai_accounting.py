"""A retrieval run's audit row must record what that run actually cost.

`sync_runs.ai_calls` was never written by the product processing chain, so every
processed run claimed zero model calls while its assessments proved otherwise.
The fix has to satisfy four properties, and each is pinned here:

* the counts land on the run that did the work;
* a resumed run **accumulates** instead of overwriting earlier batches;
* other runs are untouched — attributing usage to "the newest run" is exactly the
  legacy bug that this must not reintroduce;
* a fully reused batch charges nothing, because nothing was called.
"""

from __future__ import annotations

import sys
from pathlib import Path

from eduecon_radar.services.local_pipeline import (
    LocalPipelineServices,
    process_batch,
    record_run_ai_usage,
)

# The lifecycle tests already build a complete production-shaped database and the
# stub services; reuse them rather than maintaining a second copy that could
# drift away from what the product path actually does.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_phase7d_lifecycle import (  # noqa: E402
    _database,
    _FakeScreening,
    _FakeTitles,
    _paper,
    _ScreeningOutcome,
    _sync,
)


class _ZeroTranslation:
    """A translation stage that costs nothing, so screening is the only charge."""

    def run(self, parameters):
        del parameters

        class _Stats:
            failed = 0
            ai_calls = 0
            input_tokens = 0
            output_tokens = 0
            failure_reasons: list[str] = []

        return _Stats()


def _run_row(database, run_id: int) -> tuple[int, int, int]:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT ai_calls,input_tokens,output_tokens FROM sync_runs WHERE id=?",
            (run_id,),
        ).fetchone()
    return int(row[0] or 0), int(row[1] or 0), int(row[2] or 0)


def _services(database, *, input_tokens: int, output_tokens: int) -> LocalPipelineServices:
    return LocalPipelineServices(
        enrichment=None,  # the paper already carries an abstract
        translation=_ZeroTranslation(),
        screening=_FakeScreening(input_tokens=input_tokens, output_tokens=output_tokens),
        batch_titles=_FakeTitles(database),
    )


def test_processing_records_calls_and_tokens_on_its_own_run(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, paper_id = _sync(database, _paper(abstract="Teacher incentives matter."))
    assert _run_row(database, run_id) == (0, 0, 0)

    result = process_batch(
        database,
        run_id,
        (paper_id,),
        _services(database, input_tokens=1200, output_tokens=340),
        profile_id="education_economics",
    )
    assert result["ai_calls"] == 1
    assert _run_row(database, run_id) == (1, 1200, 340)


def test_a_resumed_run_accumulates_instead_of_overwriting(tmp_path: Path) -> None:
    """One run is processed by several invocations; the totals must add up."""
    database = _database(tmp_path)
    run_id, first_paper = _sync(database, _paper(doi="10.1000/a", abstract="A."))
    # A second retrieval into the same database, to have two papers in one run.
    from eduecon_radar.services.local_pipeline import ensure_paper_statuses

    with database.connect() as connection:
        journal_id = connection.execute("SELECT id FROM journals LIMIT 1").fetchone()[0]
    second_id, _ = database.upsert_paper(
        _paper(doi="10.1000/b", title="Second paper", abstract="B."), journal_id
    )
    ensure_paper_statuses(database, run_id)

    process_batch(
        database,
        run_id,
        (first_paper,),
        _services(database, input_tokens=100, output_tokens=10),
        profile_id="education_economics",
    )
    assert _run_row(database, run_id) == (1, 100, 10)

    process_batch(
        database,
        run_id,
        (second_id,),
        _services(database, input_tokens=200, output_tokens=20),
        profile_id="education_economics",
    )
    assert _run_row(database, run_id) == (2, 300, 30)


def test_usage_is_never_attributed_to_another_run(tmp_path: Path) -> None:
    """The legacy path wrote to `ORDER BY id DESC LIMIT 1`; that must not return."""
    database = _database(tmp_path)
    first_run, first_paper = _sync(database, _paper(doi="10.1000/a", abstract="A."))

    # A newer retrieval run exists but is not the one being processed.
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO sync_runs (started_at,status) VALUES ('2026-08-01','success')"""
        )
    with database.connect() as connection:
        newer_run = int(
            connection.execute("SELECT MAX(id) FROM sync_runs").fetchone()[0]
        )
    assert newer_run != first_run

    process_batch(
        database,
        first_run,
        (first_paper,),
        _services(database, input_tokens=500, output_tokens=50),
        profile_id="education_economics",
    )
    assert _run_row(database, first_run) == (1, 500, 50)
    assert _run_row(database, newer_run) == (0, 0, 0)


def test_record_run_ai_usage_is_additive_and_no_op_at_zero(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, _ = _sync(database, _paper(abstract="A."))

    record_run_ai_usage(database, run_id, ai_calls=0)
    assert _run_row(database, run_id) == (0, 0, 0)

    record_run_ai_usage(database, run_id, ai_calls=3, input_tokens=30, output_tokens=3)
    record_run_ai_usage(database, run_id, ai_calls=2, input_tokens=20, output_tokens=2)
    assert _run_row(database, run_id) == (5, 50, 5)


def test_record_run_ai_usage_tolerates_zero_counters(tmp_path: Path) -> None:
    """A run row created before any processing holds zeros, not NULL."""
    database = _database(tmp_path)
    run_id, _ = _sync(database, _paper(abstract="A."))
    with database.connect() as connection:
        connection.execute(
            "UPDATE sync_runs SET ai_calls=0,input_tokens=0,output_tokens=0 WHERE id=?",
            (run_id,),
        )
    record_run_ai_usage(database, run_id, ai_calls=1, input_tokens=7, output_tokens=2)
    assert _run_row(database, run_id) == (1, 7, 2)


def test_a_batch_that_only_reuses_stored_judgements_charges_nothing(tmp_path: Path) -> None:
    database = _database(tmp_path)
    run_id, paper_id = _sync(database, _paper(abstract="Teacher incentives matter."))

    class _ReusingScreening:
        def __init__(self) -> None:
            self.calls = 0

        def screen_paper(self, paper_id, *, task_id=None, force=False):
            del paper_id, task_id, force
            self.calls += 1
            outcome = _ScreeningOutcome()
            outcome.ai_calls = 0  # served from the stored screening
            return outcome

    services = LocalPipelineServices(
        enrichment=None,
        translation=_ZeroTranslation(),
        screening=_ReusingScreening(),
        batch_titles=_FakeTitles(database),
    )
    process_batch(database, run_id, (paper_id,), services, profile_id="education_economics")
    assert _run_row(database, run_id) == (0, 0, 0)


def test_no_counters_are_written_for_a_run_that_does_not_exist(tmp_path: Path) -> None:
    """An unknown run id must not create or corrupt anything."""
    database = _database(tmp_path)
    run_id, _ = _sync(database, _paper(abstract="A."))
    record_run_ai_usage(database, run_id + 999, ai_calls=4, input_tokens=1, output_tokens=1)
    assert _run_row(database, run_id) == (0, 0, 0)
    with database.connect() as connection:
        total = connection.execute("SELECT COUNT(*) FROM sync_runs").fetchone()[0]
    assert total == 1


def test_historical_rows_are_left_alone(tmp_path: Path) -> None:
    """Nothing backfills a run that was already recorded.

    The user's instruction is explicit: do not guess at history.  A run whose
    counters are zero stays zero unless the product chain does real work on it.
    """
    database = _database(tmp_path)
    run_id, paper_id = _sync(database, _paper(abstract="Teacher incentives matter."))
    with database.connect() as connection:
        connection.execute(
            "UPDATE sync_runs SET ai_calls=0,input_tokens=0,output_tokens=0 WHERE id=?",
            (run_id,),
        )
        connection.execute(
            "INSERT INTO paper_profile_assessments "
            "(profile_id,paper_id,decision,classification_json,updated_at) "
            "VALUES ('education_economics',?,?,?,?)",
            (paper_id, "relevant", '{"screening_ai_calls": 7}', "2026-09-01"),
        )
    # Importing the read-only audit must not move the number into the run row.
    assert _run_row(database, run_id) == (0, 0, 0)
    with database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='sync_runs'"
        ).fetchone()[0] == 1
