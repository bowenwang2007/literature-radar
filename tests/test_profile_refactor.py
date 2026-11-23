"""Profile-aware semantics: paper facts are global, judgements are not.

These tests exercise the real migration path (a fresh database migrated to the
production profile) rather than a hand-built schema, so a change that breaks the
migration also breaks these.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eduecon_radar.db import Database
from eduecon_radar.services import local_pipeline as pipeline
from eduecon_radar.services import profile_store
from eduecon_radar.services.migration_integrity import migrate_database
from eduecon_radar.services.profile_registry import (
    ProfileConfigError,
    load_profile_registry,
)
from eduecon_radar.ui.views.paper_library import (
    _export_scope_papers,
    _library_count,
    resolve_library_scope,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "config" / "profiles"
CATALOG = ROOT / "config" / "journals" / "catalog.json"


def _database(tmp_path: Path) -> Database:
    path = tmp_path / "profile.db"
    migrate_database(path, "production", dry_run=False)
    return Database(path)


def _paper(database: Database, *, doi: str, title: str, abstract: str) -> int:
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO journals (name,short_name,issn,priority,enabled,created_at,updated_at)
            VALUES ('Journal','J','0000-0001','high',1,'t','t')
            ON CONFLICT(short_name) DO NOTHING"""
        )
        journal_id = connection.execute(
            "SELECT id FROM journals WHERE short_name='J'"
        ).fetchone()["id"]
        cursor = connection.execute(
            """INSERT INTO papers
            (doi,fingerprint,title,abstract,authors_json,journal_id,source_metadata_json,
             first_seen_at,last_seen_at,created_at,updated_at)
            VALUES (?,?,?,?,'[]',?,'{}','t','t','t','t')""",
            (doi, f"fp-{doi}", title, abstract, journal_id),
        )
        return int(cursor.lastrowid)


def _run(database: Database, *, profile_id: str, paper_ids: tuple[int, ...]) -> int:
    with database.connect() as connection:
        cursor = connection.execute(
            """INSERT INTO sync_runs
            (started_at,status,requested_limit,matched_paper_ids,profile_id)
            VALUES ('t','success',10,?,?)""",
            (json.dumps(list(paper_ids)), profile_id),
        )
        run_id = int(cursor.lastrowid)
    # The product always attaches per-paper processing rows right after a
    # retrieval run; membership and processing state both rely on that.
    pipeline.ensure_paper_statuses_for_paper_ids(database, run_id, paper_ids)
    return run_id


def _assess(
    database: Database,
    *,
    profile_id: str,
    run_id: int,
    paper_id: int,
    decision: str,
    basis: str,
    prompt_version: str,
) -> None:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT title,abstract FROM papers WHERE id=?", (paper_id,)
        ).fetchone()
    pipeline._record_profile_assessment(
        database,
        profiles_ready=True,
        profile_id=profile_id,
        run_id=run_id,
        paper_id=paper_id,
        classification={"decision": decision, "screening_basis": basis},
        model_name="test-model",
        prompt_version=prompt_version,
        title=row["title"],
        abstract=row["abstract"],
    )


# --- profile configuration -------------------------------------------------


def test_profile_registry_exposes_default_and_second_profile() -> None:
    registry = load_profile_registry(PROFILES, catalog_path=CATALOG)
    assert registry.default_profile_id == "education_economics"
    assert [item[0] for item in registry.profile_choices()][0] == "education_economics"
    profile_ids = {item[0] for item in registry.profile_choices()}
    # The registry only grows; education stays first and labour stays available.
    assert {"education_economics", "labor_economics"} <= profile_ids


def test_journals_are_shared_between_profiles() -> None:
    registry = load_profile_registry(PROFILES, catalog_path=CATALOG)
    education = set(registry.profile("education_economics").journal_keys)
    labor = set(registry.profile("labor_economics").journal_keys)
    shared = education & labor
    # A journal is not owned by a profile. Two kinds of sharing have to hold:
    # the flagship tier is identical everywhere, and journals whose subject is
    # relevant to two directions appear in both recommendations.
    assert "general_qje" in shared
    assert "journal_of_public_economics" in shared
    assert "journal_of_human_resources" in shared
    # And a direction still keeps journals of its own.
    assert "industrial_and_labor_relations_review" in labor - education
    assert "economics_of_education_review" in education - labor


def test_the_global_library_is_shared_and_no_journal_is_duplicated() -> None:
    """The catalog holds journals; a profile only recommends them.

    Two properties are checked, neither of which forces every journal to be
    multi-queue (political-science journals legitimately serve only the
    political-economy queue):

    * the flagship tier is recommended by all four queues, from one entry;
    * no journal appears twice within a profile, and the catalog holds each
      journal exactly once.
    """
    registry = load_profile_registry(PROFILES, catalog_path=CATALOG)
    membership: dict[str, set[str]] = {}
    for profile_id, _ in registry.profile_choices():
        keys = list(registry.profile(profile_id).journal_keys)
        assert len(keys) == len(set(keys)), f"{profile_id} lists a journal twice"
        for key in keys:
            membership.setdefault(key, set()).add(profile_id)

    # Every catalog journal is reachable from at least one queue.
    assert len(membership) == len(registry.catalog)
    # The flagship tier is shared by all four queues, through one catalog entry.
    assert membership["general_qje"] == {
        "education_economics",
        "labor_economics",
        "behavioral_economics",
        "political_economy",
    }
    for profile_id, _ in registry.profile_choices():
        profile = registry.profile(profile_id)
        group_keys = {
            journal.journal_key
            for journal in profile.registry().expand_group("general_economics")
        }
        assert "general_qje" in group_keys, profile_id
    # Identity lives in the catalog: one entry per key, one owner per ISSN.
    assert len({entry.journal_key for entry in registry.catalog.values()}) == len(
        registry.catalog
    )


def test_each_profile_declares_its_own_screening_prompt() -> None:
    registry = load_profile_registry(PROFILES, catalog_path=CATALOG)
    education = registry.profile("education_economics")
    labor = registry.profile("labor_economics")
    assert education.prompts_dir != labor.prompts_dir
    assert education.screening_prompt_version != labor.screening_prompt_version
    text = (labor.prompts_dir / "screening_primary_system.txt").read_text(
        encoding="utf-8"
    )
    assert "labour" in text.lower()


def test_unknown_profile_is_rejected() -> None:
    registry = load_profile_registry(PROFILES, catalog_path=CATALOG)
    with pytest.raises(ProfileConfigError):
        registry.profile("history_economics")


# --- storage semantics -----------------------------------------------------


def test_fresh_database_has_profile_schema(tmp_path: Path) -> None:
    database = _database(tmp_path)
    assert profile_store.profile_schema_ready(database)
    with database.connect() as connection:
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(sync_runs)")
        }
    assert "profile_id" in columns


def test_same_paper_can_be_judged_differently_by_two_profiles(tmp_path: Path) -> None:
    database = _database(tmp_path)
    paper_id = _paper(
        database,
        doi="10.1/shared",
        title="Education, wages and the labour market",
        abstract="We estimate the effect of schooling on wages.",
    )
    education_run = _run(database, profile_id="education_economics", paper_ids=(paper_id,))
    labor_run = _run(database, profile_id="labor_economics", paper_ids=(paper_id,))
    _assess(
        database,
        profile_id="education_economics",
        run_id=education_run,
        paper_id=paper_id,
        decision="relevant",
        basis="title_and_abstract",
        prompt_version="ee-v1",
    )
    _assess(
        database,
        profile_id="labor_economics",
        run_id=labor_run,
        paper_id=paper_id,
        decision="irrelevant",
        basis="title_and_abstract",
        prompt_version="lab-v1",
    )

    education = pipeline.sync_run_papers(database, education_run)[0]
    labor = pipeline.sync_run_papers(database, labor_run)[0]
    assert education["processing"]["classification"]["decision"] == "relevant"
    assert labor["processing"]["classification"]["decision"] == "irrelevant"
    assert education["processing"]["prompt_version"] == "ee-v1"
    assert labor["processing"]["prompt_version"] == "lab-v1"
    # The underlying paper facts are stored exactly once.
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
    assert education["title"] == labor["title"]
    assert education["abstract"] == labor["abstract"]


def test_human_review_is_independent_per_profile(tmp_path: Path) -> None:
    database = _database(tmp_path)
    paper_id = _paper(database, doi="10.1/human", title="T", abstract="A")
    education_run = _run(database, profile_id="education_economics", paper_ids=(paper_id,))
    labor_run = _run(database, profile_id="labor_economics", paper_ids=(paper_id,))
    pipeline.ensure_paper_statuses_for_paper_ids(database, education_run, (paper_id,))
    pipeline.ensure_paper_statuses_for_paper_ids(database, labor_run, (paper_id,))

    assert pipeline.save_human_decisions(
        database, education_run, {paper_id: ("related", "keep for education")}
    ) == (paper_id,)
    assert pipeline.save_human_decisions(
        database, labor_run, {paper_id: ("irrelevant", "not labour")}
    ) == (paper_id,)

    education = pipeline.sync_run_papers(database, education_run)[0]
    labor = pipeline.sync_run_papers(database, labor_run)[0]
    assert (education["human_decision"], education["reviewer_notes"]) == (
        "related",
        "keep for education",
    )
    assert (labor["human_decision"], labor["reviewer_notes"]) == (
        "irrelevant",
        "not labour",
    )
    # Only the profile's own runs are mirrored.
    with database.connect() as connection:
        rows = {
            int(row["sync_run_id"]): row["human_decision"]
            for row in connection.execute(
                "SELECT sync_run_id,human_decision FROM paper_processing_status "
                "WHERE paper_id=?",
                (paper_id,),
            )
        }
    assert rows[education_run] == "related"
    assert rows[labor_run] == "irrelevant"


def test_ai_assessment_never_overwrites_human_review(tmp_path: Path) -> None:
    database = _database(tmp_path)
    paper_id = _paper(database, doi="10.1/protect", title="T", abstract="A")
    run_id = _run(database, profile_id="education_economics", paper_ids=(paper_id,))
    pipeline.ensure_paper_statuses_for_paper_ids(database, run_id, (paper_id,))
    pipeline.save_human_decisions(database, run_id, {paper_id: ("related", "mine")})
    _assess(
        database,
        profile_id="education_economics",
        run_id=run_id,
        paper_id=paper_id,
        decision="irrelevant",
        basis="title_only",
        prompt_version="ee-v1",
    )
    paper = pipeline.sync_run_papers(database, run_id)[0]
    assert paper["human_decision"] == "related"
    assert paper["reviewer_notes"] == "mine"
    assert paper["processing"]["classification"]["decision"] == "irrelevant"


def test_repeated_retrieval_deduplicates_the_paper(tmp_path: Path) -> None:
    database = _database(tmp_path)
    paper_id = _paper(database, doi="10.1/dup", title="T", abstract="A")
    first = _run(database, profile_id="education_economics", paper_ids=(paper_id,))
    second = _run(database, profile_id="labor_economics", paper_ids=(paper_id,))
    assert profile_store.profile_paper_ids(database, "education_economics") == (paper_id,)
    assert profile_store.profile_paper_ids(database, "labor_economics") == (paper_id,)
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
    assert first != second


def test_assessment_fingerprint_changes_when_abstract_arrives(tmp_path: Path) -> None:
    title_only = profile_store.assessment_fingerprint("p", "v1", "T", None)
    with_abstract = profile_store.assessment_fingerprint("p", "v1", "T", "A")
    assert title_only != with_abstract


# --- legacy backfill -------------------------------------------------------


def _legacy_database(tmp_path: Path) -> tuple[Database, int]:
    """A database that still stores judgements in the run-scoped columns."""
    database = _database(tmp_path)
    paper_id = _paper(database, doi="10.1/legacy", title="T", abstract="A")
    run_id = _run(database, profile_id="education_economics", paper_ids=(paper_id,))
    with database.connect() as connection:
        connection.execute(
            """UPDATE paper_processing_status
            SET status='classified',stage='complete',classification_json=?,
                model_name='deepseek-v4-pro',prompt_version='2026-07-31-screening-v1',
                classified_at='2026-08-03T00:00:00+00:00',human_decision='irrelevant',
                human_note='legacy note',human_decided_at='2026-08-03T01:00:00+00:00'
            WHERE sync_run_id=? AND paper_id=?""",
            (
                json.dumps({"decision": "irrelevant", "screening_basis": "title_only"}),
                run_id,
                paper_id,
            ),
        )
    return database, paper_id


def test_backfill_preserves_unknown_basis_without_inventing_one(tmp_path: Path) -> None:
    database, paper_id = _legacy_database(tmp_path)
    result = profile_store.migrate_legacy_assessments(
        database, profile_id="education_economics"
    )
    assert result["assessments_stored"] == 1
    assessment = profile_store.assessments_for(
        database, "education_economics", (paper_id,)
    )[paper_id]
    assert assessment.screening_basis == "title_only"
    assert assessment.decision == "irrelevant"

    # A judgement with no recorded basis stays unknown; it is never upgraded.
    with database.connect() as connection:
        connection.execute(
            "UPDATE paper_processing_status SET classification_json=? WHERE paper_id=?",
            (json.dumps({"decision": "irrelevant"}), paper_id),
        )
        connection.execute(
            "DELETE FROM paper_profile_assessments WHERE paper_id=?", (paper_id,)
        )
    profile_store.migrate_legacy_assessments(database, profile_id="education_economics")
    assessment = profile_store.assessments_for(
        database, "education_economics", (paper_id,)
    )[paper_id]
    assert assessment.screening_basis is None


def test_backfill_is_idempotent_and_keeps_human_decisions(tmp_path: Path) -> None:
    database, paper_id = _legacy_database(tmp_path)
    first = profile_store.migrate_legacy_assessments(
        database, profile_id="education_economics"
    )
    second = profile_store.migrate_legacy_assessments(
        database, profile_id="education_economics"
    )
    assert first["assessments_stored"] == second["assessments_stored"] == 1
    assert first["reviews_stored"] == second["reviews_stored"] == 1
    review = profile_store.reviews_for(database, "education_economics", (paper_id,))[
        paper_id
    ]
    assert (review.human_decision, review.human_note) == ("irrelevant", "legacy note")


# --- library and export ----------------------------------------------------


def test_library_and_export_are_profile_scoped(tmp_path: Path) -> None:
    database = _database(tmp_path)
    education_paper = _paper(database, doi="10.1/ee", title="EE paper", abstract="A")
    labor_paper = _paper(database, doi="10.1/lab", title="LAB paper", abstract="A")
    _run(database, profile_id="education_economics", paper_ids=(education_paper,))
    _run(database, profile_id="labor_economics", paper_ids=(labor_paper,))

    education_scope = resolve_library_scope(database, "education_economics")
    labor_scope = resolve_library_scope(database, "labor_economics")
    assert education_scope.uses_profile and labor_scope.uses_profile

    filters = {
        "journal": "全部",
        "abstract_state": "全部",
        "decision": "全部",
        "auto_decision": "全部",
        "year": "全部",
        "search": "",
    }
    assert (
        _library_count(database, **filters, has_status=True, scope=education_scope) == 1
    )
    assert _library_count(database, **filters, has_status=True, scope=labor_scope) == 1
    education_rows = _export_scope_papers(
        database,
        "论文库全部",
        filters=filters,
        has_status=True,
        library_scope=education_scope,
    )
    labor_rows = _export_scope_papers(
        database,
        "论文库全部",
        filters=filters,
        has_status=True,
        library_scope=labor_scope,
    )
    assert {row["doi"] for row in education_rows} == {"10.1/ee"}
    assert {row["doi"] for row in labor_rows} == {"10.1/lab"}


def test_screening_reuse_is_scoped_to_the_profile_prompt(tmp_path: Path) -> None:
    """One profile must never reuse another profile's stored screening.

    ``paper_screenings`` is keyed by input hash *and* prompt version; looking up
    the module default instead of the profile's own version silently turned a
    labor screening into a copy of the education judgement.
    """
    from eduecon_radar.services.deterministic_screening import (
        DeterministicScreeningService,
    )

    database = _database(tmp_path)
    paper_id = _paper(database, doi="10.1/reuse", title="T", abstract="A")
    education = DeterministicScreeningService(
        database, prompts_dir=PROFILES / "education_economics" / "prompts"
    )
    labor = DeterministicScreeningService(
        database,
        prompts_dir=PROFILES / "labor_economics" / "prompts",
        prompt_version="2026-09-21-labor-v1",
    )
    digest = "digest-of-title-and-abstract"
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO paper_screenings
            (paper_id,input_hash,model_decision,evidence_json,review_priority,
             prompt_version,parser_version,evidence_validation_version,
             screened_at,created_at,updated_at)
            VALUES (?,?,?,'[]','medium',?,'p','e','t','t','t')""",
            (paper_id, digest, "irrelevant", education.prompt_version),
        )
    # Same input, different profile: no reuse.
    assert labor._latest_existing(paper_id, digest) is None
    # Same input, same profile: reuse is allowed.
    assert education._latest_existing(paper_id, digest) is not None
    other = _paper(database, doi="10.1/other-paper", title="T", abstract="A")
    assert education._latest_existing(other, digest) is None


def test_unmigrated_database_still_reads_legacy_columns(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    database = Database(path)
    database.init_schema()
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO journals (name,short_name,issn,priority,enabled,created_at,updated_at)"
            " VALUES ('J','J','0000-0001','high',1,'t','t')"
        )
        connection.execute(
            "INSERT INTO papers (doi,fingerprint,title,abstract,authors_json,journal_id,"
            "source_metadata_json,first_seen_at,last_seen_at,created_at,updated_at)"
            " VALUES ('10.1/old','fp','T','A','[]',1,'{}','t','t','t','t')"
        )
    assert not profile_store.profile_schema_ready(database)
    scope = resolve_library_scope(database, "education_economics")
    assert not scope.uses_profile

def test_shared_paper_review_and_repeat_retrieval_stay_in_direction(tmp_path: Path) -> None:
    from dataclasses import replace

    database = _database(tmp_path)
    paper = _paper(database, doi="10.1000/shared", title="Shared paper", abstract="Evidence")
    education = _run(database, profile_id="education_economics", paper_ids=(paper,))
    pipeline.save_human_decisions(database, education, {paper: ("related", "education note")})

    labor = replace(resolve_library_scope(database, "labor_economics"), shared=True)
    rows = _export_scope_papers(
        database, "当前筛选结果", filters={"decision": "未决定"}, has_status=True,
        library_scope=labor,
    )
    assert [(r["id"], r["reviewer_notes"]) for r in rows] == [(paper, None)]
    library_run = pipeline.library_processing_run(database, paper, "labor_economics")
    assert library_run != education
    assert pipeline.library_processing_run(database, paper, "labor_economics") == library_run
    pipeline.save_human_decisions(database, library_run, {paper: ("uncertain", "labor note")})
    assert pipeline.sync_run_papers(database, education)[0]["reviewer_notes"] == "education note"
    new_run = _run(database, profile_id="political_economy", paper_ids=(paper,))
    new_paper = pipeline.sync_run_papers(database, new_run)[0]
    assert new_paper["human_decision"] == "undecided"
    assert new_paper["reviewer_notes"] is None

def test_rediscovery_reuses_only_matching_direction_and_evidence(tmp_path):
    db = _database(tmp_path)
    edu = "education_economics"
    pid = _paper(db, doi="10.1000/repeat", title="School", abstract="Original evidence.")
    first = _run(db, profile_id=edu, paper_ids=(pid,))
    _assess(db, profile_id=edu, run_id=first, paper_id=pid, decision="relevant",
            basis="title_abstract", prompt_version="v1")
    with db.connect() as conn:
        conn.execute(
            """INSERT INTO paper_translations(paper_id,chinese_title,chinese_abstract,
            title_translation_status,abstract_translation_status,updated_at)
            VALUES(?, '标题', '摘要', 'ai_generated', 'ai_generated', 't')""", (pid,),
        )
    second = _run(db, profile_id=edu, paper_ids=(pid,))
    assert pipeline.sync_run_snapshot(db, second)["counts"]["automatic_judgments"] == 1
    assert pipeline.resumable_paper_ids(db, second) == ()
    labor = _run(db, profile_id="labor_economics", paper_ids=(pid,))
    assert pipeline.resumable_paper_ids(db, labor) == (pid,)
    with db.connect() as conn:
        conn.execute("UPDATE papers SET abstract='Additional evidence.' WHERE id=?", (pid,))
    assert pipeline.resumable_paper_ids(db, second) == (pid,)
