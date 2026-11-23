"""A third research profile, and the contract that made it possible.

The Behavioural Economics profile is the second stress test of the profile
abstraction: it must reuse the same storage, journals, prompts and lifecycle,
with its own meaning for the judgement dimensions.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from eduecon_radar.db import Database
from eduecon_radar.services import local_pipeline as pipeline
from eduecon_radar.services import profile_store
from eduecon_radar.services.deterministic_screening import (
    PROMPT_FIELDS,
    PrimaryScreeningOutput,
    render_prompt,
    validate_primary,
)
from eduecon_radar.services.migration_integrity import migrate_database
from eduecon_radar.services.profile_registry import load_profile_registry
from eduecon_radar.ui.views.paper_library import _library_count, resolve_library_scope

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "config" / "profiles"
CATALOG = ROOT / "config" / "journals" / "catalog.json"
BEHAVIORAL = "behavioral_economics"
FILTERS = {
    "journal": "全部",
    "abstract_state": "全部",
    "decision": "全部",
    "auto_decision": "全部",
    "year": "全部",
    "search": "",
}


def _registry():
    return load_profile_registry(PROFILES, catalog_path=CATALOG)


# --- the profile itself ------------------------------------------------------


def test_three_profiles_load_with_their_own_meaning() -> None:
    registry = _registry()
    ids = [item[0] for item in registry.profile_choices()]
    assert ids[0] == "education_economics"
    # The three profiles this work stream introduced must exist; a later profile
    # is allowed to join without rewriting this test.
    assert {"education_economics", "labor_economics", BEHAVIORAL} <= set(ids)
    labels = {
        profile_id: registry.profile(profile_id).prompt_context()["centrality_label"]
        for profile_id in ids
    }
    # Every profile must declare a distinct centrality dimension: if two queues
    # shared a label, "central" would mean two different things in one product.
    assert len(set(labels.values())) == len(labels)
    for profile_id in ids:
        profile = registry.profile(profile_id)
        assert profile.screening_prompt_version
        assert profile.subject_signals.get("jel_codes")
        for name in (
            "screening_primary_system.txt",
            "screening_primary_user.txt",
            "screening_review_system.txt",
            "screening_review_user.txt",
        ):
            assert (profile.prompts_dir / name).is_file()


def test_behavioral_journals_are_in_the_global_catalog_and_shared() -> None:
    registry = _registry()
    behavioral = set(registry.profile(BEHAVIORAL).journal_keys)
    education = set(registry.profile("education_economics").journal_keys)
    labor = set(registry.profile("labor_economics").journal_keys)
    for key in (
        "journal_of_economic_behavior_and_organization",
        "experimental_economics",
        "games_and_economic_behavior",
        "journal_of_risk_and_uncertainty",
    ):
        assert key in behavioral
    # The catalog is shared: general journals appear in every profile and the
    # behavioural set is not a copy of another profile's list.
    assert "general_qje" in behavioral & education & labor
    assert behavioral != education and behavioral != labor


def test_behavioral_prompts_state_the_mechanism_and_the_ambiguous_cases() -> None:
    prompts = PROFILES / BEHAVIORAL / "prompts"
    primary = (prompts / "screening_primary_system.txt").read_text(encoding="utf-8")
    assert "{centrality_definition}" in primary
    assert "{subject_signals}" in primary
    # Ambiguous cases must be decided by a stated rule, not left to taste.
    # Assert on flattened text so prompt line wrapping cannot break the test.
    flat = " ".join(primary.split()).casefold()
    for token in (
        "nudges",
        "market design",
        "statistical discrimination",
        "present bias",
        "well-being",
    ):
        assert token in flat
    # Title-only and conflict resolution are separate stages of the same contract.
    user = (prompts / "screening_primary_user.txt").read_text(encoding="utf-8")
    assert "needs_abstract" in user
    review = (prompts / "screening_review_system.txt").read_text(encoding="utf-8")
    assert "conflict-resolution" in review
    assert "{centrality_label}" in review


def test_every_shipped_prompt_placeholder_is_supported() -> None:
    pattern = re.compile(r"\{([a-z_]+)\}")
    for profile_dir in sorted(path for path in PROFILES.iterdir() if path.is_dir()):
        for prompt in sorted((profile_dir / "prompts").glob("*.txt")):
            unknown = {
                name
                for name in pattern.findall(prompt.read_text(encoding="utf-8"))
                if name not in PROMPT_FIELDS
            }
            assert unknown == set(), f"{prompt}: unsupported placeholders {sorted(unknown)}"


# --- prompt rendering is additive --------------------------------------------


def test_render_prompt_ignores_json_braces_and_missing_values() -> None:
    template = 'Return {"decision": "relevant"} for {title} in {journal} ({jel_codes}).'
    rendered = render_prompt(template, {"title": "T", "journal": "J"})
    assert rendered == 'Return {"decision": "relevant"} for T in J ().'


def test_existing_profiles_render_unchanged() -> None:
    """A prompt that uses none of the new fields renders exactly as before."""
    registry = _registry()
    education = registry.profile("education_economics")
    template = (education.prompts_dir / "screening_primary_user.txt").read_text(
        encoding="utf-8"
    )
    legacy = template.format(title="T", abstract="A", journal="J", year="2026")
    rendered = render_prompt(
        template,
        {
            "title": "T",
            "abstract": "A",
            "journal": "J",
            "year": "2026",
            **education.prompt_context(),
        },
    )
    assert rendered == legacy


# --- the validator no longer claims the dimension is education ---------------


def test_validation_rules_are_profile_neutral() -> None:
    output = PrimaryScreeningOutput(
        decision="relevant",
        education_centrality="secondary",
        economic_relevance="substantive",
        evidence_sufficiency="sufficient",
        confidence="high",
        evidence=[{"quote": "a sufficiently long quote", "supports": "economics"}],
        reason="r",
        topic_tags=[],
        review_priority="medium",
    )
    result = validate_primary(output, title="t", abstract="a sufficiently long quote")
    assert "relevant_requires_central_subject" in result.errors
    assert not any("education" in error for error in result.errors)

    output.education_centrality = "none"
    result = validate_primary(output, title="t", abstract="a sufficiently long quote")
    assert "subject_none_cannot_be_relevant" in result.errors


# --- three profiles over one paper ------------------------------------------


def _database(tmp_path: Path) -> Database:
    path = tmp_path / "expansion.db"
    migrate_database(path, "production", dry_run=False)
    return Database(path)


def _paper(database: Database, *, doi: str = "10.1/three") -> int:
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
            (
                doi,
                f"fp-{doi}",
                "Stereotypes, beliefs and study choice",
                "We run a field experiment on beliefs about fields of study.",
                journal_id,
            ),
        )
        paper_id = int(cursor.lastrowid)
    return paper_id


def _run(database: Database, profile_id: str, paper_id: int) -> int:
    with database.connect() as connection:
        cursor = connection.execute(
            """INSERT INTO sync_runs
            (started_at,status,requested_limit,matched_paper_ids,profile_id)
            VALUES ('t','success',10,?,?)""",
            (json.dumps([paper_id]), profile_id),
        )
        run_id = int(cursor.lastrowid)
    pipeline.ensure_paper_statuses_for_paper_ids(database, run_id, (paper_id,))
    return run_id


def _assess(database: Database, profile_id: str, run_id: int, paper_id: int, decision: str) -> None:
    pipeline._record_profile_assessment(
        database,
        profiles_ready=True,
        profile_id=profile_id,
        run_id=run_id,
        paper_id=paper_id,
        classification={"decision": decision, "screening_basis": "title_and_abstract"},
        model_name="m",
        prompt_version="v1",
        title="T",
        abstract="A",
    )


def test_three_profiles_judge_and_review_the_same_paper_independently(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    paper_id = _paper(database)
    decisions = {
        "education_economics": "relevant",
        "labor_economics": "irrelevant",
        BEHAVIORAL: "relevant",
    }
    runs = {}
    for profile_id, decision in decisions.items():
        runs[profile_id] = _run(database, profile_id, paper_id)
        _assess(database, profile_id, runs[profile_id], paper_id, decision)

    # Different judgements for the same paper row.
    for profile_id, decision in decisions.items():
        item = pipeline.sync_run_papers(database, runs[profile_id])[0]
        assert item["assessment_source"] == "profile"
        assert item["processing"]["classification"]["decision"] == decision

    # Independent human decisions, three ways.
    assert pipeline.save_human_decisions(
        database, runs["education_economics"], {paper_id: ("related", "education")}
    ) == (paper_id,)
    assert pipeline.save_human_decisions(
        database, runs["labor_economics"], {paper_id: ("irrelevant", "not labour")}
    ) == (paper_id,)
    assert pipeline.save_human_decisions(
        database, runs[BEHAVIORAL], {paper_id: ("uncertain", "behavioural?")}
    ) == (paper_id,)
    assert (
        pipeline.sync_run_papers(database, runs["education_economics"])[0]["human_decision"]
        == "related"
    )
    assert (
        pipeline.sync_run_papers(database, runs["labor_economics"])[0]["human_decision"]
        == "irrelevant"
    )
    assert (
        pipeline.sync_run_papers(database, runs[BEHAVIORAL])[0]["human_decision"]
        == "uncertain"
    )

    # Paper facts stay single: one row, one translation, three judgements.
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
        assert (
            connection.execute("SELECT COUNT(*) FROM paper_profile_assessments").fetchone()[0]
            == 3
        )
        reviews = connection.execute(
            "SELECT COUNT(*) FROM paper_profile_reviews"
        ).fetchone()[0]
        assert reviews == 3
    assert profile_store.assessment_drift(database) == ()


def test_library_is_scoped_per_profile_including_the_new_one(tmp_path: Path) -> None:
    database = _database(tmp_path)
    paper_id = _paper(database)
    _run(database, "education_economics", paper_id)
    _run(database, BEHAVIORAL, paper_id)
    education = _library_count(
        database,
        **FILTERS,
        has_status=True,
        scope=resolve_library_scope(database, "education_economics"),
    )
    behavioral = _library_count(
        database,
        **FILTERS,
        has_status=True,
        scope=resolve_library_scope(database, BEHAVIORAL),
    )
    labor = _library_count(
        database,
        **FILTERS,
        has_status=True,
        scope=resolve_library_scope(database, "labor_economics"),
    )
    assert education == 1 and behavioral == 1 and labor == 0


def test_prompt_context_reaches_the_service(tmp_path: Path) -> None:
    """The service exposes the profile vocabulary to its prompts."""
    from eduecon_radar.services.deterministic_screening import (
        DeterministicScreeningService,
    )

    database = _database(tmp_path)
    profile = _registry().profile(BEHAVIORAL)
    service = DeterministicScreeningService(
        database,
        prompts_dir=profile.prompts_dir,
        prompt_version=profile.screening_prompt_version,
        prompt_context=profile.prompt_context(),
    )
    assert service.prompt_version == profile.screening_prompt_version
    values = service._prompt_values(
        {"title": "T", "abstract": "A", "journal": "J", "year": "2026", "jel_codes": ["D90"]}
    )
    assert values["centrality_label"] == "行为机制中心性"
    assert values["jel_codes"] == "D90"
    assert "D90" in values["subject_signals"]


def test_behavioral_placeholder_fills_from_the_profile() -> None:
    profile = _registry().profile(BEHAVIORAL)
    template = (profile.prompts_dir / "screening_primary_system.txt").read_text(
        encoding="utf-8"
    )
    rendered = render_prompt(template, profile.prompt_context())
    assert "{centrality_definition}" not in rendered
    assert "{subject_signals}" not in rendered
    assert "行为机制中心性" in rendered or "behavioural mechanism" in rendered
    assert "D90" in rendered


@pytest.mark.parametrize(
    "profile_id",
    ["education_economics", "labor_economics", BEHAVIORAL, "political_economy"],
)
def test_profile_prompt_version_is_unique(profile_id: str) -> None:
    """Two queues must never share a prompt version.

    The version is what makes a stored judgement traceable to the prompt that
    produced it; if two profiles declared the same string, an assessment could
    not be attributed to its prompt.
    """
    registry = _registry()
    versions = [
        registry.profile(item[0]).screening_prompt_version
        for item in registry.profile_choices()
    ]
    assert len(set(versions)) == len(versions)
    assert registry.profile(profile_id).screening_prompt_version in versions


def test_repo_has_no_duplicate_prompt_or_journal_sources() -> None:
    """The third profile must not have introduced a second copy of anything."""
    assert not (ROOT / "config" / "journals" / "behavioral_economics.json").exists()
    for name in (
        "screening_primary_system.txt",
        "screening_primary_user.txt",
        "screening_review_system.txt",
        "screening_review_user.txt",
    ):
        assert not (ROOT / "prompts" / name).exists()
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    keys = [item["journal_key"] for item in catalog["journals"]]
    assert len(keys) == len(set(keys))
    issns = [
        issn
        for item in catalog["journals"]
        for issn in (item.get("print_issn"), item.get("online_issn"))
        if issn
    ]
    assert len(issns) == len(set(issns)), "an ISSN is claimed twice in the catalog"



# --- knobs the third profile needed -----------------------------------------


def test_profile_may_declare_its_own_screening_latency_budget() -> None:
    """A longer analysis may need more than the installation default."""
    registry = _registry()
    behavioral = registry.profile(BEHAVIORAL)
    assert behavioral.screening_timeout_seconds == 150
    # Other profiles keep the installation default by declaring nothing.
    assert registry.profile("education_economics").screening_timeout_seconds is None
    assert registry.profile("labor_economics").screening_timeout_seconds is None


def test_invalid_latency_budget_is_rejected(tmp_path: Path) -> None:
    import shutil

    from eduecon_radar.services.profile_registry import (
        ProfileConfigError,
        load_profile_registry,
    )

    profiles = tmp_path / "profiles"
    shutil.copytree(PROFILES, profiles)
    target = profiles / BEHAVIORAL / "profile.yaml"
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            "screening_timeout_seconds: 150", "screening_timeout_seconds: -5"
        ),
        encoding="utf-8",
    )
    try:
        load_profile_registry(profiles, catalog_path=CATALOG)
    except ProfileConfigError as exc:
        assert "screening_timeout_seconds" in str(exc)
    else:  # pragma: no cover - the assertion above is the contract
        raise AssertionError("a non-positive latency budget must be rejected")


def test_every_stored_prompt_version_is_declared_by_a_profile() -> None:
    """Guards the defect class where a run was judged by another queue's prompts."""
    from eduecon_radar.services.deterministic_screening import (
        FRONT_MATTER_PROMPT_VERSION,
    )

    registry = _registry()
    declared = {
        registry.profile(item[0]).screening_prompt_version
        for item in registry.profile_choices()
    } | {FRONT_MATTER_PROMPT_VERSION}
    assert "2026-07-31-screening-v1" in declared
    assert "2026-08-02-front-matter-v1" in declared
    assert "2026-09-21-labor-v1" in declared
    assert "2026-09-24-behavioral-v2" in declared
    assert "2026-09-25-political-economy-v1" in declared
    # Every profile declares its own version, so the count is profiles + the
    # deterministic front-matter rule, whatever the number of profiles is.
    assert len(declared) == len(registry.profile_choices()) + 1


def test_behavioral_boundary_covers_decision_theory_and_experiments() -> None:
    """The refined boundary must keep decision-theoretic work in scope.

    The first version of this prompt under-called papers on ambiguity
    elicitation, belief updating and consideration sets; these rules are what
    fixed it, so they are pinned.
    """
    profile = _registry().profile(BEHAVIORAL)
    flat = " ".join(
        (profile.prompts_dir / "screening_primary_system.txt")
        .read_text(encoding="utf-8")
        .split()
    ).casefold()
    for phrase in (
        "decision-theoretic and measurement contributions are in scope",
        "consideration sets",
        "belief formation and updating",
        "diagnostic expectations",
        "rather than \"irrelevant\"",
    ):
        assert phrase in flat

def test_packaging_includes_the_new_profile() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_packaging import main

    assert main([]) == 0
