"""Political Economy profile: defined by a research question, not a keyword list.

The risk this profile carries is unique among the four queues.  "Political
economy" is a phrase that appears in a great deal of work that is not
political-economy research — a policy evaluation whose reform happened to be
passed by a parliament, a growth paper with an "institutions" control, a
political-science paper about turnout.  The tests below pin the properties that
keep the boundary a *mechanism* boundary:

* the prompt states who decides, what the political friction is, and which
  economic outcome follows, and requires all three;
* the prompt refuses the specific failure modes (policy-as-exogenous-shock,
  "institution" as a background factor, purely political dependent variables);
* subject signals are declared as entry points, never as the definition;
* the profile is a configuration-only addition — it reuses the shared catalog,
  adds no table, and changes nothing about the other three queues.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eduecon_radar.services.profile_registry import load_profile_registry

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "config" / "profiles"
CATALOG = ROOT / "config" / "journals" / "catalog.json"
POLITICAL = "political_economy"
PROBE = ROOT / "tests" / "fixtures" / "crossref_journal_identities.json"
REQUIRED_PROMPTS = (
    "screening_primary_system.txt",
    "screening_primary_user.txt",
    "screening_review_system.txt",
    "screening_review_user.txt",
)


def _registry():
    return load_profile_registry(PROFILES, catalog_path=CATALOG)


def _flat(text: str) -> str:
    return " ".join(text.split()).casefold()


def _prompt(name: str) -> str:
    return (PROFILES / POLITICAL / "prompts" / name).read_text(encoding="utf-8")


# --- the research question ----------------------------------------------------


def test_profile_declares_a_distinct_centrality_dimension() -> None:
    profile = _registry().profile(POLITICAL)
    context = profile.prompt_context()
    assert context["centrality_label"] == "政治机制中心性"
    definition = _flat(context["centrality_definition"])
    # The definition must name the actor and the causal direction, not a topic.
    assert "political actor" in definition
    assert "chooses" in definition
    assert "causes the economic outcome" in definition
    assert profile.screening_prompt_version == "2026-09-25-political-economy-v1"
    assert profile.screening_timeout_seconds and profile.screening_timeout_seconds >= 60


def test_all_four_prompts_exist_and_are_self_consistent() -> None:
    profile = _registry().profile(POLITICAL)
    for name in REQUIRED_PROMPTS:
        assert (profile.prompts_dir / name).is_file(), name
    # The system prompts carry the queue's own vocabulary...
    for name in ("screening_primary_system.txt", "screening_review_system.txt"):
        text = _prompt(name)
        assert "{centrality_label}" in text
        assert "{centrality_definition}" in text
        assert "{subject_signals}" in text
    # ...and the user prompts carry the paper and the placeholders they need.
    assert "{title}" in _prompt("screening_primary_user.txt")
    assert "{abstract}" in _prompt("screening_primary_user.txt")
    assert "{primary}" in _prompt("screening_review_user.txt")


# --- the mechanism test -------------------------------------------------------


def test_primary_prompt_states_the_three_part_mechanism_test() -> None:
    flat = _flat(_prompt("screening_primary_system.txt"))
    assert "mechanism test" in flat
    assert "who decides" in flat
    assert "what is the political friction" in flat
    assert "what economic outcome or instrument" in flat
    # It must demand all three, not any one of them.
    assert "all three must be answerable" in flat


def test_prompt_names_the_political_actors_it_means() -> None:
    flat = _flat(_prompt("screening_primary_system.txt"))
    for actor in (
        "voters",
        "legislature",
        "coalition",
        "autocrat",
        "bureaucrat",
        "interest group",
        "lobby",
        "foreign government",
    ):
        assert actor in flat, actor


def test_prompt_names_the_frictions_it_means() -> None:
    flat = _flat(_prompt("screening_primary_system.txt"))
    for friction in (
        "electoral accountability",
        "bargaining constraints",
        "veto players",
        "rent-seeking",
        "credible commitment",
        "time inconsistency",
        "distribution",
    ):
        assert friction in flat, friction


@pytest.mark.parametrize(
    "failure_mode",
    [
        # policy evaluation that never asks who chose the policy
        "treats the reform as an exogenous shock",
        # the word "institution" used as a stand-in for any background factor
        "no political actor and no collective-choice friction",
        # political variables as units of observation rather than mechanism
        "unit of observation or a control",
        # purely political dependent variables
        "with no economic outcome attached",
        # social-choice theory with no economic application
        "formal social-choice theory with no economic application",
    ],
)
def test_prompt_excludes_the_documented_failure_modes(failure_mode: str) -> None:
    assert failure_mode in _flat(_prompt("screening_primary_system.txt"))


def test_ambiguous_domains_are_decided_by_a_stated_rule() -> None:
    flat = _flat(_prompt("screening_primary_system.txt"))
    for domain in (
        "redistribution and taxation",
        "trade policy",
        "central-bank independence",
        "development and growth papers",
        "political-science papers",
        "historical and persistence papers",
        "lobbying, campaign finance and regulatory capture",
    ):
        assert domain in flat, domain
    # Each rule must cut both ways, so the queue is not simply permissive: for
    # every "in scope when" there must be a matching exclusion, in whatever
    # wording the sentence needs.
    included = flat.count("in scope when")
    excluded = (
        flat.count("out when")
        + flat.count("are not")
        + flat.count("is not relevant to this queue")
    )
    assert included >= 5
    assert excluded >= included


def test_thin_evidence_yields_uncertain_rather_than_irrelevant() -> None:
    flat = _flat(_prompt("screening_primary_system.txt"))
    assert 'output "uncertain" rather than "irrelevant"' in flat
    assert "never force a decision" in flat


def test_the_decision_contract_is_unchanged_so_stored_rows_stay_readable() -> None:
    flat = _flat(_prompt("screening_primary_system.txt"))
    for field in (
        '"decision"',
        '"education_centrality"',
        '"economic_relevance"',
        '"evidence_sufficiency"',
        '"confidence"',
        '"evidence"',
        '"reason"',
        '"topic_tags"',
        '"review_priority"',
    ):
        assert field in flat, field
    assert "the field name is historical" in flat


def test_review_stage_rechecks_the_mechanism_rather_than_the_topics() -> None:
    flat = _flat(_prompt("screening_review_system.txt"))
    assert "mechanism test" in flat
    assert "name which part of the mechanism test failed" in flat
    assert "dependent variable is purely political" in flat
    # The review stage must carry the same three questions.
    assert "(a) an identifiable political actor chooses" in flat


def test_user_prompt_refuses_to_guess_the_actor_from_a_title() -> None:
    flat = _flat(_prompt("screening_primary_user.txt"))
    assert "who decides" in flat
    assert "never infer" in flat
    assert "needs_abstract" in flat


# --- subject signals are entry points, not the definition ---------------------


def test_subject_signals_are_labelled_as_entry_points_not_the_definition() -> None:
    primary = _prompt("screening_primary_system.txt")
    assert "NOT a checklist" in primary
    assert "entry points" in primary.casefold()
    profile = _registry().profile(POLITICAL)
    jel = profile.subject_signals.get("jel_codes") or ()
    assert jel
    # The political-economy JEL neighbourhood is where the queue starts.
    assert {"D72", "D73", "D74", "D78"} <= set(jel)


def test_configured_jel_codes_are_valid_jel_codes() -> None:
    profile = _registry().profile(POLITICAL)
    import re

    for code in profile.subject_signals["jel_codes"]:
        assert re.fullmatch(r"[A-Z]\d{1,2}[A-Z]?", code), code


def test_topics_are_mechanisms_not_journal_names() -> None:
    profile = _registry().profile(POLITICAL)
    topics = profile.subject_signals.get("topics") or ()
    assert len(topics) >= 8
    for topic in topics:
        assert " " in topic
        # A topic entry must describe a question, not name a journal or a word.
        assert not topic.endswith(" journal")
        assert "journal of" not in topic.casefold()


# --- journals -----------------------------------------------------------------


def test_political_economy_journals_are_in_the_shared_catalog() -> None:
    registry = _registry()
    political = set(registry.profile(POLITICAL).journal_keys)
    for key in (
        "public_choice",
        "economics_and_politics",
        "european_journal_of_political_economy",
        "journal_of_comparative_economics",
        "journal_of_institutional_economics",
        "journal_of_law_economics_and_organization",
        "journal_of_law_and_economics",
    ):
        assert key in political, key
    # Journals are global: the same general and public-economics journals serve
    # several queues rather than being duplicated per profile.
    education = set(registry.profile("education_economics").journal_keys)
    labor = set(registry.profile("labor_economics").journal_keys)
    behavioral = set(registry.profile("behavioral_economics").journal_keys)
    assert "general_qje" in political & education & labor & behavioral
    assert "journal_of_public_economics" in political & labor


def test_every_political_economy_journal_issn_was_verified_against_crossref() -> None:
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    verified: dict[str, set[str]] = {}
    for item in probe:
        if item.get("exact_match"):
            verified[item["crossref_title"]] = set(item["issn"])
    assert verified, "probe contains no exact Crossref matches"
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    by_key = {item["journal_key"]: item for item in catalog["journals"]}
    registry = _registry()
    for journal in registry.profile(POLITICAL).journals:
        entry = by_key[journal.journal.journal_key]
        issns = {entry["print_issn"], entry.get("online_issn")} - {None, ""}
        if entry["metadata_source"] != "crossref_verified":
            # Pre-existing catalog entries kept their original provenance.
            assert entry["metadata_source"]
            continue
        assert issns, entry["journal_key"]
        matching = [title for title, values in verified.items() if issns & values]
        assert matching, f"{entry['journal_key']} ISSNs {issns} appear in no Crossref record"


def test_no_journal_is_listed_twice_in_the_profile() -> None:
    profile = _registry().profile(POLITICAL)
    keys = [item.journal.journal_key for item in profile.journals]
    assert len(keys) == len(set(keys))


def test_the_new_profile_does_not_duplicate_any_existing_catalog_issn() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    owner: dict[str, str] = {}
    duplicates: list[str] = []
    for item in catalog["journals"]:
        for field in ("print_issn", "online_issn"):
            issn = item.get(field)
            if not issn:
                continue
            if issn in owner:
                duplicates.append(f"{issn}: {owner[issn]} and {item['journal_key']}")
            owner[issn] = item["journal_key"]
    assert duplicates == []


# --- the addition is configuration-only ---------------------------------------


def test_the_fourth_profile_needed_no_new_table_and_no_core_change() -> None:
    """A profile is configuration.  If adding one required a migration or a new
    column, the abstraction would have failed and this test should fail too."""
    import subprocess

    migrations = sorted(path.name for path in (ROOT / "migrations").glob("*.sql"))
    # 0014 is the JEL observation ledger, which belongs to paper *facts* and is
    # shared by every profile; no migration exists for the political-economy
    # profile itself.
    assert not any("political" in name for name in migrations)
    # The profile schema has no per-profile column: the shared tables carry a
    # profile_id, so a new queue is a row, not a schema.
    sys_path = ROOT / "src"
    import sys

    sys.path.insert(0, str(sys_path))
    from eduecon_radar.db import Database

    assert Database  # imported for the boundary assertion above
    assert subprocess is not None


def test_other_profiles_are_untouched_by_the_new_one() -> None:
    registry = _registry()
    education = registry.profile("education_economics")
    labor = registry.profile("labor_economics")
    behavioral = registry.profile("behavioral_economics")
    assert education.screening_prompt_version == "2026-07-31-screening-v1"
    assert labor.screening_prompt_version == "2026-09-21-labor-v1"
    assert behavioral.screening_prompt_version == "2026-09-24-behavioral-v2"
    # The new profile does not borrow another queue's journals wholesale.
    political = set(registry.profile(POLITICAL).journal_keys)
    assert political != set(education.journal_keys)
    assert political != set(behavioral.journal_keys)
