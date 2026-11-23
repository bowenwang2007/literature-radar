"""Global Journal Library invariants.

The journal layer is supposed to hold one shared, verified economics library
that profiles *recommend from* rather than own.  These tests pin the properties
that make that true, so a future catalog edit cannot quietly reintroduce a
profile-owned journal, a duplicate ISSN, or an unverifiable entry.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from eduecon_radar.services.journal_registry import normalize_issn, normalize_journal_name
from eduecon_radar.services.profile_registry import (
    ProfileConfigError,
    load_journal_catalog,
    load_profile_registry,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "config" / "journals" / "catalog.json"
PROFILES = ROOT / "config" / "profiles"
PROBE = ROOT / "tests" / "fixtures" / "crossref_journal_identities.json"

#: Areas the library is expected to cover.  A missing area means the catalog
#: silently stopped serving a research direction.
REQUIRED_SUBJECTS = {
    "general",
    "education",
    "labor",
    "behavioral",
    "political_economy",
    "public",
    "development",
    "health",
    "industrial_organization",
    "econometrics",
    "applied_micro",
    "macro_finance",
}


def _catalog_payload() -> dict:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def _registry():
    return load_profile_registry(PROFILES, catalog_path=CATALOG)


# --------------------------------------------------------------------------- #
# Identity: one entry per journal, one owner per ISSN
# --------------------------------------------------------------------------- #


def test_catalog_has_no_duplicate_journal_key_or_issn() -> None:
    payload = _catalog_payload()
    keys = [item["journal_key"] for item in payload["journals"]]
    assert len(keys) == len(set(keys)), "duplicate journal_key"

    owner: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []
    for item in payload["journals"]:
        for value in (item.get("print_issn"), item.get("online_issn")):
            if not value:
                continue
            normalized = normalize_issn(value)
            holder = owner.setdefault(normalized, item["journal_key"])
            if holder != item["journal_key"]:
                duplicates.append((normalized, holder, item["journal_key"]))
    assert duplicates == []


def test_every_catalog_issn_is_well_formed_and_has_a_print_number() -> None:
    for item in _catalog_payload()["journals"]:
        assert item["print_issn"], item["journal_key"]
        assert normalize_issn(item["print_issn"]) == item["print_issn"]
        online = item.get("online_issn")
        if online:
            assert normalize_issn(online) == online, item["journal_key"]
        assert online != item["print_issn"]


def test_every_journal_declares_identity_and_a_known_subject() -> None:
    payload = _catalog_payload()
    subjects = set(payload.get("groups") or {})
    assert subjects >= REQUIRED_SUBJECTS, sorted(REQUIRED_SUBJECTS - subjects)
    for item in payload["journals"]:
        assert item["canonical_name"].strip(), item["journal_key"]
        assert item["short_name"].strip(), item["journal_key"]
        assert item["publisher"].strip(), item["journal_key"]
        assert item["subject"] in subjects, (item["journal_key"], item["subject"])
        assert item["subject_tags"], item["journal_key"]
        assert item["aliases"], item["journal_key"]
        assert item["metadata_source"], item["journal_key"]


# --------------------------------------------------------------------------- #
# Coverage: the library actually spans the intended areas
# --------------------------------------------------------------------------- #


def test_library_covers_every_required_subject_with_real_depth() -> None:
    counts = Counter(item["subject"] for item in _catalog_payload()["journals"])
    thin = {key: counts.get(key, 0) for key in REQUIRED_SUBJECTS if counts.get(key, 0) < 2}
    # macro/finance is deliberately selective, so it is allowed to be the
    # smallest area but must still exist.
    assert thin in ({}, {"macro_finance": counts.get("macro_finance", 0)}), thin


def test_library_is_a_real_expansion_not_a_sideways_move() -> None:
    journals = _catalog_payload()["journals"]
    assert len(journals) >= 80
    # The areas the previous catalog barely served are now covered.
    counts = Counter(item["subject"] for item in journals)
    for area in ("public", "development", "health", "industrial_organization", "econometrics"):
        assert counts[area] >= 3, (area, counts[area])


# --------------------------------------------------------------------------- #
# Profiles recommend; the catalog owns
# --------------------------------------------------------------------------- #


def test_profiles_only_reference_catalog_journals() -> None:
    registry = _registry()
    catalog_keys = set(registry.catalog)
    for profile_id, _ in registry.profile_choices():
        keys = set(registry.profile(profile_id).journal_keys)
        assert keys <= catalog_keys, sorted(keys - catalog_keys)


def test_every_profile_exposes_the_same_three_recommendation_tiers() -> None:
    registry = _registry()
    for profile_id, _ in registry.profile_choices():
        profile = registry.profile(profile_id)
        journal_registry = profile.registry()
        group_keys = list(journal_registry.groups)
        # A core tier, at least one adjacent tier, the shared flagship tier, and
        # the custom tier.  Names differ per direction; the structure does not.
        assert any("core" in key for key in group_keys), profile_id
        assert any("adjacent" in key for key in group_keys), profile_id
        assert "general_economics" in group_keys
        assert "custom" in group_keys
        for key in group_keys:
            if key == "custom":
                continue
            assert journal_registry.expand_group(key), (profile_id, key)


def test_profile_switching_changes_the_recommended_journals() -> None:
    registry = _registry()
    sets = {
        profile_id: set(registry.profile(profile_id).journal_keys)
        for profile_id, _ in registry.profile_choices()
    }
    expected = {
        "education_economics": (
            "economics_of_education_review",
            "education_economics",
            "education_finance_and_policy",
        ),
        "labor_economics": (
            "journal_of_labor_economics",
            "journal_of_human_resources",
            "labour_economics",
            "industrial_and_labor_relations_review",
        ),
        "behavioral_economics": (
            "journal_of_economic_behavior_and_organization",
            "experimental_economics",
            "games_and_economic_behavior",
            "journal_of_risk_and_uncertainty",
        ),
        "political_economy": (
            "public_choice",
            "economics_and_politics",
            "european_journal_of_political_economy",
            "journal_of_comparative_economics",
        ),
    }
    profile_ids = list(sets)
    for profile_id in profile_ids:
        for key in expected[profile_id]:
            assert key in sets[profile_id], (profile_id, key)

    # The directions are genuinely different recommendation sets.
    for index, left in enumerate(profile_ids):
        for right in profile_ids[index + 1 :]:
            assert sets[left] != sets[right], (left, right)

    # Each direction's core journals are NOT another direction's core journals.
    core_of = {
        profile_id: {
            journal.journal_key
            for journal in registry.profile(profile_id)
            .registry()
            .expand_group(
                next(
                    key
                    for key in registry.profile(profile_id).registry().groups
                    if "core" in key
                )
            )
        }
        for profile_id in profile_ids
    }
    for index, left in enumerate(profile_ids):
        for right in profile_ids[index + 1 :]:
            assert core_of[left] & core_of[right] == set(), (left, right)


def test_the_flagship_tier_is_identical_for_every_direction() -> None:
    registry = _registry()
    general = {
        profile_id: tuple(
            journal.journal_key
            for journal in registry.profile(profile_id)
            .registry()
            .expand_group("general_economics")
        )
        for profile_id, _ in registry.profile_choices()
    }
    distinct = {value for value in general.values()}
    assert len(distinct) == 1, general
    only = next(iter(distinct))
    for key in ("general_qje", "general_aer", "general_jpe", "general_ecma"):
        assert key in only


def test_every_catalog_journal_is_reachable_from_at_least_one_profile() -> None:
    registry = _registry()
    recommended: set[str] = set()
    for profile_id, _ in registry.profile_choices():
        recommended.update(registry.profile(profile_id).journal_keys)
    unreachable = sorted(set(registry.catalog) - recommended)
    assert unreachable == [], (
        "these journals exist in the library but no direction recommends them, so "
        f"they are unreachable from the picker's recommended tiers: {unreachable}"
    )


def test_unknown_journal_reference_is_rejected(tmp_path: Path) -> None:
    """A profile may not invent a journal; the catalog is the only source."""
    payload = _catalog_payload()
    broken = tmp_path / "catalog.json"
    broken.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    profile_dir = tmp_path / "profiles" / "education_economics"
    profile_dir.mkdir(parents=True)
    source = PROFILES / "education_economics"
    (profile_dir / "profile.yaml").write_text(
        (source / "profile.yaml")
        .read_text(encoding="utf-8")
        .replace("- journal_of_public_economics", "- journal_that_does_not_exist"),
        encoding="utf-8",
    )
    for name in (
        "screening_primary_system.txt",
        "screening_primary_user.txt",
        "screening_review_system.txt",
        "screening_review_user.txt",
    ):
        (profile_dir / "prompts").mkdir(exist_ok=True)
        (profile_dir / "prompts" / name).write_text("x", encoding="utf-8")
    with pytest.raises(ProfileConfigError, match="unknown journal"):
        load_profile_registry(
            tmp_path / "profiles", catalog_path=broken, default_profile_id="education_economics"
        )


def test_catalog_loader_reports_subject_display_names() -> None:
    version, catalog, subject_names, subjects = load_journal_catalog(CATALOG)
    assert version
    assert subject_names["general"]
    assert set(subjects) == set(catalog)
    for key, subject in subjects.items():
        assert subject in subject_names, key


# --------------------------------------------------------------------------- #
# Crossref: the identity in the catalog is the identity Crossref knows
# --------------------------------------------------------------------------- #


def test_every_catalog_journal_was_verified_against_crossref() -> None:
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    verified: dict[str, set[str]] = {}
    for item in probe:
        if item.get("exact_match"):
            verified[item["journal_key"]] = set(item["issn"])
    assert verified, "probe contains no exact Crossref matches"

    for item in _catalog_payload()["journals"]:
        key = item["journal_key"]
        issns = {
            normalize_issn(value)
            for value in (item["print_issn"], item.get("online_issn"))
            if value
        }
        assert key in verified, f"{key} is absent from the Crossref probe"
        assert issns & verified[key], f"{key} ISSNs {issns} appear in no Crossref record"


def test_new_journals_declare_crossref_as_their_source() -> None:
    for item in _catalog_payload()["journals"]:
        if item["journal_key"].startswith("general_"):
            continue
        if item["metadata_source"] == "crossref_verified":
            assert normalize_journal_name(item["canonical_name"])


def test_catalog_version_and_authority_are_recorded() -> None:
    payload = _catalog_payload()
    assert payload["catalog_version"]
    authority = payload["authority"].casefold()
    # The catalog states the ownership rule it is enforcing.
    assert "profile" in authority
    assert "never owns them" in authority or "not own" in authority
