"""Research profiles: what a reading queue is *for*.

A profile is the unit of assessment.  A paper fact (DOI, title, abstract,
translation, provenance) is global and stored once; whether that paper is
relevant, and whether the researcher wants to keep it, depends on the profile
that is asking.  Education Economics is the default profile and behaves exactly
like the single-domain application did.

The journals themselves are global (``config/journals/catalog.json``).  A
profile only selects which of them it monitors and in what order, so QJE can be
monitored by several profiles without being duplicated.

Layout::

    config/profiles/<profile_id>/profile.yaml
    config/profiles/<profile_id>/prompts/screening_primary_system.txt
    config/profiles/<profile_id>/prompts/screening_primary_user.txt
    config/profiles/<profile_id>/prompts/screening_review_system.txt
    config/profiles/<profile_id>/prompts/screening_review_user.txt
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from eduecon_radar.resources import resource
from eduecon_radar.services.journal_registry import (
    CUSTOM_GROUP,
    JournalGroup,
    JournalRegistry,
    RegistryJournal,
    normalize_issn,
)

DEFAULT_PROFILE_ID = "education_economics"
CATALOG_VERSION_FIELD = "catalog_version"
REQUIRED_PROMPTS = (
    "screening_primary_system.txt",
    "screening_primary_user.txt",
    "screening_review_system.txt",
    "screening_review_user.txt",
)


class ProfileConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class CatalogJournal:
    journal_key: str
    canonical_name: str
    short_name: str
    aliases: tuple[str, ...]
    print_issn: str
    online_issn: str | None
    publisher: str
    subject_tags: tuple[str, ...]
    default_priority: int
    metadata_source: str
    catalog_version: str
    #: Subject area the catalog files this journal under.  Presentation only:
    #: it lets the UI group the global library without inventing a second
    #: taxonomy, and it never decides what a profile monitors.
    subject: str = ""

    @property
    def issns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                item for item in (self.print_issn, self.online_issn) if item
            )
        )


@dataclass(frozen=True)
class ProfileJournal:
    journal: CatalogJournal
    priority: int
    group_key: str


@dataclass(frozen=True)
class ScreeningDimensions:
    """What this queue's judgement dimensions mean.

    The stored classification keeps stable field names (``education_centrality``
    and ``economic_relevance``) so historical rows stay readable, but the
    *meaning* of "central" is profile content: education centrality for one
    queue, behavioural-mechanism centrality for another.  Declaring it here —
    instead of hard-coding education wording in the validator and the prompts —
    is what lets several research areas share one screening contract.
    """

    centrality_label: str
    centrality_definition: str
    economics_definition: str

    def as_prompt_context(self) -> dict[str, str]:
        return {
            "centrality_label": self.centrality_label,
            "centrality_definition": self.centrality_definition,
            "economics_definition": self.economics_definition,
        }


def _default_dimensions(display_name: str) -> ScreeningDimensions:
    return ScreeningDimensions(
        centrality_label=f"{display_name}中心性",
        centrality_definition=(
            f"the paper's subject is the core research object of the {display_name} queue"
        ),
        economics_definition=(
            "the paper contains substantive economic content: incentives, markets, "
            "policy, welfare, or credible causal identification"
        ),
    )


@dataclass(frozen=True)
class ProductProfile:
    profile_id: str
    display_name: str
    display_name_en: str
    description: str
    status: str
    screening_prompt_version: str
    prompts_dir: Path
    journals: tuple[ProfileJournal, ...]
    groups: tuple[JournalGroup, ...]
    dimensions: ScreeningDimensions
    subject_signals: dict[str, tuple[str, ...]]
    #: Optional per-profile latency budget for a screening call.  A queue whose
    #: prompt asks for a longer analysis may legitimately need more than the
    #: installation default; leaving it unset keeps the default behaviour.
    screening_timeout_seconds: float | None = None

    @property
    def journal_keys(self) -> tuple[str, ...]:
        return tuple(item.journal.journal_key for item in self.journals)

    def prompt_context(self) -> dict[str, str]:
        """Values a screening prompt of this profile may reference."""
        signals: list[str] = []
        jel = self.subject_signals.get("jel_codes") or ()
        topics = self.subject_signals.get("topics") or ()
        if jel:
            signals.append("JEL codes: " + ", ".join(jel))
        if topics:
            signals.append("Typical topics: " + "; ".join(topics))
        return {
            **self.dimensions.as_prompt_context(),
            "subject_signals": "\n".join(signals),
        }

    def registry(self, *, extra_journals: tuple[RegistryJournal, ...] = ()) -> JournalRegistry:
        """Adapt the profile selection to the UI's journal registry shape.

        The picker keeps working with ``RegistryJournal`` objects, so a profile
        only changes *which* journals appear and how they are grouped.
        """
        journals = [
            RegistryJournal(
                journal_key=item.journal.journal_key,
                canonical_name=item.journal.canonical_name,
                short_name=item.journal.short_name,
                aliases=item.journal.aliases,
                print_issn=item.journal.print_issn,
                online_issn=item.journal.online_issn,
                publisher=item.journal.publisher,
                journal_group=item.group_key,
                subject_tags=item.journal.subject_tags,
                enabled=True,
                display_order=index + 1,
                retrieval_priority=item.priority,
                metadata_source=item.journal.metadata_source,
                registry_version=item.journal.catalog_version,
            )
            for index, item in enumerate(self.journals)
        ]
        journals.extend(extra_journals)
        groups = {group.key: group for group in self.groups}
        groups.setdefault(
            CUSTOM_GROUP,
            JournalGroup(
                key=CUSTOM_GROUP,
                display_name="自定义期刊",
                description="由本人添加的期刊；保存在本地配置中。",
            ),
        )
        return JournalRegistry(
            version=self.profile_id,
            groups=groups,
            journals=tuple(journals),
        )


@dataclass(frozen=True)
class ProfileRegistry:
    catalog_version: str
    catalog: dict[str, CatalogJournal]
    profiles: dict[str, ProductProfile]
    default_profile_id: str
    #: subject key -> display name, from the global catalog.  Empty when the
    #: catalog declares no subject groups.
    catalog_subject_names: dict[str, str] = field(default_factory=dict)

    def catalog_subjects(self) -> tuple[tuple[str, str, int], ...]:
        """(subject key, display name, journal count), largest first.

        Used by the UI to let a researcher browse the whole journal world by
        area.  Derived from the catalog on every call so it cannot drift from it.
        """
        counts: dict[str, int] = {}
        for entry in self.catalog.values():
            counts[entry.subject] = counts.get(entry.subject, 0) + 1
        ordered = sorted(
            counts.items(),
            key=lambda item: (-item[1], self.catalog_subject_names.get(item[0], item[0])),
        )
        return tuple(
            (key, self.catalog_subject_names.get(key) or "其他", count)
            for key, count in ordered
        )

    def profile(self, profile_id: str | None = None) -> ProductProfile:
        key = (profile_id or self.default_profile_id).strip()
        if key not in self.profiles:
            raise ProfileConfigError(
                f"Unknown research profile: {key!r}. Available: "
                f"{', '.join(sorted(self.profiles))}"
            )
        return self.profiles[key]

    def profile_choices(self) -> tuple[tuple[str, str], ...]:
        """(profile_id, display_name) pairs, default first."""
        ordered = sorted(
            self.profiles.values(),
            key=lambda item: (item.profile_id != self.default_profile_id, item.display_name),
        )
        return tuple((item.profile_id, item.display_name) for item in ordered)


def _catalog_path(path: Path | None) -> Path:
    if path is not None:
        return path
    candidate = Path(str(resource("config/journals/catalog.json")))
    if not candidate.is_file():
        raise ProfileConfigError("Global journal catalog is missing")
    return candidate


def load_journal_catalog(
    path: Path | None = None,
) -> tuple[str, dict[str, CatalogJournal], dict[str, str], dict[str, str]]:
    """Return (version, catalog, subject display names, key -> subject).

    The catalog's subject groups are optional presentation metadata.  A catalog
    without them stays valid, and in that case every journal reports an empty
    subject rather than being forced into an invented one.
    """
    target = _catalog_path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    version = str(payload.get(CATALOG_VERSION_FIELD) or "").strip()
    if not version:
        raise ProfileConfigError("Journal catalog version is required")
    subject_names = {
        str(key): str(value.get("display_name") or key)
        for key, value in dict(payload.get("groups") or {}).items()
    }
    catalog: dict[str, CatalogJournal] = {}
    issn_owner: dict[str, str] = {}
    for item in payload.get("journals") or []:
        entry = CatalogJournal(
            journal_key=str(item["journal_key"]),
            canonical_name=str(item["canonical_name"]),
            short_name=str(item["short_name"]),
            aliases=tuple(str(value) for value in item.get("aliases") or ()),
            print_issn=normalize_issn(str(item.get("print_issn") or "")) or "",
            online_issn=(
                normalize_issn(str(item["online_issn"]))
                if item.get("online_issn")
                else None
            ),
            publisher=str(item.get("publisher") or ""),
            subject_tags=tuple(str(value) for value in item.get("subject_tags") or ()),
            default_priority=int(item.get("default_priority") or 3),
            metadata_source=str(item.get("metadata_source") or ""),
            catalog_version=version,
            subject=str(item.get("subject") or ""),
        )
        if entry.journal_key in catalog:
            raise ProfileConfigError(f"Duplicate journal_key in catalog: {entry.journal_key}")
        catalog[entry.journal_key] = entry
        for issn in entry.issns:
            owner = issn_owner.setdefault(issn, entry.journal_key)
            if owner != entry.journal_key:
                raise ProfileConfigError(
                    f"ISSN {issn} is claimed by both {owner} and {entry.journal_key}"
                )
    if not catalog:
        raise ProfileConfigError("Journal catalog is empty")
    return version, catalog, subject_names, {
        key: entry.subject for key, entry in catalog.items()
    }


def load_profile(profile_dir: Path, *, catalog: dict[str, CatalogJournal]) -> ProductProfile:
    profile_file = profile_dir / "profile.yaml"
    if not profile_file.is_file():
        raise ProfileConfigError(f"Profile definition is missing: {profile_file}")
    payload = yaml.safe_load(profile_file.read_text(encoding="utf-8")) or {}
    profile_id = str(payload.get("profile_id") or "").strip()
    if not profile_id:
        raise ProfileConfigError("profile_id is required")
    if profile_id != profile_dir.name:
        raise ProfileConfigError(
            f"profile_id {profile_id!r} must match its directory {profile_dir.name!r}"
        )
    prompts_dir = profile_dir / "prompts"
    missing = [name for name in REQUIRED_PROMPTS if not (prompts_dir / name).is_file()]
    if missing:
        raise ProfileConfigError(
            f"Profile {profile_id} is missing screening prompts: {missing}"
        )
    journals: list[ProfileJournal] = []
    groups: list[JournalGroup] = []
    seen: set[str] = set()
    for position, group in enumerate(payload.get("journal_groups") or [], start=1):
        key = str(group.get("key") or "").strip()
        if not key:
            raise ProfileConfigError(f"Profile {profile_id} has a journal group without a key")
        groups.append(
            JournalGroup(
                key=key,
                display_name=str(group.get("display_name") or key),
                description=str(group.get("description") or ""),
            )
        )
        for journal_key in group.get("journal_keys") or []:
            key_value = str(journal_key)
            if key_value not in catalog:
                raise ProfileConfigError(
                    f"Profile {profile_id} references unknown journal {key_value!r}"
                )
            if key_value in seen:
                raise ProfileConfigError(
                    f"Profile {profile_id} lists journal {key_value!r} twice"
                )
            seen.add(key_value)
            journals.append(
                ProfileJournal(journal=catalog[key_value], priority=position, group_key=key)
            )
    if not journals:
        raise ProfileConfigError(f"Profile {profile_id} monitors no journals")
    display_name = str(payload.get("display_name") or profile_id)
    return ProductProfile(
        profile_id=profile_id,
        display_name=display_name,
        display_name_en=str(payload.get("display_name_en") or profile_id),
        description=str(payload.get("description") or ""),
        status=str(payload.get("status") or "active"),
        screening_prompt_version=str(
            payload.get("screening_prompt_version") or "unversioned"
        ),
        prompts_dir=prompts_dir,
        journals=tuple(journals),
        groups=tuple(groups),
        dimensions=_dimensions_from_payload(payload, display_name),
        subject_signals=_subject_signals_from_payload(payload),
        screening_timeout_seconds=_optional_positive_float(
            payload.get("screening_timeout_seconds")
        ),
    )


def _optional_positive_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProfileConfigError(f"Invalid screening_timeout_seconds: {value!r}") from exc
    if number <= 0:
        raise ProfileConfigError("screening_timeout_seconds must be positive")
    return number


def _dimensions_from_payload(
    payload: dict, display_name: str
) -> ScreeningDimensions:
    """Profile-declared meaning of the judgement dimensions.

    Optional: a profile that declares nothing keeps the neutral default, which
    is what an existing profile file did before this became configurable.
    """
    declared = payload.get("screening_dimensions")
    if not isinstance(declared, dict):
        return _default_dimensions(display_name)
    fallback = _default_dimensions(display_name)
    return ScreeningDimensions(
        centrality_label=str(
            declared.get("centrality_label") or fallback.centrality_label
        ),
        centrality_definition=str(
            declared.get("centrality_definition") or fallback.centrality_definition
        ),
        economics_definition=str(
            declared.get("economics_definition") or fallback.economics_definition
        ),
    )


def _subject_signals_from_payload(payload: dict) -> dict[str, tuple[str, ...]]:
    declared = payload.get("subject_signals")
    if not isinstance(declared, dict):
        return {}
    signals: dict[str, tuple[str, ...]] = {}
    for key, value in declared.items():
        if isinstance(value, (list, tuple)):
            signals[str(key)] = tuple(str(item) for item in value)
    return signals


def load_profile_registry(
    profiles_root: Path | None = None,
    *,
    catalog_path: Path | None = None,
    default_profile_id: str = DEFAULT_PROFILE_ID,
) -> ProfileRegistry:
    version, catalog, subject_names, _subjects = load_journal_catalog(catalog_path)
    root = profiles_root
    if root is None:
        root = Path(str(resource("config/profiles")))
    if not root.is_dir():
        raise ProfileConfigError(f"Profile root is missing: {root}")
    profiles: dict[str, ProductProfile] = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        profile = load_profile(directory, catalog=catalog)
        profiles[profile.profile_id] = profile
    if not profiles:
        raise ProfileConfigError(f"No profiles found under {root}")
    if default_profile_id not in profiles:
        raise ProfileConfigError(
            f"Default profile {default_profile_id!r} is not defined under {root}"
        )
    return ProfileRegistry(
        catalog_version=version,
        catalog=catalog,
        profiles=profiles,
        default_profile_id=default_profile_id,
        catalog_subject_names=subject_names,
    )


def default_profile_registry(
    profiles_root: Path | None = None,
    *,
    catalog_path: Path | None = None,
    profile_id: str | None = None,
) -> JournalRegistry:
    """The journal registry of one profile, in the shape the UI and CLI expect.

    Callers that used to load a per-domain registry file now ask the profile
    registry, so there is exactly one place that decides which journals belong
    to a reading queue.
    """
    registry = load_profile_registry(
        profiles_root,
        catalog_path=catalog_path,
        default_profile_id=profile_id or DEFAULT_PROFILE_ID,
    )
    return registry.profile(profile_id).registry()
