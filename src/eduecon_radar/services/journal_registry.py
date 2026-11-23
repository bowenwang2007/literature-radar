from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from eduecon_radar.journals import load_journals
from eduecon_radar.resources import resource
from eduecon_radar.schemas import JournalConfig

REGISTRY_VERSION_FIELD = "registry_version"
#: The global catalog names its version differently; both are accepted so
#: one loader serves the catalog and any legacy registry file.
CATALOG_VERSION_FIELD = "catalog_version"
#: Group key used when a registry file carries no groups (the global catalog).
ALL_GROUP = "all"
CORE_GROUP = "education_economics_core"
ADJACENT_GROUP = "education_economics_adjacent"
GENERAL_GROUP = "general_economics"
CUSTOM_GROUP = "custom"
DEFAULT_PERSONAL_JOURNALS = (
    "economics_of_education_review",
    "education_economics",
    "education_finance_and_policy",
)
_ISSN_RE = re.compile(r"^[0-9]{7}[0-9X]$")


def normalize_issn(value: str | None) -> str | None:
    compact = re.sub(r"[^0-9Xx]", "", value or "").upper()
    if not compact:
        return None
    if not _ISSN_RE.fullmatch(compact):
        raise ValueError(f"Invalid ISSN format: {value}")
    total = sum(
        int(character) * weight
        for character, weight in zip(compact[:7], range(8, 1, -1), strict=True)
    )
    check_value = (11 - total % 11) % 11
    expected = "X" if check_value == 10 else str(check_value)
    if compact[-1] != expected:
        raise ValueError(f"Invalid ISSN checksum: {value}")
    return f"{compact[:4]}-{compact[4:]}"


def normalize_journal_name(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", value or "").casefold().replace("&", " and ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


@dataclass(frozen=True)
class JournalGroup:
    key: str
    display_name: str
    description: str


@dataclass(frozen=True)
class RegistryJournal:
    journal_key: str
    canonical_name: str
    short_name: str
    aliases: tuple[str, ...]
    print_issn: str
    online_issn: str | None
    publisher: str
    journal_group: str
    subject_tags: tuple[str, ...]
    enabled: bool
    display_order: int
    retrieval_priority: int
    metadata_source: str
    registry_version: str

    @property
    def issns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item for item in (self.print_issn, self.online_issn) if item))

    @property
    def search_values(self) -> tuple[str, ...]:
        return (
            self.canonical_name,
            self.short_name,
            *self.aliases,
            *self.issns,
        )

    def as_journal_config(self) -> JournalConfig:
        return JournalConfig(
            name=self.canonical_name,
            short_name=self.short_name,
            issn=self.print_issn,
            eissn=self.online_issn,
            priority="high" if self.retrieval_priority == 1 else "medium",
            rank_source=f"Phase 7A journal registry {self.registry_version}",
            enabled=self.enabled,
        )


@dataclass(frozen=True)
class JournalRegistry:
    version: str
    groups: dict[str, JournalGroup]
    journals: tuple[RegistryJournal, ...]

    def by_key(self, journal_key: str) -> RegistryJournal:
        for journal in self.journals:
            if journal.journal_key == journal_key:
                return journal
        raise KeyError(journal_key)

    def expand_group(self, group: str) -> tuple[RegistryJournal, ...]:
        return tuple(
            sorted(
                (
                    journal
                    for journal in self.journals
                    if journal.enabled and journal.journal_group == group
                ),
                key=lambda journal: (journal.display_order, journal.canonical_name),
            )
        )

    def search(self, query: str) -> tuple[RegistryJournal, ...]:
        needle_name = normalize_journal_name(query)
        needle_issn = re.sub(r"[^0-9Xx]", "", query).upper()
        matches = []
        for journal in self.journals:
            names = tuple(normalize_journal_name(value) for value in journal.search_values)
            issns = tuple(re.sub(r"[^0-9X]", "", value) for value in journal.issns)
            if (
                (needle_name and any(needle_name in value for value in names))
                or (needle_issn and any(needle_issn in value for value in issns))
            ):
                matches.append(journal)
        return tuple(matches)

    def resolve(self, value: str) -> RegistryJournal | None:
        try:
            issn = normalize_issn(value)
        except ValueError:
            issn = None
        if issn:
            matches = [journal for journal in self.journals if issn in journal.issns]
            return matches[0] if len(matches) == 1 else None
        normalized = normalize_journal_name(value)
        matches = [
            journal
            for journal in self.journals
            if normalized
            in {
                normalize_journal_name(journal.canonical_name),
                normalize_journal_name(journal.short_name),
                *(normalize_journal_name(alias) for alias in journal.aliases),
            }
        ]
        return matches[0] if len(matches) == 1 else None

    def conflict_for_issn(
        self, print_issn: str, online_issn: str | None = None
    ) -> RegistryJournal | None:
        requested = {normalize_issn(print_issn)}
        if online_issn:
            requested.add(normalize_issn(online_issn))
        for journal in self.journals:
            if requested.intersection(journal.issns):
                return journal
        return None


def load_journal_registry(
    path: Path | None = None,
    *,
    general_journals_path: Path | None = None,
) -> JournalRegistry:
    registry_path = path or Path(str(resource("config/journals/catalog.json")))
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    version = str(
        payload.get(REGISTRY_VERSION_FIELD) or payload.get(CATALOG_VERSION_FIELD) or ""
    ).strip()
    if not version:
        raise ValueError("Journal registry version is required")
    groups = {
        key: JournalGroup(
            key=key,
            display_name=str(value["display_name"]),
            description=str(value.get("description") or ""),
        )
        for key, value in dict(payload.get("groups") or {}).items()
    }
    journals = [_journal_from_payload(item, version) for item in payload.get("journals") or []]
    if not groups:
        groups = {
            ALL_GROUP: JournalGroup(
                key=ALL_GROUP,
                display_name="全部期刊",
                description="全局期刊清单；研究方向决定监控哪些期刊。",
            )
        }
    if len({journal.journal_key for journal in journals}) != len(journals):
        raise ValueError("Journal registry contains duplicate journal_key values")
    seen_issns: dict[str, str] = {}
    for journal in journals:
        for issn in journal.issns:
            owner = seen_issns.setdefault(issn, journal.journal_key)
            if owner != journal.journal_key:
                raise ValueError(f"ISSN {issn} belongs to multiple registry journals")
    if general_journals_path is not None:
        known_issns = set(seen_issns)
        for position, item in enumerate(
            load_journals(general_journals_path).journals, start=100
        ):
            item_issns = {normalize_issn(item.issn)}
            if item.eissn:
                item_issns.add(normalize_issn(item.eissn))
            if item_issns.intersection(known_issns):
                continue
            key = f"general_{normalize_journal_name(item.short_name).replace(' ', '_')}"
            journals.append(
                RegistryJournal(
                    journal_key=key,
                    canonical_name=item.name,
                    short_name=item.short_name,
                    aliases=(item.name, item.short_name),
                    print_issn=normalize_issn(item.issn) or "",
                    online_issn=normalize_issn(item.eissn) if item.eissn else None,
                    publisher="",
                    journal_group=GENERAL_GROUP,
                    subject_tags=("general_economics",),
                    enabled=item.enabled,
                    display_order=position,
                    retrieval_priority=3,
                    metadata_source="journals.yaml",
                    registry_version=version,
                )
            )
            known_issns.update(item_issns)
    return JournalRegistry(version=version, groups=groups, journals=tuple(journals))


def make_custom_journal(
    registry: JournalRegistry,
    *,
    canonical_name: str,
    short_name: str,
    print_issn: str,
    online_issn: str | None = None,
) -> RegistryJournal:
    name = canonical_name.strip()
    short = short_name.strip()
    if not name or not short:
        raise ValueError("Custom journal name and short name are required")
    normalized_print = normalize_issn(print_issn)
    normalized_online = normalize_issn(online_issn) if online_issn else None
    conflict = registry.conflict_for_issn(normalized_print or "", normalized_online)
    if conflict:
        raise ValueError(
            f"ISSN conflicts with registered journal: {conflict.canonical_name}"
        )
    key = f"custom_{normalize_journal_name(name).replace(' ', '_')}"
    return RegistryJournal(
        journal_key=key,
        canonical_name=name,
        short_name=short,
        aliases=(name, short),
        print_issn=normalized_print or "",
        online_issn=normalized_online,
        publisher="",
        journal_group=CUSTOM_GROUP,
        subject_tags=(),
        enabled=True,
        display_order=999,
        retrieval_priority=4,
        metadata_source="personal_config",
        registry_version=registry.version,
    )


def save_personal_journals(path: Path, journals: tuple[RegistryJournal, ...]) -> None:
    """Persist custom journals only after an explicit user action."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "registry_version": journals[0].registry_version if journals else "",
        "journals": [
            {
                "journal_key": journal.journal_key,
                "canonical_name": journal.canonical_name,
                "short_name": journal.short_name,
                "aliases": list(journal.aliases),
                "print_issn": journal.print_issn,
                "online_issn": journal.online_issn,
                "publisher": journal.publisher,
                "journal_group": CUSTOM_GROUP,
                "subject_tags": list(journal.subject_tags),
                "enabled": journal.enabled,
                "display_order": journal.display_order,
                "retrieval_priority": journal.retrieval_priority,
                "metadata_source": "personal_config",
                "registry_version": journal.registry_version,
            }
            for journal in journals
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _journal_from_payload(payload: dict, version: str) -> RegistryJournal:
    """One journal, from the catalog or from a legacy registry file.

    Only identity is required.  A catalog entry carries a default priority and
    no per-entry version; a legacy entry carries its own group and version.
    """
    required = {
        "journal_key",
        "canonical_name",
        "short_name",
        "aliases",
        "print_issn",
        "online_issn",
        "publisher",
        "subject_tags",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Journal registry entry missing fields: {sorted(missing)}")
    if payload.get("registry_version") and str(payload["registry_version"]) != version:
        raise ValueError("Journal entry registry_version does not match registry")
    return RegistryJournal(
        journal_key=str(payload["journal_key"]),
        canonical_name=str(payload["canonical_name"]),
        short_name=str(payload["short_name"]),
        aliases=tuple(str(item) for item in payload["aliases"]),
        print_issn=normalize_issn(str(payload["print_issn"])) or "",
        online_issn=(
            normalize_issn(str(payload["online_issn"]))
            if payload["online_issn"]
            else None
        ),
        publisher=str(payload["publisher"]),
        journal_group=str(payload.get("journal_group") or ALL_GROUP),
        subject_tags=tuple(str(item) for item in payload["subject_tags"]),
        enabled=bool(payload.get("enabled", True)),
        display_order=int(payload.get("display_order") or 0),
        retrieval_priority=int(
            payload.get("retrieval_priority") or payload.get("default_priority") or 3
        ),
        metadata_source=str(payload.get("metadata_source") or "catalog"),
        registry_version=version,
    )
