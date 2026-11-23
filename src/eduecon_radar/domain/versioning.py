"""Read-only domain configuration version registry and rollback selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DomainVersionRecord:
    bundle_version: str
    status: str
    ontology_version: str
    exclusion_context_version: str
    scoring_version: str
    prompt_version: str
    semantic_version: str
    immutable: bool
    notes: str


class DomainVersionRegistry:
    def __init__(self, path: Path) -> None:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))["version_registry"]
        self.profile_id = str(payload["profile_id"])
        self.current_version = str(payload["current"])
        self._versions = {
            item["bundle_version"]: DomainVersionRecord(**item) for item in payload["versions"]
        }
        if self.current_version not in self._versions:
            raise ValueError("current domain bundle is absent from version registry")

    @property
    def versions(self) -> tuple[DomainVersionRecord, ...]:
        return tuple(self._versions.values())

    def load(self, bundle_version: str | None = None) -> DomainVersionRecord:
        selected = bundle_version or self.current_version
        try:
            return self._versions[selected]
        except KeyError as exc:
            raise ValueError(f"unknown domain bundle version: {selected}") from exc

    def rollback_target(self) -> DomainVersionRecord:
        versions = list(self._versions)
        index = versions.index(self.current_version)
        if index == 0:
            raise ValueError("no earlier domain bundle is registered")
        return self._versions[versions[index - 1]]
