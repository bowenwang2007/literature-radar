from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from eduecon_radar.schemas import JournalConfig


class RankingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    frozen_at: str


class JournalFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ranking: RankingConfig
    journals: list[JournalConfig]


def load_journals(path: Path) -> JournalFile:
    if not path.exists():
        raise FileNotFoundError(f"Journal configuration not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return JournalFile.model_validate(raw)
