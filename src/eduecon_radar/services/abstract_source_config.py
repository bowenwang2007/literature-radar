"""Versioned abstract source configuration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AbstractSourceConfig:
    name: str
    enabled: bool
    priority: int
    timeout_seconds: float
    max_retries: int
    rate_limit_per_second: float
    requires_api_key: bool
    quality_threshold: float
    allow_title_fallback: bool
    feature_status: str


@dataclass(frozen=True)
class AbstractSourcesConfig:
    version: str
    sources: tuple[AbstractSourceConfig, ...]
    fingerprint: str


def load_abstract_sources(path: Path) -> AbstractSourcesConfig:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw) or {}
    version = str(payload.get("version") or "")
    if not version or not isinstance(payload.get("sources"), dict):
        raise ValueError("abstract source config requires version and sources")
    sources = []
    for name, value in payload["sources"].items():
        config = AbstractSourceConfig(name=name, **value)
        if config.priority < 1 or config.timeout_seconds < 0 or config.max_retries < 0:
            raise ValueError(f"invalid abstract source config: {name}")
        if not 0 <= config.quality_threshold <= 1:
            raise ValueError(f"invalid quality threshold: {name}")
        sources.append(config)
    sources.sort(key=lambda item: item.priority)
    return AbstractSourcesConfig(version, tuple(sources), hashlib.sha256(raw).hexdigest())
