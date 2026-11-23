from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from eduecon_radar.domain.semantic_models import (
    EscalationDecision,
    FastSemanticResult,
    SemanticEngineConfig,
    SemanticPaperInput,
)


class PromptConfigurationError(RuntimeError):
    """Raised when a versioned semantic prompt bundle is invalid."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PromptEntry(_StrictModel):
    prompt_id: str
    version: str
    role: Literal["fast", "strong_review"]
    system_file: str
    user_file: str
    system_sha256: str
    user_sha256: str
    status: Literal["draft", "test_fixture", "released"]
    created_at: date
    notes: str


class PromptManifestBody(_StrictModel):
    profile_id: str
    version: str
    status: Literal["draft", "test_fixture", "released"]
    prompts: tuple[PromptEntry, ...]


class PromptManifest(_StrictModel):
    prompt_manifest: PromptManifestBody


@dataclass(frozen=True)
class PromptBundle:
    prompt_id: str
    prompt_version: str
    role: str
    system_prompt: str
    user_prompt: str
    system_sha256: str
    user_sha256: str


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest().upper()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_semantic_engine_config(config_dir: Path) -> SemanticEngineConfig:
    path = config_dir / "semantic_models.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return SemanticEngineConfig.model_validate(payload["semantic_engine"])
    except (OSError, KeyError, TypeError, yaml.YAMLError, ValidationError) as exc:
        raise PromptConfigurationError(f"Invalid semantic configuration {path}: {exc}") from exc


class SemanticPromptBuilder:
    """Build prompts from an explicit allow-list and verify immutable prompt hashes."""

    def __init__(self, prompts_dir: Path) -> None:
        self.prompts_dir = prompts_dir.resolve()
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> PromptManifestBody:
        path = self.prompts_dir / "manifest.yaml"
        try:
            parsed = PromptManifest.model_validate(
                yaml.safe_load(path.read_text(encoding="utf-8"))
            ).prompt_manifest
        except (OSError, TypeError, yaml.YAMLError, ValidationError) as exc:
            raise PromptConfigurationError(f"Invalid prompt manifest {path}: {exc}") from exc
        seen: set[tuple[str, str]] = set()
        for entry in parsed.prompts:
            identity = (entry.prompt_id, entry.version)
            if identity in seen:
                raise PromptConfigurationError(f"Duplicate prompt identity: {identity}")
            seen.add(identity)
            for filename, expected in (
                (entry.system_file, entry.system_sha256),
                (entry.user_file, entry.user_sha256),
            ):
                prompt_path = self._safe_path(filename)
                actual = file_sha256(prompt_path)
                if actual != expected.upper():
                    raise PromptConfigurationError(
                        f"Prompt hash mismatch for {filename}: expected {expected}, got {actual}"
                    )
        return parsed

    def _safe_path(self, filename: str) -> Path:
        path = (self.prompts_dir / filename).resolve()
        if path.parent != self.prompts_dir or not path.is_file():
            raise PromptConfigurationError(f"Unsafe or missing prompt path: {filename}")
        return path

    def validate(self) -> dict[str, object]:
        return {
            "profile_id": self.manifest.profile_id,
            "version": self.manifest.version,
            "status": self.manifest.status,
            "prompt_count": len(self.manifest.prompts),
            "valid": True,
        }

    def build_fast(self, paper: SemanticPaperInput) -> PromptBundle:
        entry = self._entry("fast")
        return self._bundle(entry, {"paper": paper.model_dump(mode="json")})

    def build_strong(
        self,
        paper: SemanticPaperInput,
        fast: FastSemanticResult,
        escalation: EscalationDecision,
    ) -> PromptBundle:
        entry = self._entry("strong_review")
        payload = {
            "paper": paper.model_dump(mode="json"),
            "fast_result": fast.model_dump(mode="json"),
            "escalation": escalation.model_dump(mode="json"),
        }
        return self._bundle(entry, payload)

    def _entry(self, role: str) -> PromptEntry:
        matches = [entry for entry in self.manifest.prompts if entry.role == role]
        if len(matches) != 1:
            raise PromptConfigurationError(f"Expected one {role} prompt, found {len(matches)}")
        return matches[0]

    def _bundle(self, entry: PromptEntry, payload: dict[str, object]) -> PromptBundle:
        system = self._safe_path(entry.system_file).read_text(encoding="utf-8")
        template = self._safe_path(entry.user_file).read_text(encoding="utf-8")
        paper_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        marker = "{{PAPER_DATA_JSON}}"
        if template.count(marker) != 1:
            raise PromptConfigurationError(
                f"Prompt template {entry.user_file} must contain exactly one data marker"
            )
        user = template.replace(marker, paper_json)
        return PromptBundle(
            prompt_id=entry.prompt_id,
            prompt_version=entry.version,
            role=entry.role,
            system_prompt=system,
            user_prompt=user,
            system_sha256=entry.system_sha256,
            user_sha256=entry.user_sha256,
        )


def semantic_input_fingerprint(
    *,
    paper: SemanticPaperInput,
    prompt: PromptBundle,
    model_role: str,
    model_name: str,
    engine_version: str,
    escalation_policy_version: str,
    fast_result: FastSemanticResult | None = None,
) -> str:
    return canonical_sha256(
        {
            "paper": paper.model_dump(mode="json"),
            "prompt_id": prompt.prompt_id,
            "prompt_version": prompt.prompt_version,
            "prompt_hashes": [prompt.system_sha256, prompt.user_sha256],
            "model_role": model_role,
            "model_name": model_name,
            "engine_version": engine_version,
            "escalation_policy_version": escalation_policy_version,
            "fast_result": fast_result.model_dump(mode="json") if fast_result else None,
        }
    )
