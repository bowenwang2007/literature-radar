# --- not part of the product core: see docs/PRODUCT_CORE.md ---
# RESEARCH/GOVERNANCE. Runtime bundle trust model; not on the product path.

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from eduecon_radar.domain.loader import DomainConfigLoader
from eduecon_radar.domain.models import DomainConfigBundle
from eduecon_radar.domain.semantic_models import SemanticEngineConfig
from eduecon_radar.domain.semantic_prompting import (
    PromptConfigurationError,
    SemanticPromptBuilder,
    canonical_sha256,
    file_sha256,
    load_semantic_engine_config,
)
from eduecon_radar.resources import resource

RuntimeMode = Literal[
    "production", "development", "test", "validation", "controlled_pilot"
]


class RuntimeBundleTrustError(RuntimeError):
    """A production AI stage cannot establish its immutable configuration trust chain."""

    def __init__(self, failure_type: str, message: str) -> None:
        self.failure_type = failure_type
        super().__init__(f"{failure_type}: {message}")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeModelIdentity(_StrictModel):
    provider: str
    fast: str
    strong: str
    translation: str


class RuntimeBundleManifest(_StrictModel):
    schema_version: Literal[1]
    status: Literal["released"]
    bundle_version: str = Field(min_length=1)
    domain_config_sha256: str = Field(pattern=r"^[A-Fa-f0-9]{64}$")
    prompt_bundle_sha256: str = Field(pattern=r"^[A-Fa-f0-9]{64}$")
    models: RuntimeModelIdentity
    resources: dict[str, str]
    created_at: str
    released_at: str
    bundle_sha256: str = Field(pattern=r"^[A-Fa-f0-9]{64}$")


@dataclass(frozen=True)
class RuntimeBundleResolution:
    mode: RuntimeMode
    status: str
    bundle_version: str
    bundle_sha256: str
    manifest_path: Path | None
    domain_bundle: DomainConfigBundle
    semantic_config: SemanticEngineConfig
    prompt_builder: SemanticPromptBuilder
    models: RuntimeModelIdentity
    unreleased_override: bool = False

    def audit_summary(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "status": self.status,
            "bundle_version": self.bundle_version,
            "bundle_sha256": self.bundle_sha256,
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "provider": self.models.provider,
            "models": self.models.model_dump(),
            "unreleased_override": self.unreleased_override,
        }


class RuntimeBundleResolver:
    """The only resolver allowed to construct a production semantic runtime."""

    MANIFEST_RELATIVE_PATH = Path("config/runtime_bundles/released/current.json")
    REQUIRED_RESOURCE_PATHS = (
        "config/domains/education_economics/profile.yaml",
        "config/domains/education_economics/ontology.yaml",
        "config/domains/education_economics/exclusion_contexts.yaml",
        "config/domains/education_economics/subject_codes.yaml",
        "config/domains/education_economics/journal_priors.yaml",
        "config/domains/education_economics/scoring.yaml",
        "config/domains/education_economics/candidates.yaml",
        "config/domains/education_economics/metadata.yaml",
        "config/domains/education_economics/semantic_models.yaml",
        "config/domains/education_economics/prompts/manifest.yaml",
        "config/domains/education_economics/prompts/education_economics_fast_semantic_system.txt",
        "config/domains/education_economics/prompts/education_economics_fast_semantic_user.txt",
        "config/domains/education_economics/prompts/education_economics_strong_review_system.txt",
        "config/domains/education_economics/prompts/education_economics_strong_review_user.txt",
        "prompts/translation_title_system.txt",
        "prompts/translation_title_user.txt",
        "prompts/translation_abstract_system.txt",
        "prompts/translation_abstract_user.txt",
    )

    def __init__(self, project_root: Path) -> None:
        requested = project_root.resolve()
        self.project_root = (
            requested
            if (requested / "config").is_dir()
            else Path(str(resource("config"))).resolve().parent
        )
        self.config_dir = self.project_root / "config" / "domains" / "education_economics"

    def resolve(
        self,
        *,
        mode: RuntimeMode = "production",
        manifest_path: Path | None = None,
        allow_unreleased_bundle: bool = False,
        development_models: RuntimeModelIdentity | None = None,
    ) -> RuntimeBundleResolution:
        if allow_unreleased_bundle:
            if mode not in {"development", "test", "validation"}:
                raise RuntimeBundleTrustError(
                    "developer_override_forbidden",
                    "Unreleased bundles are never allowed in production mode",
                )
            if development_models is None:
                raise RuntimeBundleTrustError(
                    "developer_model_identity_missing",
                    "An explicit model identity is required for an unreleased bundle",
                )
            return self._resolve_unreleased(mode, development_models)
        if mode != "production":
            raise RuntimeBundleTrustError(
                "explicit_override_required",
                "Development, test and validation modes require --allow-unreleased-bundle",
            )
        selected = manifest_path or self.project_root / self.MANIFEST_RELATIVE_PATH
        return self._resolve_released(selected)

    def inspect(self, manifest_path: Path | None = None) -> dict[str, object]:
        try:
            resolution = self.resolve(manifest_path=manifest_path)
        except RuntimeBundleTrustError as exc:
            return {
                "trusted": False,
                "mode": "production",
                "failure_type": exc.failure_type,
                "message": str(exc),
            }
        return {"trusted": True, **resolution.audit_summary()}

    def resolve_controlled_pilot(
        self,
        authorization: object,
        *,
        models: RuntimeModelIdentity,
    ) -> RuntimeBundleResolution:
        """Resolve an unreleased runtime only when a persisted authorization matches."""
        runtime = self._resolve_unreleased("controlled_pilot", models)
        runtime_config_sha256 = canonical_sha256(
            {
                "domain": runtime.domain_bundle.config_sha256,
                "semantic": runtime.semantic_config.model_dump(mode="json"),
            }
        )
        prompt_bundle_sha256 = canonical_sha256(
            {
                relative: file_sha256(self.project_root / relative)
                for relative in self.REQUIRED_RESOURCE_PATHS
                if "prompts/" in relative.replace("\\", "/")
            }
        )
        expected_runtime = getattr(authorization, "runtime_config_sha256", None)
        expected_prompts = getattr(authorization, "prompt_bundle_sha256", None)
        expected_models = getattr(authorization, "models", None)
        if expected_runtime != runtime_config_sha256:
            raise RuntimeBundleTrustError(
                "pilot_runtime_config_mismatch",
                "Controlled pilot runtime configuration changed after authorization",
            )
        if expected_prompts != prompt_bundle_sha256:
            raise RuntimeBundleTrustError(
                "pilot_prompt_bundle_mismatch",
                "Controlled pilot prompt bundle changed after authorization",
            )
        if expected_models != models.model_dump():
            raise RuntimeBundleTrustError(
                "pilot_model_identity_mismatch",
                "Controlled pilot model identities changed after authorization",
            )
        return replace(runtime, status="controlled_pilot")

    def _resolve_released(self, manifest_path: Path) -> RuntimeBundleResolution:
        manifest_path = manifest_path.resolve()
        self._require_within_root(manifest_path)
        if not manifest_path.is_file():
            raise RuntimeBundleTrustError(
                "released_bundle_missing", f"Released manifest not found: {manifest_path}"
            )
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = RuntimeBundleManifest.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise RuntimeBundleTrustError(
                "invalid_bundle_manifest", f"Released manifest cannot be parsed: {exc}"
            ) from exc
        body = dict(raw)
        stored_hash = str(body.pop("bundle_sha256", "")).upper()
        actual_hash = canonical_sha256(body)
        if stored_hash != actual_hash:
            raise RuntimeBundleTrustError(
                "bundle_hash_mismatch",
                f"Expected {stored_hash or '[missing]'}, calculated {actual_hash}",
            )
        self._reject_duplicate_version(manifest_path, manifest.bundle_version, stored_hash)
        missing = sorted(set(self.REQUIRED_RESOURCE_PATHS) - set(manifest.resources))
        if missing:
            raise RuntimeBundleTrustError(
                "bundle_resource_missing",
                "Manifest omits required resources: " + ", ".join(missing),
            )
        for relative, expected in manifest.resources.items():
            path = (self.project_root / relative).resolve()
            self._require_within_root(path)
            if not path.is_file():
                raise RuntimeBundleTrustError(
                    "bundle_resource_missing", f"Released resource is missing: {relative}"
                )
            actual = file_sha256(path)
            if actual != expected.upper():
                failure = (
                    "prompt_hash_mismatch"
                    if relative.startswith("prompts/")
                    or "/prompts/" in relative.replace("\\", "/")
                    else "resource_hash_mismatch"
                )
                raise RuntimeBundleTrustError(
                    failure, f"Hash mismatch for {relative}: expected {expected}, got {actual}"
                )
        try:
            domain = DomainConfigLoader(self.config_dir).load()
            semantic = load_semantic_engine_config(self.config_dir)
            prompts = SemanticPromptBuilder(self.config_dir / "prompts")
        except (RuntimeError, PromptConfigurationError) as exc:
            raise RuntimeBundleTrustError("runtime_resource_invalid", str(exc)) from exc
        self._validate_released_contract(manifest, domain, semantic, prompts)
        return RuntimeBundleResolution(
            mode="production",
            status="released",
            bundle_version=manifest.bundle_version,
            bundle_sha256=stored_hash,
            manifest_path=manifest_path,
            domain_bundle=domain,
            semantic_config=semantic,
            prompt_builder=prompts,
            models=manifest.models,
        )

    def _resolve_unreleased(
        self, mode: RuntimeMode, models: RuntimeModelIdentity
    ) -> RuntimeBundleResolution:
        try:
            domain = DomainConfigLoader(self.config_dir).load()
            semantic = load_semantic_engine_config(self.config_dir)
            prompts = SemanticPromptBuilder(self.config_dir / "prompts")
        except (RuntimeError, PromptConfigurationError) as exc:
            raise RuntimeBundleTrustError("runtime_resource_invalid", str(exc)) from exc
        return RuntimeBundleResolution(
            mode=mode,
            status=semantic.status,
            bundle_version=domain.metadata.bundle_version,
            bundle_sha256=canonical_sha256(
                {
                    "domain": domain.config_sha256,
                    "semantic": semantic.model_dump(mode="json"),
                    "models": models.model_dump(),
                    "mode": mode,
                }
            ),
            manifest_path=None,
            domain_bundle=domain,
            semantic_config=semantic,
            prompt_builder=prompts,
            models=models,
            unreleased_override=True,
        )

    def _validate_released_contract(
        self,
        manifest: RuntimeBundleManifest,
        domain: DomainConfigBundle,
        semantic: SemanticEngineConfig,
        prompts: SemanticPromptBuilder,
    ) -> None:
        if domain.metadata.status != "released" or domain.profile.status != "released":
            raise RuntimeBundleTrustError(
                "domain_bundle_unreleased", "Domain metadata and profile must both be released"
            )
        domain_statuses = {
            "ontology": domain.ontology.status,
            "exclusions": domain.exclusions.status,
            "subject_codes": domain.subject_codes.status,
            "journal_priors": domain.journal_priors.status,
            "scoring": domain.scoring.status,
        }
        unreleased = sorted(
            name for name, status in domain_statuses.items() if status != "released"
        )
        if unreleased:
            raise RuntimeBundleTrustError(
                "domain_resource_unreleased",
                "Domain resources are not released: " + ", ".join(unreleased),
            )
        if domain.metadata.bundle_version != manifest.bundle_version:
            raise RuntimeBundleTrustError(
                "bundle_version_mismatch", "Manifest and domain bundle versions differ"
            )
        if domain.config_sha256 != manifest.domain_config_sha256.upper():
            raise RuntimeBundleTrustError(
                "domain_config_hash_mismatch", "Canonical domain configuration hash differs"
            )
        prompt_status = prompts.validate()
        if prompt_status["status"] != "released":
            raise RuntimeBundleTrustError(
                "prompt_bundle_unreleased", "Prompt manifest is not released"
            )
        if any(entry.status != "released" for entry in prompts.manifest.prompts):
            raise RuntimeBundleTrustError(
                "prompt_entry_unreleased", "Every prompt entry must be released"
            )
        prompt_hash = canonical_sha256(
            {
                key: manifest.resources[key]
                for key in sorted(manifest.resources)
                if "/prompts/" in key.replace("\\", "/")
            }
        )
        if prompt_hash != manifest.prompt_bundle_sha256.upper():
            raise RuntimeBundleTrustError(
                "prompt_bundle_hash_mismatch", "Prompt bundle aggregate hash differs"
            )
        if semantic.status != "released" or not semantic.live_ai_allowed:
            raise RuntimeBundleTrustError(
                "semantic_bundle_unreleased", "Semantic engine is not released for live AI"
            )
        for role in (semantic.fast, semantic.strong_review):
            if role.status != "released" or not role.enabled:
                raise RuntimeBundleTrustError(
                    "semantic_role_unreleased", "Fast and Strong roles must be enabled and released"
                )
        prompt_by_role = {entry.role: entry for entry in prompts.manifest.prompts}
        for role_name, role in (
            ("fast", semantic.fast),
            ("strong_review", semantic.strong_review),
        ):
            entry = prompt_by_role.get(role_name)
            if (
                entry is None
                or entry.prompt_id != role.prompt_id
                or entry.version != role.prompt_version
            ):
                raise RuntimeBundleTrustError(
                    "prompt_identity_mismatch",
                    f"{role_name} prompt identity differs from semantic configuration",
                )
        if semantic.fast.provider != manifest.models.provider:
            raise RuntimeBundleTrustError(
                "provider_mismatch", "Fast provider differs from manifest"
            )
        if semantic.strong_review.provider != manifest.models.provider:
            raise RuntimeBundleTrustError(
                "provider_mismatch", "Strong provider differs from manifest"
            )
        if semantic.fast.model != manifest.models.fast:
            raise RuntimeBundleTrustError("model_mismatch", "Fast model differs from manifest")
        if semantic.strong_review.model != manifest.models.strong:
            raise RuntimeBundleTrustError("model_mismatch", "Strong model differs from manifest")

    def _reject_duplicate_version(
        self, selected: Path, version: str, bundle_sha256: str
    ) -> None:
        directory = selected.parent
        if not directory.is_dir():
            return
        for candidate in directory.glob("*.json"):
            if candidate.resolve() == selected:
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                payload.get("bundle_version") == version
                and str(payload.get("bundle_sha256", "")).upper() != bundle_sha256
            ):
                raise RuntimeBundleTrustError(
                    "released_version_collision",
                    f"Released version {version} has more than one bundle hash",
                )

    def _require_within_root(self, path: Path) -> None:
        try:
            path.relative_to(self.project_root)
        except ValueError as exc:
            raise RuntimeBundleTrustError(
                "unsafe_bundle_path", f"Path escapes project root: {path}"
            ) from exc
