from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from eduecon_radar.resources import resource


def _load_dotenv(path: Path) -> None:
    """Load a small .env file without overriding the process environment."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_profile_slug(value: str) -> str:
    """Validate a domain profile name before using it as a path segment.

    The value selects files under ``config/``, so a separator or a traversal
    segment must be rejected instead of silently resolving somewhere else.
    """
    cleaned = value.strip()
    if (
        not cleaned
        or cleaned in {".", ".."}
        or "/" in cleaned
        or "\\" in cleaned
        or ".." in cleaned
    ):
        raise ValueError(
            "EDUERADAR_DOMAIN_PROFILE must be a simple directory name, "
            f"got {value!r}"
        )
    return cleaned


@dataclass(frozen=True)
class Settings:
    project_root: Path
    database_url: str
    journals_path: Path
    prompts_dir: Path
    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_model_fast: str
    deepseek_model_strong: str
    deepseek_reasoning_enabled: bool
    deepseek_timeout_seconds: float
    deepseek_max_retries: int
    openalex_api_key: str | None
    unpaywall_email: str | None
    log_level: str
    app_timezone: str
    ui_version: str
    domain_config_dir: Path
    max_per_journal_limit: int = 100
    journal_registry_path: Path | None = None
    #: Root of the product profiles (``config/profiles/<id>/profile.yaml``).
    profiles_root: Path | None = None
    #: Global journal catalog shared by every profile.
    journal_catalog_path: Path | None = None
    #: Profile used when nothing else is selected; existing behaviour.
    default_profile: str = "education_economics"
    #: Locally added journals (kept out of the global catalog).
    personal_journals_path: Path | None = None
    # Defaulted (not required) so callers that build Settings positionally keep
    # working; Settings.load() always sets it explicitly from the environment.
    domain_profile: str = "education_economics"

    @property
    def screening_prompts_dir(self) -> Path:
        """Screening prompts of the default research profile.

        Screening prompts are profile content: they define what relevance means
        for that reading queue.  The product therefore reads them from the
        profile directory instead of keeping a second copy under ``prompts/``,
        which now holds only profile-independent prompts (translation, and the
        legacy v0.4 relevance/summary pair).
        """
        root = self.profiles_root or Path(str(resource("config/profiles")))
        return root / self.default_profile / "prompts"

    @property
    def database_path(self) -> Path:
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            raise ValueError("MVP only supports sqlite:/// database URLs")
        # sqlite:///data/file.db is relative; sqlite:////absolute/path.db is absolute.
        raw = unquote(self.database_url[len(prefix) :])
        path = Path(raw)
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    @classmethod
    def load(cls, project_root: Path | None = None) -> Settings:
        root = (project_root or Path.cwd()).resolve()
        _load_dotenv(root / ".env")
        # v05 is the default production UI. v04 is available for developer compatibility.
        # Set EDUECON_UI_VERSION=v04 in .env to restore the old 4-page interface.
        ui_version = os.getenv("EDUECON_UI_VERSION", "v05").strip().lower()
        if ui_version not in {"v04", "v05"}:
            raise ValueError("EDUECON_UI_VERSION must be either v04 or v05")
        # The domain package is the only place that encodes what "relevant"
        # means for this literature. Keep the selector configurable so adding a
        # domain is a configuration change rather than editing assembly code.
        domain_profile = os.getenv(
            "EDUERADAR_DOMAIN_PROFILE", "education_economics"
        ).strip()
        if not domain_profile:
            raise ValueError("EDUERADAR_DOMAIN_PROFILE must not be empty")
        domain_slug = _safe_profile_slug(domain_profile)
        timeout_seconds = float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "45"))
        max_retries = int(os.getenv("DEEPSEEK_MAX_RETRIES", "2"))
        if timeout_seconds <= 0:
            raise ValueError("DEEPSEEK_TIMEOUT_SECONDS must be positive")
        if max_retries < 1:
            raise ValueError("DEEPSEEK_MAX_RETRIES must be at least 1")
        max_per_journal_limit = int(
            os.getenv("EDUERADAR_PER_JOURNAL_LIMIT_MAX", "100")
        )
        if not 1 <= max_per_journal_limit <= 100:
            raise ValueError("EDUERADAR_PER_JOURNAL_LIMIT_MAX must be between 1 and 100")
        return cls(
            project_root=root,
            database_url=os.getenv("DATABASE_URL", "sqlite:///data/eduecon_radar.db"),
            journals_path=(
                root / "journals.yaml"
                if (root / "journals.yaml").is_file()
                else Path(str(resource("journals.yaml")))
            ),
            prompts_dir=(
                root / "prompts"
                if (root / "prompts").is_dir()
                else Path(str(resource("prompts")))
            ),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip() or None,
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_model_fast=os.getenv("DEEPSEEK_MODEL_FAST", "deepseek-v4-flash"),
            deepseek_model_strong=os.getenv("DEEPSEEK_MODEL_STRONG", "deepseek-v4-pro"),
            deepseek_reasoning_enabled=_as_bool(os.getenv("DEEPSEEK_REASONING_ENABLED", "false")),
            deepseek_timeout_seconds=timeout_seconds,
            deepseek_max_retries=max_retries,
            openalex_api_key=os.getenv("OPENALEX_API_KEY") or None,
            unpaywall_email=os.getenv("UNPAYWALL_EMAIL") or None,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            app_timezone=os.getenv("APP_TIMEZONE", "Asia/Shanghai"),
            ui_version=ui_version,
            domain_profile=domain_profile,
            domain_config_dir=(
                root / "config" / "domains" / domain_slug
                if (root / "config" / "domains" / domain_slug).is_dir()
                else Path(str(resource(f"config/domains/{domain_slug}")))
            ),
            max_per_journal_limit=max_per_journal_limit,
            journal_registry_path=(
                # One global catalog; a profile decides which journals it
                # monitors (see services/profile_registry.py).
                root / "config" / "journals" / "catalog.json"
                if (root / "config" / "journals" / "catalog.json").is_file()
                else Path(str(resource("config/journals/catalog.json")))
            ),
            profiles_root=(
                root / "config" / "profiles"
                if (root / "config" / "profiles").is_dir()
                else Path(str(resource("config/profiles")))
            ),
            journal_catalog_path=(
                root / "config" / "journals" / "catalog.json"
                if (root / "config" / "journals" / "catalog.json").is_file()
                else Path(str(resource("config/journals/catalog.json")))
            ),
            default_profile=domain_profile,
            personal_journals_path=root / "config" / "personal_journals.json",
        )
