"""Shared construction of the product processing chain.

Both the workbench and the paper library trigger the exact same pipeline;
keeping the builders here avoids a second abstract/translation/screening
implementation while preventing a UI-module import cycle.
"""

from __future__ import annotations

import logging

from eduecon_radar.config import Settings
from eduecon_radar.db import Database
from eduecon_radar.resources import runtime_path
from eduecon_radar.services.local_pipeline import LocalPipelineServices


def build_enrichment_service(
    database: Database, settings: Settings
) -> object:
    from eduecon_radar.services.abstract_enrichment import (
        AbstractEnrichmentService,
        CrossrefEnrichmentAdapter,
        DOAJEnrichmentAdapter,
        OpenAIREEnrichmentAdapter,
        OpenAlexEnrichmentAdapter,
        RePEcOAIAdapter,
        SemanticScholarEnrichmentAdapter,
        load_repec_series_registry,
    )
    from eduecon_radar.services.abstract_source_config import load_abstract_sources
    from eduecon_radar.services.resilient_http import ResilientHttpClient

    source_bundle = load_abstract_sources(
        runtime_path(settings.project_root, "config/abstract_sources.yaml")
    )
    source_configs = {item.name: item for item in source_bundle.sources}

    def http_for(source: str) -> ResilientHttpClient:
        value = source_configs[source]
        return ResilientHttpClient(
            timeout=value.timeout_seconds,
            max_retries=value.max_retries,
            rate_limit_per_second=value.rate_limit_per_second,
            max_inline_sleep_seconds=1.0,
        )

    return AbstractEnrichmentService(
        database,
        adapters=[
            CrossrefEnrichmentAdapter(client=http_for("crossref")),
            RePEcOAIAdapter(
                client=http_for("repec"),
                crossref_client=http_for("crossref"),
                series_by_issn=load_repec_series_registry(
                    runtime_path(settings.project_root, "config/repec_series.json")
                ),
            ),
            OpenAlexEnrichmentAdapter(settings.openalex_api_key, client=http_for("openalex")),
            SemanticScholarEnrichmentAdapter(client=http_for("semantic_scholar")),
            DOAJEnrichmentAdapter(client=http_for("doaj")),
            OpenAIREEnrichmentAdapter(client=http_for("openaire")),
        ],
        source_configs=source_configs,
        policy_version=f"{source_bundle.version}-personal-overnight-7c1",
    )


def build_batch_title_service(
    database: Database, settings: Settings
) -> object:
    from eduecon_radar.services.batch_translation import BatchTitleTranslationService
    from eduecon_radar.services.deepseek import DeepSeekClient

    if not settings.deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY is required")
    client = DeepSeekClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout_seconds=settings.deepseek_timeout_seconds,
        max_attempts=settings.deepseek_max_retries,
    )
    return BatchTitleTranslationService(
        database,
        settings.prompts_dir,
        client,
        settings.deepseek_model_strong,
    )


def build_translation_service(database: Database, settings: Settings) -> object:
    from eduecon_radar.services.deepseek import DeepSeekClient
    from eduecon_radar.services.translation import TranslationService

    if not settings.deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY is required")
    client = DeepSeekClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout_seconds=settings.deepseek_timeout_seconds,
        max_attempts=settings.deepseek_max_retries,
    )
    return TranslationService(
        database,
        settings.prompts_dir,
        client,
        settings.deepseek_model_strong,
    )


def _screening_client(settings: Settings, translation: object, profile: object | None):
    """The client the screening service should call.

    Normally the shared translation client is reused.  A profile may declare its
    own latency budget (``screening_timeout_seconds``) because its prompt asks
    for a longer analysis; only then is a dedicated client built, so every other
    profile keeps exactly the previous behaviour.
    """
    timeout = getattr(profile, "screening_timeout_seconds", None)
    if not timeout:
        return translation.client
    from eduecon_radar.services.deepseek import DeepSeekClient

    return DeepSeekClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout_seconds=float(timeout),
        max_attempts=settings.deepseek_max_retries,
    )


def build_local_pipeline_services(
    database: Database,
    settings: Settings,
    *,
    profile: object | None = None,
) -> LocalPipelineServices:
    """Assemble the product chain for one research profile.

    ``profile`` is a ``services.profile_registry.ProductProfile``; when omitted
    the default profile's prompts are used, which is what the single-domain
    application did.
    """
    profile_id = getattr(profile, "profile_id", None)
    if profile is None:
        # Callers that do not name a profile get the default one.  Screening
        # prompts are profile content, so the fallback must be the default
        # profile's directory — never a second copy under ``prompts/``.
        screening_prompts_dir = settings.screening_prompts_dir
        screening_prompt_version = "2026-07-31-screening-v1"
        prompt_context: dict[str, str] = {}
        from eduecon_radar.services.profile_registry import load_profile_registry

        try:
            registry = load_profile_registry(
                settings.profiles_root,
                catalog_path=settings.journal_catalog_path,
                default_profile_id=settings.default_profile,
            )
            default_profile = registry.profile(settings.default_profile)
            screening_prompts_dir = default_profile.prompts_dir
            screening_prompt_version = default_profile.screening_prompt_version
            prompt_context = default_profile.prompt_context()
            profile_id = default_profile.profile_id
        except Exception:
            logging.getLogger(__name__).exception(
                "Unable to load the default research profile; using its prompts directory"
            )
    else:
        screening_prompts_dir = getattr(profile, "prompts_dir", None) or settings.prompts_dir
        screening_prompt_version = getattr(
            profile, "screening_prompt_version", None
        ) or "2026-07-31-screening-v1"
        # The profile's own vocabulary — what "central" means for this queue,
        # plus its JEL and subject signals — travels into its prompts.
        prompt_context = profile.prompt_context() if hasattr(profile, "prompt_context") else {}
    if not getattr(settings, "deepseek_api_key", None):
        return LocalPipelineServices(
            enrichment=build_enrichment_service(database, settings),
            translation=None,
            screening=None,
            batch_titles=None,
            profile_id=profile_id,
        )
    translation = build_translation_service(database, settings)
    screening = None
    try:
        from eduecon_radar.services.deterministic_screening import (
            DeterministicScreeningService,
        )
        from eduecon_radar.services.migrations import overnight_schema_ready

        if overnight_schema_ready(database):
            screening = DeterministicScreeningService(
                database,
                client=_screening_client(settings, translation, profile),
                model=settings.deepseek_model_strong,
                prompts_dir=screening_prompts_dir,
                prompt_version=screening_prompt_version,
                prompt_context=prompt_context,
            )
    except Exception:
        logging.getLogger(__name__).exception("Unable to initialize paper screening")
        screening = None
    return LocalPipelineServices(
        enrichment=build_enrichment_service(database, settings),
        translation=translation,
        screening=screening,
        batch_titles=build_batch_title_service(database, settings),
        profile_id=profile_id,
    )
