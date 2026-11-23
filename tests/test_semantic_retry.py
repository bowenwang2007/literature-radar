from test_phase4_semantic import QueueClient, _fast, _prepared_database

from eduecon_radar.domain.semantic_prompting import (
    SemanticPromptBuilder,
    load_semantic_engine_config,
)
from eduecon_radar.services.semantic_screening import SemanticScreeningService


def test_historical_fast_failure_is_not_reused_after_schema_contract_fix(tmp_path):
    database, paper_id = _prepared_database(tmp_path)
    from test_phase4_semantic import CONFIG, PROMPTS

    first_client = QueueClient(RuntimeError("schema_validation_failed"))
    first = SemanticScreeningService(
        database=database,
        config=load_semantic_engine_config(CONFIG),
        prompts=SemanticPromptBuilder(PROMPTS),
        client=first_client,
        fixture_mode=True,
    )
    failed = first.analyze_many((paper_id,), run_strong_if_escalated=False)
    assert paper_id in failed.failures

    second_client = QueueClient(_fast().model_copy(update={"paper_id": paper_id}))
    second = SemanticScreeningService(
        database=database,
        config=load_semantic_engine_config(CONFIG),
        prompts=SemanticPromptBuilder(PROMPTS),
        client=second_client,
        fixture_mode=True,
    )
    recovered = second.analyze_many((paper_id,), run_strong_if_escalated=False)
    assert paper_id in recovered.completed
    assert second_client.calls
