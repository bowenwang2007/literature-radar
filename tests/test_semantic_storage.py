import json

import pytest

from eduecon_radar.domain.semantic_prompting import SemanticPromptBuilder
from eduecon_radar.domain.semantic_storage import DomainSemanticRepository
from eduecon_radar.domain.storage import DomainEvidenceRepository
from eduecon_radar.services.deepseek import DeepSeekError
from eduecon_radar.services.semantic_screening import load_semantic_paper_input


def test_successful_fast_result_uses_json_mode_atomic_storage_and_leaves_review_untouched(tmp_path):
    from test_phase4_semantic import CONFIG, _fast, _prepared_database

    database, paper_id = _prepared_database(tmp_path)
    evidence = DomainEvidenceRepository(database).latest_for_paper(paper_id)
    paper = load_semantic_paper_input(database, paper_id, evidence)
    prompt = SemanticPromptBuilder(CONFIG / "prompts").build_fast(paper)
    result = _fast().model_copy(
        update={
            "paper_id": paper_id,
            "adjacent_domain": None,
            "suspected_false_friend": False,
            "prompt_version": "model-returned-version-is-not-storage-identity",
        }
    )
    repository = DomainSemanticRepository(database)
    with database.connect() as connection:
        review_count_before = connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    stored = repository.save_success(
        paper=paper,
        role="fast",
        fingerprint="storage-json-mode",
        prompt=prompt,
        model_provider="fixture",
        model_name="fixture",
        engine_version="fixture",
        escalation_policy_version="fixture",
        result=result,
        input_tokens=0,
        output_tokens=0,
        estimated_cost=None,
    )
    with database.connect() as connection:
        row = connection.execute(
            "SELECT result_json,prompt_version,decision,confidence,education_is_core,"
            "economics_mechanism_present,"
            "information_sufficiency,title_evidence_sufficiency,suspected_false_friend,"
            "adjacent_domain,succeeded,failure_type,failure_message "
            "FROM domain_semantic_runs WHERE id=?",
            (stored.id,),
        ).fetchone()
        review_count = connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    payload = json.loads(row["result_json"])
    assert payload["decision"] == "include"
    assert row["prompt_version"] == prompt.prompt_version
    assert row["prompt_version"] != payload["prompt_version"]
    assert row["decision"] == "include" and row["succeeded"] == 1
    assert row["adjacent_domain"] is None
    assert row["failure_type"] is None and row["failure_message"] is None
    assert review_count == review_count_before


def test_serialization_failure_inserts_no_partial_success_row(tmp_path):
    from test_phase4_semantic import CONFIG, _prepared_database

    database, paper_id = _prepared_database(tmp_path)
    evidence = DomainEvidenceRepository(database).latest_for_paper(paper_id)
    paper = load_semantic_paper_input(database, paper_id, evidence)
    prompt = SemanticPromptBuilder(CONFIG / "prompts").build_fast(paper)
    repository = DomainSemanticRepository(database)
    with pytest.raises(TypeError):
        repository.save_success(
            paper=paper,
            role="fast",
            fingerprint="bad-serialization",
            prompt=prompt,
            model_provider="fixture",
            model_name="fixture",
            engine_version="fixture",
            escalation_policy_version="fixture",
            result=object(),
            input_tokens=0,
            output_tokens=0,
            estimated_cost=None,
        )
    with database.connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM domain_semantic_runs WHERE input_fingerprint='bad-serialization'"
        ).fetchone()[0]
    assert count == 0


def test_failure_preserves_sanitized_raw_response_and_reproducibility_identity(tmp_path):
    from test_phase4_semantic import CONFIG, _prepared_database

    database, paper_id = _prepared_database(tmp_path)
    evidence = DomainEvidenceRepository(database).latest_for_paper(paper_id)
    paper = load_semantic_paper_input(database, paper_id, evidence)
    prompt = SemanticPromptBuilder(CONFIG / "prompts").build_fast(paper)
    repository = DomainSemanticRepository(database)

    stored = repository.save_failure(
        paper=paper,
        role="fast",
        fingerprint="failure-diagnostic",
        prompt=prompt,
        model_provider="deepseek-compatible",
        model_name="requested-model",
        engine_version="engine-v1",
        escalation_policy_version="policy-v1",
        error=DeepSeekError(
            "schema validation failed",
            raw_response='{"decision":"INCLUDE","note":"Authorization: Bearer sk-secret"}',
            model_name="actual-model",
            validation_error="confidence: expected number",
        ),
    )

    with database.connect() as connection:
        row = connection.execute(
            "SELECT result_json,succeeded FROM domain_semantic_runs WHERE id=?",
            (stored.id,),
        ).fetchone()
    payload = json.loads(row["result_json"])
    assert row["succeeded"] == 0
    assert payload["execution_state"] == "retryable_error"
    assert payload["model_identity"] == {
        "provider": "deepseek-compatible",
        "model": "actual-model",
    }
    assert payload["validation_error"] == "confidence: expected number"
    assert "sk-secret" not in payload["raw_response"]
    assert payload["prompt_hashes"] == {
        "system": prompt.system_sha256,
        "user": prompt.user_sha256,
    }
    assert payload["runtime_fingerprint"]
