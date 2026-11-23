from types import SimpleNamespace

import pytest

from eduecon_radar.schemas import AIScreeningResult
from eduecon_radar.services.deepseek import (
    DeepSeekClient,
    DeepSeekError,
    StructuredResponseError,
    parse_json_content,
)

VALID = (
    '{"decision":"include","confidence":0.9,"reason":"core topic",'
    '"topics":["教师"],"education_level":null,"country_or_region":null,'
    '"method":null,"research_question":null,"main_result":null,'
    '"missing_information":["sample"]}'
)


def response(content, *, model=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        model=model,
    )


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeOpenAI:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


class HttpFailure(RuntimeError):
    def __init__(self, status: int):
        self.response = SimpleNamespace(status_code=status, headers={"x-request-id": "req-test"})
        super().__init__(f"HTTP {status}")


def test_parse_valid_json_and_fence():
    parsed = parse_json_content(f"```json\n{VALID}\n```", AIScreeningResult)
    assert parsed.decision == "include"


def test_parse_markdown_and_explanatory_text_extracts_first_json_object():
    content = f"Here is the result:\n```json\n{VALID}\n```\nDone."
    assert parse_json_content(content, AIScreeningResult).decision == "include"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("INCLUDE", "include"),
        ("Included", "include"),
        ("EXCLUDED", "exclude"),
        ("Border-Line", "borderline"),
    ],
)
def test_parse_normalizes_only_approved_decision_enum_aliases(value, expected):
    content = VALID.replace('"include"', f'"{value}"', 1)
    assert parse_json_content(content, AIScreeningResult).decision == expected


def test_parse_does_not_synthesize_missing_required_fields():
    with pytest.raises(StructuredResponseError, match="schema_validation_failed"):
        parse_json_content('{"decision":"INCLUDE"}', AIScreeningResult)


@pytest.mark.parametrize(
    "content,kind",
    [
        ('{"decision":"include"}', "schema_validation_failed"),
        (VALID.replace('"include"', '"maybe"'), "schema_validation_failed"),
        ("not json", "json_parse_failed"),
    ],
)
def test_schema_and_json_errors_are_explicit_and_safe(content, kind):
    with pytest.raises(StructuredResponseError) as exc:
        parse_json_content(content, AIScreeningResult)
    assert exc.value.kind == kind


def test_invalid_json_retries_once():
    fake = FakeOpenAI([response("not-json"), response(VALID)])
    client = DeepSeekClient(
        api_key="test", base_url="https://example.test", client=fake, sleeper=lambda _: None
    )
    result = client.request_json(
        model="fast", system_prompt="system", user_prompt="user", schema=AIScreeningResult
    )
    assert result.data.decision == "include"
    assert fake.completions.calls == 2


@pytest.mark.parametrize("error", [TimeoutError("timeout"), RuntimeError("429")])
def test_network_and_rate_limit_failures_do_not_return_partial_data(error):
    fake = FakeOpenAI([error, error])
    client = DeepSeekClient(
        api_key="test", base_url="https://example.test", client=fake, sleeper=lambda _: None
    )
    with pytest.raises(DeepSeekError):
        client.request_json(
            model="fast", system_prompt="system", user_prompt="user", schema=AIScreeningResult
        )


def test_request_includes_pydantic_output_contract():
    fake = FakeOpenAI([response(VALID)])
    client = DeepSeekClient(api_key="test", base_url="https://example.test", client=fake)
    client.request_json(
        model="fast", system_prompt="system", user_prompt="user", schema=AIScreeningResult
    )
    request = fake.completions.calls
    assert request == 1
    assert "OUTPUT CONTRACT" in fake.completions.requests[0]["messages"][0]["content"]
    assert client.last_diagnostic and client.last_diagnostic.http_status == 200


def test_response_model_name_uses_transport_metadata_when_available():
    fake = FakeOpenAI([response(VALID, model="fast")])
    client = DeepSeekClient(api_key="test", base_url="https://example.test", client=fake)

    result = client.request_json(
        model="requested-fast", system_prompt="system", user_prompt="user", schema=AIScreeningResult
    )

    assert result.model_name == "fast"


def test_failure_diagnostic_and_error_do_not_leak_authorization_or_key():
    error = RuntimeError("Authorization: Bearer sk-secret-value")
    fake = FakeOpenAI([error, error])
    client = DeepSeekClient(
        api_key="test", base_url="https://example.test", client=fake, sleeper=lambda _: None
    )
    with pytest.raises(DeepSeekError) as exc:
        client.request_json(
            model="fast", system_prompt="system", user_prompt="user", schema=AIScreeningResult
        )
    assert "sk-secret" not in str(exc.value)
    assert client.last_diagnostic and client.last_diagnostic.error_type == "RuntimeError"


@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_failures_are_classified_without_response_body(status):
    fake = FakeOpenAI([HttpFailure(status), HttpFailure(status)])
    client = DeepSeekClient(
        api_key="test", base_url="https://example.test", client=fake, sleeper=lambda _: None
    )
    with pytest.raises(DeepSeekError, match=f"http_{status}"):
        client.request_json(
            model="fast", system_prompt="system", user_prompt="user", schema=AIScreeningResult
        )
    assert client.last_diagnostic and client.last_diagnostic.http_status == status
    assert client.last_diagnostic.request_id == "req-test"
