from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

SchemaT = TypeVar("SchemaT", bound=BaseModel)


BATCH_REQUEST_LIMIT = 24
_request_guard: ContextVar[Callable[[], bool] | None] = ContextVar("request_guard", default=None)


@contextmanager
def limit_batch_requests(reserve=None):
    """Reserve each physical provider attempt, including SDK-level manual retries."""
    used = 0
    def reserve_local():
        nonlocal used
        used += 1
        return used <= BATCH_REQUEST_LIMIT

    token = _request_guard.set(reserve or reserve_local)
    try:
        yield
    finally:
        _request_guard.reset(token)


class DeepSeekError(RuntimeError):
    """Raised after a DeepSeek request cannot be safely parsed or completed."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str | None = None,
        model_name: str | None = None,
        validation_error: str | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.model_name = model_name
        self.validation_error = validation_error


class StructuredResponseError(ValueError):
    def __init__(
        self,
        kind: str,
        fields: tuple[str, ...] = (),
        *,
        raw_response: str | None = None,
    ) -> None:
        super().__init__(kind)
        self.kind = kind
        self.fields = fields
        self.raw_response = raw_response


@dataclass(frozen=True)
class DeepSeekDiagnostic:
    http_status: int | None
    request_id: str | None
    elapsed_ms: int | None
    json_parseable: bool
    schema_valid: bool
    error_type: str | None = None


@dataclass(frozen=True)
class DeepSeekResponse:
    data: BaseModel
    model_name: str
    input_tokens: int
    output_tokens: int


def parse_json_content(content: str, schema: type[SchemaT]) -> SchemaT:
    try:
        cleaned = extract_first_json_object(content)
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, StructuredResponseError):
            raise
        raise StructuredResponseError("json_parse_failed", raw_response=content) from exc
    payload = _normalize_approved_enum_aliases(payload)
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        fields = tuple(".".join(str(part) for part in item["loc"]) for item in exc.errors()[:5])
        raise StructuredResponseError(
            "schema_validation_failed", fields, raw_response=content
        ) from exc
    except TypeError as exc:
        raise StructuredResponseError(
            "schema_validation_failed", raw_response=content
        ) from exc


_DECISION_FIELDS = {
    "decision",
    "title_only_decision",
    "abstract_aware_decision",
    "independent_decision",
    "recommended_final_decision",
}
_DECISION_ALIASES = {
    "include": "include",
    "included": "include",
    "exclude": "exclude",
    "excluded": "exclude",
    "borderline": "borderline",
    "border_line": "borderline",
    "border-line": "borderline",
}
_CASE_INSENSITIVE_ENUM_FIELDS = {
    "title_evidence_sufficiency",
    "abstract_evidence_sufficiency",
    "information_sufficiency",
    "decision_sufficiency",
}
_SUFFICIENCY_VALUES = {"high", "medium", "low", "insufficient"}


def _normalize_approved_enum_aliases(payload: object) -> object:
    """Normalize only allow-listed enum spellings; never synthesize missing fields."""
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    for field in _DECISION_FIELDS:
        value = normalized.get(field)
        if isinstance(value, str):
            alias = _DECISION_ALIASES.get(value.strip().casefold())
            if alias is not None:
                normalized[field] = alias
    for field in _CASE_INSENSITIVE_ENUM_FIELDS:
        value = normalized.get(field)
        if isinstance(value, str) and value.strip().casefold() in _SUFFICIENCY_VALUES:
            normalized[field] = value.strip().casefold()
    return normalized


def extract_first_json_object(content: str) -> str:
    """Extract one JSON object without accepting non-object or trailing prose as data."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("Invalid structured AI response: json_object_not_found")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("Invalid structured AI response: unterminated_json_object")


class DeepSeekClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        client: object | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        max_attempts: int = 2,
        timeout_seconds: float = 45,
    ) -> None:
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for AI screening")
        if client is not None:
            self.client = client
        else:
            # Imported here on purpose: the OpenAI SDK pulls in ~780 modules and
            # is reached from the UI import chain through this service, so a
            # module-level import slows every cold start even though the app
            # only talks to a provider after an explicit user action.
            from openai import OpenAI

            self.client = OpenAI(
                api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=0
            )
        self.sleeper = sleeper
        self.clock = clock
        self.max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds
        self.last_diagnostic: DeepSeekDiagnostic | None = None

    def request_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
    ) -> DeepSeekResponse:
        last_error: Exception | None = None
        last_raw_response: str | None = None
        last_model_name: str | None = None
        for attempt in range(self.max_attempts):
            guard = _request_guard.get()
            if guard and not guard():
                raise RuntimeError("本批 AI 请求额度已用完")
            started = self.clock()
            try:
                contract = json.dumps(
                    schema.model_json_schema(), ensure_ascii=False, separators=(",", ":")
                )
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                            + "\n\nOUTPUT CONTRACT: Return exactly one JSON object conforming to "
                            + contract,
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("DeepSeek returned an empty response")
                last_raw_response = content
                last_model_name = _response_model_name(response, model)
                data = parse_json_content(content, schema)
                usage = getattr(response, "usage", None)
                self.last_diagnostic = DeepSeekDiagnostic(
                    http_status=_response_status(response),
                    request_id=_response_request_id(response),
                    elapsed_ms=int((self.clock() - started) * 1000),
                    json_parseable=True,
                    schema_valid=True,
                )
                return DeepSeekResponse(
                    data=data,
                    model_name=last_model_name,
                    input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                )
            except Exception as exc:  # SDK error subclasses vary between versions.
                last_error = exc
                self.last_diagnostic = DeepSeekDiagnostic(
                    http_status=_exception_status(exc),
                    request_id=_exception_request_id(exc),
                    elapsed_ms=int((self.clock() - started) * 1000),
                    json_parseable=not (
                        isinstance(exc, json.JSONDecodeError)
                        or (
                            isinstance(exc, StructuredResponseError)
                            and exc.kind == "json_parse_failed"
                        )
                        or (
                            isinstance(exc, ValueError)
                            and not isinstance(exc, StructuredResponseError)
                        )
                    ),
                    schema_valid=False,
                    error_type=_safe_error_type(exc),
                )
                if attempt + 1 < self.max_attempts:
                    self.sleeper(float(2**attempt))
        validation_error = (
            _safe_error_summary(last_error)
            if isinstance(last_error, (StructuredResponseError, ValidationError, ValueError))
            else None
        )
        raise DeepSeekError(
            f"DeepSeek request failed after {self.max_attempts} attempts: "
            f"{_safe_error_summary(last_error)}",
            raw_response=last_raw_response,
            model_name=last_model_name or model,
            validation_error=validation_error,
        ) from last_error


def _response_status(response: object) -> int | None:
    value = getattr(response, "status_code", None)
    # The OpenAI SDK's successful ChatCompletion object usually omits transport status.
    # Reaching this point means the SDK completed a successful HTTP request.
    return value if isinstance(value, int) else 200


def _response_request_id(response: object) -> str | None:
    value = getattr(response, "_request_id", None)
    if value:
        return str(value)
    headers = getattr(response, "headers", None)
    if headers:
        return headers.get("x-request-id") or headers.get("request-id")
    return None


def _response_model_name(response: object, requested_model: str) -> str:
    value = getattr(response, "model", None)
    return str(value) if value else requested_model


def _exception_status(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _exception_request_id(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    return _response_request_id(response) if response else None


def _safe_error_type(exc: Exception) -> str:
    status = _exception_status(exc)
    if status is not None:
        return f"http_{status}"
    if isinstance(exc, StructuredResponseError):
        return exc.kind
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return "timeout"
    if isinstance(exc, ValidationError):
        return "schema_validation_failed"
    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        return "json_parse_failed"
    return type(exc).__name__


def _safe_error_summary(exc: Exception | None) -> str:
    if exc is None:
        return "unknown_error"
    if isinstance(exc, StructuredResponseError):
        suffix = ":" + ",".join(exc.fields) if exc.fields else ""
        return exc.kind + suffix
    if isinstance(exc, ValidationError):
        locations = [".".join(str(part) for part in item["loc"]) for item in exc.errors()[:5]]
        return "schema_validation_failed:" + ",".join(locations)
    return _safe_error_type(exc)
