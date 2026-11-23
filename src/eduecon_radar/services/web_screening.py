from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass

from eduecon_radar.db import Database
from eduecon_radar.services.ai_screening import preview_screening_candidates


@dataclass(frozen=True)
class ScreeningParameters:
    source: str = "awaiting_screening"
    limit: int = 10
    allow_strong: bool = True

    def __post_init__(self) -> None:
        if self.source not in {"awaiting_screening", "screening_failed"}:
            raise ValueError("Unsupported screening source")
        if not 1 <= self.limit <= 100:
            raise ValueError("Screening limit must be between 1 and 100")

    @property
    def signature(self) -> tuple[object, ...]:
        return (self.source, self.limit, self.allow_strong)


def screening_preview(database: Database, parameters: ScreeningParameters) -> dict:
    total, candidates = preview_screening_candidates(
        database, source=parameters.source, limit=parameters.limit
    )
    return {
        "matching_total": total,
        "planned_count": len(candidates),
        "candidates": candidates,
        "source": parameters.source,
        "retry_failed": parameters.source == "screening_failed",
    }


def screening_preview_is_valid(state: MutableMapping, signature: tuple[object, ...]) -> bool:
    return (
        bool(state.get("screen_preview_result"))
        and state.get("screen_preview_signature") == signature
    )


def claim_screening_run(state: MutableMapping, signature: tuple[object, ...]) -> bool:
    if state.get("screen_running") or state.get("screen_consumed_signature") == signature:
        return False
    if not screening_preview_is_valid(state, signature) or not state.get("screen_confirm"):
        return False
    state["screen_running"] = True
    state["screen_consumed_signature"] = signature
    return True


def finish_screening_run(state: MutableMapping) -> None:
    state["screen_running"] = False
    state["screen_confirm"] = False
    state["screen_preview_signature"] = None
