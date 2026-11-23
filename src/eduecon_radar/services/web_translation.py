from __future__ import annotations

from collections.abc import MutableMapping


def translation_preview_is_valid(state: MutableMapping, signature: tuple[object, ...]) -> bool:
    return (
        state.get("translation_preview") is not None
        and state.get("translation_preview_signature") == signature
    )


def claim_translation_run(
    state: MutableMapping, signature: tuple[object, ...], *, overwrite_human: bool
) -> bool:
    if state.get("translation_running") or state.get("translation_consumed_signature") == signature:
        return False
    if not translation_preview_is_valid(state, signature) or not state.get("translation_confirm"):
        return False
    if overwrite_human and not state.get("translation_overwrite_confirm"):
        return False
    state["translation_running"] = True
    state["translation_consumed_signature"] = signature
    return True


def finish_translation_run(state: MutableMapping) -> None:
    state["translation_running"] = False
    state["translation_confirm"] = False
    state["translation_overwrite_confirm"] = False
    state["translation_preview_signature"] = None
