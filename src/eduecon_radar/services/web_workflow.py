from __future__ import annotations

from collections.abc import MutableMapping


def workflow_preview_is_valid(
    state: MutableMapping[str, object], signature: tuple[object, ...]
) -> bool:
    return (
        bool(state.get("workflow_preview")) and state.get("workflow_preview_signature") == signature
    )


def claim_workflow_run(state: MutableMapping[str, object], signature: tuple[object, ...]) -> bool:
    if state.get("workflow_running"):
        return False
    if state.get("workflow_consumed_signature") == signature:
        return False
    if not workflow_preview_is_valid(state, signature):
        return False
    if not state.get("workflow_confirm"):
        return False
    state["workflow_running"] = True
    state["workflow_consumed_signature"] = signature
    return True


def finish_workflow_run(state: MutableMapping[str, object]) -> None:
    state["workflow_running"] = False
    state["workflow_confirm"] = False
    state["workflow_preview_signature"] = None
