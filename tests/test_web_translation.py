from eduecon_radar.services.translation import TranslationParameters
from eduecon_radar.services.web_translation import (
    claim_translation_run,
    finish_translation_run,
    translation_preview_is_valid,
)


def _state(parameters: TranslationParameters) -> dict:
    return {
        "translation_preview": object(),
        "translation_preview_signature": parameters.signature,
        "translation_confirm": True,
        "translation_overwrite_confirm": False,
    }


def test_translation_parameter_change_invalidates_preview():
    parameters = TranslationParameters()
    state = _state(parameters)

    assert translation_preview_is_valid(state, parameters.signature)
    assert not translation_preview_is_valid(state, TranslationParameters(limit=11).signature)


def test_translation_requires_cost_confirmation():
    parameters = TranslationParameters()
    state = _state(parameters)
    state["translation_confirm"] = False

    assert not claim_translation_run(state, parameters.signature, overwrite_human=False)


def test_translation_requires_separate_human_overwrite_confirmation():
    parameters = TranslationParameters(overwrite_human=True)
    state = _state(parameters)

    assert not claim_translation_run(state, parameters.signature, overwrite_human=True)
    state["translation_overwrite_confirm"] = True
    assert claim_translation_run(state, parameters.signature, overwrite_human=True)


def test_translation_rerun_cannot_claim_same_signature_twice():
    parameters = TranslationParameters()
    state = _state(parameters)

    assert claim_translation_run(state, parameters.signature, overwrite_human=False)
    finish_translation_run(state)
    state["translation_preview"] = object()
    state["translation_preview_signature"] = parameters.signature
    state["translation_confirm"] = True
    assert not claim_translation_run(state, parameters.signature, overwrite_human=False)
