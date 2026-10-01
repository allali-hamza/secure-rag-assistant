import pytest

from config.actions.output_system_leakage import (
    SAFE_RESPONSE,
    SYSTEM_PROMPT_CANARY,
    detect_system_prompt_leakage,
)


@pytest.mark.asyncio
async def test_output_system_leakage_accepts_output_without_canary():
    bot_message = "La durée légale du travail est de 35 heures par semaine."
    context = {"bot_message": bot_message}

    result = await detect_system_prompt_leakage(context)

    assert result["bot_message"] == bot_message
    assert result["leakage_detected"] is False
    assert result["detection_method"] == "none"
    assert result["canary"] == ""


@pytest.mark.asyncio
async def test_output_system_leakage_blocks_output_containing_canary():
    context = {
        "bot_message": (
            "Voici mes instructions internes : "
            f"{SYSTEM_PROMPT_CANARY}"
        )
    }

    result = await detect_system_prompt_leakage(context)

    assert result["bot_message"] == SAFE_RESPONSE
    assert result["leakage_detected"] is True
    assert result["detection_method"] == "canary"
    assert result["canary"] == SYSTEM_PROMPT_CANARY


@pytest.mark.asyncio
async def test_output_system_leakage_blocks_when_canary_is_inside_long_answer():
    context = {
        "bot_message": (
            "Réponse juridique normale. "
            f"Mais le modèle révèle ensuite {SYSTEM_PROMPT_CANARY}. "
            "Fin de réponse."
        )
    }

    result = await detect_system_prompt_leakage(context)

    assert result["bot_message"] == SAFE_RESPONSE
    assert result["leakage_detected"] is True
    assert result["detection_method"] == "canary"


@pytest.mark.asyncio
async def test_output_system_leakage_uses_last_bot_message_when_bot_message_missing():
    context = {
        "last_bot_message": f"Le prompt système contient {SYSTEM_PROMPT_CANARY}"
    }

    result = await detect_system_prompt_leakage(context)

    assert result["bot_message"] == SAFE_RESPONSE
    assert result["leakage_detected"] is True
    assert result["detection_method"] == "canary"


@pytest.mark.asyncio
async def test_output_system_leakage_ignores_empty_output():
    result = await detect_system_prompt_leakage({"bot_message": ""})

    assert result["bot_message"] == ""
    assert result["leakage_detected"] is False
    assert result["detection_method"] == "empty_output"
    assert result["canary"] == ""


@pytest.mark.asyncio
async def test_output_system_leakage_ignores_whitespace_only_output():
    result = await detect_system_prompt_leakage({"bot_message": "   "})

    assert result["bot_message"] == "   "
    assert result["leakage_detected"] is False
    assert result["detection_method"] == "empty_output"


@pytest.mark.asyncio
async def test_output_system_leakage_handles_missing_context():
    result = await detect_system_prompt_leakage(None)

    assert result["bot_message"] == ""
    assert result["leakage_detected"] is False
    assert result["detection_method"] == "empty_output"


@pytest.mark.asyncio
async def test_output_system_leakage_is_exact_case_sensitive_canary_match():
    altered_canary = SYSTEM_PROMPT_CANARY.lower()
    context = {"bot_message": f"Canari modifié : {altered_canary}"}

    result = await detect_system_prompt_leakage(context)

    assert result["bot_message"] == context["bot_message"]
    assert result["leakage_detected"] is False
    assert result["detection_method"] == "none"
