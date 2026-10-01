import pytest

from config.actions.input_PII_leakage import mask_pii


@pytest.mark.asyncio
async def test_email_is_masked_in_simple_prompt():
    context = {
        "user_message": "Mon email est hamza.test@gmail.com. Comment contester une sanction disciplinaire ?"
    }

    result = await mask_pii(context)

    masked_message = result["user_message"]
    assert "hamza.test@gmail.com" not in masked_message
    assert "<EMAIL_ADDRESS>" in masked_message
    assert "Comment contester une sanction disciplinaire" in masked_message


@pytest.mark.asyncio
async def test_phone_number_is_masked_in_simple_prompt():
    context = {
        "user_message": "Mon numéro est 06 12 34 56 78. Quelle est la durée légale du travail ?"
    }

    result = await mask_pii(context)

    masked_message = result["user_message"]
    assert "06 12 34 56 78" not in masked_message
    assert "<PHONE_NUMBER>" in masked_message
    assert "Quelle est la durée légale du travail" in masked_message


@pytest.mark.asyncio
async def test_prompt_without_pii_is_not_modified():
    original_prompt = "Quelle est la durée légale du travail ?"
    context = {
        "user_message": original_prompt
    }

    result = await mask_pii(context)

    assert result["user_message"] == original_prompt


@pytest.mark.asyncio
async def test_only_user_question_is_masked_in_enriched_prompt():
    prompt = """
<LLM_SYSTEM>
Tu es un assistant juridique.
</LLM_SYSTEM>

<RAG_CONTEXT>
Contact source documentaire : source@example.com
</RAG_CONTEXT>

<USER_QUESTION>
Mon email est hamza.test@gmail.com. Comment contester une sanction disciplinaire ?
</USER_QUESTION>
"""

    context = {
        "user_message": prompt
    }

    result = await mask_pii(context)
    masked_prompt = result["user_message"]

    assert "hamza.test@gmail.com" not in masked_prompt
    assert "<EMAIL_ADDRESS>" in masked_prompt
    assert "source@example.com" in masked_prompt
    assert "<RAG_CONTEXT>" in masked_prompt
    assert "<USER_QUESTION>" in masked_prompt


@pytest.mark.asyncio
async def test_email_and_phone_are_masked_together():
    context = {
        "user_message": "Mon email est hamza@gmail.com et mon téléphone est 07 11 22 33 44. Quels sont mes droits ?"
    }

    result = await mask_pii(context)

    masked_message = result["user_message"]
    assert "hamza@gmail.com" not in masked_message
    assert "07 11 22 33 44" not in masked_message
    assert "<EMAIL_ADDRESS>" in masked_message
    assert "<PHONE_NUMBER>" in masked_message
