import pytest

from config.actions.input_check_length import check_input_length


@pytest.mark.asyncio
async def test_short_prompt_is_valid():
    context = {
        "user_message": "Quelle est la durée légale du travail ?"
    }

    result = await check_input_length(context)

    assert result is True


@pytest.mark.asyncio
async def test_long_prompt_is_rejected():
    context = {
        "user_message": "a" * 501
    }

    result = await check_input_length(context)

    assert result is False


@pytest.mark.asyncio
async def test_only_user_question_is_measured_in_enriched_prompt():
    long_rag_context = "a" * 5000

    prompt = f"""
<LLM_SYSTEM>
Tu es un assistant juridique.
</LLM_SYSTEM>

<RAG_CONTEXT>
{long_rag_context}
</RAG_CONTEXT>

<USER_QUESTION>
Quelle est la durée légale du travail ?
</USER_QUESTION>
"""

    context = {
        "user_message": prompt
    }

    result = await check_input_length(context)

    assert result is True


@pytest.mark.asyncio
async def test_long_user_question_inside_enriched_prompt_is_rejected():
    long_question = "a" * 501

    prompt = f"""
<LLM_SYSTEM>
Tu es un assistant juridique.
</LLM_SYSTEM>

<RAG_CONTEXT>
Contexte court.
</RAG_CONTEXT>

<USER_QUESTION>
{long_question}
</USER_QUESTION>
"""

    context = {
        "user_message": prompt
    }

    result = await check_input_length(context)

    assert result is False


@pytest.mark.asyncio
async def test_empty_prompt_is_valid():
    context = {
        "user_message": ""
    }

    result = await check_input_length(context)

    assert result is True
