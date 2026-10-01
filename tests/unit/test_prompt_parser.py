from config.actions.prompt_parser import extract_user_question, replace_user_question, has_user_question_section


def test_extract_user_question():
    prompt = """
<RAG_CONTEXT>
Contexte quelconque
</RAG_CONTEXT>

<USER_QUESTION>
Quelle est la durée légale du travail ?
</USER_QUESTION>
"""

    result = extract_user_question(prompt)

    assert result == "Quelle est la durée légale du travail ?"



def test_replace_user_question():
    prompt = """
<USER_QUESTION>
Ancienne question
</USER_QUESTION>
"""

    result = replace_user_question(prompt, "Nouvelle question")

    assert "Nouvelle question" in result
    assert "Ancienne question" not in result


def test_has_user_question_section():
    prompt = """
<USER_QUESTION>
Une question
</USER_QUESTION>
"""

    assert has_user_question_section(prompt) is True


def test_has_no_user_question_section():
    prompt = "Prompt sans section utilisateur"

    assert has_user_question_section(prompt) is False