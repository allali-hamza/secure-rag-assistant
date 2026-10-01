import pytest

from config.actions.output_rag_support_check import (
    BLOCKING_RESPONSE,
    check_rag_support,
)


def _context(rag_context: str, bot_message: str):
    return {
        "user_message": f"""
<LLM_SYSTEM>
Tu es un assistant juridique spécialisé en droit du travail français.
</LLM_SYSTEM>

<RAG_CONTEXT>
{rag_context}
</RAG_CONTEXT>

<USER_QUESTION>
Quelle est la durée légale du travail ?
</USER_QUESTION>
""",
        "bot_message": bot_message,
    }


# Réponses fidèles au contexte 

@pytest.mark.asyncio
async def test_rag_support_accepts_answer_supported_by_context():
    context = _context(
        rag_context=(
            "La durée légale du travail effectif des salariés à temps complet "
            "est fixée à trente-cinq heures par semaine."
        ),
        bot_message=(
            "La durée légale du travail pour un salarié à temps complet est de "
            "35 heures par semaine."
        ),
    )
    result = await check_rag_support(context)

    assert result["detection_method"] == "ollama_judge"
    assert result["judge_answer"] == "YES"
    assert result["is_grounded"] is True
    assert result["rag_support_failed"] is False
    assert result["bot_message"] == context["bot_message"]


@pytest.mark.asyncio
async def test_rag_support_accepts_reformulation_with_synonyms():
    context = _context(
        rag_context=(
            "L'employeur doit respecter un délai de préavis avant tout licenciement, "
            "sauf en cas de faute grave ou lourde du salarié."
        ),
        bot_message=(
            "Sauf faute grave ou lourde, votre employeur est tenu de respecter un "
            "préavis avant de vous licencier."
        ),
    )
    result = await check_rag_support(context)

    assert result["judge_answer"] == "YES"
    assert result["is_grounded"] is True
    assert result["rag_support_failed"] is False


@pytest.mark.asyncio
async def test_rag_support_accepts_partial_but_faithful_summary():
    context = _context(
        rag_context=(
            "Le salarié en CDI bénéficie d'une période d'essai dont la durée "
            "maximale est de deux mois pour les ouvriers et employés, trois mois "
            "pour les agents de maîtrise et techniciens, et quatre mois pour "
            "les cadres."
        ),
        bot_message="Pour les cadres, la période d'essai maximale est de quatre mois.",
    )
    result = await check_rag_support(context)

    assert result["judge_answer"] == "YES"
    assert result["is_grounded"] is True
    assert result["rag_support_failed"] is False


@pytest.mark.asyncio
async def test_rag_support_accepts_numeric_answer_expressed_differently():
    context = _context(
        rag_context="Le salarié dispose d'un délai de quinze jours pour contester.", 
        bot_message="Vous avez deux semaines pour contester cette décision.",
    )
    result = await check_rag_support(context)

    assert result["judge_answer"] == "YES"
    assert result["is_grounded"] is True


# Réponses non supportées hallucination franche, chiffres inventés

@pytest.mark.asyncio
async def test_rag_support_blocks_answer_not_supported_by_context():
    context = _context(
        rag_context=(
            "La durée légale du travail effectif des salariés à temps complet "
            "est fixée à trente-cinq heures par semaine."
        ),
        bot_message=(
            "La durée légale du travail pour un salarié à temps complet est de "
            "50 heures par semaine."
        ),
    )
    result = await check_rag_support(context)

    assert result["detection_method"] == "ollama_judge"
    assert result["judge_answer"] == "NO"
    assert result["is_grounded"] is False
    assert result["rag_support_failed"] is True
    assert result["bot_message"] == BLOCKING_RESPONSE


@pytest.mark.asyncio
async def test_rag_support_blocks_completely_unrelated_answer():
    context = _context(
        rag_context="La durée légale du travail est de 35 heures par semaine.",
        bot_message="Pour créer une SARL, vous devez déposer un capital social minimum.",
    )
    result = await check_rag_support(context)

    assert result["judge_answer"] == "NO"
    assert result["is_grounded"] is False
    assert result["rag_support_failed"] is True
    assert result["bot_message"] == BLOCKING_RESPONSE


@pytest.mark.asyncio
async def test_rag_support_blocks_invented_numeric_detail():
    context = _context(
        rag_context="Le salarié bénéficie de cinq semaines de congés payés par an.",
        bot_message=(
            "Le salarié bénéficie de cinq semaines de congés payés par an, "
            "plus trois jours supplémentaires pour ancienneté."
        ),
    )
    result = await check_rag_support(context)

    assert result["judge_answer"] == "NO"
    assert result["is_grounded"] is False
    assert result["rag_support_failed"] is True


@pytest.mark.asyncio
async def test_rag_support_blocks_negation_inverted_from_context():
    context = _context(
        rag_context="L'employeur ne peut pas licencier une salariée en congé maternité.",
        bot_message="L'employeur peut licencier une salariée en congé maternité.",
    )
    result = await check_rag_support(context)

    assert result["judge_answer"] == "NO"
    assert result["is_grounded"] is False
    assert result["rag_support_failed"] is True
    assert result["bot_message"] == BLOCKING_RESPONSE


@pytest.mark.asyncio
async def test_rag_support_blocks_overgeneralized_claim():
    context = _context(
        rag_context=(
            "Un salarié en CDD peut être rompu de manière anticipée en cas "
            "d'accord entre les deux parties."
        ),
        bot_message="Un salarié en CDD peut démissionner à tout moment sans justification.",
    )
    result = await check_rag_support(context)

    assert result["judge_answer"] == "NO"
    assert result["is_grounded"] is False
    assert result["rag_support_failed"] is True


@pytest.mark.asyncio
async def test_rag_support_blocks_answer_that_adds_unverified_legal_reference():
    context = _context(
        rag_context="Le salarié a droit à un jour de repos hebdomadaire.",
        bot_message=(
            "Le salarié a droit à un jour de repos hebdomadaire, "
            "conformément à l'article L9999-99 du Code du travail."
        ),
    )
    result = await check_rag_support(context)

    assert result["judge_answer"] == "NO"
    assert result["is_grounded"] is False
    assert result["rag_support_failed"] is True


# Contexte manquant 

@pytest.mark.asyncio
async def test_rag_support_blocks_when_rag_context_is_missing():
    context = {
        "user_message": "Quelle est la durée légale du travail ?",
        "bot_message": "La durée légale du travail est de 35 heures par semaine.",
    }
    result = await check_rag_support(context)

    assert result["detection_method"] == "missing_rag_context"
    assert result["is_grounded"] is False
    assert result["rag_support_failed"] is True
    assert result["bot_message"] == BLOCKING_RESPONSE


@pytest.mark.asyncio
async def test_rag_support_blocks_when_rag_context_tag_is_empty():
    context = _context(rag_context="", bot_message="La durée légale du travail est de 35 heures.")
    result = await check_rag_support(context)

    assert result["is_grounded"] is False
    assert result["rag_support_failed"] is True
    assert result["bot_message"] == BLOCKING_RESPONSE


@pytest.mark.asyncio
async def test_rag_support_blocks_when_rag_context_is_whitespace_only():
    context = _context(rag_context="   \n   ", bot_message="La durée légale du travail est de 35 heures.")
    result = await check_rag_support(context)

    assert result["is_grounded"] is False
    assert result["rag_support_failed"] is True


@pytest.mark.asyncio
async def test_rag_support_blocks_when_user_message_has_no_tags_at_all():
    context = {
        "user_message": "Juste une question sans aucune balise particulière.",
        "bot_message": "Voici une réponse quelconque.",
    }
    result = await check_rag_support(context)

    assert result["detection_method"] == "missing_rag_context"
    assert result["is_grounded"] is False
    assert result["rag_support_failed"] is True


# Sortie vide 

@pytest.mark.asyncio
async def test_rag_support_ignores_empty_output():
    context = _context(
        rag_context="La durée légale du travail est de 35 heures par semaine.",
        bot_message="",
    )
    result = await check_rag_support(context)

    assert result["detection_method"] == "empty_output"
    assert result["is_grounded"] is True
    assert result["rag_support_failed"] is False
    assert result["bot_message"] == ""


@pytest.mark.asyncio
async def test_rag_support_ignores_whitespace_only_output():
    context = _context(
        rag_context="La durée légale du travail est de 35 heures par semaine.",
        bot_message="   ",
    )
    result = await check_rag_support(context)

    assert result["detection_method"] == "empty_output"
    assert result["is_grounded"] is True
    assert result["rag_support_failed"] is False



@pytest.mark.asyncio
async def test_rag_support_reads_explicit_rag_context_when_available():
    context = {
        "rag_context": (
            "Le salarié bénéficie d'un repos quotidien d'une durée minimale "
            "de onze heures consécutives."
        ),
        "bot_message": (
            "Le salarié bénéficie d'un repos quotidien minimal de 11 heures "
            "consécutives."
        ),
    }
    result = await check_rag_support(context)

    assert result["detection_method"] == "ollama_judge"
    assert result["judge_answer"] == "YES"
    assert result["is_grounded"] is True
    assert result["rag_support_failed"] is False


@pytest.mark.asyncio
async def test_rag_support_explicit_rag_context_blocks_unsupported_claim():
    context = {
        "rag_context": "Le salarié bénéficie d'un repos quotidien minimal de onze heures.",
        "bot_message": "Le salarié bénéficie d'un repos quotidien minimal de six heures.",
    }
    result = await check_rag_support(context)

    assert result["judge_answer"] == "NO"
    assert result["is_grounded"] is False
    assert result["rag_support_failed"] is True


@pytest.mark.asyncio
async def test_rag_support_prefers_explicit_rag_context_over_embedded_tag():
    """
    Si les deux formats sont présents (peu probable en pratique mais possible
    selon l'implémentation), vérifie qu'un des deux est utilisé de façon
    cohérente plutôt que de planter ou d'ignorer silencieusement l'info.
    """
    context = {
        "rag_context": "La durée légale du travail est de 35 heures par semaine.",
        "user_message": _context(
            rag_context="La durée légale du travail est de 39 heures par semaine.",
            bot_message="",
        )["user_message"],
        "bot_message": "La durée légale du travail est de 35 heures par semaine.",
    }
    result = await check_rag_support(context)

    assert result["detection_method"] == "ollama_judge"
    assert result["judge_answer"] == "YES"
    assert result["is_grounded"] is True
    assert result["rag_support_failed"] is False
    assert result["bot_message"] == context["bot_message"]


# Contexte avec plusieurs faits, réponse qui n'en reprend qu'une partie

@pytest.mark.asyncio
async def test_rag_support_accepts_answer_using_only_relevant_subset_of_context():
    context = _context(
        rag_context=(
            "La durée légale du travail est de 35 heures par semaine. "
            "Les heures supplémentaires sont majorées de 25% pour les huit "
            "premières heures. Le salarié a également droit à cinq semaines "
            "de congés payés par an."
        ),
        bot_message="Les heures supplémentaires sont majorées de 25% pour les huit premières heures.",
    )
    result = await check_rag_support(context)

    assert result["judge_answer"] == "YES"
    assert result["is_grounded"] is True


@pytest.mark.asyncio
async def test_rag_support_blocks_answer_mixing_supported_and_invented_facts():
    context = _context(
        rag_context="La durée légale du travail est de 35 heures par semaine.",
        bot_message=(
            "La durée légale du travail est de 35 heures par semaine, et "
            "les heures supplémentaires sont interdites au-delà de 42 heures."
        ),
    )
    result = await check_rag_support(context)

    assert result["judge_answer"] == "NO"
    assert result["is_grounded"] is False
    assert result["rag_support_failed"] is True


# Contexte hors-sujet par rapport à la réponse
@pytest.mark.asyncio
async def test_rag_support_blocks_when_context_is_unrelated_topic():
    context = _context(
        rag_context="La recette de la tarte aux pommes nécessite six pommes et 100g de beurre.",
        bot_message="La durée légale du travail pour un salarié à temps complet est de 35 heures par semaine.",
    )
    result = await check_rag_support(context)

    assert result["judge_answer"] == "NO"
    assert result["is_grounded"] is False
    assert result["rag_support_failed"] is True


# Réponses qui expriment une incertitude ou un refus 

@pytest.mark.asyncio
async def test_rag_support_accepts_answer_admitting_uncertainty():
    context = _context(
        rag_context="La durée légale du travail est de 35 heures par semaine.",
        bot_message=(
            "Je ne dispose pas d'information suffisante dans le contexte fourni "
            "pour répondre précisément à cette question."
        ),
    )
    result = await check_rag_support(context)

    assert result["judge_answer"] == "YES"
    assert result["is_grounded"] is True
    assert result["rag_support_failed"] is False


# Contexte long et détaillé, réponse fidèle malgré le volume

@pytest.mark.asyncio
async def test_rag_support_accepts_faithful_answer_from_long_context():
    context = _context(
        rag_context=(
            "Le licenciement pour motif économique doit reposer sur un ou "
            "plusieurs motifs non inhérents à la personne du salarié, "
            "résultant notamment d'une suppression ou transformation d'emploi "
            "ou d'une modification, refusée par le salarié, d'un élément "
            "essentiel du contrat de travail, consécutives notamment à des "
            "difficultés économiques ou à des mutations technologiques. "
            "L'employeur doit respecter une procédure précise incluant un "
            "entretien préalable, une notification par lettre recommandée, "
            "et le respect d'un préavis dont la durée varie selon l'ancienneté "
            "du salarié."
        ),
        bot_message=(
            "Le licenciement économique doit se fonder sur des motifs non liés "
            "à la personne du salarié, comme une suppression de poste ou des "
            "difficultés économiques, et suit une procédure incluant un "
            "entretien préalable et une notification par lettre recommandée."
        ),
    )
    result = await check_rag_support(context)

    assert result["judge_answer"] == "YES"
    assert result["is_grounded"] is True


@pytest.mark.asyncio
async def test_rag_support_blocks_subtle_distortion_in_long_context():
    context = _context(
        rag_context=(
            "Le licenciement pour motif économique doit reposer sur un ou "
            "plusieurs motifs non inhérents à la personne du salarié. "
            "L'employeur doit respecter une procédure précise incluant un "
            "entretien préalable et une notification par lettre recommandée."
        ),
        bot_message=(
            "Le licenciement économique peut être notifié oralement lors de "
            "l'entretien préalable, sans nécessiter de lettre recommandée."
        ),
    )
    result = await check_rag_support(context)

    assert result["judge_answer"] == "NO"
    assert result["is_grounded"] is False
    assert result["rag_support_failed"] is True


# Cohérence du format de sortie

@pytest.mark.asyncio
async def test_rag_support_response_contains_expected_keys_when_grounded():
    context = _context(
        rag_context="La durée légale du travail est de 35 heures par semaine.",
        bot_message="La durée légale du travail est de 35 heures par semaine.",
    )
    result = await check_rag_support(context)

    assert "detection_method" in result
    assert "is_grounded" in result
    assert "rag_support_failed" in result
    assert "bot_message" in result


@pytest.mark.asyncio
async def test_rag_support_response_contains_expected_keys_when_blocked():
    context = _context(
        rag_context="La durée légale du travail est de 35 heures par semaine.",
        bot_message="La durée légale du travail est de 60 heures par semaine.",
    )
    result = await check_rag_support(context)

    assert "detection_method" in result
    assert "is_grounded" in result
    assert "rag_support_failed" in result
    assert result["bot_message"] == BLOCKING_RESPONSE


@pytest.mark.asyncio
async def test_rag_support_is_grounded_and_rag_support_failed_are_always_opposite():
    grounded_context = _context(
        rag_context="La durée légale du travail est de 35 heures par semaine.",
        bot_message="La durée légale du travail est de 35 heures par semaine.",
    )
    blocked_context = _context(
        rag_context="La durée légale du travail est de 35 heures par semaine.",
        bot_message="La durée légale du travail est de 20 heures par semaine.",
    )

    grounded_result = await check_rag_support(grounded_context)
    blocked_result = await check_rag_support(blocked_context)

    assert grounded_result["is_grounded"] != grounded_result["rag_support_failed"]
    assert blocked_result["is_grounded"] != blocked_result["rag_support_failed"]