from __future__ import annotations

import asyncio
from nemoguardrails.actions import action
from ollama import Client

from utils.console import log_section_end, log_section_start, log_step
from config.actions.rails_settings import get_rail_settings


# On charge la config du rail depuis le fichier de settings, avec des valeurs
# par défaut si rien n'est précisé quelque part.
SETTINGS = get_rail_settings(
    "output_rag_support",
    {
        "ollama_url": "http://localhost:11434",
        "judge_model": "qwen2.5:3b",
        "temperature": 0,
        "num_predict": 8,
        "blocking_response": "Je ne suis pas en mesure de confirmer cette reponse a partir des documents fournis.",
    },
)

OLLAMA_URL = SETTINGS["ollama_url"]
JUDGE_MODEL = SETTINGS["judge_model"]
JUDGE_TEMPERATURE = SETTINGS["temperature"]
JUDGE_NUM_PREDICT = int(SETTINGS["num_predict"])
# Message renvoyé à la place de la réponse du bot quand on juge qu'elle n'est pas fiable.
BLOCKING_RESPONSE = SETTINGS["blocking_response"]


def _extract_tag_content(text: str, tag_name: str) -> str:
    """Extrait le contenu placé entre deux balises simples."""
    if not text:
        return ""

    # Exemple : tag_name="RAG_CONTEXT" cherche <RAG_CONTEXT>...</RAG_CONTEXT>.
    start_tag = f"<{tag_name}>"
    end_tag = f"</{tag_name}>"
    start = text.find(start_tag)
    # On cherche la balise de fin après la balise de début, pas avant,
    # sinon on risque de couper le texte au mauvais endroit.
    end = text.find(end_tag, start + len(start_tag))

    # Si une des deux balises n'existe pas, on considère qu'il n'y a rien à extraire.
    if start == -1 or end == -1:
        return ""

    return text[start + len(start_tag):end].strip()


def _extract_rag_context(context: dict | None) -> str:
    """Récupère le contexte RAG depuis NeMo ou depuis le prompt enrichi."""
    if not context:
        return ""
    
    
    explicit_context = context.get("rag_context") or context.get("retrieved_context")
    if explicit_context:
        return str(explicit_context).strip()


    # le contexte est dans la section <RAG_CONTEXT> du prompt
    # envoyé par l'utilisateur (injecté en amont par la chaîne RAG).
    user_message = str(
        context.get("user_message")
        or context.get("last_user_message")
        or ""
    )
    return _extract_tag_content(user_message, "RAG_CONTEXT")


def _extract_bot_message(context: dict | None) -> str:
    """Récupère la réponse générée par le LLM depuis le contexte NeMo."""
    if not context:
        return ""

    return str(
        context.get("bot_message")
        or context.get("last_bot_message")
        or ""
    ).strip()


def _build_judge_prompt(rag_context: str, bot_message: str) -> str:
    """Construit le prompt donné au LLM judge pour vérifier le support RAG."""
    # Le prompt est volontairement très directif
    # pour limiter les réponses bavardes du petit modèle juge.
    return f"""Tu es un juge de conformite RAG.

Ta tache est de verifier si la reponse du modele est entierement supportee par le contexte fourni.

Regles :
- Utilise uniquement le contexte fourni.
- N'utilise aucune connaissance externe.
- Si une information importante de la reponse n'apparait pas dans le contexte, reponds NO.
- Si la reponse contredit le contexte, reponds NO.
- Si la reponse est vague mais ne contredit pas le contexte, reponds YES.
- Si la reponse dit qu'il n'y a pas assez de sources, reponds YES.
- Reponds uniquement par YES ou NO.

CONTEXTE RAG :
{rag_context}

REPONSE A VERIFIER :
{bot_message}

La reponse est-elle entierement supportee par le contexte ?"""


def _parse_judge_answer(answer):
    """Convertit la réponse YES/NO du judge en décision exploitable."""
    cleaned = answer.strip().upper()

    # On accepte aussi les réponses qui commencent par YES/NO pour être tolérant aux petits écarts
    # (par exemple si le modèle répond "YES, car..." au lieu de juste "YES").
    if cleaned.startswith("YES"):
        return True, "YES"
    if cleaned.startswith("NO"):
        return False, "NO"

    # Si YES/NO n'est pas au début, on regarde si un seul des deux mots apparaît
    # quelque part dans la réponse.
    if "YES" in cleaned and "NO" not in cleaned:
        return True, "YES"
    if "NO" in cleaned and "YES" not in cleaned:
        return False, "NO"

    # Cas ambigu (les deux mots présents, ou aucun) : par sécurité on bloque,
    return False, "UNKNOWN"


def _call_rag_support_judge(rag_context, bot_message):
    """Appelle Ollama pour vérifier si la réponse est supportée par les chunks RAG."""
    client = Client(host=OLLAMA_URL)
    prompt = _build_judge_prompt(rag_context, bot_message)

    response = client.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": JUDGE_TEMPERATURE,
            # num_predict est volontairement petit : on n'attend qu'un YES ou NO,
            # pas besoin de laisser le modèle générer plus de texte.
            "num_predict": JUDGE_NUM_PREDICT,
        },
    )

    answer = response["message"]["content"]
    is_grounded, parsed_answer = _parse_judge_answer(answer)

    return {
        "is_grounded": is_grounded,
        "judge_answer": parsed_answer,
        "raw_judge_answer": answer.strip(),
    }


@action(is_system_action=True, name="check_rag_support")
async def check_rag_support(context: dict):
    """Bloque ou remplace une réponse qui n'est pas justifiée par le contexte RAG."""
    log_section_start("OUTPUT_RAG_SUPPORT", "Debut de l'analyse")

    bot_message = _extract_bot_message(context)
    rag_context = _extract_rag_context(context)

    # Rien à vérifier si le bot n'a rien répondu : on laisse passer tel quel.
    if not bot_message:
        log_step("OUTPUT_RAG_SUPPORT", "Message vide, verification ignoree")
        log_section_end()
        return {
            "bot_message": bot_message,
            "is_grounded": True,
            "rag_support_failed": False,
            "detection_method": "empty_output",
            "judge_answer": "",
        }

    # Sans contexte RAG, impossible de vérifier quoi que ce soit : on bloque par précaution
    # plutôt que de laisser passer une réponse potentiellement non fondée.
    if not rag_context:
        log_step("OUTPUT_RAG_SUPPORT", "Contexte RAG absent")
        log_section_end()
        return {
            "bot_message": BLOCKING_RESPONSE,
            "is_grounded": False,
            "rag_support_failed": True,
            "detection_method": "missing_rag_context",
            "judge_answer": "",
        }

    # L'appel Ollama est bloquant, donc il est exécuté dans un thread séparé
    # pour ne pas bloquer la boucle asyncio de NeMo Guardrails.
    try:
        result = await asyncio.to_thread(
            _call_rag_support_judge,
            rag_context,
            bot_message,
        )
    except Exception as exc:
        # Si le juge Ollama plante (service down, timeout, etc.), on bloque par précaution 
        log_step("OUTPUT_RAG_SUPPORT", f"Erreur juge Ollama : {exc}")
        log_section_end()
        return {
            "bot_message": BLOCKING_RESPONSE,
            "is_grounded": False,
            "rag_support_failed": True,
            "detection_method": "judge_error",
            "judge_answer": "",
        }

    is_grounded = result["is_grounded"]
    log_step(
        "OUTPUT_RAG_SUPPORT",
        f"Juge={result['judge_answer']} - grounded={is_grounded}",
    )
    log_section_end()

    # Si le juge estime que la réponse n'est pas fondée, on la remplace
    # par le message de blocage plutôt que de la renvoyer telle quelle.
    return {
        "bot_message": bot_message if is_grounded else BLOCKING_RESPONSE,
        "is_grounded": is_grounded,
        "rag_support_failed": not is_grounded,
        "detection_method": "ollama_judge",
        "judge_answer": result["judge_answer"],
        "raw_judge_answer": result["raw_judge_answer"],
    }