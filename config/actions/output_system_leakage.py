
from __future__ import annotations

from nemoguardrails.actions import action
from utils.console import log_section_end, log_section_start, log_step
from config.actions.rails_settings import get_rail_settings


# On charge le canari et la réponse de sécurité depuis rails_settings.yml.
SETTINGS = get_rail_settings(
    "output_system_leakage",
    {
        "canary": "STAGE2026_SYSTEM_CANARY_9F3A7C",
        "safe_response": "Je ne peux pas reveler mes instructions internes. Je peux uniquement repondre dans le cadre du code du travail.",
    },
)

# Le canari est une chaîne secrète présente dans le prompt système.
SYSTEM_PROMPT_CANARY = SETTINGS["canary"]
SAFE_RESPONSE = SETTINGS["safe_response"]


def _extract_bot_message(context) -> str:
    """Récupère la réponse du bot depuis le contexte NeMo."""
    if not context:
        return ""

    return (
        context.get("bot_message")
        or context.get("last_bot_message")
        or ""
    )


def _detect_system_prompt_leakage(bot_message: str) -> dict:
    """Cherche le canari secret dans la réponse du modèle."""
    # Si le canari apparaît, cela indique que le prompt système a fui.
    leakage_detected = SYSTEM_PROMPT_CANARY in bot_message

    return {
        "leakage_detected": leakage_detected,
        "detection_method": "canary" if leakage_detected else "none",
        "canary": SYSTEM_PROMPT_CANARY if leakage_detected else "",
    }


@action(is_system_action=True, name="detect_system_prompt_leakage")
async def detect_system_prompt_leakage(context):
    """Detecte une fuite du prompt systeme avec un canari secret."""
    log_section_start("OUTPUT_SYSTEM_LEAKAGE", "Debut de l'analyse")

    bot_message = _extract_bot_message(context)
    # Si le modèle ne répond rien, il ne peut pas y avoir de fuite à détecter.
    if not bot_message.strip():
        log_step("OUTPUT_SYSTEM_LEAKAGE", "Message vide, aucune analyse necessaire.")
        log_section_end()
        return {
            "bot_message": bot_message,
            "leakage_detected": False,
            "detection_method": "empty_output",
            "canary": "",
        }

    # Le canari est une vérification simple et déterministe, sans modèle supplémentaire.
    result = _detect_system_prompt_leakage(bot_message)

    # Si le canari apparaît, on remplace la réponse par un message sûr.
    if result["leakage_detected"]:
        log_step("OUTPUT_SYSTEM_LEAKAGE", "Fuite detectee - canari retrouve dans la sortie")
        log_section_end()
        return {
            "bot_message": SAFE_RESPONSE,
            **result,
        }

    log_step("OUTPUT_SYSTEM_LEAKAGE", "OK - canari absent de la sortie")
    log_section_end()
    return {
        "bot_message": bot_message,
        **result,
    }
