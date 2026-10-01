from nemoguardrails.actions import action

from config.actions.pii_utils import mask_text_pii
from config.actions.rails_settings import get_rail_settings
from utils.console import log_section_end, log_section_start, log_step


# On charge la configuration PII de sortie depuis rails_settings.yml.
SETTINGS = get_rail_settings(
    "output_pii",
    {
        "language": "fr",
        "model_name": "fr_core_news_md",
        "entities": ["EMAIL_ADDRESS", "PHONE_NUMBER"],
        "labels_to_ignore": ["MISC"],
    },
)


@action(name="output_mask_pii")
async def output_mask_pii(context: dict):
    """Analyse la réponse du bot et masque les données personnelles configurées."""
    log_section_start("OUTPUT_MASK_PII", "Debut")

    # NeMo peut stocker la réponse sous bot_message ou last_bot_message selon le flow.
    bot_message = context.get("bot_message") or context.get("last_bot_message") or ""

    # La logique Presidio commune est dans pii_utils.py.
    # Ici on l applique directement à la réponse du modèle.
    masked_text, has_pii = mask_text_pii(
        bot_message,
        language=SETTINGS["language"],
        model_name=SETTINGS["model_name"],
        entities=SETTINGS["entities"],
        labels_to_ignore=SETTINGS["labels_to_ignore"],
    )

    # Si aucune donnée personnelle n est trouvée, la réponse reste inchangée.
    if not has_pii:
        log_step("OUTPUT_MASK_PII", "Aucun PII detecte.")
        log_step("OUTPUT_MASK_PII", "OK")
        log_section_end()
        return {"bot_message": bot_message}

    # En cas de masquage, on logge l avant/après pour le debug
    log_step("OUTPUT_MASK_PII", f"Reponse originale : {bot_message}")
    log_step("OUTPUT_MASK_PII", f"Reponse masquee   : {masked_text}")
    log_step("OUTPUT_MASK_PII", "PII detecte et masque.")
    log_step("OUTPUT_MASK_PII", "OK")
    log_section_end()

    return {"bot_message": masked_text}
