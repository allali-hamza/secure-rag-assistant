from nemoguardrails.actions import action

from config.actions.pii_utils import mask_text_pii
from config.actions.prompt_parser import extract_user_question, replace_user_question
from config.actions.rails_settings import get_rail_settings
from utils.console import log_section_end, log_section_start, log_step


# On charge la configuration PII depuis rails_settings.yml.
# Les valeurs par défaut évitent de planter si la section est absente.
SETTINGS = get_rail_settings(
    "input_pii",
    {
        "language": "fr",
        "model_name": "fr_core_news_md",
        "entities": ["EMAIL_ADDRESS", "PHONE_NUMBER"],
        "labels_to_ignore": ["MISC"],
    },
)


@action(name="mask_pii")
async def mask_pii(context: dict):
    """Masque les données personnelles présentes dans la question utilisateur."""
    log_section_start("INPUT_PII_LEAKAGE", "Debut")

    # Le prompt complet contient le système et le RAG : on isole seulement la question.
    full_prompt = context.get("user_message") or context.get("last_user_message") or ""
    user_question = extract_user_question(full_prompt)

    # La logique Presidio commune est dans pii_utils.py.
    # Cette action décide seulement où réinjecter le texte masqué.
    masked_question, has_pii = mask_text_pii(
        user_question,
        language=SETTINGS["language"],
        model_name=SETTINGS["model_name"],
        entities=SETTINGS["entities"],
        labels_to_ignore=SETTINGS["labels_to_ignore"],
    )

    # Si aucune donnée sensible n'est trouvée, on garde le prompt inchangé.
    if not has_pii:
        log_step("INPUT_PII_LEAKAGE", "Aucun PII detecte.")
        log_step("INPUT_PII_LEAKAGE", "OK")
        log_section_end()
        return {"user_message": full_prompt}

    # On remplace uniquement la question utilisateur, pas le contexte RAG ni le système.
    masked_prompt = replace_user_question(full_prompt, masked_question)

    # Affichage debug pour montrer ce qui a été détecté et masqué.
    print(f"\nPrompt original : {user_question}")
    print(f"Prompt masqué   : {masked_question}\n")

    # Le fichier .co récupère cette valeur pour remplacer user_message dans le pipeline.
    log_step("INPUT_PII_LEAKAGE", "PII detecte et masque.")
    log_step("INPUT_PII_LEAKAGE", "OK")
    log_section_end()
    return {"user_message": masked_prompt}
