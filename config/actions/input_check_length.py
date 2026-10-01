from nemoguardrails.actions import action
from config.actions.prompt_parser import extract_user_question
from utils.console import log_section_end, log_section_start, log_step
from config.actions.rails_settings import get_rail_settings

# On charge la limite de taille depuis rails_settings.yml, avec une valeur par défaut.
SETTINGS = get_rail_settings("input_check_length", {"max_chars": 500})

@action(name="check_input_length")
async def check_input_length(context):
    """Vérifie que la question utilisateur ne dépasse pas la limite configurée."""
    log_section_start("INPUT_CHECK_LENGTH", "Debut")
    len_max = int(SETTINGS["max_chars"])

    # NeMo peut fournir le message sous plusieurs clés selon le moment du pipeline.
    user_message = context.get("user_message") or context.get("last_user_message") or ""

    # Le prompt contient le système et le RAG.
    # Ici on mesure seulement la vraie question utilisateur.
    user_message = extract_user_question(user_message)
    len_usr_message = len(user_message)
    log_step("INPUT_CHECK_LENGTH", f"Longueur du message utilisateur : {len_usr_message} caracteres")
    
    # Si la question est trop longue, le flow Colang bloque la requête.
    if len_usr_message > len_max:
        log_step("INPUT_CHECK_LENGTH", "Le message utilisateur depasse la longueur maximale autorisee.")
        log_section_end()
        return False
    
    log_step("INPUT_CHECK_LENGTH", "OK")
    log_section_end()
    return True