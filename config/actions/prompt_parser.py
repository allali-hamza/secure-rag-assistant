
def has_user_question_section(prompt):
    """Vérifie si le prompt contient une section USER_QUESTION explicite."""
    # Ces balises sont ajoutées dans app.py quand le prompt est enrichi avec le RAG.
    return prompt.find("<USER_QUESTION>") != -1 and prompt.find("</USER_QUESTION>") != -1

def extract_user_question(prompt):
    """Extrait uniquement la question utilisateur depuis le prompt enrichi."""
    if not prompt:
        return ""

    # Si le prompt n est pas enrichi, on considère que tout le texte est la question.
    if not has_user_question_section(prompt):
        return prompt.strip()

    # On prend uniquement le texte situé entre les deux balises USER_QUESTION.
    start = prompt.find("<USER_QUESTION>") + len("<USER_QUESTION>")
    end = prompt.find("</USER_QUESTION>", start)

    # Si la balise de fin est absente, on évite de couper le prompt au mauvais endroit.
    if end == -1:
        return prompt.strip()

    return prompt[start:end].strip()

def replace_user_question(prompt, new_question):
    """Remplace la question utilisateur sans toucher au contexte RAG ni au système."""
    if not prompt:
        return new_question

    start = prompt.find("<USER_QUESTION>")
    end = prompt.find("</USER_QUESTION>")

    # Si les balises sont absentes, on retourne directement la nouvelle question.
    if start == -1 or end == -1:
        return new_question

    # On conserve les balises et on remplace seulement leur contenu.
    content_start = start + len("<USER_QUESTION>")

    return (prompt[:content_start] + "\n" + new_question.strip() + "\n" + prompt[end:])



