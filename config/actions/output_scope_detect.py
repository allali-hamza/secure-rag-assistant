

from __future__ import annotations

import asyncio
from threading import Lock

import numpy as np
from nemoguardrails.actions import action
from ollama import Client
from utils.console import log_section_end, log_section_start, log_step
from config.actions.rails_settings import get_rail_settings


# On charge les paramètres du rail depuis rails_settings.yml.
# Les seuils peuvent être ajustés sans toucher au code.
SETTINGS = get_rail_settings(
    "output_scope",
    {
        "ollama_url": "http://localhost:11434",
        "embedding_model": "mxbai-embed-large",
        "top_n": 3,
        "min_in_scope_similarity": 0.48,
        "min_score_margin": 0.02,
    },
)

OLLAMA_URL = SETTINGS["ollama_url"]
EMBEDDING_MODEL = SETTINGS["embedding_model"]
TOP_N = int(SETTINGS["top_n"])
MIN_IN_SCOPE_SIMILARITY = float(SETTINGS["min_in_scope_similarity"])
MIN_SCORE_MARGIN = float(SETTINGS["min_score_margin"])

# Exemples de réponses acceptables : ils donnent au modèle d embedding une idée du périmètre.
_ALLOWED_OUTPUT_EXAMPLES = [
    "La duree legale du travail pour un salarie a temps complet est encadree par le Code du travail.",
    "Les heures supplementaires donnent lieu a une majoration ou a un repos compensateur selon les regles applicables.",
    "Le contrat de travail organise la relation entre l'employeur et le salarie.",
    "La periode d'essai permet a l'employeur et au salarie d'evaluer la relation de travail.",
    "Les conges payes sont des droits du salarie prevus par le droit du travail.",
    "Le licenciement doit respecter une procedure et reposer sur un motif juridiquement valable.",
    "La rupture conventionnelle permet a l'employeur et au salarie de rompre le contrat d'un commun accord.",
    "Le conseil de prud'hommes peut etre saisi en cas de litige entre un salarie et un employeur.",
    "L'employeur a des obligations en matiere de sante et de securite au travail.",
    "Le comite social et economique represente le personnel dans l'entreprise.",
    "Le teletravail est encadre par le Code du travail, un accord collectif ou une charte.",
    "La convention collective peut completer les regles du Code du travail applicables aux salaries.",
    "Une sanction disciplinaire peut etre contestee par le salarie si elle est injustifiee ou disproportionnee.",
]

# Exemples de réponses hors-sujet : ils servent de contre-exemples.
_DENIED_OUTPUT_EXAMPLES = [
    "Le meilleur buteur du monde est un joueur de football connu pour ses records sportifs.",
    "Pour preparer un gateau au chocolat, il faut melanger les ingredients puis cuire la pate.",
    "Voici un programme Python qui trie une liste de nombres.",
    "La meteo de demain prevoit de la pluie et une baisse des temperatures.",
    "Ce film est recommande pour son scenario et la qualite de ses acteurs.",
    "Pour investir en bourse, il faut analyser les actions et diversifier son portefeuille.",
    "La Revolution francaise est un evenement politique majeur de l'histoire de France.",
    "Le score du match depend des buts marques par chaque equipe.",
    "Cette traduction anglaise reformule le texte original.",
    "La resolution de cette equation mathematique donne une valeur numerique.",
]

_client = None
_embedding_model = EMBEDDING_MODEL
_reference_lock = Lock()
_allowed_embeddings = None
_denied_embeddings = None


def _get_client() -> Client:
    """Retourne le client Ollama utilisé pour les embeddings de sortie."""
    global _client, _embedding_model

    # Le client Ollama est conservé en mémoire pour éviter de le recréer à chaque appel.
    if _client is None:
        _client = Client(host=OLLAMA_URL)

    return _client


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    """Normalise les embeddings pour comparer les textes avec une similarité cosinus."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Protection contre une division par zéro si un embedding vide apparaît.
    norms[norms == 0.0] = 1.0
    return embeddings / norms


def _embed(texts: list[str]) -> np.ndarray:
    """Convertit une liste de textes en embeddings Ollama normalisés."""
    response = _get_client().embed(model=_embedding_model, input=texts)
    embeddings = getattr(response, "embeddings", None)

    # Selon la version du client Ollama, la réponse peut être un objet ou un dictionnaire.
    if embeddings is None:
        embeddings = response.get("embeddings")

    if not embeddings:
        raise RuntimeError("Ollama n'a retourne aucun embedding.")

    return _normalize(np.asarray(embeddings, dtype=np.float32))


def _load_reference_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Charge les embeddings des exemples de réponses autorisées et hors-sujet."""
    global _allowed_embeddings, _denied_embeddings

    if _allowed_embeddings is not None and _denied_embeddings is not None:
        return _allowed_embeddings, _denied_embeddings

    # Le lock évite de construire les embeddings de référence deux fois en parallèle.
    with _reference_lock:
        if _allowed_embeddings is None or _denied_embeddings is None:
            log_step("OUTPUT_SCOPE", f"Chargement des embeddings de reference avec {_embedding_model}")
            _allowed_embeddings = _embed(_ALLOWED_OUTPUT_EXAMPLES)
            _denied_embeddings = _embed(_DENIED_OUTPUT_EXAMPLES)
            log_step("OUTPUT_SCOPE", "Embeddings de reference charges")

    return _allowed_embeddings, _denied_embeddings


def _mean_top_n(similarities: np.ndarray, top_n: int) -> float:
    """Calcule la moyenne des meilleurs scores pour stabiliser la décision."""
    if similarities.size == 0:
        return 0.0

    n = min(top_n, similarities.size)
    # np.partition récupère les meilleurs scores sans trier toute la liste.
    top_values = np.partition(similarities, -n)[-n:]
    return float(top_values.mean())


def _classify_output_scope(bot_message: str) -> dict:
    """Vérifie par embeddings si la réponse reste dans le domaine du droit du travail."""
    allowed_embeddings, denied_embeddings = _load_reference_embeddings()
    output_embedding = _embed([bot_message])[0]

    # Les embeddings sont normalisés : le produit scalaire donne une similarité cosinus.
    allowed_similarities = allowed_embeddings @ output_embedding
    denied_similarities = denied_embeddings @ output_embedding

    scope_score = _mean_top_n(allowed_similarities, TOP_N)
    offtopic_score = _mean_top_n(denied_similarities, TOP_N)
    # La marge mesure l avance du score autorisé sur le score hors-sujet.
    score_margin = scope_score - offtopic_score

    is_in_scope = (
        scope_score >= MIN_IN_SCOPE_SIMILARITY
        and score_margin >= MIN_SCORE_MARGIN
    )

    return {
        "is_in_scope": is_in_scope,
        "detection_method": "ollama_embedding",
        "scope_score": round(scope_score, 4),
        "offtopic_score": round(offtopic_score, 4),
        "score_margin": round(score_margin, 4),
    }


def _extract_bot_message(context, user_input):
    """Récupère la réponse du bot depuis le contexte NeMo ou depuis user_input."""
    # NeMo peut fournir la réponse dans le contexte, mais certains tests passent user_input.
    if context:
        bot_message = context.get("bot_message") or context.get("last_bot_message")
        if bot_message:
            return bot_message

    return user_input or ""


@action(is_system_action=True, name="detect_scope")
async def detect_scope(context = None, user_input = None):
    """Verifie si la reponse du LLM reste dans le perimetre code du travail."""
    log_section_start("OUTPUT_SCOPE", "Debut de l'analyse semantique")

    bot_message = _extract_bot_message(context, user_input)
    if not bot_message or not bot_message.strip():
        log_step("OUTPUT_SCOPE", "Message vide, aucune analyse necessaire.")
        log_section_end()
        return {
            "is_in_scope": True,
            "detection_method": "empty_output",
            "scope_score": 0.0,
            "offtopic_score": 0.0,
            "score_margin": 0.0,
        }

    # L appel Ollama est bloquant, donc on l exécute dans un thread séparé.
    try:
        result = await asyncio.to_thread(_classify_output_scope, bot_message.strip())
    except Exception as exc:
        # Si les embeddings échouent, on bloque la sortie par précaution.
        log_step("OUTPUT_SCOPE", f"Erreur Ollama embeddings : {exc}")
        log_section_end()
        return {
            "is_in_scope": False,
            "detection_method": "embedding_error",
            "scope_score": 0.0,
            "offtopic_score": 0.0,
            "score_margin": 0.0,
        }

    status = "In-scope" if result["is_in_scope"] else "Hors-sujet"
    log_step(
        "OUTPUT_SCOPE",
        f"{status} - scope={result['scope_score']:.3f}, "
        f"off={result['offtopic_score']:.3f}, "
        f"margin={result['score_margin']:.3f}",
    )
    log_section_end()
    return result
