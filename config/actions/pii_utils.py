from __future__ import annotations

from functools import lru_cache
from typing import Any

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine


def _get_pii_engines(
    language: str,
    model_name: str,
    labels_to_ignore,
) -> tuple[AnalyzerEngine, AnonymizerEngine]:
    """Crée ou réutilise les moteurs Presidio pour une configuration donnée."""
    # Presidio utilise spaCy pour détecter les entités dans le texte français.
    configuration: dict[str, Any] = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": language, "model_name": model_name},
        ],
        "ner_model_configuration": {
            "labels_to_ignore": list(labels_to_ignore),
        },
    }

    # Le provider construit le moteur NLP à partir de la configuration spaCy.
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    # Analyzer détecte les PII, anonymizer remplace les valeurs détectées.
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[language])
    anonymizer = AnonymizerEngine()

    return analyzer, anonymizer


def mask_text_pii(text: str, *, language: str, model_name: str, entities: list[str], labels_to_ignore: list[str]):
    """Masque les PII d'un texte et indique si une donnée sensible a été trouvée."""
    # Texte vide : rien à analyser, rien à masquer.
    if not text:
        return "", False

    analyzer, anonymizer = _get_pii_engines(
        language=language,
        model_name=model_name,
        labels_to_ignore=tuple(labels_to_ignore),
    )

    # Presidio cherche uniquement les types de PII configurés dans rails_settings.yml.
    analysis_results = analyzer.analyze(
        text=text,
        entities=entities,
        language=language,
    )

    # Aucun résultat signifie que Presidio n a trouvé aucune PII demandée.
    if not analysis_results:
        return text, False

    # Exemple : "Mon mail est x@y.com" devient "Mon mail est <EMAIL_ADDRESS>".
    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=analysis_results,
    )

    return anonymized_result.text, True
