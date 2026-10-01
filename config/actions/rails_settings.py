from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[1]
RAILS_SETTINGS_PATH = CONFIG_DIR / "settings" / "rails_settings.yml"


def load_rails_settings():
    """Charge le fichier YAML qui contient les paramètres des rails."""
    if not RAILS_SETTINGS_PATH.exists():
        return {}

    with RAILS_SETTINGS_PATH.open("r", encoding="utf-8") as settings_file:
        return yaml.safe_load(settings_file) or {}


def get_rail_settings(section: str, defaults: dict[str, Any] = None) -> dict[str, Any]:
    """Retourne les paramètres d'une rail avec des valeurs par défaut si besoin."""
    settings = load_rails_settings()
    section_settings = settings.get(section, {}) or {}

    if defaults is None:
        return section_settings

    # Les valeurs du YAML remplacent les valeurs par défaut du code.
    return {
        **defaults,
        **section_settings,
    }
