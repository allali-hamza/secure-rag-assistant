#!/bin/bash




set -e


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"






VERSION_MIN=10
VERSION_MAX=13   # 3.14 est EXCLU (NemoGuardrails ne le supporte pas encore)


PYT_OK=false
PYTHON_BIN=""


for minor in 13 12 11 10; do
   if command -v "python3.${minor}" &>/dev/null; then
       PYTHON_BIN=$(command -v "python3.${minor}")
       PYT_OK=true
       break
   fi
done


if [ "$PYT_OK" = false ]; then
   echo "Aucune version Python 3.10–3.13 détectée. Installation de Python 3.13 via le PPA deadsnakes..."
   sudo apt install -y software-properties-common
   sudo add-apt-repository -y ppa:deadsnakes/ppa
   sudo apt update
   sudo apt install -y python3.13 python3.13-venv python3.13-dev
   if command -v python3.13 &>/dev/null; then
       PYTHON_BIN=$(command -v python3.13)
       PYT_OK=true
   else
       echo "[ERREUR] L'installation de Python 3.13 a échoué." >&2
       exit 1
   fi
fi


FULL_VERSION=$($PYTHON_BIN -V 2>&1 | awk '{print $2}')
echo "Python compatible détecté : ${FULL_VERSION} ($PYTHON_BIN)"







PYTHON_VER=$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')


echo "Mise à jour de apt"
sudo apt update


echo "Installation des dépendances système et des paquets de développement pour Python ${PYTHON_VER}"
sudo apt install -y \
   poppler-utils \
   build-essential \
   curl \
   software-properties-common \
   zstd || true


# Installation des paquets python versionnés (venv + dev) en ignorant les paquets absents
for PKG in "python${PYTHON_VER}-venv" "python${PYTHON_VER}-dev"; do
   if apt-cache show "$PKG" &>/dev/null 2>&1; then
       sudo apt install -y "$PKG"
   else
       echo "(Paquet $PKG non disponible dans apt, on continue)"
   fi
done


echo "Installation terminée avec succès."








echo "Création de l'environnement virtuel avec $($PYTHON_BIN -V)..."
$PYTHON_BIN -m venv --clear "$SCRIPT_DIR/.venv"


echo "Activation de l'environnement virtuel"
source "$SCRIPT_DIR/.venv/bin/activate"


echo "Mise à jour de pip"
"$SCRIPT_DIR/.venv/bin/pip" install --upgrade pip

PIP="$SCRIPT_DIR/.venv/bin/pip"
VENV_PY=$(head -1 "$PIP" | sed 's/^#!//')
echo "-> Interpréteur venv : $VENV_PY ($($VENV_PY -V))"


echo "Environnement virtuel prêt."








PIP="$SCRIPT_DIR/.venv/bin/pip"


echo "Installation des dépendances principales (avec versions explicites)..."
"$PIP" install \
   "nemoguardrails==0.22.0" \
   "presidio-analyzer==2.2.362" \
   "presidio-anonymizer==2.2.362" \
   "numpy==2.5.0" \
   "spacy==3.8.14" \
   "pytest==9.1.1" \
   "pytest-asyncio==1.4.0" \
   "transformers" \
   "torch"


echo "Installation des bibliothèques supplémentaires..."
"$PIP" install \
   chromadb \
   langchain-text-splitters \
   rank-bm25 \
   ollama


echo "Toutes les dépendances Python ont été installées avec succès."






# Définition de l'URL du modèle
URL_MODEL="https://github.com/explosion/spacy-models/releases/download/fr_core_news_md-3.8.0/fr_core_news_md-3.8.0-py3-none-any.whl"


echo "Installation du modèle spaCy (fr_core_news_md)"
"$PIP" install "$URL_MODEL"


echo "Modèle spaCy installé avec succès (utilisé directement depuis le venv)."









OLLAMA_LOCAL_BIN="$SCRIPT_DIR/models/ollama-bin/bin/ollama"


# Fonction qui vérifie que llama-server est bien présent dans l'installation Ollama
_ollama_is_complete() {
   local cmd="$1"
   # llama-server doit se trouver dans /usr/local/lib/ollama/ (installation système)
   if [ -f "/usr/local/lib/ollama/llama-server" ]; then
       return 0
   fi
   return 1
}


_ollama_uninstall() {
   echo "   -> Désinstallation de l'ancienne version d'Ollama..."
   sudo systemctl stop ollama 2>/dev/null || true
   sudo rm -f /usr/local/bin/ollama
   sudo rm -rf /usr/local/lib/ollama
   sudo rm -f /etc/systemd/system/ollama.service
   sudo systemctl daemon-reload 2>/dev/null || true
   echo "   -> Ancienne installation supprimée."
}


if command -v ollama &>/dev/null; then
   OLLAMA_CMD="ollama"
   echo "-> Ollama détecté au niveau système : $($OLLAMA_CMD --version)"
   # Vérification que llama-server est bien présent (installation complète)
   if ! _ollama_is_complete "$OLLAMA_CMD"; then
       echo "   [ATTENTION] llama-server manquant : installation Ollama incomplète."
       echo "   Réinstallation propre d'Ollama..."
       _ollama_uninstall
       curl -fsSL https://ollama.com/install.sh | sh
       if command -v ollama &>/dev/null && _ollama_is_complete "ollama"; then
           OLLAMA_CMD="ollama"
           echo "   ✓ Ollama réinstallé avec succès : $($OLLAMA_CMD --version)"
       else
           echo "[ERREUR] La réinstallation d'Ollama a échoué ou llama-server toujours absent." >&2
           echo "   Vérifiez votre connexion internet et relancez le script." >&2
           exit 1
       fi
   else
       echo "   ✓ llama-server présent — installation complète."
   fi
elif [ -f "$OLLAMA_LOCAL_BIN" ]; then
   OLLAMA_CMD="$OLLAMA_LOCAL_BIN"
   echo "-> Ollama local déjà présent : $($OLLAMA_CMD --version)"
else
   echo "-> Ollama non détecté. Installation via le script officiel"
   curl -fsSL https://ollama.com/install.sh | sh
   if command -v ollama &>/dev/null && _ollama_is_complete "ollama"; then
       OLLAMA_CMD="ollama"
       echo "   Ollama installé avec succès : $($OLLAMA_CMD --version)"
   else
       echo "[ERREUR] L'installation d'Ollama a échoué ou llama-server absent." >&2
       exit 1
   fi
fi


if ! $OLLAMA_CMD list &>/dev/null 2>&1; then
   echo "-> Démarrage du service Ollama en arrière-plan..."
   $OLLAMA_CMD serve &>/dev/null &
   echo "   Attente du démarrage..."
   for i in {1..30}; do
       if $OLLAMA_CMD list &>/dev/null 2>&1; then
           echo "   Ollama prêt."
           break
       fi
       sleep 1
   done
else
   echo "-> Service Ollama déjà actif."
fi









"$VENV_PY" -c "
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_ID = 'Verm1ion/injection-sentry-xlmr'
print(f'Téléchargement du modèle jailbreak : {MODEL_ID}')
AutoTokenizer.from_pretrained(MODEL_ID)
AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
print('Modèle jailbreak téléchargé avec succès.')
"




if $OLLAMA_CMD list 2>/dev/null | grep -q "mxbai-embed-large"; then
   echo "-> Modèle mxbai-embed-large déjà téléchargé, skip."
else
   echo "-> Téléchargement du modèle mxbai-embed-large..."
   $OLLAMA_CMD pull mxbai-embed-large
   echo "   mxbai-embed-large téléchargé."
fi


if $OLLAMA_CMD list 2>/dev/null | grep -q "qwen3:8b"; then
   echo "-> Modèle qwen3:8b déjà téléchargé, skip."
else
   echo "-> Téléchargement du modèle qwen3:8b..."
   $OLLAMA_CMD pull qwen3:8b
   echo "   qwen3:8b téléchargé."
fi


if $OLLAMA_CMD list 2>/dev/null | grep -q "qwen2.5:3b"; then
   echo "-> Modèle qwen2.5:3b déjà téléchargé, skip."
else
   echo "-> Téléchargement du modèle qwen2.5:3b..."
   $OLLAMA_CMD pull qwen2.5:3b
   echo "   qwen2.5:3b téléchargé."
fi






