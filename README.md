# Security LLM Guardrails

Projet autour d'un LLM local protege par NeMo Guardrails, avec un RAG
local sur des documents de droit du travail francais.

Le pipeline est :

1. `app.py` recupere un contexte RAG avec `rag.py`.
2. Le prompt enrichi est envoye a NeMo Guardrails.
3. Les rails d'entree filtrent la question utilisateur.
4. Le modele local genere une reponse avec Ollama.
5. Les rails de sortie verifient la reponse avant affichage.

## Prerequis

- Ubuntu/Debian recommande ;
- Git ;
- Ollama ;
- une connexion Internet au premier lancement pour telecharger les dependances
  et les modeles.

<span style="color:red"><strong>Important : utiliser Python 3.10 a 3.13. Python 3.14 n'est pas compatible avec NeMo Guardrails.</strong></span>

Les modeles LLM peuvent occuper plusieurs gigaoctets.

## Installation manuelle (installation automatique disponible plus bas)

### 1. Cloner le projet

```bash
git clone <URL_DU_DEPOT_GITLAB>
cd stage2026-security-LLM
```

### 2. Installer les dependances systeme

Sous Ubuntu/Debian :

```bash
sudo apt update
sudo apt install python3 python3-venv python3-dev curl build-essential
```

### 3. Installer Ollama

Sous Linux :

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

Si le service Ollama n'est pas actif :

```bash
ollama serve
```

Verifier que le serveur repond :

```bash
curl http://localhost:11434/api/tags
```

Pour Windows ou macOS, utiliser l'installateur officiel :

https://ollama.com/download

### 4. Creer l'environnement Python

Verifier d'abord la version de Python :

```bash
python3 --version
```

La version doit etre comprise entre Python 3.10 et Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Si l'installation echoue avec une erreur de quota ou d'espace disque dans `/tmp`,
creer un dossier temporaire dans le projet et demander a pip de l'utiliser :

```bash
mkdir -p tmp
TMPDIR=$(pwd)/tmp python -m pip install -r requirements.txt --cache-dir=$(pwd)/tmp
```

Cette situation peut arriver si la partition `/tmp` est trop petite pour les
gros paquets Python telecharges pendant l'installation.

### 5. Telecharger les modeles Ollama

```bash
ollama pull qwen3:8b
ollama pull qwen2.5:3b
ollama pull mxbai-embed-large
ollama list
```

Role des modeles :

- `qwen3:8b` : modele principal de generation, declare dans
  `config/config.yml` ;
- `qwen2.5:3b` : modele juge pour certains controles, notamment le support RAG
  et les cas ambigus du scope input ;
- `mxbai-embed-large` : modele d'embedding utilise par le RAG, le scope input
  et le scope output.

### 6. Lancer l'application

```bash
python app.py
```

Pour tester une autre question, modifier la variable `user_prompt` dans
`app.py`. Pour modifier le comportement global du modele, modifier
`system_prompt` dans `app.py`.

## Architecture du pipeline

### RAG externe

Le RAG est execute avant NeMo Guardrails, directement dans `app.py`.
La fonction `retrieve_context(user_prompt)` construit un contexte a partir des
documents presents dans `config/database`.

La recherche est hybride :

- recherche vectorielle avec ChromaDB et `mxbai-embed-large` ;
- recherche lexicale BM25 avec `rank-bm25`.

Les parametres du RAG sont dans :

```text
config/settings/rag_settings.yml
```

Ce fichier permet de modifier les parametres simples sans toucher au code
complexe : taille des chunks, `top_k`, seuil de similarite, taille maximale du
contexte, chemins de stockage.

### NeMo Guardrails

La configuration principale est dans :

```text
config/config.yml
```

Les flows Colang sont dans :

```text
config/rails/
```

Les actions Python appelees par les flows sont dans :

```text
config/actions/
```

Les rails d'entree actifs sont :

- `check prompt length` : bloque les prompts trop longs ;
- `mask pii input` : masque les emails et numeros de telephone ;
- `block jailbreak model attempt` : detecte les tentatives de jailbreak avec
  `Verm1ion/injection-sentry-xlmr` ;
- `restrict input scope` : verifie que la question concerne le droit du travail
  francais avec embeddings Ollama, puis LLM-as-judge uniquement en zone grise.

Les rails de sortie actifs sont :

- `detect system prompt leakage` : detecte la fuite du canari systeme ;
- `check rag support` : verifie avec `qwen2.5:3b` que la reponse est supportee
  par le contexte RAG ;
- `restrict scope` : verifie que la reponse reste dans le perimetre ;
- `output mask pii` : masque les PII dans la reponse finale.

Les parametres des rails sont centralises dans :

```text
config/settings/rails_settings.yml
```

La section de fin explique comment changer les modeles et les parametres sans
modifier le code Python.

## Tests

Les tests utilisent `pytest`. 

Options utiles :

- `-v` affiche le nom de chaque test ;
- `-s` affiche les logs et les `print` du pipeline pendant l'execution.

Lancer tous les tests :

```bash
python -m pytest tests/ -v
```

Lancer tous les tests avec les logs visibles :

```bash
python -m pytest tests/ -v -s
```

Lancer uniquement les tests unitaires :

```bash
python -m pytest tests/unit -v
```

Lancer uniquement les tests du pipeline complet :

```bash
python -m pytest tests/pipeline -v -s
```

Lancer uniquement une rail precise, par exemple la rail de jailbreak input :

```bash
python -m pytest tests/unit/test_input_jailbreak.py -v -s
```

Autres exemples utiles :

```bash
python -m pytest tests/unit/test_input_scope.py -v -s
python -m pytest tests/unit/test_input_pii.py -v -s
python -m pytest tests/unit/test_output_rag_support.py -v -s
python -m pytest tests/unit/test_output_system_leakage.py -v -s
```

Les tests du dossier `tests/pipeline` executent le pipeline complet avec RAG,
NeMo Guardrails et rails actifs. Ils servent surtout a verifier que le pipeline
fonctionne de bout en bout.

### Limites

Certains tests unitaires servent aussi a documenter les limites actuelles des
modeles utilises dans le POC. Ils peuvent donc echouer meme si le pipeline
principal fonctionne.

Le test limite de `output_rag_support` sur l'equivalence entre `quinze jours`
et `deux semaines` peut echouer parce que le juge `qwen2.5:3b` est un petit
modele. Il peut refuser une reformulation numerique pourtant correcte.

Certains tests de jailbreak peuvent aussi echouer parce que le modele de
detection n'est pas entraine specifiquement sur ce domaine et sur toutes les
formulations possibles en francais.

## Structure du projet

```text
.
├── app.py                         # Point d'entree principal du POC
├── rag.py                         # RAG externe : recherche vectorielle + BM25
├── requirements.txt               # Dependances Python du projet
├── setup.sh                       # Installation automatique
├── utils/                         # Fonctions utilitaires : logs console et logger
├── config/
│   ├── config.yml                 # Configuration NeMo Guardrails
│   ├── actions/                   # Actions Python appelees par les rails
│   ├── rails/                     # Flows Colang (.co)
│   ├── settings/                  # Parametres simples modifiables
│   │   ├── rag_settings.yml       # Parametres du RAG
│   │   └── rails_settings.yml     # Parametres des rails
│   └── database/                  # Documents Markdown utilises par le RAG
└── tests/
    ├── unit/                      # Tests rail par rail
    └── pipeline/                  # Tests du pipeline complet
```

## Installation automatique

Une installation automatique est disponible avec `setup.sh` :

```bash
chmod +x setup.sh
./setup.sh
```

Par defaut, `setup.sh` installe Python si besoin, cree le `.venv`, installe les
dependances, installe Ollama si necessaire et telecharge les modeles Ollama
prevus pour ce POC :

- `qwen3:8b` ;
- `qwen2.5:3b` ;
- `mxbai-embed-large`.

Si vous voulez utiliser vos propres modeles, utilisez plutot l'installation
manuelle decrite plus haut : modifiez les fichiers de configuration, puis
telechargez les modeles correspondants avec `ollama pull`.

## Changer les modeles et les parametres

Les modeles peuvent etre modifies sans changer le code Python.

Le modele principal de generation se change dans `config/config.yml` :

```yaml
models:
  - type: main
    model: qwen3:8b
```

Apres modification, telecharger le modele Ollama correspondant :

```bash
ollama pull <nom-du-modele>
```

Les modeles d'embedding se changent dans :

```text
config/settings/rag_settings.yml
config/settings/rails_settings.yml
```

Si le modele d'embedding du RAG est change, supprimer l'ancien index ChromaDB
pour forcer une reindexation :

```bash
rm -rf config/database/.chroma_db
```

Les modeles juges utilises par certaines rails se changent dans
`config/settings/rails_settings.yml` :

```yaml
input_scope:
  judge_model: "qwen2.5:3b"

output_rag_support:
  judge_model: "qwen2.5:3b"
```

Les parametres simples du RAG se changent dans :

```text
config/settings/rag_settings.yml
```

Ce fichier permet notamment de modifier la taille des chunks, le nombre de
chunks recuperes, le seuil de similarite et la taille maximale du contexte.

Les parametres simples des rails se changent dans :

```text
config/settings/rails_settings.yml
```

Ce fichier permet notamment de modifier les seuils, les modeles juges, les
messages de blocage et les parametres des controles input/output.

