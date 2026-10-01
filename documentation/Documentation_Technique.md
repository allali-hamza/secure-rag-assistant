# Documentation technique: LLM Guardrails

## 1. Présentation technique du projet

Ce projet est un POC d’assistant LLM local sécurisé par NeMo Guardrails.
L’assistant est spécialisé, dans sa version actuelle, sur le droit du travail
français.

Le système combine :

- un LLM local exécuté avec Ollama ;
- un RAG local basé sur des documents Markdown ;
- des rails d’entrée pour filtrer la question utilisateur ;
- des rails de sortie pour vérifier la réponse avant affichage ;
- des tests unitaires et des tests de pipeline complet.

Les principaux composants utilisés sont :

- NeMo Guardrails : orchestration des rails input/output ;
- Ollama : exécution locale des modèles LLM et embeddings ;
- ChromaDB : stockage de l’index vectoriel du RAG ;
- BM25 : recherche lexicale complémentaire ;
- Presidio + spaCy : détection et masquage des PII ;
- Hugging Face Transformers : détection de jailbreak/prompt injection ;
- pytest : tests unitaires et tests du pipeline.

## 2. Vue globale du pipeline

Le point d’entrée principal est :

```bash
python app.py
```

Le pipeline complet suit cet ordre :

```text
python app.py
    |
    v
Chargement de config/config.yml
    |
    v
Initialisation de NeMo Guardrails
    |
    v
Récupération du contexte RAG avec rag.py
    |
    v
Construction du prompt enrichi
    |
    v
Rails d’entrée NeMo
    |
    v
Génération de la réponse par qwen3:8b
    |
    v
Rails de sortie NeMo
    |
    v
Affichage de la réponse finale
```

Le RAG est exécuté avant NeMo Guardrails. Ensuite, NeMo reçoit un prompt déjà
enrichi avec le contexte documentaire.

## 3. Fonctionnement de `app.py`

Le fichier `app.py` orchestre le lancement du POC.

La fonction `load_config` charge la configuration NeMo :

```python
RailsConfig.from_path("./config")
```

Cette configuration lit notamment :

- `config/config.yml` ;
- les fichiers `.co` dans `config/rails/` ;
- les actions Python dans `config/actions/`.

Ensuite, l’application crée l’objet principal NeMo :

```python
app = LLMRails(config)
```

Cet objet est responsable de l’exécution des rails et de l’appel au modèle de
génération.

### Canari système

`app.py` récupère le canari défini dans :

```text
config/settings/rails_settings.yml
```

Le canari est une chaîne secrète placée dans le prompt système. Il sert à
détecter une fuite du prompt système dans la réponse du modèle.

Si cette chaîne apparaît dans la réponse finale, la rail
`detect system prompt leakage` bloque la sortie.

### Prompt système

Le `system_prompt` impose au modèle plusieurs règles :

- répondre uniquement à partir du contexte fourni ;
- ne jamais répéter le canari ;
- ne pas utiliser de connaissances générales ;
- ne pas citer d’article absent du contexte ;
- répondre par un message spécifique si le contexte ne contient pas assez
  d’informations.

### Question utilisateur

La variable `user_prompt` contient la question testée pendant l’exécution du
POC.

Pour tester une autre question, il faut modifier cette variable dans `app.py`.

### Appel au RAG

Le contexte documentaire est récupéré avec :

```python
rag_context = retrieve_context(user_prompt)
```

Cette fonction est définie dans `rag.py`.

### Prompt enrichi

Le prompt final envoyé à NeMo est structuré avec trois balises :

```text
<LLM_SYSTEM>
Instructions système
</LLM_SYSTEM>

<RAG_CONTEXT>
Contexte documentaire récupéré par le RAG
</RAG_CONTEXT>

<USER_QUESTION>
Question utilisateur
</USER_QUESTION>
```

Cette structure est importante car plusieurs actions utilisent
`prompt_parser.py` pour extraire uniquement la vraie question utilisateur et
éviter d’analyser tout le contexte RAG.

### Génération

La génération est lancée avec :

```python
response = await app.generate_async(prompt=prompt)
```

À partir de ce moment, NeMo applique les rails d’entrée, appelle le modèle LLM,
puis applique les rails de sortie.

## 4. RAG externe

Le RAG est implémenté dans `rag.py`.

Il est volontairement exécuté en dehors des actions NeMo, avant l’appel à
`LLMRails.generate_async`.

Ce choix permet de garder un pipeline simple :

- `app.py` récupère le contexte ;
- NeMo contrôle ensuite le prompt enrichi ;
- les rails peuvent analyser uniquement la question utilisateur grâce aux
  balises `<USER_QUESTION>`.

### Base documentaire

Les documents utilisés par le RAG sont placés dans :

```text
config/database/
```

Dans le POC actuel, la base principale est :

```text
config/database/code_du_travail.md
```

Les fichiers Markdown sont chargés, puis transformés en documents LangChain.

### Découpage en chunks

Les documents sont découpés en chunks avec :

```python
RecursiveCharacterTextSplitter
```

Les paramètres sont définis dans :

```text
config/settings/rag_settings.yml
```

Les paramètres importants sont :

- `chunk_size` : taille maximale d’un chunk ;
- `chunk_overlap` : chevauchement entre deux chunks voisins.

Le chevauchement évite de couper une information importante entre deux chunks.

### Recherche vectorielle

La recherche vectorielle utilise :

- ChromaDB pour stocker l’index ;
- Ollama pour calculer les embeddings ;
- le modèle `mxbai-embed-large`.

L’index est stocké dans :

```text
config/database/.chroma_db
```

Si la collection ChromaDB est vide, `rag.py` indexe les documents Markdown.
Sinon, il réutilise l’index existant.

Si le modèle d’embedding ou les documents changent fortement, il peut être
nécessaire de supprimer l’index :

```bash
rm -rf config/database/.chroma_db
```

### Recherche BM25

BM25 complète la recherche vectorielle avec une recherche lexicale.

Cette recherche est utile quand la question contient des termes précis :

- noms d’articles ;
- notions juridiques exactes ;
- mots importants du Code du travail.

Le texte est tokenisé avant d’être envoyé à BM25. La tokenisation garde les
mots utiles et retire certains mots fréquents comme `le`, `la`, `de`, `dans`,
etc.

### Fusion des résultats

Le RAG exécute deux recherches :

- recherche vectorielle ;
- recherche BM25.

Les résultats sont ensuite fusionnés. Si un même chunk apparaît deux fois, il
est conservé une seule fois.

Le contexte final contient :

- l’identifiant du chunk ;
- la méthode utilisée (`vector` ou `bm25`) ;
- la source ;
- le score ;
- le contenu du chunk.

La taille maximale du contexte est limitée par :

```yaml
max_context_chars
```

Cela évite d’envoyer un prompt trop long au modèle.

## 5. Configuration NeMo Guardrails

La configuration principale est :

```text
config/config.yml
```

Elle déclare :

- le modèle principal ;
- les rails d’entrée ;
- les rails de sortie.

### Modèle principal

Le modèle principal est déclaré comme un modèle OpenAI-compatible :

```yaml
models:
  - type: main
    engine: openai
    model: qwen3:8b
    parameters:
      base_url: "http://localhost:11434/v1"
      api_key: "na"
```

Même si `engine` vaut `openai`, le modèle est local. Ollama expose une API
compatible OpenAI sur :

```text
http://localhost:11434/v1
```

### Rails d’entrée

Les rails d’entrée sont exécutées avant la génération :

```yaml
input:
  flows:
    - check prompt length
    - mask pii input
    - block jailbreak model attempt
    - restrict input scope
```

Si une rail bloque avec `stop`, les rails suivantes ne sont pas exécutées et le
modèle principal n’est pas appelé.

### Rails de sortie

Les rails de sortie sont exécutées après la génération :

```yaml
output:
  flows:
    - detect system prompt leakage
    - check rag support
    - restrict scope
    - output mask pii
```

Si une rail de sortie bloque, la réponse finale est remplacée par un message de
refus.

## 6. Rôle des fichiers `.co`

Les fichiers `.co` décrivent les flows Colang utilisés par NeMo Guardrails.

Ils ne contiennent pas la logique complexe. Ils appellent les actions Python et
décident si le pipeline doit continuer ou s’arrêter.

Exemple simplifié :

```text
define flow check prompt length
  $is_valid_length = execute check_input_length

  if not $is_valid_length
    bot refuse because too long
    stop
```

La logique de calcul est dans Python. Le fichier `.co` décrit seulement :

- quelle action appeler ;
- quelle variable récupérer ;
- quelle condition vérifier ;
- quel message retourner si la condition échoue.

## 7. Rails d’entrée

### 7.1 `check prompt length`

Fichiers :

- `config/rails/input_length_check.co`
- `config/actions/input_check_length.py`

Objectif :

- bloquer les questions utilisateur trop longues.

Fonctionnement :

1. NeMo appelle l’action `check_input_length`.
2. L’action récupère `user_message` ou `last_user_message`.
3. Elle utilise `extract_user_question` pour mesurer uniquement le contenu de
   `<USER_QUESTION>`.
4. Elle compare la longueur à `max_chars`.

Paramètre :

```yaml
input_check_length:
  max_chars: 500
```

Si la question dépasse la limite, le flow retourne :

```text
Désolé, votre message est trop long. Veuillez formuler une question plus courte.
```

### 7.2 `mask pii input`

Fichiers :

- `config/rails/input_pii_detect.co`
- `config/actions/input_PII_leakage.py`
- `config/actions/pii_utils.py`

Objectif :

- masquer les données personnelles envoyées par l’utilisateur.

Dans le POC actuel, les entités ciblées sont :

- adresses email ;
- numéros de téléphone.

Fonctionnement :

1. L’action extrait la vraie question utilisateur.
2. Elle appelle `mask_text_pii`.
3. Presidio détecte les entités configurées.
4. Les valeurs détectées sont remplacées par des balises génériques.
5. Le prompt enrichi est reconstruit avec la question masquée.

Exemple :

```text
Mon numéro est 06 12 34 56 78.
```

devient :

```text
Mon numéro est <PHONE_NUMBER>.
```

Le contexte RAG et le prompt système ne sont pas modifiés.

### 7.3 `block jailbreak model attempt`

Fichiers :

- `config/rails/jailbreak_model.co`
- `config/actions/jailbreak_model.py`

Objectif :

- détecter les tentatives de jailbreak ou prompt injection.

Le modèle utilisé est :

```text
Verm1ion/injection-sentry-xlmr
```

Fonctionnement :

1. L’action extrait la vraie question utilisateur.
2. Le tokenizer Hugging Face transforme le texte en tenseurs.
3. Le modèle calcule deux scores :
   - `SAFE`
   - `INJECTION`
4. Si le score `INJECTION` dépasse `injection_threshold`, la requête est
   bloquée.

Paramètres principaux :

```yaml
input_jailbreak_model:
  model_name: "Verm1ion/injection-sentry-xlmr"
  device: "cpu"
  max_length: 512
  injection_threshold: 0.6
```

En cas d’erreur du modèle, l’action bloque par précaution.

Message de refus :

```text
Violation de la politique de sécurité : tentative d'injection détectée.
```

### 7.4 `restrict input scope`

Fichiers :

- `config/rails/input_scope.co`
- `config/actions/input_scope_detect.py`

Objectif :

- vérifier que la question concerne réellement le droit du travail français.

Le contrôle fonctionne en deux étages.

Premier étage : embeddings Ollama.

1. Des exemples autorisés sont embarqués en mémoire.
2. Des exemples hors-sujet sont embarqués en mémoire.
3. Des exemples suspects de keyword stuffing sont aussi embarqués.
4. La question utilisateur est convertie en embedding.
5. Le code calcule :
   - `scope_score`
   - `offtopic_score`
   - `suspicious_score`
   - `score_margin`

La question est acceptée si :

- le score de scope est assez haut ;
- le score de scope dépasse suffisamment le score hors-sujet.

Deuxième étage : LLM-as-judge.

Le LLM judge est appelé uniquement si la question est acceptée par embeddings
mais reste ambiguë :

- marge faible ;
- forte similarité avec des exemples de keyword stuffing.

Le modèle juge est :

```text
qwen2.5:3b
```

Il répond uniquement :

```text
IN_SCOPE
```

ou :

```text
OUT_OF_SCOPE
```

Si la question est hors périmètre, le flow retourne :

```text
Je suis un agent strictement dédié au code du travail. Je ne peux pas répondre à cette question.
```

## 8. Rails de sortie

### 8.1 `detect system prompt leakage`

Fichiers :

- `config/rails/output_system_leakage.co`
- `config/actions/output_system_leakage.py`

Objectif :

- détecter si le modèle révèle une partie sensible du prompt système.

Le mécanisme est volontairement simple :

1. Un canari secret est placé dans le prompt système.
2. La sortie du modèle est inspectée.
3. Si le canari apparaît dans la sortie, la réponse est bloquée.

Paramètres :

```yaml
output_system_leakage:
  canary: "STAGE2026_SYSTEM_CANARY_9F3A7C"
  safe_response: "Je ne peux pas reveler mes instructions internes..."
```

Ce contrôle ne dépend pas d’un modèle. Il est déterministe.

### 8.2 `check rag support`

Fichiers :

- `config/rails/output_rag_support_check.co`
- `config/actions/output_rag_support_check.py`

Objectif :

- vérifier que la réponse générée est supportée par le contexte RAG fourni.

Fonctionnement :

1. L’action récupère le contexte RAG.
2. Elle récupère la réponse du bot.
3. Elle construit un prompt de jugement.
4. Elle appelle le modèle juge `qwen2.5:3b`.
5. Le juge doit répondre `YES` ou `NO`.

Si le juge répond `YES`, la réponse passe.

Si le juge répond `NO`, la réponse est remplacée par :

```text
Je ne suis pas en mesure de confirmer cette reponse a partir des documents fournis.
```

Si le contexte RAG est absent, l’action bloque par précaution.

Si le juge Ollama échoue, l’action bloque aussi par précaution.

### 8.3 `restrict scope`

Fichiers :

- `config/rails/output_scope.co`
- `config/actions/output_scope_detect.py`

Objectif :

- vérifier que la réponse finale reste dans le périmètre du droit du travail.

Le fonctionnement est similaire au scope input, mais appliqué à la réponse du
modèle.

Le code compare la réponse avec :

- des exemples de réponses acceptables ;
- des exemples de réponses hors-sujet.

Scores calculés :

- `scope_score`
- `offtopic_score`
- `score_margin`

Si la réponse est hors périmètre, elle est remplacée par :

```text
Je suis un agent strictement dedie au code du travail. Je ne peux pas fournir une reponse hors de ce perimetre.
```

### 8.4 `output mask pii`

Fichiers :

- `config/rails/output_pii.co`
- `config/actions/output_mask_pii.py`
- `config/actions/pii_utils.py`

Objectif :

- masquer les données personnelles qui apparaîtraient dans la réponse finale.

Le fonctionnement est le même que pour l’input PII, mais appliqué à
`bot_message`.

Si aucune PII n’est détectée, la réponse est renvoyée sans modification.

## 9. Actions utilitaires

### `prompt_parser.py`

Ce fichier contient les fonctions qui manipulent le prompt enrichi :

- `has_user_question_section` ;
- `extract_user_question` ;
- `replace_user_question`.

Ces fonctions évitent que les rails d’entrée analysent le contexte RAG ou les
instructions système à la place de la vraie question utilisateur.

### `rails_settings.py`

Ce fichier charge :

```text
config/settings/rails_settings.yml
```

Il fournit une fonction :

```python
get_rail_settings(section, defaults)
```

Les valeurs du YAML remplacent les valeurs par défaut du code.

### `pii_utils.py`

Ce fichier factorise la logique Presidio utilisée par :

- `input_PII_leakage.py` ;
- `output_mask_pii.py`.

Il évite de dupliquer :

- la configuration spaCy ;
- la création du moteur Presidio ;
- l’appel à l’anonymizer.


## 10. Paramètres configurables

### RAG

Les paramètres RAG sont dans :

```text
config/settings/rag_settings.yml
```

Paramètres importants :

- `db_dir` : dossier contenant les documents Markdown ;
- `chroma_dir` : dossier de l’index ChromaDB ;
- `collection_name` : nom de la collection ;
- `embedding_model` : modèle d’embedding Ollama ;
- `chunk_size` : taille des chunks ;
- `chunk_overlap` : chevauchement entre chunks ;
- `vector_top_k` : nombre de résultats vectoriels ;
- `bm25_top_k` : nombre de résultats BM25 ;
- `min_similarity` : seuil minimal de similarité vectorielle ;
- `max_context_chars` : taille maximale du contexte injecté.

### Rails

Les paramètres des rails sont dans :

```text
config/settings/rails_settings.yml
```

Ce fichier permet de régler :

- les seuils ;
- les modèles ;
- les messages de blocage ;
- les entités PII ;
- les timeouts ;
- les paramètres du LLM judge.

### Changement de modèles

Le modèle principal se change dans :

```text
config/config.yml
```

Les modèles d’embedding et les modèles juges se changent dans :

```text
config/settings/rag_settings.yml
config/settings/rails_settings.yml
```

Après avoir changé un modèle Ollama, il faut le télécharger :

```bash
ollama pull <nom-du-modele>
```

Après avoir changé le modèle d’embedding du RAG, supprimer l’ancien index :

```bash
rm -rf config/database/.chroma_db
```

Sinon, l’index existant peut avoir été construit avec l’ancien modèle.

## 11. Tests

Les tests sont dans :

```text
tests/
```

Il existe deux types de tests.

### Tests unitaires

Les tests unitaires vérifient les rails ou fonctions séparément.

Exemples :

```bash
python -m pytest tests/unit/test_input_check_length.py -v
python -m pytest tests/unit/test_input_pii.py -v
python -m pytest tests/unit/test_input_scope.py -v -s
python -m pytest tests/unit/test_input_jailbreak.py -v -s
python -m pytest tests/unit/test_output_rag_support.py -v -s
python -m pytest tests/unit/test_output_system_leakage.py -v
```

Certains tests appellent réellement des modèles locaux. Ils nécessitent donc
Ollama lancé et les modèles téléchargés.

### Tests pipeline

Les tests pipeline sont dans :

```text
tests/pipeline/
```

Ils exécutent le pipeline complet :

- RAG ;
- NeMo Guardrails ;
- rails d’entrée ;
- génération LLM ;
- rails de sortie.

Commande :

```bash
python -m pytest tests/pipeline -v -s
```

Les cas testés sont définis dans :

```text
tests/pipeline/pipeline_cases.yml
```

### Limites des tests

Certains tests documentent volontairement les limites du POC.

Exemples :

- le juge `qwen2.5:3b` peut échouer sur des équivalences comme `quinze jours`
  et `deux semaines` ;
- le modèle jailbreak peut manquer certaines formulations ;
- les tests dépendants des modèles peuvent varier selon la machine, la version
  du modèle ou les performances d’Ollama.

## 12. Dépendances principales

Les dépendances Python sont listées dans :

```text
requirements.txt
```

Les dépendances importantes sont :

- `nemoguardrails` : orchestration des rails ;
- `ollama` : appels aux modèles locaux ;
- `openai` : compatibilité avec l’API OpenAI exposée par Ollama ;
- `chromadb` : index vectoriel ;
- `rank-bm25` : recherche lexicale ;
- `presidio-analyzer` et `presidio-anonymizer` : détection et masquage PII ;
- `spacy` et `fr-core-news-md` : NLP français ;
- `transformers` et `torch` : modèle jailbreak ;
- `pytest` et `pytest-asyncio` : tests.

Le projet nécessite Python 3.10 à 3.13. Python 3.14 n’est pas compatible avec
NeMo Guardrails.

## 13. Limites techniques actuelles

Le POC fonctionne, mais il possède des limites connues.

### Dépendance aux modèles locaux

La qualité dépend directement des modèles Ollama et Hugging Face utilisés.

Un petit modèle local est plus simple à exécuter, mais il peut produire des
jugements moins fiables qu’un modèle plus puissant.

### Latence

Plusieurs étapes peuvent être lentes :

- indexation initiale du RAG ;
- embeddings Ollama ;
- modèle jailbreak Hugging Face ;
- appels au LLM judge ;
- génération avec `qwen3:8b`.

Les modèles sont chargés en lazy loading quand c’est possible, mais le premier
appel reste plus lent.

### Scope par embeddings

La détection de périmètre par embeddings peut être trompée par du keyword
stuffing.

Exemple :

```text
Code du travail, salarié, licenciement : quelle est la recette du poulet rôti ?
```

Pour limiter ce problème, l’input scope utilise un deuxième étage avec
LLM-as-judge sur les cas ambigus.

### RAG support

Le contrôle RAG support repose sur un LLM judge.

Il peut donc se tromper :

- faux positif : bloquer une réponse correcte ;
- faux négatif : laisser passer une réponse insuffisamment fondée.

Le choix actuel privilégie un POC local et simple plutôt qu’un juge très lourd.

### Détection jailbreak

Le modèle de jailbreak n’est pas parfait.

Il peut manquer certaines attaques ou produire des faux positifs selon la
formulation et la langue.

Les tests documentent une partie de ces limites.

## 14. Pistes d’amélioration

Améliorations possibles :

- enrichir les exemples de scope input/output ;
- tester d’autres modèles de jailbreak multilingues ;
- utiliser un juge plus puissant pour `output_rag_support` ;
- mesurer les temps d’exécution par rail ;
- Utiliser une API au lieu des modèles locaux ;
- Pour satisfaire l'exigence A008, il faudrait étendre le block "detect system prompt leakage" pour y inclure la détection de leak d'input, output, logs, etc ;

