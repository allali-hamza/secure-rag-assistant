# Security LLM Guardrails

A project built around a local LLM protected by NeMo Guardrails, with local RAG
over French labor law documents.

The pipeline works as follows:

1. `app.py` retrieves RAG context using `rag.py`.
2. The enriched prompt is sent to NeMo Guardrails.
3. Input rails filter the user's question.
4. The local model generates a response using Ollama.
5. Output rails check the response before it is displayed.

## Prerequisites

- Ubuntu/Debian recommended;
- Git;
- Ollama;
- an Internet connection on the first run to download dependencies
  and models.

<span style="color:red"><strong>Important: use Python 3.10 to 3.13. Python 3.14 is not compatible with NeMo Guardrails.</strong></span>

LLM models can take up several gigabytes of disk space.

## Manual installation (automatic installation available below)

### 1. Clone the project

```bash
git clone <GITLAB_REPOSITORY_URL>
cd stage2026-security-LLM
```

### 2. Install system dependencies

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-dev curl build-essential
```

### 3. Install Ollama

On Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

If the Ollama service is not running:

```bash
ollama serve
```

Check that the server responds:

```bash
curl http://localhost:11434/api/tags
```

For Windows or macOS, use the official installer:

https://ollama.com/download

### 4. Create the Python environment

First, check the Python version:

```bash
python3 --version
```

The version must be between Python 3.10 and Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If installation fails with a quota or disk space error in `/tmp`,
create a temporary directory in the project and tell pip to use it:

```bash
mkdir -p tmp
TMPDIR=$(pwd)/tmp python -m pip install -r requirements.txt --cache-dir=$(pwd)/tmp
```

This can happen if the `/tmp` partition is too small for the
large Python packages downloaded during installation.

### 5. Download the Ollama models

```bash
ollama pull qwen3:8b
ollama pull qwen2.5:3b
ollama pull mxbai-embed-large
ollama list
```

Model roles:

- `qwen3:8b`: the main generation model, declared in
  `config/config.yml`;
- `qwen2.5:3b`: the judge model for certain checks, including RAG support
  and ambiguous input scope cases;
- `mxbai-embed-large`: the embedding model used by RAG, input scope checks,
  and output scope checks.

### 6. Run the application

```bash
python app.py
```

To test another question, edit the `user_prompt` variable in
`app.py`. To change the model's overall behavior, edit
`system_prompt` in `app.py`.

## Pipeline architecture

### External RAG

RAG runs before NeMo Guardrails, directly in `app.py`.
The `retrieve_context(user_prompt)` function builds context from the
documents in `config/database`.

Retrieval is hybrid:

- vector search with ChromaDB and `mxbai-embed-large`;
- BM25 lexical search with `rank-bm25`.

RAG settings are in:

```text
config/settings/rag_settings.yml
```

This file lets you adjust basic settings without editing complex code:
chunk size, `top_k`, similarity threshold, maximum context size,
and storage paths.

### NeMo Guardrails

The main configuration is in:

```text
config/config.yml
```

Colang flows are in:

```text
config/rails/
```

The Python actions called by the flows are in:

```text
config/actions/
```

The active input rails are:

- `check prompt length`: blocks prompts that are too long;
- `mask pii input`: masks email addresses and phone numbers;
- `block jailbreak model attempt`: detects jailbreak attempts using
  `Verm1ion/injection-sentry-xlmr`;
- `restrict input scope`: checks whether the question concerns French labor
  law using Ollama embeddings, then uses an LLM as a judge only for borderline cases.

The active output rails are:

- `detect system prompt leakage`: detects leakage of the system canary;
- `check rag support`: uses `qwen2.5:3b` to check that the response is supported
  by the RAG context;
- `restrict scope`: checks that the response stays within scope;
- `output mask pii`: masks personally identifiable information (PII) in the final response.

Rail settings are centralized in:

```text
config/settings/rails_settings.yml
```

The final section explains how to change models and settings without
editing the Python code.

## Tests

The tests use `pytest`.

Useful options:

- `-v` displays the name of each test;
- `-s` displays pipeline logs and `print` output during execution.

Run all tests:

```bash
python -m pytest tests/ -v
```

Run all tests with visible logs:

```bash
python -m pytest tests/ -v -s
```

Run only the unit tests:

```bash
python -m pytest tests/unit -v
```

Run only the full pipeline tests:

```bash
python -m pytest tests/pipeline -v -s
```

Test a specific rail, such as the input jailbreak rail:

```bash
python -m pytest tests/unit/test_input_jailbreak.py -v -s
```

Other useful examples:

```bash
python -m pytest tests/unit/test_input_scope.py -v -s
python -m pytest tests/unit/test_input_pii.py -v -s
python -m pytest tests/unit/test_output_rag_support.py -v -s
python -m pytest tests/unit/test_output_system_leakage.py -v -s
```

The tests in `tests/pipeline` run the full pipeline with RAG,
NeMo Guardrails, and active rails. Their main purpose is to check that the
pipeline works end to end.

### Limitations

Some unit tests also document the current limitations of the
models used in the proof of concept (POC). They may therefore fail even if the main
pipeline works.

The `output_rag_support` edge case test for equivalence between `quinze jours`
(fifteen days) and `deux semaines` (two weeks) may fail because the `qwen2.5:3b`
judge is a small model. It may reject an otherwise valid numerical paraphrase.

Some jailbreak tests may also fail because the detection model
is not specifically trained for this domain or for every possible
phrasing in French.

## Project structure

```text
.
├── app.py                         # Main entry point for the POC
├── rag.py                         # External RAG: vector search + BM25
├── requirements.txt               # Project Python dependencies
├── setup.sh                       # Automatic installation
├── utils/                         # Utility functions: console logs and logger
├── config/
│   ├── config.yml                 # NeMo Guardrails configuration
│   ├── actions/                   # Python actions called by the rails
│   ├── rails/                     # Colang flows (.co)
│   ├── settings/                  # Basic configurable settings
│   │   ├── rag_settings.yml       # RAG settings
│   │   └── rails_settings.yml     # Rail settings
│   └── database/                  # Markdown documents used by RAG
└── tests/
    ├── unit/                      # Tests for individual rails
    └── pipeline/                  # Full pipeline tests
```

## Automatic installation

Automatic installation is available through `setup.sh`:

```bash
chmod +x setup.sh
./setup.sh
```

By default, `setup.sh` installs Python if needed, creates the `.venv`, installs
dependencies, installs Ollama if necessary, and downloads the Ollama models
used by this POC:

- `qwen3:8b`;
- `qwen2.5:3b`;
- `mxbai-embed-large`.

If you want to use your own models, follow the manual installation
instructions above: edit the configuration files, then
download the corresponding models with `ollama pull`.

## Changing models and settings

Models can be changed without editing the Python code.

Change the main generation model in `config/config.yml`:

```yaml
models:
  - type: main
    model: qwen3:8b
```

After making the change, download the corresponding Ollama model:

```bash
ollama pull <model-name>
```

Change the embedding models in:

```text
config/settings/rag_settings.yml
config/settings/rails_settings.yml
```

If you change the RAG embedding model, delete the old ChromaDB index
to force reindexing:

```bash
rm -rf config/database/.chroma_db
```

Change the judge models used by certain rails in
`config/settings/rails_settings.yml`:

```yaml
input_scope:
  judge_model: "qwen2.5:3b"

output_rag_support:
  judge_model: "qwen2.5:3b"
```

Change basic RAG settings in:

```text
config/settings/rag_settings.yml
```

This file lets you adjust the chunk size, number of retrieved chunks,
similarity threshold, and maximum context size.

Change basic rail settings in:

```text
config/settings/rails_settings.yml
```

This file lets you adjust thresholds, judge models, blocking messages,
and input/output check settings.
