from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import re
import yaml
from rank_bm25 import BM25Okapi
from utils.console import log_section_start, log_step

RAG_SETTINGS_PATH = Path("config/settings/rag_settings.yml")


def _load_rag_settings(path = RAG_SETTINGS_PATH):
    """Charge les paramètres du RAG depuis le fichier YAML."""
    # Tous les paramètres RAG sont centralisés ici pour éviter de modifier le code.
    if not path.exists():
        raise FileNotFoundError(f"Fichier de configuration RAG introuvable : {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


RAG_SETTINGS = _load_rag_settings()

DB_DIR = Path(RAG_SETTINGS["database"]["db_dir"])
CHROMA_DIR = Path(RAG_SETTINGS["database"]["chroma_dir"])
COLLECTION_NAME = RAG_SETTINGS["database"]["collection_name"]

OLLAMA_URL = RAG_SETTINGS["ollama"]["url"]
EMBED_MODEL = RAG_SETTINGS["ollama"]["embedding_model"]

TOP_K = int(RAG_SETTINGS["retrieval"]["vector_top_k"])
BM25_TOP_K = int(RAG_SETTINGS["retrieval"]["bm25_top_k"])
MIN_SIMILARITY = float(RAG_SETTINGS["retrieval"]["min_similarity"])
MAX_CONTEXT_CHARS = int(RAG_SETTINGS["retrieval"]["max_context_chars"])

CHUNK_SIZE = int(RAG_SETTINGS["chunking"]["chunk_size"])
CHUNK_OVERLAP = int(RAG_SETTINGS["chunking"]["chunk_overlap"])

NO_CONTEXT_MESSAGE = RAG_SETTINGS["messages"]["no_context"]

_COLLECTION = None
_BM25_INDEX = None
_BM25_CHUNKS = None


def _tokenize(text):
    """Transforme un texte en liste de mots utilisables par BM25."""
    # Tokenisation simple pour BM25 : on garde les mots utiles et on retire les mots très fréquents.
    stopwords = {
        "le", "la", "les", "un", "une", "des", "du", "de", "d", "l",
        "à", "a", "au", "aux", "et", "ou", "pour", "par", "sur", "dans",
        "est", "sont", "être", "etre", "avec", "comme", "selon",
        "quelle", "quelles", "quel", "quels", "comment",
        "moi", "donne", "source", "réponse", "reponse",
    }

    # Cette regex extrait uniquement les suites de lettres/chiffres.
    # Elle garde aussi les lettres accentuées françaises, puis met tout en minuscules.
    tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text.lower())
    return [
        token
        for token in tokens
        if len(token) >= 3 and token not in stopwords
    ]


def _embedding_function():
    """Crée la fonction d'embedding utilisée par ChromaDB avec Ollama."""
    # ChromaDB appelle Ollama pour transformer les textes en vecteurs.
    return embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_URL,
        model_name=EMBED_MODEL,
        timeout=120,
    )


def _load_markdown_files() -> list[Path]:
    """Retourne la liste des fichiers Markdown à indexer."""
    # Les documents sources du RAG sont les fichiers Markdown présents dans la base documentaire.
    if not DB_DIR.exists():
        return []
    return sorted(DB_DIR.glob("*.md"))



def load_documents() -> list[Document]:
    """Lit les fichiers Markdown et les convertit en documents LangChain."""
    # Chaque fichier Markdown devient un document LangChain avec sa source en metadata.
    documents = []
    for path in _load_markdown_files():
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        if content:
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": path.name,
                        "path": str(path)
                    }
                )
            )
    return documents



def split_text(documents: list[Document]):
    """Découpe les documents en petits morceaux recherchables."""
    # Les documents sont découpés en chunks pour retrouver seulement les passages utiles.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents=documents)
    return chunks




def build_rag_context(top_chunks: list[Document]) -> str:
    """Construit un bloc de contexte lisible à partir des chunks sélectionnés."""
    # Format lisible pour injecter les chunks dans le prompt du LLM.
    parts = []
    for i, chunk in enumerate(top_chunks):
        source = chunk.metadata["source"]
        content = chunk.page_content.strip()

        if content:
            parts.append(f"[chunk {i} | source: {source}] \n {content}")
    return "\n".join(parts)




def _collection_builder():
    """Crée ou recharge la collection ChromaDB contenant l'index vectoriel."""
    # ChromaDB garde l'index vectoriel sur disque pour éviter de tout recalculer à chaque lancement.
    client = chromadb.PersistentClient(str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function(),
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        }
    )

    # Si la collection est vide, on découpe puis indexe les documents Markdown.
    if collection.count() == 0:
        documents = load_documents()
        if documents:
            chunks = split_text(documents=documents)
            ids = [f"{chunk.metadata['source']}-{i}" for i, chunk in enumerate(chunks)]
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]

            batch_size = 50

            # L'ajout par batch évite de surcharger ChromaDB et Ollama pendant l'indexation.
            for i in range(0, len(ids), batch_size):
                frontiere = i + batch_size if (i + batch_size) <= len(ids) else len(ids)
                tmp_ids = ids[i: frontiere]
                tmps_texts = texts[i: frontiere]
                tmps_metadatas = metadatas[i: frontiere]
                collection.add(
                    ids=tmp_ids,
                    documents=tmps_texts,
                    metadatas=tmps_metadatas
                )
                log_step("RETRIEVAL_INDEXING", f"{frontiere} chunks indexes sur {len(ids)}")
            log_step("RETRIEVAL_INDEXING", f"{len(chunks)} chunks indexes dans ChromaDB.")
        else:
            log_step("RETRIEVAL_INDEXING", "Aucun document Markdown trouve.")
    else:
        log_section_start("RETRIEVAL_INDEXING", "Initialisation des index")
        log_step("RETRIEVAL_INDEXING", f"ChromaDB deja indexee avec {collection.count()} chunks")

    return collection


def _get_collection():
    """Retourne la collection ChromaDB en la construisant seulement si nécessaire."""
    # Lazy loading : l'index vectoriel est chargé seulement au premier besoin.
    global _COLLECTION
    if _COLLECTION is None:
        _COLLECTION = _collection_builder()
    return _COLLECTION




def _get_bm25_index():
    """Construit ou retourne l'index BM25 utilisé pour la recherche lexicale."""
    global _BM25_INDEX, _BM25_CHUNKS

    # Lazy loading : l'index BM25 est construit une seule fois par exécution.
    if _BM25_INDEX is not None and _BM25_CHUNKS is not None:
        return _BM25_INDEX, _BM25_CHUNKS

    documents = load_documents()
    chunks = split_text(documents)
    tokenized_chunks = [_tokenize(chunk.page_content) for chunk in chunks]

    _BM25_INDEX = BM25Okapi(tokenized_chunks)
    _BM25_CHUNKS = chunks

    log_step("RETRIEVAL_INDEXING", f"BM25 construit avec {len(chunks)} chunks")
    return _BM25_INDEX, _BM25_CHUNKS



def _bm25_search(query, top_k=BM25_TOP_K):
    """Recherche les chunks qui partagent les mots importants de la question."""
    # BM25 complète la recherche vectorielle avec une recherche par mots importants.
    if not query or not query.strip():
        return []

    bm25, chunks = _get_bm25_index()
    query_tokens = _tokenize(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked = sorted(
        enumerate(scores),
        key=lambda item: item[1],
        reverse=True,
    )

    results = []
    for idx, score in ranked[:top_k]:
        if score <= 0:
            continue

        chunk = chunks[idx]
        results.append({
            "source": chunk.metadata.get("source"),
            "start_index": chunk.metadata.get("start_index"),
            "score": float(score),
            "content": chunk.page_content,
            "method": "bm25",
        })

    return results



def _vector_search(query, top_k = TOP_K, min_similarity = MIN_SIMILARITY):
    """Recherche les chunks les plus proches du sens de la question."""
    # Recherche sémantique : on récupère les chunks proches du sens de la question.
    if not query or not query.strip():
        return []

    collection = _get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadats = results["metadatas"][0]
    distances = results["distances"][0]

    output = []
    for doc, meta, distance in zip(documents, metadats, distances):
        similarity = 1 - distance
        if similarity < min_similarity:
            continue

        output.append({
            "source": meta.get("source"),
            "start_index": meta.get("start_index"),
            "score": float(similarity),
            "content": doc,
            "method": "vector",
        })

    return output



def _merge_results(vector_results, bm25_results):
    """Fusionne les résultats BM25 et vectoriels en supprimant les doublons."""
    # On fusionne les résultats des deux méthodes sans répéter le même chunk.
    merged = []
    seen = set()

    for result in bm25_results + vector_results:
        key = (
            result.get("source"),
            result.get("start_index"),
        )

        if key in seen:
            continue

        seen.add(key)
        merged.append(result)

    return merged



def retrieve_context(query, top_k=TOP_K, min_similarity=MIN_SIMILARITY):
    """Récupère le contexte final qui sera ajouté au prompt envoyé au LLM."""
    # Point d'entrée utilisé par app.py : retourne le contexte RAG final à injecter.
    if not query:
        return NO_CONTEXT_MESSAGE

    # Les deux recherches sont complémentaires : vecteurs pour le sens, BM25 pour les mots précis.
    vector_results = _vector_search(
        query=query,
        top_k=top_k,
        min_similarity=min_similarity,
    )

    bm25_results = _bm25_search(query=query, top_k=BM25_TOP_K)
    results = _merge_results(vector_results, bm25_results)

    if not results:
      return NO_CONTEXT_MESSAGE

    # Le contexte final garde les scores et les sources pour faciliter le debug du POC.
    parts = []
    for i, result in enumerate(results):
        parts.append(
            f"[chunk {i} | method: {result['method']} | "
            f"source: {result['source']} | score: {result['score']:.4f}]\n"
            f"{result['content']}"
        )

    context = "\n\n".join(parts)
    return context[:MAX_CONTEXT_CHARS]

