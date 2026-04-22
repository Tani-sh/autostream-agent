"""
RAG Embedder: Chunks the AutoStream knowledge base and builds a persistent FAISS index.

Run directly to (re)build the index:
    python -m rag.embedder
"""

import os
import pickle

# ---------------------------------------------------------------------------
# Redirect HuggingFace cache into the project directory.
# This avoids permission errors on systems where ~/.cache is restricted.
# ---------------------------------------------------------------------------
_HERE_EMBED = os.path.dirname(os.path.abspath(__file__))
_ROOT_CACHE = os.path.dirname(_HERE_EMBED)
_LOCAL_HF_CACHE = os.path.join(_ROOT_CACHE, ".hf_cache")
os.makedirs(_LOCAL_HF_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME", _LOCAL_HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _LOCAL_HF_CACHE)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _LOCAL_HF_CACHE)

from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Paths ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

KB_PATH = os.path.join(_ROOT, "knowledge_base", "autostream_kb.md")
INDEX_DIR = os.path.join(_ROOT, "faiss_index")
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _load_and_chunk(path: str) -> List[str]:
    """
    Loads the markdown knowledge base and splits it into semantic chunks.

    Strategy:
    1. Split on level-2 headings (##) to preserve topic coherence.
    2. For large sections, further split into overlapping word windows.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by ## headings
    raw_sections: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        if line.startswith("## ") and current:
            raw_sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        raw_sections.append("\n".join(current).strip())

    # Sub-chunk large sections to stay within ~200 words
    CHUNK_SIZE = 200
    OVERLAP = 40
    chunks: list[str] = []

    for section in raw_sections:
        words = section.split()
        if len(words) <= CHUNK_SIZE:
            if len(section.strip()) > 30:
                chunks.append(section.strip())
        else:
            for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
                chunk = " ".join(words[i : i + CHUNK_SIZE])
                if len(chunk.strip()) > 30:
                    chunks.append(chunk)

    return chunks


def build_index(verbose: bool = True) -> Tuple:
    """
    Embeds knowledge base chunks and persists a FAISS IndexFlatIP (cosine similarity).

    Returns:
        (index, chunks)
    """
    if verbose:
        print("🔨  Building FAISS index from knowledge base …")

    model = SentenceTransformer(MODEL_NAME)
    chunks = _load_and_chunk(KB_PATH)

    if verbose:
        print(f"   → {len(chunks)} chunks extracted")

    embeddings = model.encode(
        chunks,
        show_progress_bar=verbose,
        normalize_embeddings=True,  # Enables cosine similarity via inner product
        convert_to_numpy=True,
    ).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    if verbose:
        print(f"✅  FAISS index saved ({index.ntotal} vectors, dim={dim})")

    return index, chunks


def load_index(verbose: bool = False) -> Tuple:
    """
    Loads the persisted FAISS index, rebuilding it if not found.

    Returns:
        (index, chunks)
    """
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        if verbose:
            print(f"✅  FAISS index loaded ({index.ntotal} vectors)")
        return index, chunks

    return build_index(verbose=True)


if __name__ == "__main__":
    build_index(verbose=True)
