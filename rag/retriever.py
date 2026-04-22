"""
RAG Retriever: Semantic search over the FAISS index.

Usage:
    from rag.retriever import retrieve
    context = retrieve("what are the pricing plans?")
"""

from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.embedder import load_index, MODEL_NAME

# --- Module-level singletons (loaded once per process) ---
_model: Optional[SentenceTransformer] = None
_index = None
_chunks: Optional[list] = None

# Minimum cosine similarity score to include a chunk
RELEVANCE_THRESHOLD = 0.25


def _ensure_loaded() -> None:
    """Lazily load the embedding model and FAISS index."""
    global _model, _index, _chunks
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    if _index is None:
        _index, _chunks = load_index(verbose=False)


def retrieve(query: str, top_k: int = 3) -> str:
    """
    Retrieve the top-k most relevant passages for a query.

    Anti-hallucination design:
    - Only returns chunks that exceed the RELEVANCE_THRESHOLD.
    - If nothing passes the threshold, returns a clear "not found" message
      so the LLM can respond honestly rather than guessing.

    Args:
        query:  Natural-language question from the user.
        top_k:  Maximum number of passages to return.

    Returns:
        A single string containing relevant passages separated by dividers,
        or a "no information found" message.
    """
    _ensure_loaded()

    query_embedding = _model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    scores, indices = _index.search(query_embedding, top_k)

    results: list[str] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and float(score) >= RELEVANCE_THRESHOLD:
            results.append(_chunks[idx].strip())

    if not results:
        return (
            "No relevant information was found in the AutoStream knowledge base "
            "for this query."
        )

    return "\n\n---\n\n".join(results)
