from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL


# ==========================================
# 1. Model Caching Mechanism
# ==========================================
# The @lru_cache decorator caches the return value of this function.
# Setting maxsize=1 ensures the heavy weights of the transformer model are loaded into RAM
# only once upon the first call, preventing expensive disk reloads on subsequent pipeline steps.
@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Loads the designated embedding model once and caches its instance in memory."""
    return SentenceTransformer(EMBEDDING_MODEL)


# ==========================================
# 2. Embedding Vector Generation Pipeline
# ==========================================
def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Converts an array of clean text segments into normalized floating-point vector lists."""

    # Performance short-circuit: avoid executing model math routines if the input text list is empty
    if not texts:
        return []

    # Safeguard check to ensure no blank strings or pure whitespace values pass through,
    # as empty strings generate garbage embeddings that degrade vector index search accuracy
    if any(not text.strip() for text in texts):
        raise ValueError("Texts must not contain empty strings.")

    # Retrieve the singleton instance of the embedding model from the memory cache
    model = get_embedding_model()

    # Encode the text segments into dense vector matrices using the underlying engine.
    # - normalize_embeddings=True calculates the L2 norm, rescaling vectors to a unit length of 1.
    #   This converts standard dot-product calculations directly into fast cosine similarity operations.
    # - convert_to_numpy=True outputs a high-performance raw NumPy matrix for faster internal handling.
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Convert the multidimensional NumPy float array structure into native Python float lists
    # because JSON-based stores like ChromaDB require standard Python arrays rather than raw NumPy objects.
    return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings


# ==========================================
# 3. Core Architectural & Engineering Decisions
# ==========================================
"""
Key Engineering Design Notes:
- Model Caching: Using @lru_cache(maxsize=1) guarantees that subsequent calls reuse the same
  in-memory weights, saving significant compute cycles and CPU/GPU memory footprint overheads.
- Mathematical Alignment: Enabling normalize_embeddings=True standardizes vector length scales,
  which makes mathematical similarity calculations (such as dot-product or cosine) far more accurate.
- Schema Compatibility: Setting convert_to_numpy=True processes arrays efficiently, while
  calling .tolist() handles standard type serialization required by local client frameworks like ChromaDB.
- Guardrail Logic: Adding check validations blocks empty inputs early, preserving resources and
  stopping empty text inputs from skewing database calculation matrices.
"""
