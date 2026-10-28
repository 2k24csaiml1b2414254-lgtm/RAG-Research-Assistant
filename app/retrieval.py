from app.config import TOP_K
from app.embeddings import generate_embeddings
from app.vector_store import get_collection


def retrieve_chunks(
    question: str,
    k: int = TOP_K,
    source: str | None = None,
) -> list[dict]:
    """Retrieves the most semantically relevant text chunks from the vector database

    based on a natural language user query.
    """

    # --- Input Validation Safeguards ---
    # Intercept empty or pure whitespace queries to prevent meaningless vector operations
    if not question.strip():
        raise ValueError("Question must not be empty.")

    # Prevent negative or zero boundary query limits which cause processing script failures
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    # Establish an active session handle link to the local ChromaDB database collection
    collection = get_collection()

    # Short-circuit immediately if the index is completely empty to preserve compute resources
    if collection.count() == 0:
        return []

    # Convert the string query into a dense mathematical embedding vector layer.
    # We take index [0] because generate_embeddings expects a list and returns a matrix.
    query_embedding = generate_embeddings([question])[0]

    # Package core parameter keys required by the ChromaDB matching query execution pipeline
    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": min(k, collection.count()),  # Bounds k to protect against index overflow
        "include": ["documents", "metadatas", "distances"],  # Explicitly query all field properties
    }

    # If a specific filename string is provided, apply a metadata filter layer to restrict the search space
    if source:
        query_args["where"] = {"source": source}

    # Execute the vector spatial distance search operation using unpack syntax (**kwargs)
    results = collection.query(**query_args)

    retrieved = []

    # Map the separated data arrays returned by ChromaDB back into unified structured list dict objects.
    # We iterate over the first row index ([0]) of the batch result matrix arrays.
    for i, chunk_id in enumerate(results["ids"][0]):
        retrieved.append(
            {
                "chunk_id": chunk_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],  # Distance score indicating semantic similarity
            }
        )

    return retrieved
