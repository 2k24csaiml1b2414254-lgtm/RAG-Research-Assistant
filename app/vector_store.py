import chromadb

from app.config import CHROMA_DB_DIR, COLLECTION_NAME


# ==========================================
# 1. Database Connection & Initialisation
# ==========================================
def get_collection():
    """Connects to or initialises a persistent local ChromaDB collection."""

    # Instantiate a persistent database client that writes updates straight to disk storage
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # Safely open an existing collection index or instantiate a brand-new one if it doesn't exist.
    # The 'metadata' dict configures the mathematical metric space to calculate vector similarity. 
    # Cosine similarity measures directional alignment rather than absolute length.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


# ==========================================
# 2. Batch Validation and Vector Storage
# ==========================================
def upsert_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    """Validates raw data chunks alongside their generated vector embeddings,

    then safely upserts them into ChromaDB storage.
    """

    # Integrity safeguard checking that text arrays perfectly map to embedding vector arrays
    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have exactly one embedding.")

    # Graceful short-circuit layout preventing crashes if empty lists are passed down the pipeline
    if not chunks:
        return

    # Initialize separate matching arrays required by ChromaDB's native collection schema format
    ids = []
    documents = []
    metadatas = []

    # Iteratively parse and rigorously extract parameters out from the dictionary lists
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text")
        source = chunk.get("source")
        chunk_index = chunk.get("chunk_index")

        # --- Strict Data Type & Boundary Validation Framework ---
        if not chunk_id or not isinstance(chunk_id, str):
            raise ValueError("Each chunk needs a valid chunk_id.")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Each chunk needs non-empty text.")

        if not isinstance(source, str) or not source.strip():
            raise ValueError("Each chunk needs a source.")

        if not isinstance(chunk_index, int) or chunk_index < 0:
            raise ValueError("chunk_index must be a non-negative integer.")

        # Append verified attributes into distinct synchronous positioning arrays
        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(
            {
                "source": source,
                "chunk_index": chunk_index,
            }
        )

    # Establish connection handle parameters to the indexing framework target
    collection = get_collection()

    # Commit records to database storage using upsert operations.
    # Upsert avoids indexing duplicates: it writes new documents if IDs are missing,
    # or overrides existing records if overlapping matches are found on disk.
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
