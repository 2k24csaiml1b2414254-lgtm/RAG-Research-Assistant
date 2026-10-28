from pathlib import Path

from app.chunking import chunk_text
from app.embeddings import generate_embeddings
from app.ingestion import load_document
from app.vector_store import get_collection, upsert_chunks


def index_document(
    file_path: str | Path,
    chunk_size: int = 500,
    overlap: int = 50,
) -> dict:
    """Orchestrates the end-to-end ingestion pipeline: reads files, segments text,

    generates vector matrices, commits to storage, and purges stale records.
    """

    # --- Step 1: Document Parsing & Extraction ---
    # Load document payload dictionary elements using the specialized text/PDF file reader
    document = load_document(file_path)

    source = document["source"]  # The filename acting as a primary reference tag (e.g., 'data.pdf')
    text = document["text"]  # The extracted clean unified string block containing raw characters

    # --- Step 2: Boundary-Aware Content Chunking ---
    # Slice the continuous body text down into smaller semantic slices using character target thresholds
    texts = chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    # Safeguard check ensuring execution stops immediately if a file yields no structural text segments
    if not texts:
        raise ValueError("Document produced no chunks.")

    # --- Step 3: Meta-Schema Enrichment & ID Strategy ---
    # Construct distinct record objects containing structured identifiers and positioning attributes.
    # Deterministic chunk_id names make updates predictable if the source text shifts.
    chunks = [
        {
            "chunk_id": f"{source}_chunk_{index}",
            "text": chunk,
            "source": source,
            "chunk_index": index,
        }
        for index, chunk in enumerate(texts)
    ]

    # --- Step 4: Vector Matrix Transformation ---
    # Request mathematical float vectors for all text blocks simultaneously via the cached model pipeline
    embeddings = generate_embeddings(texts)

    # --- Step 5: Historical Data Drift Analysis ---
    # Connect directly to the local persistent database collection target setup
    collection = get_collection()

    # Query the existing database for document segments linked specifically to the source file name.
    # This keeps track of previously inserted versions to prevent file update collisions.
    previous = collection.get(
        where={"source": source},
        include=["metadatas"],
    )

    # Convert array keys into lookup sets to calculate additions or removals accurately
    previous_ids = set(previous["ids"])
    current_ids = {chunk["chunk_id"] for chunk in chunks}

    # --- Step 6: Database Records Synchronization ---
    # Upsert the validated list data and normalized embedding matrices straight into the index space
    upsert_chunks(chunks, embeddings)

    # --- Step 7: Obsolete Chunk Garbage Collection ---
    # Discover dead nodes by computing the difference: IDs present previously but absent now.
    # This keeps the vector store clean if a user trims down or completely modifies a source document.
    stale_ids = previous_ids - current_ids

    # Perform hard deletion operations only if obsolete trace items are identified in storage
    if stale_ids:
        collection.delete(ids=list(stale_ids))

    # Compile dynamic telemetry logs counting system database transactions
    return {
        "source": source,
        "chunks_indexed": len(chunks),  # Count of successfully written or updated records
        "stale_chunks_removed": len(stale_ids),  # Count of old data nodes cleared out from disk
    }
