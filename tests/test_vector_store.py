import pytest

from app import vector_store


# ==========================================
# 1. Isolated Test Database Fixtures
# ==========================================
@pytest.fixture
def temporary_collection(tmp_path, monkeypatch):
    """Creates a temporary, isolated ChromaDB collection instance for testing.

    Ensures that test operations do not read or write data to the actual development database.
    """
    # Hot-swap the production directory configuration with an isolated, temporary path
    monkeypatch.setattr(
        vector_store,
        "CHROMA_DB_DIR",
        tmp_path / "chroma_db",
    )
    # Re-route database connection handlers to use a designated test collection index space
    monkeypatch.setattr(
        vector_store,
        "COLLECTION_NAME",
        "test_documents",
    )

    # Initialise and return the scoped collection pointer to the dependent test functions
    return vector_store.get_collection()


# ==========================================
# 2. Integration & Validation Workflows
# ==========================================
def test_upsert_and_retrieve(temporary_collection, monkeypatch):
    """Validates that valid chunks and embeddings are safely stored inside the database

    index space, and can be queried back with accurate metadata.
    """
    # Intercept and patch the global collection retriever to fetch our isolated testing fixture instance
    monkeypatch.setattr(
        vector_store,
        "get_collection",
        lambda: temporary_collection,
    )

    # Build structured mock sample data records matching the ingestion pipeline schema
    chunks = [
        {
            "chunk_id": "doc_chunk_0",
            "text": "Machine learning learns patterns.",
            "source": "notes.txt",
            "chunk_index": 0,
        }
    ]

    # Provide a simple 3-dimensional floating point mock vector list
    embeddings = [[0.1, 0.2, 0.3]]

    # Execute the injection routine to store records inside the local index block
    vector_store.upsert_chunks(chunks, embeddings)

    # Directly query the database container using explicit ID pointers to extract data details
    result = temporary_collection.get(
        ids=["doc_chunk_0"],
        include=["documents", "metadatas"],
    )

    # --- Structural & Integrity Assertions ---
    # Confirm that the returned document string body matches the injected text content perfectly
    assert result["documents"][0] == chunks[0]["text"]
    # Verify that structural tracking keys match original input metadata assignments
    assert result["metadatas"][0]["source"] == "notes.txt"
    assert result["metadatas"][0]["chunk_index"] == 0


def test_mismatched_embedding_count():
    """Asserts that attempting to write document slices when the vector array count

    does not exactly match the text count raises a ValueError.
    """
    # The 'match' rule validates that the console failure output contains the descriptive constraint phrase
    with pytest.raises(ValueError, match="exactly one embedding"):
        vector_store.upsert_chunks(
            [
                {
                    "chunk_id": "doc_chunk_0",
                    "text": "Some text",
                    "source": "notes.txt",
                    "chunk_index": 0,
                }
            ],
            [],  # Intentionally pass an empty array list to force a vector dimension error
        )


def test_empty_chunks_do_not_create_records(temporary_collection, monkeypatch):
    """Ensures that passing blank array structures short-circuits safely, resulting in

    zero additions or modification queries to database counts.
    """
    # Patch database calls to bind directly to our transient testing index framework
    monkeypatch.setattr(
        vector_store,
        "get_collection",
        lambda: temporary_collection,
    )

    # Run the ingestion interface function passing dual blank sequences
    vector_store.upsert_chunks([], [])

    # Confirm that the total row index count inside the data collection remains exactly zero
    assert temporary_collection.count() == 0
