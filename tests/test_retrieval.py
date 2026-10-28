import pytest

from app import retrieval


# ==========================================
# 1. Parameter Validation & Boundary Tests
# ==========================================
def test_empty_question_is_rejected():
    """Validates that passing blank string configurations or hidden whitespace characters

    correctly triggers a ValueError exception early to protect pipeline performance.
    """
    with pytest.raises(ValueError, match="Question must not be empty"):
        retrieval.retrieve_chunks("   ")


def test_invalid_k_is_rejected():
    """Ensures that setting an invalid upper bounds lookup count parameter (k <= 0)

    is blocked, raising a clear contextual validation exception.
    """
    with pytest.raises(ValueError, match="k must be greater than 0"):
        retrieval.retrieve_chunks("What is RAG?", k=0)


# ==========================================
# 2. Mocking Interface & Query State Tests
# ==========================================
def test_empty_collection_returns_empty_list(monkeypatch):
    """Ensures that if the index contains zero documents, the retrieval routing function

    short-circuits safely, returning an empty list instead of calling model math layers.
    """
    class FakeCollection:
        def count(self):
            return 0

    # Hot-swap the database connection handle with our empty mock object instance
    monkeypatch.setattr(
        retrieval, "get_collection", lambda: FakeCollection()
    )

    result = retrieval.retrieve_chunks("What is RAG?")

    # Verify that the short-circuit successfully returns an empty list mapping layout
    assert result == []


def test_retrieved_results_have_expected_fields(monkeypatch):
    """Intercepts database lookups to verify that payload matrix responses from ChromaDB

    are properly normalized and unpacked into standard structured data dictionary collections.
    """
    class FakeCollection:
        def count(self):
            return 2

        def query(self, **kwargs):
            # Assert that the correct bounds constraints are enforced down to the query layer
            assert kwargs["n_results"] == 2
            assert kwargs["include"] == [
                "documents", "metadatas", "distances"
            ]

            # Mock a standard nested batch payload response matrix matching ChromaDB's schema format
            return {
                "ids": [["chunk_1", "chunk_2"]],
                "documents": [["RAG retrieves information.", "AI uses data."]],
                "metadatas": [
                    [{"source": "notes.txt", "chunk_index": 0},
                     {"source": "notes.txt", "chunk_index": 1}]
                ],
                "distances": [[0.15, 0.42]],
            }

    # Intercept database fetch handles and heavy sentence-transformer embedding model weights
    monkeypatch.setattr(
        retrieval, "get_collection", lambda: FakeCollection()
    )
    monkeypatch.setattr(
        retrieval, "generate_embeddings", lambda texts: [[0.1, 0.2, 0.3]]
    )

    results = retrieval.retrieve_chunks("What is RAG?", k=2)

    # --- Structural Integrity Assertions ---
    # Ensure that passing down 2 target items extracts exactly 2 mapped objects
    assert len(results) == 2
    # Verify that data columns are correctly mapped back into unified structured keys
    assert results[0]["chunk_id"] == "chunk_1"
    assert results[0]["text"] == "RAG retrieves information."
    assert results[0]["metadata"]["source"] == "notes.txt"
    assert results[0]["distance"] == 0.15  # Semantic proximity calculation score match


# ==========================================
# 3. Metadata Filtering Constraints Tests
# ==========================================
def test_source_filter_is_passed_to_chromadb(monkeypatch):
    """Verifies that when a specific document name filter constraint is passed in,

    the system accurately updates the dictionary arguments to pass the 'where' filter block down to ChromaDB.
    """
    class FakeCollection:
        def count(self):
            return 1

        def query(self, **kwargs):
            # Validate that the filtering parameter keys match the underlying database logic requirements
            assert kwargs["where"] == {"source": "notes.txt"}

            return {
                "ids": [["chunk_1"]],
                "documents": [["RAG retrieves information."]],
                "metadatas": [[{"source": "notes.txt", "chunk_index": 0}]],
                "distances": [[0.15]],
            }

    # Apply mock patches to run the targeted check safely in memory
    monkeypatch.setattr(
        retrieval, "get_collection", lambda: FakeCollection()
    )
    monkeypatch.setattr(
        retrieval, "generate_embeddings", lambda texts: [[0.1, 0.2, 0.3]]
    )

    # Execute the retrieval interface using an active file identifier filter constraint
    results = retrieval.retrieve_chunks(
        "What is RAG?", source="notes.txt"
    )

    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "notes.txt"
