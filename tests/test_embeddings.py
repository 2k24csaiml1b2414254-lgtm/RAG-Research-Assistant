import pytest

from app import embeddings


# ==========================================
# 1. Mock Objects & Fake Implementations
# ==========================================
class FakeModel:
    """A lightweight mock replica of SentenceTransformer.

    It mimics the transformer model interface to return hardcoded vector shapes instantly,
    removing dependencies on heavy ML weights or hardware accelerators during testing.
    """
    def encode(self, texts, **kwargs):
        # Generate a standard mock 3-dimensional float array matching the size of the text array input
        return [[0.1, 0.2, 0.3] for _ in texts]


# ==========================================
# 2. Input Integrity & Validation Tests
# ==========================================
def test_empty_input():
    """Verifies that an empty text input sequence returns an empty collection list
    without needlessly triggering the embedding model pipeline.
    """
    assert embeddings.generate_embeddings([]) == []


def test_empty_string_is_rejected():
    """Asserts that sending collections that contain hidden blank strings or whitespaces
    correctly triggers a ValueError to maintain clean database index states.
    """
    # The 'match' parameter verifies that the thrown error message mentions the explicit constraint
    with pytest.raises(ValueError, match="empty strings"):
        embeddings.generate_embeddings(["hello", " "])


# ==========================================
# 3. Execution Patching & Structure Tests
# ==========================================
def test_embeddings_are_returned_as_lists(monkeypatch):
    """Uses pytest's monkeypatch fixture to intercept internal model loading functions,
    verifying that the matrix outputs are properly restructured into standard Python lists.
    """
    # Dynamically hot-swap the module's real model loader with a lambda that returns our FakeModel instance.
    # This prevents the test suite from loading gigabytes of transformer weights into RAM.
    monkeypatch.setattr(
        embeddings,
        "get_embedding_model",
        lambda: FakeModel(),
    )

    # Execute the target embedding extraction pipeline step using the patched mock framework
    result = embeddings.generate_embeddings(["hello", "world"])

    # --- Structural Integrity Assertions ---
    # Ensure that passing down 2 input text units returns exactly 2 vector payloads
    assert len(result) == 2
    # Verify that the vector values match our designated hardcoded mock outputs exactly
    assert result[0] == [0.1, 0.2, 0.3]
    # Check that all returning elements are native lists (compatible with storage architectures like ChromaDB)
    assert all(isinstance(vector, list) for vector in result)
