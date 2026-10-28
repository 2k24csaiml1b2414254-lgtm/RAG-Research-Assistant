import pytest

from app import indexing


# ==========================================
# 1. State-Tracking Database Mock Setup
# ==========================================
class FakeCollection:
    """A lightweight mock replica of a ChromaDB collection object.

    Tracks state transitions (adds, gets, deletes) in memory to avoid physical database overhead during
    testing.
    """

    def __init__(self, existing_ids=None):
        # Initialize internal storage arrays representing current indexed records on disk
        self.ids = existing_ids or []
        # Trace array recording list keys targeting standard delete cleanups
        self.deleted = []

    def get(self, where, include):
        """Simulates querying the index by returning the stored keys."""
        return {"ids": self.ids, "metadatas": []}

    def delete(self, ids):
        """Simulates records deletion by moving targeted keys from index to trash history lists."""
        self.deleted.extend(ids)
        self.ids = [item for item in self.ids if item not in ids]


# ==========================================
# 2. Pipeline E2E Orchestration Tests
# ==========================================
def test_index_document(monkeypatch, tmp_path):
    """Verifies the core orchestration path: safely loading, segmenting, embedding,

    and storing a standard document.
    """
    # Create a physical file fixture block to accurately trigger path parameters
    file_path = tmp_path / "notes.txt"
    file_path.write_text(
        "Machine learning learns patterns. AI systems can reason.",
        encoding="utf-8",
    )

    # --- Isolation Mocking Block (Intercept external pipeline modules) ---
    monkeypatch.setattr(
        indexing,
        "load_document",
        lambda path: {
            "source": "notes.txt",
            "file_type": "txt",
            "text": file_path.read_text(encoding="utf-8"),
            "character_count": len(file_path.read_text(encoding="utf-8")),
        },
    )

    monkeypatch.setattr(
        indexing,
        "chunk_text",
        lambda text, chunk_size, overlap: [
            "Machine learning learns patterns.",
            "AI systems can reason.",
        ],
    )

    monkeypatch.setattr(
        indexing,
        "generate_embeddings",
        lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
    )

    # Initialize our fresh in-memory collection mock object
    collection = FakeCollection()
    monkeypatch.setattr(indexing, "get_collection", lambda: collection)

    # Intercept storage operations to evaluate parameters submitted to our saving functions
    stored = {}

    def fake_upsert(chunks, embeddings):
        stored["chunks"] = chunks
        stored["embeddings"] = embeddings

    monkeypatch.setattr(indexing, "upsert_chunks", fake_upsert)

    # Run orchestrator
    result = indexing.index_document(file_path)

    # --- Structural Target Assertions ---
    assert result["source"] == "notes.txt"
    assert result["chunks_indexed"] == 2
    assert result["stale_chunks_removed"] == 0  # No previous data rows existed
    assert len(stored["chunks"]) == 2
    assert stored["chunks"][0]["chunk_id"] == "notes.txt_chunk_0"


# ==========================================
# 3. Deduplication & Cleanup Tests
# ==========================================
def test_stale_chunks_are_removed(monkeypatch, tmp_path):
    """Verifies data drift logic: when an updated document requires fewer segments,

    the obsolete historical chunks are dropped cleanly from index records.
    """
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Updated document.", encoding="utf-8")

    # --- Setup mock adjustments reflecting a trimmed down file configuration ---
    monkeypatch.setattr(
        indexing,
        "load_document",
        lambda path: {
            "source": "notes.txt",
            "file_type": "txt",
            "text": "Updated document.",
            "character_count": 17,
        },
    )

    monkeypatch.setattr(
        indexing,
        "chunk_text",
        lambda text, chunk_size, overlap: ["Updated document."],  # Yields 1 chunk instead of 3
    )

    monkeypatch.setattr(
        indexing,
        "generate_embeddings",
        lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
    )

    # Pre-populate the mock database with 3 legacy records representing the older version
    collection = FakeCollection(
        [
            "notes.txt_chunk_0",
            "notes.txt_chunk_1",
            "notes.txt_chunk_2",
        ]
    )

    monkeypatch.setattr(indexing, "get_collection", lambda: collection)
    monkeypatch.setattr(
        indexing,
        "upsert_chunks",
        lambda chunks, embeddings: None,
    )

    # Run orchestrator to trace updates and deletions
    result = indexing.index_document(file_path)

    # --- Data Drift Metric Assertions ---
    assert result["chunks_indexed"] == 1
    # Verify that the delta computation precisely identifies and triggers the deletion of 2 stale items
    assert result["stale_chunks_removed"] == 2
    # Check that index tracking accurately records the specific obsolete record keys that were purged
    assert set(collection.deleted) == {
        "notes.txt_chunk_1",
        "notes.txt_chunk_2",
    }
