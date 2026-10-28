import pytest

from app import rag_pipeline


# ==========================================
# 1. Parameter Validation & Boundary Tests
# ==========================================
def test_empty_question_is_rejected():
    """Validates that empty questions or whitespace inputs are rejected early

    to block invalid pipeline processing down the stack.
    """
    with pytest.raises(ValueError, match="Question must not be empty"):
        rag_pipeline.answer_question("   ")


# ==========================================
# 2. Empty Vector Storage & Short-Circuit Tests
# ==========================================
def test_no_retrieved_chunks_returns_empty_sources(monkeypatch):
    """Ensures that if the vector database returns an empty list of chunks,

    the orchestrator short-circuits instantly, avoids making an LLM API call,
    and returns a clean fallback dictionary payload structure.
    """
    # Intercept document retrieval queries to return an empty array list context block
    monkeypatch.setattr(rag_pipeline, "retrieve_chunks", lambda question, k: [])

    # Stub out the generation call to intentionally throw an assertion error if executed.
    # This proves the pipeline logic accurately isolates the text engine if context data is missing.
    def unexpected_generation(*args, **kwargs):
        pytest.fail("Generation should not run without retrieved chunks.")

    monkeypatch.setattr(
        rag_pipeline, "generate_answer", unexpected_generation
    )

    # Execute the primary answering pipeline orchestrator function
    result = rag_pipeline.answer_question("What is RAG?")

    # --- Structural Short-Circuit Fallback Assertions ---
    assert result["sources"] == []
    assert result["citations"] == []
    assert result["retrieved_chunks"] == []
    assert "could not find" in result["answer"].lower()


# ==========================================
# 3. Target Verification & Citation Mapping Tests
# ==========================================
def test_answer_and_citations_are_returned(monkeypatch):
    """Validates successful execution paths: checks context composition logic,

    verifies unique source deduplication rows, and confirms citation tracking values.
    """
    # Build structured mock sample data rows mimicking active vector storage records
    fake_chunks = [
        {
            "chunk_id": "notes.txt_chunk_0",
            "text": "RAG retrieves external information.",
            "metadata": {
                "source": "notes.txt",
                "chunk_index": 0,
            },
            "distance": 0.15,
        },
        {
            "chunk_id": "notes.txt_chunk_1",
            "text": "RAG uses retrieved context to generate answers.",
            "metadata": {
                "source": "notes.txt",
                "chunk_index": 1,
            },
            "distance": 0.25,
        },
    ]

    # Patch the retrieval function to return the static test chunk data matrix lists array
    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_chunks",
        lambda question, k: fake_chunks,
    )

    # Intercept text generation to verify the format of incoming context variables
    def fake_generate_answer(question, context):
        assert question == "What is RAG?"
        # Verify metadata formatting strings exist inside the combined prompt matrix boundary
        assert "Source: notes.txt, Chunk: 0" in context
        assert "Source: notes.txt, Chunk: 1" in context
        return "RAG uses retrieved information to generate answers."

    monkeypatch.setattr(
        rag_pipeline, "generate_answer", fake_generate_answer
    )

    # Run the full integrated answering test routine setting boundary k count rules
    result = rag_pipeline.answer_question("What is RAG?", k=2)

    # --- Citation and Telemetry Validation Assertions ---
    assert result["answer"].startswith("RAG uses")
    assert result["sources"] == ["notes.txt"]  # Confirms source file tracking is deduplicated cleanly
    assert len(result["citations"]) == 2
    assert result["citations"][0]["chunk_id"] == "notes.txt_chunk_0"
    assert result["citations"][0]["chunk_index"] == 0
    assert result["citations"][0]["distance"] == 0.15  # Semantic proximity calculation value match


# ==========================================
# 4. Error Propagation & Network Fault Tests
# ==========================================
def test_generation_error_propagates(monkeypatch):
    """Verifies exception handling: confirms down-stream network breaks or model server drops

    propagate cleanly through the master pipeline rather than getting swallowed silently.
    """
    fake_chunks = [
        {
            "chunk_id": "notes.txt_chunk_0",
            "text": "RAG retrieves information.",
            "metadata": {
                "source": "notes.txt",
                "chunk_index": 0,
            },
            "distance": 0.15,
        }
    ]

    # Pre-populate retrieval layers to safely pass vector checks without crash triggers
    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_chunks",
        lambda question, k: fake_chunks,
    )

    # Force a runtime error block to mimic local Ollama daemon connection drops
    def failing_generation(question, context):
        raise RuntimeError("Model unavailable")

    monkeypatch.setattr(
        rag_pipeline, "generate_answer", failing_generation
    )

    # Ensure the orchestrated stack catches the inner failure and raises it accurately
    with pytest.raises(RuntimeError, match="Model unavailable"):
        rag_pipeline.answer_question("What is RAG?")
