import pytest

from app.chunking import chunk_text


# ==========================================
# 1. Edge Case & Blank Input Tests
# ==========================================
def test_empty_text():
    """Validates that input strings consisting only of whitespace return an empty

    list rather than creating useless text elements.
    """
    assert chunk_text(" ") == []


def test_short_text_returns_one_chunk():
    """Ensures that if the entire text easily fits within the chunk size limit,

    it returns as a single, intact element without being split.
    """
    text = "AI helps machines learn."
    assert chunk_text(text, chunk_size=100) == [text]


# ==========================================
# 2. Boundary & Character Length Tests
# ==========================================
def test_chunks_respect_size():
    """Verifies that multi-sentence inputs are split into multiple elements, and

    that every generated block strictly adheres to the maximum character size.
    """
    text = (
        "Artificial intelligence is useful."
        "Machine learning discovers patterns."
        "Deep learning uses neural networks."
    )

    chunks = chunk_text(
        text,
        chunk_size=45,
        overlap=5,
    )
    # Check that the text was actually split across multiple blocks
    assert len(chunks) > 1
    # Check that no single chunk breaks the upper character threshold boundary
    assert all(len(chunk) <= 45 for chunk in chunks)


def test_long_sentence_is_split():
    """Confirms that a single continuous string exceeding the chunk limit is

    forced to break into multiple pieces instead of breaking the size boundary.
    """
    # Create an artificial 120-character unbroken text string
    text = "A" * 120

    chunks = chunk_text(
        text,
        chunk_size=40,
        overlap=5,
    )

    assert len(chunks) > 1
    # Ensure even forced slices respect the strict maximum size setting
    assert all(len(chunk) <= 40 for chunk in chunks)


# ==========================================
# 3. Parameter Validation & Exception Tests
# ==========================================
def test_invalid_chunk_size():
    """Asserts that setting an invalid or zero chunk size throws a ValueError early

    to prevent infinite looping during calculations.
    """
    with pytest.raises(ValueError):
        chunk_text("Example text", chunk_size=0)


def test_invalid_overlap():
    """Asserts that setting an overlap value equal to or greater than the chunk size

    correctly triggers a ValueError to avoid invalid sliding strides.
    """
    with pytest.raises(ValueError):
        chunk_text("Example text", chunk_size=20, overlap=20)
