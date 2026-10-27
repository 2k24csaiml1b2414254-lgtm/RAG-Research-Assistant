from pathlib import Path

import pytest

from app.ingestion import load_document


# ==========================================
# 1. Successful Document Loading Tests
# ==========================================
def test_load_txt(tmp_path: Path):
    """Validates standard text file loading and verifies that the output matches

    the expected dictionary structure and text extraction.
    """
    # Create an isolated temporary file inside pytest's managed 'tmp_path' directory
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "Artificial intelligence enables machines to learn.",
        encoding="utf-8",
    )

    # Execute the file loading pipeline function
    result = load_document(file_path)

    # Validate metadata parameters and text parsing accuracy
    assert result["source"] == "sample.txt"
    assert result["file_type"] == "txt"
    assert "artificial intelligence" in result["text"].lower()
    assert result["character_count"] > 0


# ==========================================
# 2. File and Type Exception Error Tests
# ==========================================
def test_missing_file():
    """Verifies that attempting to read a file that does not exist on disk

    correctly triggers a FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        load_document("does_not_exist.txt")


def test_unsupported_file(tmp_path: Path):
    """Validates that checking an unmapped extension category (.docx) throws a

    ValueError early with an accurate contextual message.
    """
    # Write a dummy file using an unsupported extension format
    file_path = tmp_path / "sample.docx"
    file_path.write_text("Example", encoding="utf-8")

    # Match parameter scans the exception text string for the target validation tag
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(file_path)


def test_empty_txt(tmp_path: Path):
    """Ensures files containing only empty whitespace trip the empty boundary validation checks

    instead of passing blank values down the processing workflow.
    """
    # Write a file consisting exclusively of space formatting blocks
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   ", encoding="utf-8")

    # Verify that the extraction engine stops processing and reports missing content
    with pytest.raises(ValueError, match="No extractable text"):
        load_document(file_path)
