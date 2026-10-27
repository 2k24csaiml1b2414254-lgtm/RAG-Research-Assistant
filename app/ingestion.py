from pathlib import Path

from pypdf import PdfReader

# Define the set of allowed file formats to prevent processing incompatible binaries
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_document(file_path: str | Path) -> dict:
    """Loads a PDF or TXT document, extracts its full text content, and returns

    a dictionary containing text segments and processing metadata markers.
    """
    # Convert input string paths into modern Path objects for robust OS-agnostic operations
    path = Path(file_path)

    # --- Disk Validation Guardrails ---
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, got a directory: {path}")

    # Extract the file extension suffix in lowercase to ensure case-insensitive matching (e.g., .PDF -> .pdf)
    file_type = path.suffix.lower()

    # Reject unsupported binary signatures early before attempting memory allocation or read tasks
    if file_type not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {file_type}. "
            "Supported types are PDF and TXT."
        )

    # --- Core Extraction Engine Routines ---
    if file_type == ".txt":
        # Straightforward character-by-character text stream load using global standard UTF-8 encoding
        text = path.read_text(encoding="utf-8")

    else:
        # Instantiate a pypdf reader sequence pointer mapping out the target document
        reader = PdfReader(str(path))
        pages = []

        # Iterate through the document tree sequentially extracting layer data
        for page in reader.pages:
            # Append extract_text strings, falling back to a blank string if a page contains no text elements
            pages.append(page.extract_text() or "")

        # Stitch all extracted page strings back together using newline line breaks
        text = "\n".join(pages)

    # Clean off trailing or leading whitespace margins remaining from systemic encoding conversions
    text = text.strip()

    # --- Empty Resource Exception Boundaries ---
    if not text:
        raise ValueError(
            f"No extractable text found in: {path.name}. "
            "The PDF may be scanned or empty."
        )

    # Pack metrics along with content packages into a standard transportable schema interface
    return {
        "source": path.name,  # Clean filename string (e.g., 'report.pdf')
        "file_type": file_type.lstrip("."),  # Normalised text label ('pdf' or 'txt')
        "text": text,  # Raw unified string extraction block
        "character_count": len(text),  # Metrics integer tracing character length scale
    }
