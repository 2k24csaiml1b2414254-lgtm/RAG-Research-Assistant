from pathlib import Path

from app.config import DATA_DIR
from app.indexing import index_document


# ==========================================
# Main Execution Pipeline
# ==========================================
def main():
    # Construct the absolute path pointing directly to the target source text document.
    # Uses configuration parameters to ensure paths resolve correctly across different operating systems.
    document_path = DATA_DIR / "sample_documents" / "sample.txt"

    # Trigger the end-to-end orchestration pipeline function.
    # This reads the file, handles chunking, generates embeddings, and synchronizes the vector store.
    result = index_document(document_path)

    # --- Terminal Status Telemetry Logs ---
    print("\nIndexing completed!")
    print(f"Source: {result['source']}")  # Prints the filename identifier (e.g., 'sample.txt')
    print(
        f"Chunks indexed: {result['chunks_indexed']}"
    )  # Counts written/overwritten data entries
    print(
        "Stale chunks removed: "
        f"{result['stale_chunks_removed']}"
    )  # Counts obsolete records cleaned from disk


# Standard execution safeguard ensuring the main sequence only runs when this script
# is executed directly from the command line, rather than when imported inside another module.
if __name__ == "__main__":
    main()
