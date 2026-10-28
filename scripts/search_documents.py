from app.retrieval import retrieve_chunks


# ==========================================
# Main Execution Module
# ==========================================
def main():
    # Prompt the developer or user directly in the console for their search query text
    question = input("Enter your question: ").strip()

    # Dispatch the input query string down to the vector store semantic retrieval engine
    results = retrieve_chunks(question)

    # Empty boundary fallback handling if the data index contains no relevant text matches
    if not results:
        print("No indexed documents found.")
        return

    # Loop through the list of semantically ranked dictionary matches.
    # Enumerate helps display matching items sequentially starting from 1 (the highest match).
    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Chunk ID: {result['chunk_id']}")  # The unique system storage key lookup path
        print(f"Source: {result['metadata']['source']}")  # The filename context origin track marker
        print(f"Distance: {result['distance']:.4f}")  # Formats raw float distance scores to 4 decimal precision spaces
        print(f"Text: {result['text']}")  # The core underlying string content piece matching the intent


# Standard protective guard block isolating execution patterns to absolute target shell calls
if __name__ == "__main__":
    main()
