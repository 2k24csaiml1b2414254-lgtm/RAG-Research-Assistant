from app.config import TOP_K
from app.generation import generate_answer
from app.retrieval import retrieve_chunks


def answer_question(question: str, k: int = TOP_K) -> dict:
    """Orchestrates the complete Retrieval-Augmented Generation (RAG) query execution block:

    Validates user query input strings, performs vector spatial similarity search loops,
    assembles context boundaries, extracts precise analytical citations, and returns a
    unified database result map payload.
    """

    # --- Step 1: Input Integrity Validation ---
    # Enforce strict data type boundaries and catch blank queries early to protect compute limits
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty.")

    # --- Step 2: Semantic Vector Store Lookup ---
    # Query the database to retrieve the top k most textually relevant matching document chunks
    retrieved_chunks = retrieve_chunks(question, k=k)

    # Clean short-circuit path routing if the underlying database layer returns zero matches
    if not retrieved_chunks:
        return {
            "question": question,
            "answer": (
                "I could not find any indexed documents "
                "to answer this question."
            ),
            "sources": [],
            "citations":[],
            "retrieved_chunks": [],
        }

    # --- Step 3: Context Compilation & Ingestion ---
    context_parts = []
    sources = []

    # Loop through the extracted matches to form standard text reference structural grids
    for chunk in retrieved_chunks:
        metadata = chunk["metadata"]
        source = metadata["source"]  # Tracks the base document name (e.g., 'policy.pdf')
        chunk_index = metadata["chunk_index"]  # Tracks the sequential position tracker index number

        # Format individual content slices alongside their respective metadata indices to let the LLM cite accurately
        context_parts.append(
            f"[Source: {source}, Chunk: {chunk_index}]\n" f"{chunk['text']}"
        )

        # Deduplicate the source track registry to return a flat, distinct list of document names
        if source not in sources:
            sources.append(source)

    # Compress separated lookup text structures into a unified continuous background context string block
    context = "\n\n".join(context_parts)

    # --- Step 4: Guardrailed Text Generation ---
    # Request answer token generation from the local LLM runtime engine using our structured prompt boundaries
    answer = generate_answer(
        question=question,
        context=context,
    )

    # --- Step 5: Citation Telemetry Parsing ---
    # Extract metadata properties into a clean citation trace array for UI transparency
    citations = [
        {
            "source": chunk["metadata"]["source"],
            "chunk_index": chunk["metadata"]["chunk_index"],
            "distance": chunk["distance"],  # Mathematical similarity proximity score
            "chunk_id": chunk["chunk_id"],  # Database internal lookup index reference key
        }
        for chunk in retrieved_chunks
    ]

    # --- Step 6: Unified Pipeline Telemetry Wrap ---
    # Package parameters along with logs into a structured system transmission dictionary object
    return {
        "question": question,  # The verified primary user text query
        "answer": answer,  # The anti-hallucination guardrailed output text response from the LLM
        "sources": sources,  # A flat distinct reference string list tracking source documents
        "citations": citations,  # Clean citation dictionaries tracking locations and similarity levels
        "retrieved_chunks": retrieved_chunks,  # The raw database trace list containing similarity scores
    }
