import ollama

from app.config import LLM_MODEL

# ==========================================
# 1. System Prompt Guardrails Matrix
# ==========================================
# Define the foundational ruleset loaded inside the system role.
# This strictly constrains model behavior to eliminate hallucination vectors.
SYSTEM_PROMPT = """
You are a helpful research assistant.

Rules:
1. Answer using only the provided context.
2. If the context does not contain enough information,
   clearly say that you do not have enough information.
3. Do not invent facts.
4. Give a clear, concise answer.
5. Do not claim that information comes from a source
   unless it is present in the provided context.
"""


# ==========================================
# 2. Guardrailed Answer Generation Routing
# ==========================================
def generate_answer(question: str, context: str) -> str:
    """Dispatches unified context blocks and user questions down to a local Ollama client,

    enforcing anti-hallucination guardrails and validating payload structures.
    """

    # --- Type and State Boundary Validation Guards ---
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty.")

    if not isinstance(context, str) or not context.strip():
        raise ValueError("Context must not be empty.")

    try:
        # Submit the conversation matrix down to the local Ollama daemon client worker
        response = ollama.chat(
            model=LLM_MODEL,  # Loaded directly from global project configuration modules
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question:\n{question}\n\n"
                        "Answer using only the context above."
                    ),
                },
            ],
        )

    # --- Double-Layered Connection Exception Routing ---
    # Catch specific API-level execution faults thrown by the local Ollama binary engine
    except ollama.ResponseError as exc:
        raise RuntimeError(f"Ollama model error: {exc}") from exc

    # Catch network anomalies, socket timeouts, or missing backend instance container drops
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to Ollama. "
            "Check that Ollama is running and the model is available. "
            f"Details: {exc}"
        ) from exc

    # --- Response Payload Parsing & Integrity Validation ---
    # Safely pull the outer message dictionary layer from the response payload
    message = response.get("message")

    if not message:
        raise RuntimeError(
            f"Ollama response does not contain a message: {response}"
        )

    # Extract the final textual string completion body from the nested content tracker key
    answer = message.get("content")

    # Final sanitization checkpoint ensuring data was successfully generated before passing it to the UI
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError(f"Ollama returned an empty answer: {response}")

    return answer.strip()
