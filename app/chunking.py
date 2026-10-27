import re


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Splits a string into character-based text chunks while preserving sentence boundaries

    when possible and handling word-boundary overlaps gracefully.
    """

    # --- Core Input Validation Safeguards ---
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Clean up outer padding whitespace from the source document string
    text = text.strip()

    # Gracefully catch empty inputs early to prevent processing errors
    if not text:
        return []

    # Use a positive look behind regex assertion to split text right after sentence-ending punctuation.
    # This splits on whitespace segments (\s+) that immediately follow a period, question mark, or exclamation point.
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Initialize a list to hold 'atomic text units' (valid short sentences or sliced sub-sentences)
    units = []

    # --- Step 1: Pre-process and Normalise Sentences ---
    for sentence in sentences:
        sentence = sentence.strip()

        # Skip empty string remnants created by trailing spaces or consecutive punctuation marks
        if not sentence:
            continue

        # If the sentence easily fits within our character limit, keep it completely intact
        if len(sentence) <= chunk_size:
            units.append(sentence)

        # Forcefully split long sentences that exceed the target size by slicing them into character blocks
        else:
            start = 0
            while start < len(sentence):
                end = start + chunk_size
                # Slice out the character window substring, strip extra padding, and store it
                units.append(sentence[start:end].strip())
                # Advance the pointer directly forward by the window block size (no semantic boundary search)
                start = end

    # --- Step 2: Assemble Text Segments with Sliding Overlaps ---
    chunks = []
    current = ""  # Running buffer memory string accumulating atomic text units

    for unit in units:
        # Dynamically calculate what the next text chunk would look like if we combined the buffer with the next unit
        candidate = f"{current} {unit}".strip() if current else unit

        # If the combined candidate fits inside the character limit framework, grow the current buffer
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # If the candidate breaks the threshold limit, flush the completed current buffer to our results list
            if current:
                chunks.append(current)

            # --- Overlap Extraction Logic ---
            # Reuse the tail end characters of the flushed buffer to seed the context window of the next text block
            if current and overlap > 0:
                current = current[-overlap:]

                # Avoid starting the next chunk mid-word by finding the first space character space boundary
                # and dropping the initial broken word fragment via split(separator, maxsplit)
                if " " in current:
                    current = current.split(" ", 1)[1]

            # Re-evaluate the combination using our newly trimmed sliding window history context
            candidate = f"{current} {unit}".strip() if current else unit

            # If the updated string window fits inside the boundaries, update the buffer state
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                # If a single unit combined with the overlap still breaks the limit, discard the overlap completely
                # and force the raw sentence unit to stand on its own as the next independent window anchor
                current = unit

    # --- Step 3: Pipeline Flush ---
    # Append any remaining text remnants still lingering inside the buffer execution space down to disk array
    if current:
        chunks.append(current)

    return chunks
