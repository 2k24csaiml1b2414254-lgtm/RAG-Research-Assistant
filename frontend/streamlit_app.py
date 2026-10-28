# Import the standard HTTP library to communicate with the FastAPI backend server
import requests
# Import Streamlit to assemble our web-based graphical user interface
import streamlit as st

# Base endpoint coordinate pointing to our locally running FastAPI application
API_URL = "http://127.0.0.1:8000"

# Configure the browser tab title, primary favicon emoji, and page geometry layout
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📚",
    layout="wide",
)

# Render a prominent main header title onto the interface layout canvas
st.title("📚 RAG Research Assistant")
# Render fine metadata microcopy explaining the application utility to users
st.caption(
    "Upload research documents, ask questions, "
    "and explore source-backed answers."
)

# --- SECTION: BACKEND SERVER HEALTH CHECK ---
try:
    # Query the app's health checkpoint with a strict 5-second failure threshold
    health = requests.get(
        f"{API_URL}/health",
        timeout=5,
    )
    # Assign online true status only if the server returns a successful 200 OK code
    api_online = health.ok
except requests.RequestException:
    # If a connection refusal or timeout occurs, flag the API as safely offline
    api_online = False

# Render contextual notification alerts based on backend accessibility
if api_online:
    st.success("API connected")
else:
    st.error(
        "API is not reachable. Start FastAPI on port 8000."
    )

# Render a horizontal layout separator line
st.divider()

# --- SECTION 1: DOCUMENT UPLOADING & INGESTION ---
st.header("1. Upload a document")

# Present a clean file upload container restricted exclusively to PDF and TXT data types
uploaded_file = st.file_uploader(
    "Choose a PDF or TXT file",
    type=["pdf", "txt"],
)

# Render the ingest button, instantly disabling it if no file is present or the API is offline
if st.button(
    "Upload and index",
    disabled=uploaded_file is None or not api_online,
):
    try:
        # Mask the workspace with a loading spinner while the asset processes
        with st.spinner("Uploading and indexing document..."):
            # Stream out the form multi-part files envelope toward the API
            response = requests.post(
                f"{API_URL}/documents/upload",
                files={
                    "file": (
                        uploaded_file.name,          # The original name string of the file
                        uploaded_file.getvalue(),    # The raw bytes array of the file contents
                        uploaded_file.type           # The MIME media string indicator
                        or "application/octet-stream",
                    )
                },
                timeout=300, # Large timeout to safely handle massive document ingestion scripts
            )

        # Parse responses to evaluate pipeline execution validation states
        if response.ok:
            result = response.json()
            st.success("Document indexed successfully!")
            # Present individual summary stats extracted from the successful upload envelope
            st.write(f"**File:** {result['filename']}")
            st.write(
                f"**Chunks indexed:** {result['chunks_indexed']}"
            )
        else:
            # Display granular API failure properties if the payload was rejected
            st.error(
                f"Upload failed ({response.status_code}): "
                f"{response.text}"
            )

    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")

st.divider()

# --- SECTION 2: VECTOR LOOKUP AND QUERYING ---
st.header("2. Ask a research question")

# Render a multi-line input textarea window allowing users to paste full paragraphs
question = st.text_area(
    "Your question",
    placeholder="What is Retrieval-Augmented Generation?",
)

# Render an adjustment slider tracking how many document chunks to pull from the Vector DB
top_k = st.slider(
    "Number of retrieved chunks",
    min_value=1,
    max_value=10,
    value=3, # Standard baseline cluster threshold selection
)

# Process query submissions when the execution control unit is activated
if st.button("Get answer", disabled=not api_online):
    # Intercept empty submissions before hitting networks to conserve processing pools
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            # Display a progress placeholder loader during embedding generation and LLM sweeps
            with st.spinner("Searching documents and generating answer..."):
                # Forward query parameters to the backend using standard JSON envelopes
                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question.strip(),
                        "k": top_k,
                    },
                    timeout=300,
                )

            # Evaluate response integrity before mutating interface panels
            if not response.ok:
                st.error(
                    f"Request failed ({response.status_code}): "
                    f"{response.text}"
                )
            else:
                result = response.json()

                # Render the final synthesized textual answer block safely
                st.subheader("Answer")
                st.write(result["answer"])

                # Render structural metadata source citations for complete user auditability
                st.subheader("Sources")

                # Extract citation structures safely, defaulting to an empty list if absent
                citations = result.get("citations", [])

                if citations:
                    # Loop through individual citation blocks to construct visual details boxes
                    for citation in citations:
                        # Nest references inside clean drop-down accordion click widgets
                        with st.expander(
                            f"{citation['source']} — "
                            f"Chunk {citation['chunk_index']}"
                        ):
                            # Expose internal database primary keys for system debug tasks
                            st.write(
                                f"Chunk ID: {citation['chunk_id']}"
                            )
                            # Render floating point similarity metrics down to 4 decimal precision slots
                            st.write(
                                f"Cosine distance: "
                                f"{citation['distance']:.4f}"
                            )
                else:
                    # Display an info block if no context was returned by the embedding layer
                    st.info("No source chunks were retrieved.")

        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
