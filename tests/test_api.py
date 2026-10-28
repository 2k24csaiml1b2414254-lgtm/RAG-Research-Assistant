# Import TestClient from FastAPI to simulate HTTP requests without launching a live server
from fastapi.testclient import TestClient

# Import the FastAPI application instance ('app') to bind it to the TestClient
from app.main import app
# Import the main module entirely so its internal function references can be dynamically patched (mocked)
from app import main

# Initialize the simulated HTTP client wrapper around your FastAPI application
client = TestClient(app)


# TEST 1: Verifies that the base application health check endpoint is operational
def test_health_endpoint():
    # Execute a simulated synchronous HTTP GET request to the "/health" route
    response = client.get("/health")

    # Assert that the API responds with an HTTP status code 200 (OK)
    assert response.status_code == 200
    # Assert that the response JSON body matches the expected service descriptor payload exactly
    assert response.json() == {
        "status": "healthy",
        "service": "RAG Research Assistant API",
    }


# TEST 2: Validates a successful query to the RAG backend using mock data instead of calling heavy models
def test_ask_endpoint_success(monkeypatch):
    # Define a stub/fake function to isolate the API endpoint from actual vector DB lookups or LLM calls
    def fake_answer_question(question, k):
        return {
            "question": question,
            "answer": "RAG uses retrieved information to generate answers.",
            "sources": ["sample.txt"],
            "citations": [
                {
                    "source": "sample.txt",
                    "chunk_index": 0,
                    "distance": 0.15,
                    "chunk_id": "sample.txt_chunk_0",
                }
            ],
            "retrieved_chunks": [],
        }

    # Intercept calls to 'main.answer_question' and force them to use our 'fake_answer_question' stub
    monkeypatch.setattr(
        main, "answer_question", fake_answer_question
    )

    # Dispatch a simulated HTTP POST request to the "/ask" endpoint with valid JSON inputs
    response = client.post(
        "/ask",
        json={
            "question": "What is RAG?",
            "k": 3,
        },
    )

    # Assert that the endpoint processed the request cleanly and returned an HTTP 200 (OK) status code
    assert response.status_code == 200

    # Parse the response payload to inspect specific keys and structures
    data = response.json()
    # Confirm that the returned object echoes back the original input question
    assert data["question"] == "What is RAG?"
    # Perform a substring check to ensure the generated text answer matches what our fake function returned
    assert "RAG uses retrieved information" in data["answer"]
    # Ensure source lists match expectations
    assert data["sources"] == ["sample.txt"]
    # Dig into the list of citation structures to verify metadata integrity (index position)
    assert data["citations"][0]["chunk_index"] == 0


# TEST 3: Checks if input validation properly catches and blocks empty search strings
def test_ask_endpoint_rejects_invalid_request():
    # Send a request where the question parameter is left as an empty string ""
    response = client.post(
        "/ask",
        json={
            "question": "",
            "k": 3,
        },
    )

    # Assert that FastAPI's Pydantic validation interceptor catches it and responds with 422 (Unprocessable Entity)
    assert response.status_code == 422


# TEST 4: Checks if input validation catches invalid numerical parameters (k must be > 0)
def test_ask_endpoint_rejects_invalid_k():
    # Send a request where k (number of retrieved chunks requested) is set to an invalid value of 0
    response = client.post(
        "/ask",
        json={
            "question": "What is RAG?",
            "k": 0,
        },
    )

    # Assert that the Pydantic schema schema validation fails, rejecting the request with an HTTP 422
    assert response.status_code == 422


# TEST 5: Verifies that the API gracefully handles structural backend crashes (like model timeouts)
def test_ask_endpoint_handles_pipeline_error(monkeypatch):
    # Define a stub function designed to simulate a total pipeline failure or model disconnection
    def fake_answer_question(question, k):
        raise RuntimeError("Model unavailable")

    # Patch the function reference to simulate the live application throwing a RuntimeError during execution
    monkeypatch.setattr(
        main, "answer_question", fake_answer_question
    )

    # Send a standard valid payload toward the endpoint
    response = client.post(
        "/ask",
        json={
            "question": "What is RAG?",
            "k": 3,
        },
    )

    # Assert that your FastAPI app caught the crash and translated it into a clean HTTP 503 (Service Unavailable)
    assert response.status_code == 503
    # Validate that the structured HTTP exception detail accurately relays the underlying error message
    assert response.json()["detail"] == "Model unavailable"


from pathlib import Path
# TEST 1: Verifies that a valid file path is successfully sent, parsed, and indexed by the system
def test_index_document_success(monkeypatch):
    # Define a stub function that simulates a successful document processing run without doing real file I/O
    def fake_index_document(file_path):
        return {
            "source": Path(file_path).name, # Extracts the file name (e.g., "sample.txt") from the path
            "chunks_indexed": 3,           # Simulates splitting the text into 3 vector database chunks
            "stale_chunks_removed": 0,     # Simulates no old database entries needing cleanup
        }

    # Intercept 'main.index_document' and swap it with our successful mock stub
    monkeypatch.setattr(
        main, "index_document", fake_index_document
    )

    # Dispatch a simulated HTTP POST request to the indexing endpoint with a mock payload
    response = client.post(
        "/documents/index",
        json={"file_path": "sample.txt"},
    )

    # Assert that the API executed successfully and returned an HTTP 200 (OK) status code
    assert response.status_code == 200
    # Confirm that the JSON payload contains a high-level success flag
    assert response.json()["status"] == "success"
    # Verify that the returned dictionary tracks the expected count of processed chunks
    assert response.json()["chunks_indexed"] == 3


# TEST 2: Validates that the endpoint gracefully catches missing files and exposes a clean 404 error
def test_index_document_missing_file(monkeypatch):
    # Define a stub function that throws a standard Python FileNotFoundError when triggered
    def fake_index_document(file_path):
        raise FileNotFoundError("Document not found")

    # Patch the function reference to simulate the core system failing to locate the file on disk
    monkeypatch.setattr(
        main, "index_document", fake_index_document
    )

    # Send a request pointing to a simulated non-existent document file path
    response = client.post(
        "/documents/index",
        json={"file_path": "missing.txt"},
    )

    # Assert that the app's exception handler translates a FileNotFoundError into a formal HTTP 404 (Not Found)
    assert response.status_code == 404
    # Validate that the error message detail matches the message attached to the original exception
    assert response.json()["detail"] == "Document not found"


# TEST 3: Checks that the API actively blocks prohibited or unsupported file extensions
def test_index_document_invalid_file(monkeypatch):
    # Define a stub function that raises a ValueError to signal that the data structure is corrupt or unauthorized
    def fake_index_document(file_path):
        raise ValueError("Unsupported file type")

    # Patch the reference so hitting the endpoint triggers this value constraint exception
    monkeypatch.setattr(
        main, "index_document", fake_index_document
    )

    # Send a path pointing to a dangerous or unparseable file extension (e.g., an executable)
    response = client.post(
        "/documents/index",
        json={"file_path": "document.exe"},
    )

    # Assert that a ValueError within the core pipeline is mapped to a standard HTTP 400 (Bad Request)
    assert response.status_code == 400
    # Confirm that the client receives the precise reason why their document upload payload was rejected
    assert response.json()["detail"] == "Unsupported file type"




import io


def test_upload_txt_success(monkeypatch):
    def fake_index_document(file_path):
        return {
            "source": file_path.name,
            "chunks_indexed": 2,
            "stale_chunks_removed": 0,
        }

    monkeypatch.setattr(
        main, "index_document", fake_index_document
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "notes.txt",
                io.BytesIO(b"RAG retrieves relevant information."),
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["filename"] == "notes.txt"
    assert response.json()["chunks_indexed"] == 2


def test_upload_rejects_unsupported_extension():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "malware.exe",
                io.BytesIO(b"not a supported document"),
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400
    assert "Only PDF and TXT" in response.json()["detail"]


def test_upload_rejects_oversized_file():
    oversized_content = b"x" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "large.txt",
                io.BytesIO(oversized_content),
                "text/plain",
            )
        },
    )

    assert response.status_code == 413
    assert "10 MB" in response.json()["detail"]

