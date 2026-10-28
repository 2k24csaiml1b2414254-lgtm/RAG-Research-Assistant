# 📚 RAG Research Assistant

An AI-powered research assistant that answers questions using your own PDF and TXT documents. Built with Retrieval-Augmented Generation (RAG), semantic search, a local LLM, and source references.

## 🚀 Overview

RAG Research Assistant allows users to upload research documents, retrieve relevant information through semantic search, and generate context-grounded answers using a locally hosted Large Language Model (LLM).

The application combines a **FastAPI backend**, **Streamlit frontend**, **ChromaDB vector database**, and **Ollama** to provide an end-to-end document question-answering experience.

## ✨ Features

- **Document ingestion:** Upload PDF and TXT documents.
- **Text chunking:** Split extracted text into smaller chunks for retrieval.
- **Semantic embeddings:** Convert text into vector representations using Sentence Transformers.
- **Vector storage:** Persist document embeddings in ChromaDB.
- **Semantic retrieval:** Retrieve relevant document chunks for a user query.
- **LLM-powered answers:** Generate context-grounded responses using Qwen 2.5 3B through Ollama.
- **Source references:** Return source filenames, chunk indices, chunk IDs, and retrieval distances.
- **REST API:** Expose functionality through FastAPI endpoints.
- **Interactive UI:** Upload documents and ask questions through Streamlit.
- **Automated testing:** Test core modules, RAG orchestration, and API behavior using pytest.

## 🏗️ Architecture

```text
                 ┌──────────────────────┐
                 │    Streamlit UI      │
                 │ Upload & Ask Questions│
                 └──────────┬───────────┘
                            │ HTTP
                 ┌──────────▼───────────┐
                 │      FastAPI         │
                 │  Upload / Ask / Health│
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │    RAG Pipeline      │
                 └──────────┬───────────┘
                            │
                ┌───────────▼────────────┐
                │   Semantic Retrieval   │
                │       ChromaDB         │
                └───────────┬────────────┘
                            │
                 ┌──────────▼───────────┐
                 │  Retrieved Context   │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │  Ollama + Qwen 2.5   │
                 │   Answer Generation  │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │ Answer + References  │
                 └──────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Frontend | Streamlit |
| Backend | FastAPI |
| Embeddings | Sentence Transformers |
| Vector database | ChromaDB |
| LLM runtime | Ollama |
| Language model | Qwen 2.5 3B |
| Document processing | pypdf |
| Testing | pytest |
| API server | Uvicorn |

## 📁 Project Structure

```text
RAG-Research-Assistant/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── indexing.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── rag_pipeline.py
│   └── main.py
├── frontend/
│   └── streamlit_app.py
├── scripts/
│   ├── index_document.py
│   └── search_documents.py
├── tests/
│   ├── test_ingestion.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_indexing.py
│   ├── test_retrieval.py
│   ├── test_rag_pipeline.py
│   └── test_api.py
├── data/
│   └── sample_documents/
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Getting Started

### 1. Prerequisites

Install the following:

- Python 3.12
- Git
- Ollama
- Qwen 2.5 3B model

Download Ollama from [ollama.com](https://ollama.com/).

### 2. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RAG-Research-Assistant
```

Replace the placeholder with your actual GitHub repository URL.

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the language model

```bash
ollama pull qwen2.5:3b
```

Make sure Ollama is running and the model is available locally.

### 5. Start the FastAPI backend

From the project root:

```bash
python -m uvicorn app.main:app --reload
```

The API will be available at:

- API: `http://127.0.0.1:8000`
- Interactive documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### 6. Start the Streamlit frontend

Open a second terminal from the project root:

```bash
python -m streamlit run frontend/streamlit_app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check API health |
| POST | `/ask` | Ask a question about indexed documents |
| POST | `/documents/upload` | Upload and index a PDF/TXT document |
| POST | `/documents/index` | Index a document using a server-side file path |

### Example: Ask a Question

**Request — `POST /ask`**

```json
{
  "question": "What is Retrieval-Augmented Generation?",
  "k": 3
}
```

The response includes the generated answer, source filenames, and citation metadata.

## 🧪 Running Tests

Run the test suite from the project root:

```bash
python -m pytest tests/ -v
```

The latest reported test run before the Streamlit frontend addition had **38 passing tests**. Run the suite again after any code changes to verify the current state.

## 🔍 How RAG Works

1. **Ingestion:** Extract text from an uploaded PDF or TXT document.
2. **Chunking:** Split the extracted text into smaller pieces.
3. **Embedding:** Convert each chunk into a vector representation.
4. **Indexing:** Store vectors, text, and metadata in ChromaDB.
5. **Retrieval:** Embed the user's question and retrieve relevant chunks.
6. **Generation:** Send the retrieved context and question to the local LLM.
7. **Response:** Return the generated answer with source references.

## ⚠️ Current Limitations

- Answer quality depends on document quality, retrieval relevance, and LLM behavior.
- Source references identify retrieved chunks but do not independently verify every claim in the generated answer.
- Evaluation benchmarks, systematic hallucination measurement, MLflow experiment tracking, and production monitoring are not yet implemented.
- The current application is intended for local development; production deployment requires additional security and operational hardening.

## 🛣️ Future Improvements

- RAG evaluation datasets and retrieval-quality metrics.
- MLflow experiment tracking and configuration comparison.
- DVC-based dataset versioning and reproducible evaluation.
- API latency, error, and quality monitoring.
- Improved document management and retrieval filtering.
- Automated CI/CD and deployment.
- Security hardening and production deployment.

## 👨‍💻 Author

**Himesh Mishra**


---

*Built as a hands-on project to explore Retrieval-Augmented Generation, semantic search, LLM integration, API development, and software testing.*
