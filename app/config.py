# Sabse pehle hum config.py banayenge. Iska purpose hai project ki settings ko ek jagah rakhna, taaki baad mein database path ya model change karne par multiple files edit na karni padein.

from pathlib import Path
#Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

#Directory for sample and uploaded documents
DATA_DIR = BASE_DIR / "data"

#Directory for the local ChromaDB database
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

#ChromaDB collection name
COLLECTION_NAME = "research_documents"

#Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#Local LLM Configuration
LLM_MODEL = "qwen2.5:3b"

#Number of chunks to retrieve for a question
TOP_K = 3

