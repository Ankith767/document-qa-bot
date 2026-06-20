import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Path Configuration ---
DB_PATH = "./db"
DATA_PATH = "./data"

# --- ChromaDB Configuration ---
COLLECTION_NAME = "document_knowledge_base"

# --- Chunking Configuration ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval Configuration ---
TOP_K = 3

# --- Model Configuration ---
EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_MODEL = "models/gemini-2.5-flash"