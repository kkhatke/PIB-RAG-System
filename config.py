"""
Configuration file for PIB RAG System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

# Embedding configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Chunking configuration
CHUNK_SIZE = 1000  # tokens
CHUNK_OVERLAP = 200  # tokens

# Vector store configuration
VECTOR_STORE_COLLECTION = "pib_articles"
VECTOR_STORE_PERSIST = True

# Query engine configuration
DEFAULT_TOP_K = 5
DEFAULT_RELEVANCE_THRESHOLD = 0.5
RECENT_DAYS = 30  # For "recent" queries

# LLM configuration (Ollama)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")  # Default model (llama3.2 or mistral)
OLLAMA_TIMEOUT = 120  # seconds

# Response generation configuration
MAX_CONTEXT_LENGTH = 4000  # tokens
MAX_RESPONSE_LENGTH = 1000  # tokens

# Conversation configuration
MAX_CONVERSATION_HISTORY = 10  # messages
CONVERSATION_SUMMARY_THRESHOLD = 8  # messages before summarization

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2  # exponential backoff multiplier

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "pib_rag.log"

# Create necessary directories
VECTOR_STORE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
