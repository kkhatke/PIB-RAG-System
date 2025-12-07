# PIB RAG System

A Retrieval Augmented Generation (RAG) system for intelligent question-answering about Indian government policies and announcements using Press Information Bureau (PIB) articles.

## Overview

The PIB RAG System combines vector search with large language models to provide accurate, contextual responses with proper source citations. It serves policy researchers, journalists, and citizens by enabling natural language queries about government policies.

### Key Features

- **Semantic Search**: Find relevant articles using natural language queries with vector similarity
- **Ministry Filtering**: Search within specific government ministries
- **Timeline Search**: Query articles within specific date ranges
- **Source Citations**: All responses include proper citations with article metadata
- **Conversational Interface**: Ask follow-up questions with context awareness
- **Local LLM**: Uses Ollama for privacy-preserving, cost-free response generation
- **Content Normalization**: Standardizes text formatting for consistent processing
- **Smart Chunking**: Preserves semantic coherence with paragraph-aware splitting

## Architecture

### High-Level Flow

```
JSON Articles → Data Ingestion → Embedding Generation → Vector Store
                                                              ↓
User Query → Query Engine → Response Generation (Ollama) → Answer + Citations
```

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIB RAG System                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐      ┌──────────────────┐               │
│  │ Data Ingestion   │      │ Content          │               │
│  │ - Article Loader │──────│ Normalization    │               │
│  │ - Validation     │      │ - Whitespace     │               │
│  │ - Deduplication  │      │ - HTML Entities  │               │
│  └────────┬─────────┘      │ - Unicode        │               │
│           │                 └────────┬─────────┘               │
│           │                          │                          │
│           ▼                          ▼                          │
│  ┌──────────────────┐      ┌──────────────────┐               │
│  │ Article Chunker  │      │ Embedding        │               │
│  │ - Smart Splitting│──────│ Generator        │               │
│  │ - Overlap        │      │ (MiniLM-L6-v2)   │               │
│  │ - Metadata       │      └────────┬─────────┘               │
│  └──────────────────┘               │                          │
│                                     ▼                          │
│                          ┌──────────────────┐                  │
│                          │ Vector Store     │                  │
│                          │ (ChromaDB)       │                  │
│                          │ - Embeddings     │                  │
│                          │ - Metadata       │                  │
│                          │ - Filtering      │                  │
│                          └────────┬─────────┘                  │
│                                   │                            │
│  ┌──────────────────┐            │                            │
│  │ Query Engine     │◄───────────┘                            │
│  │ - Semantic Search│                                          │
│  │ - Filtering      │                                          │
│  │ - Ranking        │                                          │
│  └────────┬─────────┘                                          │
│           │                                                    │
│           ▼                                                    │
│  ┌──────────────────┐      ┌──────────────────┐              │
│  │ Response         │      │ Ollama LLM       │              │
│  │ Generator        │──────│ (llama3.2)       │              │
│  │ - Context Format │      │ - Local Inference│              │
│  │ - Citations      │      │ - No API Costs   │              │
│  └────────┬─────────┘      └──────────────────┘              │
│           │                                                    │
│           ▼                                                    │
│  ┌──────────────────┐                                         │
│  │ Conversational   │                                         │
│  │ Interface        │                                         │
│  │ - History        │                                         │
│  │ - Commands       │                                         │
│  └──────────────────┘                                         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended for larger datasets)
- **Disk Space**: 5GB for models and vector database
- **CPU**: Multi-core processor recommended for faster embedding generation
- **Operating System**: Windows, macOS, or Linux

### Required Software

- **Ollama**: For local LLM inference (no API keys needed)
- **Git**: For cloning the repository

## Quick Start

### 1. Install Ollama

Ollama is required for response generation. It runs locally, ensuring privacy and eliminating API costs.

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
1. Download from [https://ollama.com/download](https://ollama.com/download)
2. Run the installer
3. Ollama will start automatically as a Windows service

**Verify Installation:**
```bash
ollama --version
```

### 2. Pull the Language Model

After installing Ollama, pull the required model. We recommend **llama3.2** for best results:

```bash
ollama pull llama3.2
```

**Alternative Models:**
```bash
# Smaller, faster alternative
ollama pull mistral

# Older but stable version
ollama pull llama2
```

**Model Comparison:**

| Model | Size | Speed | Quality | Recommended For |
|-------|------|-------|---------|-----------------|
| llama3.2 | ~2GB | Medium | High | Best overall results |
| mistral | ~4GB | Fast | High | Faster responses |
| llama2 | ~4GB | Medium | Good | Stable, proven |

### 3. Start Ollama Service

Ensure Ollama is running before using the system:

**macOS/Linux:**
```bash
ollama serve
```

**Windows:**
Ollama runs automatically as a service. To verify:
```bash
curl http://localhost:11434/api/tags
```

You should see a JSON response listing available models.

**Troubleshooting:**
If Ollama isn't responding:
- Check if the service is running: `ps aux | grep ollama` (macOS/Linux)
- Restart the service: `ollama serve`
- Check firewall settings (port 11434 should be accessible)

### 4. Set Up Python Environment

**Clone the Repository:**
```bash
git clone <repository-url>
cd pib-rag-system
```

**Create Virtual Environment:**

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Verify Python Version:**
```bash
python --version  # Should be 3.8 or higher
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

**Installed Packages:**

| Package | Purpose | Version |
|---------|---------|---------|
| `langchain` | RAG orchestration framework | Latest |
| `langchain-community` | Community integrations (Ollama) | Latest |
| `chromadb` | Vector database | Latest |
| `sentence-transformers` | Embedding generation | Latest |
| `ollama` | Ollama Python client | Latest |
| `hypothesis` | Property-based testing | Latest |
| `pytest` | Unit testing framework | Latest |

**Verify Installation:**
```bash
python verify_setup.py
```

This script checks:
- ✅ Python version
- ✅ All required packages
- ✅ Ollama connectivity
- ✅ Model availability

### 6. Configure Environment Variables (Optional)

Copy the example environment file:

```bash
cp .env.example .env
```

**Configuration Options:**

Edit `.env` to customize settings:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434  # Change if Ollama runs elsewhere
OLLAMA_MODEL=llama3.2                   # Change to mistral or llama2

# Query Configuration
DEFAULT_TOP_K=5                         # Number of results to retrieve
DEFAULT_RELEVANCE_THRESHOLD=0.5         # Minimum similarity score

# Chunking Configuration
CHUNK_SIZE=1000                         # Maximum chunk size in characters
CHUNK_OVERLAP=200                       # Overlap between chunks

# Logging
LOG_LEVEL=INFO                          # DEBUG, INFO, WARNING, ERROR
```

**Note:** Most users can use the default settings without creating a `.env` file.

## Configuration

The system is configured via `config.py` and optional `.env` file.

### Default Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| **Embedding Model** | `all-MiniLM-L6-v2` | Sentence transformer model (384 dimensions) |
| **Chunk Size** | 1000 characters | Maximum chunk size for article splitting |
| **Chunk Overlap** | 200 characters | Overlap between consecutive chunks |
| **Top-K Results** | 5 | Number of most relevant chunks to retrieve |
| **Relevance Threshold** | 0.5 | Minimum similarity score (0.0-1.0) |
| **Ollama Base URL** | `http://localhost:11434` | Ollama API endpoint |
| **Ollama Model** | `llama3.2` | Language model for response generation |
| **Max Conversation History** | 10 messages | Maximum messages to keep in context |

### Customizing Configuration

Edit `config.py` directly or use environment variables in `.env`:

```python
# config.py
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
```

## Usage

### Step 1: Data Ingestion

Before using the system, ingest PIB articles into the vector database:

**Basic Usage:**
```bash
python ingest_articles.py
```

**With Options:**
```bash
# Specify input file
python ingest_articles.py data/all_articles.json

# Reset vector store (clear existing data)
python ingest_articles.py --reset

# Custom chunking parameters
python ingest_articles.py --chunk-size 1200 --chunk-overlap 250

# Batch processing for large files
python ingest_articles.py --batch-size 64
```

**What Happens During Ingestion:**
1. ✅ Loads articles from JSON file
2. ✅ Normalizes content (whitespace, HTML entities, Unicode)
3. ✅ Deduplicates based on article ID
4. ✅ Chunks long articles with paragraph awareness
5. ✅ Generates vector embeddings
6. ✅ Stores in ChromaDB with metadata

**Expected Output:**
```
Loading articles from data/all_articles.json...
Loaded 1000 articles
Deduplicating articles...
After deduplication: 995 unique articles
Normalizing content...
Chunking articles...
Created 2500 chunks from 995 articles
Generating embeddings in batches...
Batch 1/79: Processing 32 chunks...
...
Successfully stored 2500 chunks in vector store
Ingestion complete in 45.2 seconds
```

### Step 2: Running the System

Start the conversational interface:

```bash
python main.py
```

**Startup Sequence:**
```
Initializing PIB RAG System...
✓ Embedding model loaded: all-MiniLM-L6-v2
✓ Vector store connected: 2500 chunks available
✓ Ollama connected: llama3.2 model ready
✓ Query engine initialized
✓ Response generator ready
✓ Conversational interface started

Type 'help' for available commands or ask a question.
>
```

### Step 3: Asking Questions

**Natural Language Queries:**
```
> What are the recent healthcare initiatives?

Searching for relevant articles...

================================================================================
ANSWER:
Based on recent PIB articles, the government has launched several healthcare 
initiatives including the expansion of Ayushman Bharat coverage to include 
senior citizens above 70 years, establishment of 50 new medical colleges in 
underserved districts, and the National Digital Health Mission rollout...

SOURCES:
1. Ayushman Bharat Expansion Announced
   Ministry: Ministry of Health and Family Welfare
   Date: 2025-01-15
   Article ID: 2025011501
   Relevance: 0.89

2. New Medical Colleges to be Established
   Ministry: Ministry of Health and Family Welfare
   Date: 2025-01-10
   Article ID: 2025011002
   Relevance: 0.85
================================================================================

> Tell me more about the medical colleges
```

### Available Commands

**Information Commands:**
```bash
> help              # Show all available commands
> ministries        # List all government ministries in database
> status            # Show system status and active filters
```

**Filter Commands:**
```bash
# Filter by ministry
> filter ministry Ministry of Health and Family Welfare
> What are the recent announcements?

# Filter by date range
> filter date 2024-01-01 to 2024-12-31
> What policies were announced?

# Combine filters
> filter ministry Ministry of Finance
> filter date 2024-06-01 to 2024-12-31
> What are the budget allocations?

# Clear filters
> filter clear
```

**Control Commands:**
```bash
> clear             # Clear conversation history
> reset             # Reset conversation and filters
> exit              # Exit the application
> quit              # Exit the application
> q                 # Exit the application
```

### Example Session

```
> What are the education policies announced in 2024?

Searching for relevant articles...

[Response with citations]

> filter ministry Ministry of Education
✓ Ministry filter set: Ministry of Education

> What about digital education initiatives?

Searching for relevant articles...
Applying filters: Ministry of Education

[Filtered response]

> ministries

Available Ministries:
1. Ministry of Health and Family Welfare
2. Ministry of Education
3. Ministry of Finance
4. Ministry of Home Affairs
...

> filter clear
✓ All filters cleared

> exit
Goodbye!
```

## Project Structure

```
pib-rag-system/
├── src/
│   ├── data_ingestion/      # Article loading and validation
│   ├── embedding/           # Vector embedding generation
│   ├── vector_store/        # ChromaDB integration
│   ├── query_engine/        # Search and retrieval
│   ├── response_generation/ # LLM response generation
│   └── interface/           # Conversational interface
├── data/                    # Input JSON files
├── vector_store/            # Persistent vector database
├── tests/                   # Unit and property-based tests
├── config.py               # System configuration
├── main.py                 # Application entry point
├── ingest_articles.py      # Data ingestion script
└── requirements.txt        # Python dependencies
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run property-based tests only
pytest -k "property"
```

## Troubleshooting

### Common Ollama Issues

#### 1. Ollama Connection Failed

**Error Message:**
```
Failed to connect to Ollama at http://localhost:11434
Error: Connection refused
```

**Possible Causes:**
- Ollama service is not running
- Ollama is running on a different port
- Firewall blocking the connection

**Solutions:**

**Check if Ollama is running:**
```bash
# Test connection
curl http://localhost:11434/api/tags

# If no response, start Ollama
ollama serve
```

**Windows Users:**
- Ollama should run automatically as a service
- Check Services app (Win+R → `services.msc`)
- Look for "Ollama" service and ensure it's running

**macOS/Linux Users:**
```bash
# Start Ollama in background
ollama serve &

# Or use systemd (Linux)
sudo systemctl start ollama
```

**Custom Port:**
If Ollama runs on a different port, update `.env`:
```env
OLLAMA_BASE_URL=http://localhost:YOUR_PORT
```

#### 2. Model Not Found

**Error Message:**
```
Model 'llama3.2' not found
Available models: []
```

**Solution:**
```bash
# Pull the required model
ollama pull llama3.2

# Verify model is available
ollama list

# Expected output:
# NAME            ID              SIZE    MODIFIED
# llama3.2:latest abc123def456    2.0 GB  2 days ago
```

**Alternative Models:**
```bash
# If llama3.2 fails, try mistral
ollama pull mistral

# Update config to use mistral
# Edit .env: OLLAMA_MODEL=mistral
```

#### 3. Slow Response Times

**Issue:** Responses take 30+ seconds

**Diagnostic:**
```bash
# Test Ollama directly
time ollama run llama3.2 "Hello"
```

**Solutions:**

**Use a Faster Model:**
```bash
ollama pull mistral  # Generally faster than llama3.2
```

Update `.env`:
```env
OLLAMA_MODEL=mistral
```

**Reduce Context Size:**
Edit `config.py`:
```python
DEFAULT_TOP_K = 3  # Reduce from 5 to 3
MAX_CONVERSATION_HISTORY = 5  # Reduce from 10
```

**Hardware Acceleration:**
- Ensure GPU drivers are up to date (if using GPU)
- Check Ollama is using GPU: `ollama ps` should show GPU usage

**System Resources:**
```bash
# Check CPU/memory usage
top  # Linux/macOS
# Task Manager on Windows

# Close unnecessary applications
```

#### 4. Ollama Timeout

**Error Message:**
```
Request to Ollama timed out after 120 seconds
```

**Solutions:**

**Increase Timeout:**
Edit `config.py`:
```python
OLLAMA_TIMEOUT = 300  # Increase from 120 to 300 seconds
```

**Check System Load:**
- Ollama may be slow if system is under heavy load
- Close other applications
- Restart Ollama service

#### 5. Context Too Long

**Error Message:**
```
Context length exceeded: 4096 tokens
```

**Solution:**
Edit `config.py`:
```python
DEFAULT_TOP_K = 3  # Reduce number of retrieved chunks
CHUNK_SIZE = 800   # Reduce chunk size
```

### Data Ingestion Issues

#### 1. File Not Found

**Error Message:**
```
FileNotFoundError: data/all_articles.json not found
```

**Solution:**
```bash
# Check if file exists
ls data/all_articles.json

# If missing, ensure you have the data file
# Place your JSON file in the data/ directory
```

#### 2. Invalid JSON Format

**Error Message:**
```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Solution:**
- Verify JSON file is valid: Use [jsonlint.com](https://jsonlint.com)
- Check file encoding (should be UTF-8)
- Ensure file is not empty or corrupted

#### 3. Memory Issues During Ingestion

**Error Message:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

**Reduce Batch Size:**
```bash
python ingest_articles.py --batch-size 16  # Reduce from default 32
```

**Process in Chunks:**
Split large JSON files into smaller files:
```python
# split_json.py
import json

with open('data/all_articles.json') as f:
    articles = json.load(f)

chunk_size = 1000
for i in range(0, len(articles), chunk_size):
    chunk = articles[i:i+chunk_size]
    with open(f'data/articles_part_{i//chunk_size}.json', 'w') as f:
        json.dump(chunk, f)
```

Then ingest each file separately:
```bash
python ingest_articles.py data/articles_part_0.json
python ingest_articles.py data/articles_part_1.json
```

#### 4. Empty Vector Store

**Error Message:**
```
Vector store connected: 0 chunks available
```

**Solution:**
```bash
# Run data ingestion first
python ingest_articles.py

# Verify ingestion succeeded
# Should see: "Successfully stored X chunks in vector store"
```

### Embedding Model Issues

#### 1. Model Download Fails

**Error Message:**
```
OSError: Can't load model 'all-MiniLM-L6-v2'
```

**Solutions:**

**Check Internet Connection:**
- Model downloads automatically on first use (~90MB)
- Ensure stable internet connection

**Manual Download:**
```python
from sentence_transformers import SentenceTransformer

# This will download the model
model = SentenceTransformer('all-MiniLM-L6-v2')
```

**Check Disk Space:**
```bash
df -h  # Linux/macOS
# Ensure at least 1GB free space
```

#### 2. CUDA/GPU Errors

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
Force CPU usage in `config.py`:
```python
import torch
DEVICE = 'cpu'  # Force CPU instead of GPU
```

### Query Issues

#### 1. No Results Found

**Issue:** Query returns no results

**Diagnostic Steps:**

**Check Vector Store:**
```python
from src.vector_store.vector_store import VectorStore

store = VectorStore()
print(f"Total chunks: {store.count()}")
print(f"Ministries: {store.get_unique_ministries()}")
```

**Adjust Relevance Threshold:**
Edit `config.py`:
```python
DEFAULT_RELEVANCE_THRESHOLD = 0.3  # Lower from 0.5
```

**Clear Filters:**
```
> filter clear
> Try your query again
```

#### 2. Irrelevant Results

**Issue:** Results don't match the query

**Solutions:**

**Increase Relevance Threshold:**
```python
DEFAULT_RELEVANCE_THRESHOLD = 0.7  # Increase from 0.5
```

**Use More Specific Queries:**
```
# Instead of: "health"
# Try: "What are the recent healthcare initiatives for rural areas?"
```

**Apply Filters:**
```
> filter ministry Ministry of Health and Family Welfare
> filter date 2024-01-01 to 2024-12-31
> Your specific query
```

### Performance Optimization

#### Slow Startup

**Issue:** Application takes long to start

**Solutions:**

**First Run:**
- Embedding model downloads on first use (~90MB)
- Subsequent runs will be faster

**Reduce Vector Store Size:**
```bash
# Re-ingest with larger chunks
python ingest_articles.py --reset --chunk-size 1500
```

**Use SSD:**
- Move `vector_store/` directory to SSD
- Update path in `config.py`

#### High Memory Usage

**Issue:** System uses too much RAM

**Solutions:**

**Reduce Batch Size:**
```python
# In config.py
EMBEDDING_BATCH_SIZE = 16  # Reduce from 32
```

**Limit Conversation History:**
```python
MAX_CONVERSATION_HISTORY = 5  # Reduce from 10
```

**Close Other Applications:**
- Free up system memory
- Restart the application

### Getting More Help

If issues persist:

1. **Check Logs:**
   ```bash
   tail -f pib_rag.log  # View real-time logs
   ```

2. **Enable Debug Mode:**
   ```env
   LOG_LEVEL=DEBUG
   ```

3. **Verify Setup:**
   ```bash
   python verify_setup.py
   ```

4. **Ollama Documentation:**
   - [https://ollama.com/docs](https://ollama.com/docs)
   - [https://github.com/ollama/ollama](https://github.com/ollama/ollama)

5. **Test Components Individually:**
   ```bash
   # Test embedding generation
   python -c "from src.embedding.embedding_generator import EmbeddingGenerator; eg = EmbeddingGenerator(); print(len(eg.generate_embedding('test')))"
   
   # Test vector store
   python -c "from src.vector_store.vector_store import VectorStore; vs = VectorStore(); print(vs.count())"
   
   # Test Ollama
   curl http://localhost:11434/api/tags
   ```

## Common Query Examples

### Policy Research Queries

```
> What are the recent healthcare initiatives announced by the government?

> Tell me about the National Education Policy implementation

> What infrastructure projects were announced in the last quarter?

> Show me all announcements related to digital India initiatives

> What are the government's plans for renewable energy?
```

### Ministry-Specific Queries

```
> filter ministry Ministry of Health and Family Welfare
> What are the recent announcements?

> filter ministry Ministry of Finance
> What are the budget allocations for education?

> filter ministry Ministry of External Affairs
> What are the recent diplomatic initiatives?
```

### Timeline-Based Queries

```
> filter date 2024-01-01 to 2024-03-31
> What policies were announced in Q1 2024?

> What announcements were made in the last 30 days?

> filter date 2024-06-01 to 2024-06-30
> Show me all June 2024 announcements

> What were the major policy changes in 2024?
```

### Combined Filter Queries

```
> filter ministry Ministry of Health and Family Welfare
> filter date 2024-01-01 to 2024-12-31
> What healthcare policies were announced in 2024?

> filter ministry Ministry of Education
> filter date 2024-04-01 to 2024-06-30
> What education initiatives were launched in Q1?
```

### Follow-up Questions

```
> What are the recent healthcare initiatives?
[System provides answer]

> Tell me more about the Ayushman Bharat program
[System uses conversation context]

> Which ministry announced this?
[System refers to previous context]

> When was this announced?
[System provides date from context]
```

### Expected Response Format

```
================================================================================
ANSWER:
[Comprehensive answer synthesized from retrieved articles, 
 written in natural language with proper context and details]

SOURCES:
1. [Article Title]
   Ministry: [Ministry Name]
   Date: [YYYY-MM-DD]
   Article ID: [ID]
   Relevance: [0.XX]

2. [Article Title]
   Ministry: [Ministry Name]
   Date: [YYYY-MM-DD]
   Article ID: [ID]
   Relevance: [0.XX]
================================================================================
```

## Development

### Running Tests

The system includes comprehensive test coverage with unit tests and property-based tests.

**Run All Tests:**
```bash
pytest
```

**Run Specific Test Categories:**
```bash
# Unit tests only
pytest tests/test_*.py -v

# Property-based tests only
pytest -k "property" -v

# Integration tests (requires Ollama running)
pytest tests/test_integration.py -v

# Specific component tests
pytest tests/test_query_engine.py -v
pytest tests/test_response_generator.py -v
```

**Run with Coverage:**
```bash
# Generate coverage report
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

**Property-Based Testing:**

The system uses Hypothesis for property-based testing to verify correctness properties:

```bash
# Run property tests with verbose output
pytest -k "property" -v --hypothesis-show-statistics

# Run with more iterations (default is 100)
pytest -k "property" --hypothesis-max-examples=1000
```

### Code Quality

**Format Code:**
```bash
# Format with black
black src/ tests/

# Check formatting without changes
black --check src/ tests/
```

**Lint Code:**
```bash
# Lint with flake8
flake8 src/ tests/

# With specific rules
flake8 src/ --max-line-length=100 --ignore=E203,W503
```

**Type Checking:**
```bash
# Type check with mypy
mypy src/

# Strict mode
mypy src/ --strict
```

### Project Structure

```
pib-rag-system/
├── .kiro/
│   └── specs/
│       └── pib-rag-system/      # Specification documents
│           ├── requirements.md   # System requirements
│           ├── design.md         # Design document
│           └── tasks.md          # Implementation tasks
├── src/
│   ├── data_ingestion/          # Article loading and validation
│   │   ├── article_loader.py    # JSON parsing and validation
│   │   ├── article_chunker.py   # Text chunking logic
│   │   └── content_normalizer.py # Content standardization
│   ├── embedding/               # Vector embedding generation
│   │   └── embedding_generator.py
│   ├── vector_store/            # ChromaDB integration
│   │   └── vector_store.py
│   ├── query_engine/            # Search and retrieval
│   │   └── query_engine.py
│   ├── response_generation/     # LLM response generation
│   │   └── response_generator.py
│   └── interface/               # Conversational interface
│       └── conversational_interface.py
├── tests/                       # Test suite
│   ├── test_article_loader.py
│   ├── test_article_chunker.py
│   ├── test_content_normalizer.py
│   ├── test_embedding_generator.py
│   ├── test_vector_store.py
│   ├── test_query_engine.py
│   ├── test_response_generator.py
│   ├── test_conversational_interface.py
│   └── test_integration.py
├── data/                        # Input data files
│   └── all_articles.json        # PIB articles
├── vector_store/                # Persistent vector database
├── config.py                    # System configuration
├── main.py                      # Application entry point
├── ingest_articles.py           # Data ingestion script
├── requirements.txt             # Python dependencies
├── .env.example                 # Example environment variables
├── README.md                    # This file
├── QUICKSTART.md                # Quick start guide
├── USAGE.md                     # Detailed usage guide
├── INGESTION_README.md          # Data ingestion guide
└── MAIN_APP_REFERENCE.md        # Main application reference
```

### Adding New Features

**1. Add New Filter Type:**

Edit `src/query_engine/query_engine.py`:
```python
def search(self, query: str, custom_filter: Optional[str] = None):
    filters = {}
    if custom_filter:
        filters['custom_field'] = custom_filter
    # ... rest of implementation
```

**2. Add New Command:**

Edit `main.py`:
```python
elif user_input.lower().startswith('mycommand'):
    self._handle_my_command(user_input)
    continue
```

**3. Customize Response Format:**

Edit `src/response_generation/response_generator.py`:
```python
def format_context(self, search_results: List[SearchResult]) -> str:
    # Customize how context is formatted for the LLM
    pass
```

## System Requirements

### Minimum Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB
- **Disk Space**: 5GB (2GB for Ollama model, 3GB for vector database and embeddings)
- **CPU**: Dual-core processor
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)

### Recommended Requirements

- **Python**: 3.10 or higher
- **RAM**: 16GB
- **Disk Space**: 10GB SSD
- **CPU**: Quad-core processor or better
- **GPU**: Optional, but improves Ollama performance
- **Operating System**: Latest stable version

### Network Requirements

- **Internet**: Required for initial setup (downloading models and dependencies)
- **Firewall**: Port 11434 must be accessible for Ollama
- **Offline Usage**: System works offline after initial setup

## Performance Benchmarks

### Typical Performance Metrics

| Operation | Time | Notes |
|-----------|------|-------|
| **Startup** | 2-5 seconds | After initial model download |
| **Query Processing** | 3-8 seconds | Depends on model and context size |
| **Embedding Generation** | 50-100ms | Per chunk |
| **Vector Search** | 50-200ms | For 10,000 chunks |
| **Data Ingestion** | 30-60 seconds | Per 1,000 articles |

### Scaling Considerations

- **10,000 articles**: ~25,000 chunks, ~500MB vector store, queries in <5 seconds
- **100,000 articles**: ~250,000 chunks, ~5GB vector store, queries in <10 seconds
- **1,000,000 articles**: Consider distributed vector store (e.g., Pinecone, Weaviate)

## Security Considerations

### Data Privacy

- **Local Processing**: All data stays on your machine
- **No External APIs**: Ollama runs locally, no data sent to external services
- **No Telemetry**: System doesn't collect or transmit usage data

### Access Control

- **File System**: Vector store and data files use standard file permissions
- **Network**: Ollama API is local-only by default (localhost:11434)
- **Authentication**: No built-in authentication (command-line interface)

### Best Practices

1. **Sensitive Data**: Don't ingest sensitive/classified information without proper security measures
2. **Network Exposure**: Don't expose Ollama port (11434) to public networks
3. **File Permissions**: Restrict access to vector store directory
4. **Regular Updates**: Keep Ollama and dependencies updated

## License

[Your License Here]

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the Repository**
2. **Create a Feature Branch**: `git checkout -b feature/your-feature`
3. **Write Tests**: Ensure new features have test coverage
4. **Follow Code Style**: Use black for formatting, flake8 for linting
5. **Update Documentation**: Update README and relevant docs
6. **Submit Pull Request**: With clear description of changes

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/pib-rag-system.git
cd pib-rag-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements.txt
pip install black flake8 mypy pytest-cov

# Run tests
pytest

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

## Support

### Getting Help

1. **Documentation**: Check README, QUICKSTART, and USAGE guides
2. **Troubleshooting**: Review the troubleshooting section above
3. **Logs**: Check `pib_rag.log` for detailed error messages
4. **Verify Setup**: Run `python verify_setup.py`

### External Resources

- **Ollama Documentation**: [https://ollama.com/docs](https://ollama.com/docs)
- **Ollama GitHub**: [https://github.com/ollama/ollama](https://github.com/ollama/ollama)
- **LangChain Documentation**: [https://python.langchain.com/docs](https://python.langchain.com/docs)
- **ChromaDB Documentation**: [https://docs.trychroma.com](https://docs.trychroma.com)
- **Sentence Transformers**: [https://www.sbert.net](https://www.sbert.net)

### Reporting Issues

When reporting issues, please include:

1. **System Information**: OS, Python version, RAM
2. **Error Messages**: Full error traceback
3. **Steps to Reproduce**: Detailed steps to reproduce the issue
4. **Configuration**: Relevant config.py or .env settings
5. **Logs**: Relevant portions of pib_rag.log

## Acknowledgments

This project builds upon excellent open-source tools and libraries:

- **Press Information Bureau (PIB)**: For providing government policy information
- **Ollama**: For making local LLM inference accessible and easy
- **LangChain**: For RAG orchestration and LLM integration
- **ChromaDB**: For efficient vector storage and retrieval
- **Sentence Transformers**: For high-quality text embeddings
- **Hypothesis**: For property-based testing framework

## Citation

If you use this system in your research or project, please cite:

```bibtex
@software{pib_rag_system,
  title = {PIB RAG System: Retrieval Augmented Generation for Indian Government Policies},
  author = {[Your Name]},
  year = {2025},
  url = {[Repository URL]}
}
```

## Changelog

### Version 1.0.0 (2025-01-XX)

- ✅ Initial release
- ✅ Complete RAG pipeline implementation
- ✅ Ollama integration for local LLM inference
- ✅ Content normalization and smart chunking
- ✅ Ministry and date filtering
- ✅ Conversational interface with history
- ✅ Comprehensive test suite with property-based tests
- ✅ Full documentation and troubleshooting guides

## Roadmap

### Planned Features

- [ ] Web interface (Flask/FastAPI)
- [ ] REST API endpoints
- [ ] Multi-user support with authentication
- [ ] Query history persistence
- [ ] Export results to PDF/Word
- [ ] Advanced filtering (boolean queries, regex)
- [ ] Semantic search refinement with user feedback
- [ ] Custom prompt templates
- [ ] Support for additional LLM providers
- [ ] Distributed vector store support
- [ ] Real-time article updates
- [ ] Analytics dashboard

### Future Enhancements

- [ ] Multi-language support
- [ ] Voice interface
- [ ] Mobile application
- [ ] Browser extension
- [ ] Slack/Teams integration
- [ ] Email digest of new policies
- [ ] Automated policy summaries
- [ ] Trend analysis and visualization
