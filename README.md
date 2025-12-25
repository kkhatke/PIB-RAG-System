# PIB RAG System

A Retrieval Augmented Generation (RAG) system for intelligent question-answering about Indian government policies and announcements using Press Information Bureau (PIB) articles.

## Overview

The PIB RAG System combines vector search with large language models to provide accurate, contextual responses with proper source citations. It serves policy researchers, journalists, and citizens by enabling natural language queries about government policies.

### Key Features

- **🌐 Web Interface**: User-friendly Streamlit web application with interactive filters
- **🔍 Semantic Search**: Find relevant articles using natural language queries
- **🏛️ Ministry Filtering**: Search within specific government ministries
- **📅 Timeline Search**: Query articles within specific date ranges
- **📝 Source Citations**: All responses include proper citations with article metadata
- **💬 Conversational Interface**: Ask follow-up questions with context awareness
- **🔒 Local LLM**: Uses Ollama for privacy-preserving, cost-free response generation
- **⚡ Model Caching**: Faster startup times with cached embedding models

## Quick Start

### 1. Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from [https://ollama.com/download](https://ollama.com/download)

### 2. Pull Language Model

```bash
ollama pull llama3.2
```

### 3. Start Ollama Service

```bash
ollama serve
```

### 4. Set Up Python Environment

```bash
# Clone repository
git clone <repository-url>
cd pib-rag-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 5. Prepare Data

```bash
# Ingest PIB articles into vector database
python ingest_articles.py
```

### 6. Start Using the System

**Option A: Web Interface (Recommended)**
```bash
streamlit run streamlit_app.py
```
Then open `http://localhost:8501` in your browser.

**Option B: Command-Line Interface**
```bash
python main.py
```

## Usage Examples

### Web Interface
1. Open `http://localhost:8501` in your browser
2. Use sidebar filters to narrow your search:
   - **Ministry**: Select specific government ministry
   - **Date Range**: Choose time period or custom dates
   - **Max Articles**: Set number of results (1-20)
3. Enter your query: "What are the recent healthcare initiatives?"
4. View formatted response with expandable citations
5. Ask follow-up questions to continue the conversation

### Command-Line Interface
```
> What are the recent healthcare initiatives?

[System provides detailed response with citations]

> filter ministry Ministry of Health and Family Welfare
> What about digital health programs?

[Filtered response about digital health programs]

> help                    # Show available commands
> ministries             # List all ministries
> clear                  # Clear conversation history
> exit                   # Exit application
```

## Project Structure

```
pib-rag-system/
├── src/                     # Source code modules
│   ├── data_ingestion/      # Article loading and validation
│   ├── embedding/           # Vector embedding generation
│   ├── vector_store/        # ChromaDB integration
│   ├── query_engine/        # Search and retrieval
│   ├── response_generation/ # LLM response generation
│   └── interface/           # User interfaces
├── data/                    # Input JSON files
├── tests/                   # Test suite
├── streamlit_app.py         # Web interface entry point
├── main.py                  # CLI entry point
├── ingest_articles.py       # Data ingestion script
├── config.py               # System configuration
└── requirements.txt        # Python dependencies
```

## Documentation

- **[Configuration Guide](CONFIGURATION.md)** - Detailed configuration options and environment variables
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions
- **[Quick Start Guide](QUICKSTART.md)** - Step-by-step setup instructions
- **[Usage Guide](USAGE.md)** - Detailed usage examples and commands

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB
- **Disk Space**: 5GB (2GB for Ollama model, 3GB for vector database)
- **CPU**: Dual-core processor

### Recommended Requirements
- **Python**: 3.10 or higher
- **RAM**: 16GB
- **Disk Space**: 10GB SSD
- **CPU**: Quad-core processor or better

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run property-based tests
pytest -k "property"
```

## Configuration

Basic configuration via environment variables:

```env
# .env file
OLLAMA_MODEL=llama3.2                   # Language model
DEFAULT_TOP_K=5                         # Number of results
MODEL_CACHE_DIR=./model_cache          # Model cache directory
STREAMLIT_SERVER_PORT=8501             # Web interface port
```

See [CONFIGURATION.md](CONFIGURATION.md) for complete configuration options.

## Troubleshooting

**Common Issues:**

- **Streamlit won't start**: `pip install streamlit`
- **Port in use**: `streamlit run streamlit_app.py --server.port 8502`
- **Ollama connection failed**: `ollama serve`
- **Model not found**: `ollama pull llama3.2`
- **No search results**: Lower relevance threshold in config
- **Empty vector store**: Run `python ingest_articles.py`

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.

## Architecture

```
JSON Articles → Data Ingestion → Embedding Generation → Vector Store
                                                              ↓
User Query → Query Engine → Response Generation (Ollama) → Answer + Citations
```

The system uses:
- **ChromaDB** for vector storage and similarity search
- **Sentence Transformers** for embedding generation (`all-MiniLM-L6-v2`)
- **Ollama** for local LLM inference (llama3.2, mistral, or llama2)
- **Streamlit** for web interface
- **LangChain** for RAG orchestration

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Write tests for new features
4. Follow code style: `black src/ tests/`
5. Submit a pull request

## License

[Your License Here]

## Acknowledgments

Built with excellent open-source tools:
- [Ollama](https://ollama.com) for local LLM inference
- [LangChain](https://python.langchain.com) for RAG orchestration
- [ChromaDB](https://docs.trychroma.com) for vector storage
- [Sentence Transformers](https://www.sbert.net) for embeddings
- [Streamlit](https://streamlit.io) for web interface