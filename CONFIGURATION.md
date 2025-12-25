# Configuration Guide

This document covers all configuration options for the PIB RAG System, including environment variables, model settings, and web interface customization.

## Environment Variables

Copy the example environment file and customize as needed:

```bash
cp .env.example .env
```

### Core Configuration

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434  # Change if Ollama runs elsewhere
OLLAMA_MODEL=llama3.2                   # Change to mistral or llama2
OLLAMA_TIMEOUT=120                      # Request timeout in seconds

# Query Configuration
DEFAULT_TOP_K=5                         # Number of results to retrieve
DEFAULT_RELEVANCE_THRESHOLD=0.5         # Minimum similarity score (0.0-1.0)

# Chunking Configuration
CHUNK_SIZE=1000                         # Maximum chunk size in characters
CHUNK_OVERLAP=200                       # Overlap between chunks

# Logging
LOG_LEVEL=INFO                          # DEBUG, INFO, WARNING, ERROR
```

### Model Caching Configuration

```env
# Model Caching (Faster Startup)
MODEL_CACHE_DIR=./model_cache          # Directory for cached embedding models
ENABLE_MODEL_CACHING=true              # Enable model caching for faster startup
```

### Streamlit Web Interface Configuration

```env
# Page Configuration
STREAMLIT_PAGE_TITLE="PIB RAG System"        # Browser tab title
STREAMLIT_PAGE_ICON="🏛️"                    # Browser tab icon
STREAMLIT_LAYOUT=wide                        # Layout: wide or centered
STREAMLIT_SIDEBAR_STATE=expanded             # Sidebar: expanded or collapsed

# Theme Configuration
STREAMLIT_PRIMARY_COLOR="#007bff"            # Primary accent color
STREAMLIT_BACKGROUND_COLOR="#ffffff"         # Main background color
STREAMLIT_SECONDARY_BG="#f0f2f6"            # Secondary background color
STREAMLIT_TEXT_COLOR="#262730"               # Text color

# Performance Configuration
STREAMLIT_CACHE_TTL=3600                     # Cache time-to-live in seconds
MAX_CONCURRENT_QUERIES=5                     # Maximum concurrent queries
DEFAULT_MAX_ARTICLES=10                      # Default number of articles to retrieve

# Error Handling and Debugging
ENABLE_GRACEFUL_DEGRADATION=true             # Enable fallback mode when Ollama unavailable
FALLBACK_MODE=demo                           # Fallback mode: demo, readonly, disabled
SHOW_SYSTEM_INFO=true                        # Show system information in sidebar
SHOW_DEBUG_INFO=false                        # Show detailed debug information
SHOW_DETAILED_ERRORS=false                   # Show detailed error messages
LOG_USER_QUERIES=true                        # Log user queries for debugging
```

## Default Configuration Values

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| **Embedding Model** | `all-MiniLM-L6-v2` | Sentence transformer model (384 dimensions) |
| **Model Cache Directory** | `./model_cache` | Directory for cached embedding models |
| **Enable Model Caching** | `true` | Cache models for faster startup |
| **Chunk Size** | 1000 characters | Maximum chunk size for article splitting |
| **Chunk Overlap** | 200 characters | Overlap between consecutive chunks |
| **Top-K Results** | 5 | Number of most relevant chunks to retrieve |
| **Relevance Threshold** | 0.5 | Minimum similarity score (0.0-1.0) |
| **Ollama Base URL** | `http://localhost:11434` | Ollama API endpoint |
| **Ollama Model** | `llama3.2` | Language model for response generation |
| **Max Conversation History** | 10 messages | Maximum messages to keep in context |
| **Streamlit Port** | 8501 | Web interface port |
| **Streamlit Theme** | `light` | Web interface theme (light/dark) |

## Customizing Configuration

### Method 1: Environment Variables (.env file)

The recommended approach for most users:

```env
# .env file
OLLAMA_MODEL=mistral
DEFAULT_TOP_K=3
CHUNK_SIZE=1200
```

### Method 2: Direct Code Modification

Edit `config.py` directly for advanced customization:

```python
# config.py
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
```

## Model Configuration

### Supported Ollama Models

| Model | Size | Speed | Quality | Recommended For |
|-------|------|-------|---------|-----------------|
| llama3.2 | ~2GB | Medium | High | Best overall results |
| mistral | ~4GB | Fast | High | Faster responses |
| llama2 | ~4GB | Medium | Good | Stable, proven |

### Embedding Models

The system uses `all-MiniLM-L6-v2` by default. To use a different model, modify `config.py`:

```python
EMBEDDING_MODEL = "all-mpnet-base-v2"  # Higher quality, slower
# or
EMBEDDING_MODEL = "all-MiniLM-L12-v2"  # Larger model, better quality
```

## Web Interface Graceful Degradation

When Ollama is not available, the web interface can operate in fallback modes:

### Demo Mode (Default)
- Shows simulated responses with actual search results
- Useful for testing and demonstrations
- Set: `FALLBACK_MODE=demo`

### Read-Only Mode
- Allows searching but disables response generation
- Shows only search results and citations
- Set: `FALLBACK_MODE=readonly`

### Disabled Mode
- Requires Ollama to be running
- Shows error message if Ollama unavailable
- Set: `FALLBACK_MODE=disabled`

## Performance Tuning

### For Faster Responses

```env
# Use faster model
OLLAMA_MODEL=mistral

# Reduce context size
DEFAULT_TOP_K=3
CHUNK_SIZE=800

# Reduce conversation history
MAX_CONVERSATION_HISTORY=5
```

### For Better Quality

```env
# Use higher quality model
OLLAMA_MODEL=llama3.2

# Increase context size
DEFAULT_TOP_K=7
CHUNK_SIZE=1200

# Lower relevance threshold for more results
DEFAULT_RELEVANCE_THRESHOLD=0.3
```

### For Large Datasets

```env
# Increase batch size for ingestion
EMBEDDING_BATCH_SIZE=64

# Increase cache TTL
STREAMLIT_CACHE_TTL=7200

# Enable model caching
ENABLE_MODEL_CACHING=true
```

## Security Considerations

### Data Privacy
- All processing happens locally
- No data sent to external services
- Ollama runs locally without internet access

### Network Security
```env
# Restrict Ollama to localhost only
OLLAMA_BASE_URL=http://127.0.0.1:11434

# Disable telemetry
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
LOG_USER_QUERIES=false
```

### File Permissions
```bash
# Secure vector store directory
chmod 750 vector_store/

# Secure model cache
chmod 750 model_cache/

# Secure configuration
chmod 600 .env
```

## Advanced Configuration

### Custom Prompt Templates

Edit `src/response_generation/response_generator.py` to customize prompts:

```python
SYSTEM_PROMPT = """
You are an AI assistant specialized in Indian government policies.
[Your custom instructions here]
"""
```

### Custom Chunking Strategy

Modify `src/data_ingestion/article_chunker.py`:

```python
def chunk_article(self, article: Article) -> List[Chunk]:
    # Your custom chunking logic
    pass
```

### Custom Filters

Add new filter types in `src/query_engine/query_engine.py`:

```python
def search(self, query: str, custom_filter: Optional[str] = None):
    filters = {}
    if custom_filter:
        filters['custom_field'] = custom_filter
    # Implementation
```

## Configuration Validation

Run the setup verification script to check your configuration:

```bash
python verify_setup.py
```

This checks:
- ✅ Python version compatibility
- ✅ Required packages installed
- ✅ Ollama connectivity and model availability
- ✅ Configuration file validity
- ✅ Directory permissions
- ✅ Model cache status

## Environment-Specific Configurations

### Development Environment
```env
LOG_LEVEL=DEBUG
SHOW_DEBUG_INFO=true
SHOW_DETAILED_ERRORS=true
ENABLE_GRACEFUL_DEGRADATION=true
```

### Production Environment
```env
LOG_LEVEL=WARNING
SHOW_DEBUG_INFO=false
SHOW_DETAILED_ERRORS=false
ENABLE_GRACEFUL_DEGRADATION=false
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### Testing Environment
```env
FALLBACK_MODE=demo
ENABLE_GRACEFUL_DEGRADATION=true
DEFAULT_MAX_ARTICLES=3
STREAMLIT_CACHE_TTL=60
```