# PIB RAG System - Usage Guide

## Prerequisites

Before running the PIB RAG System, ensure you have:

1. **Python 3.8+** installed
2. **Ollama** installed and running
3. Required Python packages installed
4. PIB articles data ingested into the vector store

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and Setup Ollama

#### Download Ollama
Visit [https://ollama.ai/download](https://ollama.ai/download) and download Ollama for your platform.

#### Start Ollama Server
```bash
ollama serve
```

#### Pull Required Model
```bash
# Pull llama3.2 (recommended)
ollama pull llama3.2

# Or pull mistral (alternative)
ollama pull mistral
```

#### Verify Installation
```bash
ollama list
```

### 3. Ingest PIB Articles

Before using the system, you need to populate the vector store with PIB articles:

```bash
python ingest_articles.py
```

This will:
- Load articles from `data/all_articles.json`
- Normalize and chunk the content
- Generate embeddings
- Store in the vector database

## Running the Application

### Start the Interactive Interface

```bash
python main.py
```

The system will:
1. Load the embedding model
2. Connect to the vector store
3. Connect to Ollama LLM
4. Start the interactive command-line interface

### Basic Usage

Once the system starts, you can:

#### Ask Questions
Simply type your question and press Enter:
```
> What are the recent healthcare initiatives?
```

#### Use Commands

**Get Help:**
```
> help
```

**List Available Ministries:**
```
> ministries
```

**Check System Status:**
```
> status
```

**Filter by Ministry:**
```
> filter ministry Ministry of Health and Family Welfare
> What are the recent announcements?
```

**Filter by Date Range:**
```
> filter date 2024-01-01 to 2024-12-31
> What policies were announced?
```

**Clear Filters:**
```
> filter clear
```

**Clear Conversation History:**
```
> clear
```

**Exit:**
```
> exit
```

## Example Session

```
> What are the recent healthcare initiatives?

Searching...

================================================================================
ANSWER:
Based on the retrieved PIB articles, recent healthcare initiatives include...
[Generated response with details]

SOURCES:
1. New Healthcare Initiative Launched
   Ministry: Ministry of Health and Family Welfare
   Date: 2024-01-15
   Article ID: 123456
   Relevance: 0.892

2. National Health Mission Update
   Ministry: Ministry of Health and Family Welfare
   Date: 2024-01-10
   Article ID: 123457
   Relevance: 0.845
================================================================================

> filter ministry Ministry of Finance
✓ Ministry filter set: Ministry of Finance

> What are the budget allocations?

Searching...
[Results filtered to Ministry of Finance only]
```

## Configuration

You can customize the system by editing `config.py`:

### Ollama Configuration
```python
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama server URL
OLLAMA_MODEL = "llama3.2"  # Model to use (llama3.2 or mistral)
OLLAMA_TIMEOUT = 120  # Request timeout in seconds
```

### Query Configuration
```python
DEFAULT_TOP_K = 5  # Number of results to retrieve
DEFAULT_RELEVANCE_THRESHOLD = 0.5  # Minimum relevance score
```

### Conversation Configuration
```python
MAX_CONVERSATION_HISTORY = 10  # Maximum messages to keep
```

## Troubleshooting

### Ollama Connection Failed

**Error:** `Failed to connect to Ollama at http://localhost:11434`

**Solution:**
1. Ensure Ollama is running: `ollama serve`
2. Check if the port is correct in `config.py`
3. Verify Ollama is accessible: `curl http://localhost:11434`

### Model Not Found

**Error:** `Model 'llama3.2' not found`

**Solution:**
```bash
ollama pull llama3.2
```

### Empty Vector Store

**Error:** `Vector store connected: 0 chunks available`

**Solution:**
Run the data ingestion script:
```bash
python ingest_articles.py
```

### Slow Response Times

If responses are slow:
1. Use a smaller model (mistral is faster than llama3.2)
2. Reduce `DEFAULT_TOP_K` in config.py
3. Increase `OLLAMA_TIMEOUT` if getting timeout errors

## Advanced Usage

### Using Different Models

To use a different Ollama model:

1. Pull the model:
```bash
ollama pull mistral
```

2. Update `config.py`:
```python
OLLAMA_MODEL = "mistral"
```

3. Restart the application

### Environment Variables

You can also configure via environment variables:

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="llama3.2"
export LOG_LEVEL="DEBUG"

python main.py
```

### Batch Processing

For batch queries, you can create a script:

```python
from main import PIBRAGSystem

system = PIBRAGSystem()
if system.initialize():
    queries = [
        "What are the healthcare initiatives?",
        "What are the education policies?",
        "What are the infrastructure projects?"
    ]
    
    for query in queries:
        response = system.interface.process_message(query)
        print(f"\nQuery: {query}")
        print(system.interface.display_response(response))
```

## Performance Tips

1. **First Run:** The first query may be slower as models load
2. **Batch Queries:** Keep the application running for multiple queries
3. **Filter Early:** Use ministry/date filters to reduce search space
4. **Clear History:** Clear conversation history if not needed to reduce context size

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the logs in `pib_rag.log`
3. Ensure all dependencies are installed correctly
4. Verify Ollama is running and models are available
