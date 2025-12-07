# Main Application Reference

## Overview

The `main.py` file is the entry point for the PIB RAG System. It provides a command-line interface for querying government policy information from PIB articles.

## Architecture

### PIBRAGSystem Class

The main application class that manages all system components:

```python
class PIBRAGSystem:
    - embedding_generator: EmbeddingGenerator
    - vector_store: VectorStore
    - query_engine: QueryEngine
    - response_generator: ResponseGenerator
    - interface: ConversationalInterface
```

### Initialization Flow

1. **Embedding Model**: Loads sentence-transformers model (all-MiniLM-L6-v2)
2. **Vector Store**: Connects to ChromaDB and checks for existing data
3. **Query Engine**: Initializes with vector store and embedding generator
4. **Response Generator**: Connects to Ollama LLM and validates connection
5. **Conversational Interface**: Sets up conversation management

### Startup Validation

The system performs comprehensive validation during initialization:

- ✅ Embedding model loads successfully
- ✅ Vector store is accessible
- ✅ Ollama server is running
- ✅ Required LLM model is available
- ✅ All components can communicate

If any validation fails, the system provides detailed troubleshooting information.

## Command Reference

### Query Commands

| Command | Description | Example |
|---------|-------------|---------|
| `<question>` | Ask a natural language question | `What are the healthcare policies?` |

### Filter Commands

| Command | Description | Example |
|---------|-------------|---------|
| `filter ministry <name>` | Filter by ministry | `filter ministry Ministry of Health` |
| `filter date <start> to <end>` | Filter by date range | `filter date 2024-01-01 to 2024-12-31` |
| `filter clear` | Clear all filters | `filter clear` |

### Information Commands

| Command | Description |
|---------|-------------|
| `ministries` | List all available ministries |
| `status` | Show system status and active filters |
| `help`, `h`, `?` | Show help message |

### Control Commands

| Command | Description |
|---------|-------------|
| `clear`, `reset` | Clear conversation history |
| `exit`, `quit`, `q` | Exit the application |

## Implementation Details

### Command Parsing

The system uses a simple string-based command parser:

```python
def run(self):
    while True:
        user_input = input("\n> ").strip()
        
        # Handle special commands
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
        elif user_input.lower() in ['help', 'h', '?']:
            self._print_help()
        # ... more commands
        else:
            # Process as query
            self._process_query(user_input)
```

### Error Handling

The system provides context-aware error messages:

- **Ollama Connection Errors**: Instructions to start Ollama and pull models
- **Model Not Found**: Instructions to pull the required model
- **Empty Vector Store**: Instructions to run data ingestion
- **Embedding Errors**: Information about model download

### Conversation Management

The system maintains conversation state through the ConversationalInterface:

- Conversation history is preserved across queries
- Filters persist until explicitly cleared
- History is automatically truncated when it exceeds limits

## Integration Points

### With Data Ingestion

The main application expects data to be ingested separately:

```bash
python ingest_articles.py  # Run before main.py
python main.py             # Then run the application
```

### With Configuration

All settings are loaded from `config.py`:

```python
import config

# Ollama settings
OLLAMA_BASE_URL = config.OLLAMA_BASE_URL
OLLAMA_MODEL = config.OLLAMA_MODEL

# Query settings
DEFAULT_TOP_K = config.DEFAULT_TOP_K
DEFAULT_RELEVANCE_THRESHOLD = config.DEFAULT_RELEVANCE_THRESHOLD
```

## Testing

### Unit Tests

Test individual components:

```bash
python test_main_init.py    # Test initialization logic
python test_main_flow.py    # Test application flow
```

### Integration Tests

Test with actual Ollama:

```bash
# Ensure Ollama is running
ollama serve

# Pull model
ollama pull llama3.2

# Run application
python main.py
```

## Troubleshooting

### Common Issues

**Issue**: `Failed to connect to Ollama`
- **Solution**: Start Ollama with `ollama serve`

**Issue**: `Model 'llama3.2' not found`
- **Solution**: Pull model with `ollama pull llama3.2`

**Issue**: `Vector store connected: 0 chunks available`
- **Solution**: Run `python ingest_articles.py` first

**Issue**: Slow response times
- **Solution**: Use a smaller model or reduce `DEFAULT_TOP_K`

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python main.py
```

Or edit `config.py`:

```python
LOG_LEVEL = "DEBUG"
```

## Extension Points

### Adding New Commands

Add new commands in the `run()` method:

```python
elif user_input.lower() == 'mycommand':
    self._handle_my_command()
    continue
```

### Custom Filters

Add new filter types in the command parsing:

```python
elif user_input.lower().startswith('filter custom'):
    self._handle_custom_filter(user_input)
    continue
```

### Alternative Interfaces

The PIBRAGSystem class can be used programmatically:

```python
from main import PIBRAGSystem

system = PIBRAGSystem()
if system.initialize():
    response = system.interface.process_message("Your question")
    print(system.interface.display_response(response))
```

## Performance Considerations

### Initialization Time

- First run: ~10-30 seconds (model downloads)
- Subsequent runs: ~2-5 seconds

### Query Response Time

- Embedding generation: ~100ms
- Vector search: ~50-200ms
- LLM generation: ~2-10 seconds (depends on model and context)

### Memory Usage

- Embedding model: ~100MB
- Vector store: ~50MB per 10,000 chunks
- LLM (via Ollama): Handled externally

## Best Practices

1. **Keep Ollama Running**: Start Ollama before the application
2. **Use Filters**: Apply ministry/date filters to reduce search space
3. **Clear History**: Clear conversation history when switching topics
4. **Monitor Logs**: Check `pib_rag.log` for detailed information
5. **Batch Queries**: Keep the application running for multiple queries

## Security Considerations

1. **Local Only**: System runs entirely locally, no external API calls
2. **Data Privacy**: All data stays on your machine
3. **No Authentication**: Command-line interface has no authentication
4. **File Permissions**: Ensure proper permissions on vector store directory

## Future Enhancements

Potential improvements:

- Web interface (Flask/FastAPI)
- REST API endpoints
- Multi-user support
- Query history persistence
- Export results to file
- Advanced filtering (boolean queries)
- Semantic search refinement
- Custom prompt templates
