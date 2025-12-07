# Quick Start Guide - PIB RAG System

This guide will help you get the PIB RAG System up and running quickly.

## Prerequisites Check

Run the setup verification script to ensure all dependencies are installed:

```bash
python verify_setup.py
```

If all checks pass, proceed with the steps below.

## Step 1: Install Ollama

### Windows
1. Download Ollama from [https://ollama.com/download](https://ollama.com/download)
2. Run the installer
3. Ollama will start automatically as a service

### macOS
```bash
brew install ollama
```

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Step 2: Pull the Language Model

Open a terminal and run:

```bash
ollama pull llama3.2
```

This will download the llama3.2 model (~2GB). Alternative models:
- `ollama pull mistral` - Smaller, faster alternative
- `ollama pull llama2` - Older but stable version

## Step 3: Verify Ollama is Running

Check if Ollama is accessible:

```bash
curl http://localhost:11434/api/tags
```

You should see a JSON response listing available models.

## Step 4: Configure Environment (Optional)

If you want to customize settings, copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to change:
- `OLLAMA_BASE_URL` - If Ollama is running on a different host/port
- `OLLAMA_MODEL` - To use a different model (mistral, llama2, etc.)
- `LOG_LEVEL` - For more/less verbose logging

## Step 5: Ingest PIB Articles

Once the data ingestion module is implemented, run:

```bash
python ingest_articles.py --input data/all_articles.json
```

This will:
- Load articles from the JSON file
- Generate embeddings
- Store them in the vector database

## Step 6: Start the Conversational Interface

Once the interface is implemented, run:

```bash
python main.py
```

## Example Usage

```
> What are the recent healthcare initiatives?

[System retrieves relevant articles and generates response with citations]

> Show me announcements from the Ministry of Health from January 2025

[Filtered results with ministry and date constraints]

> /ministries

[Lists all available ministries]

> /help

[Shows all available commands]
```

## Troubleshooting

### Ollama Not Running
**Error**: Connection refused to http://localhost:11434

**Solution**: Start Ollama service
- Windows: Ollama runs as a service automatically
- macOS/Linux: Run `ollama serve` in a terminal

### Model Not Found
**Error**: Model 'llama3.2' not found

**Solution**: Pull the model
```bash
ollama pull llama3.2
```

### Slow Performance
**Issue**: Responses take too long

**Solutions**:
1. Use a smaller model: `ollama pull mistral`
2. Update `.env`: `OLLAMA_MODEL=mistral`
3. Reduce context in `config.py`: Lower `MAX_CONTEXT_LENGTH`

### Import Errors
**Error**: ModuleNotFoundError

**Solution**: Reinstall dependencies
```bash
pip install -r requirements.txt
```

## Next Steps

1. Review the full README.md for detailed documentation
2. Explore the design document at `.kiro/specs/pib-rag-system/design.md`
3. Check the implementation tasks at `.kiro/specs/pib-rag-system/tasks.md`
4. Run tests: `pytest`

## Getting Help

- Check the troubleshooting section in README.md
- Review Ollama documentation: [https://ollama.com/docs](https://ollama.com/docs)
- Check the project's issue tracker

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Disk Space**: 5GB for models and vector database
- **OS**: Windows, macOS, or Linux
