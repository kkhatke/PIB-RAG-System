# Troubleshooting Guide

This guide covers common issues and their solutions for the PIB RAG System.

## Web Interface Issues

### 1. Streamlit App Won't Start

**Error Message:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
# Ensure Streamlit is installed
pip install streamlit

# Verify installation
streamlit --version

# If still failing, reinstall requirements
pip install -r requirements.txt
```

### 2. Port Already in Use

**Error Message:**
```
OSError: [Errno 48] Address already in use
```

**Solutions:**

**Use Different Port:**
```bash
streamlit run streamlit_app.py --server.port 8502
```

**Kill Existing Process:**
```bash
# Find process using port 8501
lsof -i :8501  # macOS/Linux
netstat -ano | findstr :8501  # Windows

# Kill the process (replace PID with actual process ID)
kill -9 PID  # macOS/Linux
taskkill /PID PID /F  # Windows
```

### 3. Model Caching Issues

**Issue:** Models re-download on every startup

**Solution:**
```bash
# Check cache directory exists
ls -la model_cache/

# If missing, create it
mkdir -p model_cache

# Verify permissions
chmod 755 model_cache
```

**Set Cache Directory:**
```env
# In .env file
MODEL_CACHE_DIR=./model_cache
ENABLE_MODEL_CACHING=true
```

### 4. Slow Web Interface Performance

**Issue:** Web interface is slow or unresponsive

**Solutions:**

**Reduce Article Count:**
- Use sidebar to set "Max Articles" to 3-5 instead of default 10

**Clear Browser Cache:**
- Hard refresh: Ctrl+F5 (Windows/Linux) or Cmd+Shift+R (macOS)
- Clear browser cache and cookies for localhost

**Restart Streamlit:**
```bash
# Stop current session (Ctrl+C)
# Restart with performance settings
streamlit run streamlit_app.py --server.maxUploadSize 200
```

### 5. Web Interface Not Loading

**Issue:** Browser shows "This site can't be reached"

**Diagnostic Steps:**
```bash
# Check if Streamlit is running
ps aux | grep streamlit  # macOS/Linux
tasklist | findstr streamlit  # Windows

# Test port accessibility
curl http://localhost:8501  # Should return HTML
telnet localhost 8501       # Should connect
```

**Solutions:**

**Check Firewall:**
- Ensure port 8501 is not blocked by firewall
- Add exception for Python/Streamlit if needed

**Try Different Browser:**
- Test in incognito/private mode
- Try different browser (Chrome, Firefox, Safari)

**Check Network Settings:**
- Disable VPN if active
- Check proxy settings

### 6. Session State Issues

**Issue:** Conversation history not persisting or filters resetting

**Solution:**
```bash
# Clear Streamlit cache
streamlit cache clear

# Restart with fresh session
streamlit run streamlit_app.py --server.enableCORS false
```

## Ollama Issues

### 1. Ollama Connection Failed

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

### 2. Model Not Found

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

### 3. Slow Response Times

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

### 4. Ollama Timeout

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

### 5. Context Too Long

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

## Data Ingestion Issues

### 1. File Not Found

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

### 2. Invalid JSON Format

**Error Message:**
```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Solution:**
- Verify JSON file is valid: Use [jsonlint.com](https://jsonlint.com)
- Check file encoding (should be UTF-8)
- Ensure file is not empty or corrupted

### 3. Memory Issues During Ingestion

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

### 4. Empty Vector Store

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

## Embedding Model Issues

### 1. Model Download Fails

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

### 2. CUDA/GPU Errors

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

## Query Issues

### 1. No Results Found

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

### 2. Irrelevant Results

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

## Performance Issues

### Slow Startup

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

### High Memory Usage

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

## Getting More Help

### Diagnostic Commands

**Check System Status:**
```bash
python verify_setup.py
```

**Check Logs:**
```bash
tail -f pib_rag.log  # View real-time logs
```

**Enable Debug Mode:**
```env
LOG_LEVEL=DEBUG
```

**Test Components Individually:**
```bash
# Test embedding generation
python -c "from src.embedding.embedding_generator import EmbeddingGenerator; eg = EmbeddingGenerator(); print(len(eg.generate_embedding('test')))"

# Test vector store
python -c "from src.vector_store.vector_store import VectorStore; vs = VectorStore(); print(vs.count())"

# Test Ollama
curl http://localhost:11434/api/tags
```

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

### Common Solutions Summary

| Issue | Quick Fix |
|-------|-----------|
| Streamlit won't start | `pip install streamlit` |
| Port in use | `streamlit run streamlit_app.py --server.port 8502` |
| Ollama not found | `ollama serve` |
| Model not found | `ollama pull llama3.2` |
| Slow responses | Use `mistral` model, reduce `DEFAULT_TOP_K` |
| No results | Lower `DEFAULT_RELEVANCE_THRESHOLD` |
| Memory issues | Reduce batch size, close other apps |
| Empty vector store | Run `python ingest_articles.py` |