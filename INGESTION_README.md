# PIB Article Ingestion Script

This script ingests PIB (Press Information Bureau) articles from JSON files into the RAG system's vector database.

## Overview

The `ingest_articles.py` script performs the following operations:

1. **Load Articles**: Reads articles from a JSON file
2. **Normalize Content**: Standardizes text formatting, HTML entities, and Unicode
3. **Deduplicate**: Removes duplicate articles based on article ID
4. **Chunk Articles**: Splits long articles into smaller, semantically coherent chunks
5. **Generate Embeddings**: Creates vector embeddings for each chunk using sentence-transformers
6. **Store in Vector Database**: Saves chunks and embeddings in ChromaDB for efficient retrieval

## Usage

### Basic Usage

```bash
# Ingest articles from default location (data/all_articles.json)
python ingest_articles.py

# Ingest articles from a specific file
python ingest_articles.py path/to/articles.json

# Reset vector store before ingestion (clears existing data)
python ingest_articles.py --reset
```

### Advanced Options

```bash
# Custom chunk size and overlap
python ingest_articles.py --chunk-size 800 --chunk-overlap 150

# Custom batch size for processing
python ingest_articles.py --batch-size 64

# Combine multiple options
python ingest_articles.py path/to/articles.json --reset --chunk-size 1200 --chunk-overlap 250 --batch-size 50
```

### Command-Line Arguments

- `filepath` (optional): Path to JSON file containing articles
  - Default: `data/all_articles.json`
  
- `--reset`: Reset vector store before ingestion (clears existing data)
  - Use this when you want to start fresh or re-ingest all articles
  
- `--chunk-size`: Maximum chunk size in characters
  - Default: 1000
  - Must be positive
  
- `--chunk-overlap`: Overlap between consecutive chunks in characters
  - Default: 200
  - Must be non-negative and less than chunk-size
  - Helps maintain context between chunks
  
- `--batch-size`: Number of chunks to process in each batch
  - Default: 32
  - Larger batches are more efficient but use more memory

## Input Format

The script expects a JSON file containing an array of article objects with the following structure:

```json
[
  {
    "date": "2025-01-15",
    "id": "123456",
    "ministry": "Ministry of Health and Family Welfare",
    "title": "Article Title",
    "content": "Full article content..."
  }
]
```

### Required Fields

Each article must have:
- `id`: Unique article identifier
- `date`: Publication date (YYYY-MM-DD format)
- `ministry`: Government ministry name
- `title`: Article title
- `content`: Full article text

## Features

### Progress Tracking

The script provides detailed progress information:
- Number of articles loaded and processed
- Number of chunks created
- Batch processing progress
- Embedding generation status
- Storage confirmation

### Error Handling

- **Invalid JSON**: Logs error and exits
- **Missing Fields**: Skips invalid articles with warnings
- **Duplicate Articles**: Automatically deduplicates based on article ID
- **Processing Errors**: Continues with remaining articles, logs errors
- **Batch Failures**: Continues with next batch instead of failing completely

### Logging

Logs are written to:
- Console (stdout) for real-time monitoring
- `ingestion.log` file for persistent records

Log levels include INFO, WARNING, and ERROR messages.

### Statistics

After completion, the script displays:
- Articles loaded and processed
- Chunks created and stored
- Errors encountered
- Total processing time
- Average time per chunk
- Vector store verification

## Performance Considerations

### Batch Processing

The script processes chunks in batches to optimize:
- **Memory Usage**: Prevents loading all embeddings at once
- **Embedding Generation**: Batch operations are more efficient
- **Database Operations**: Reduces number of database transactions

### Large Files

For very large JSON files (millions of articles):
- Consider splitting into smaller files
- Use larger batch sizes (e.g., 64 or 128) if memory allows
- Monitor system resources during ingestion

### Chunking Strategy

- **Chunk Size**: Affects retrieval granularity
  - Smaller chunks: More precise but may lose context
  - Larger chunks: More context but less precise
  
- **Overlap**: Maintains context between chunks
  - Recommended: 15-25% of chunk size
  - Too small: May lose important context
  - Too large: Redundant information

## Examples

### Example 1: Initial Setup

```bash
# First time ingestion with default settings
python ingest_articles.py data/all_articles.json --reset
```

### Example 2: Incremental Update

```bash
# Add new articles without clearing existing data
python ingest_articles.py data/new_articles.json
```

### Example 3: Fine-tuned Chunking

```bash
# Optimize for longer context windows
python ingest_articles.py --chunk-size 1500 --chunk-overlap 300
```

### Example 4: High-performance Ingestion

```bash
# Use larger batches for faster processing
python ingest_articles.py --batch-size 128 --reset
```

## Troubleshooting

### "File not found" Error

Ensure the JSON file path is correct and the file exists.

```bash
# Check if file exists
ls data/all_articles.json
```

### "Chunk size must be positive" Error

Ensure chunk-size is a positive integer.

```bash
# Correct usage
python ingest_articles.py --chunk-size 1000
```

### "Chunk overlap must be less than chunk size" Error

Ensure overlap is smaller than chunk size.

```bash
# Correct usage
python ingest_articles.py --chunk-size 1000 --chunk-overlap 200
```

### Memory Issues

If you encounter memory errors:
- Reduce batch size: `--batch-size 16`
- Process in smaller files
- Close other applications

### Slow Processing

To improve speed:
- Increase batch size: `--batch-size 64`
- Use SSD storage for vector database
- Ensure sufficient RAM available

## Integration with RAG System

After ingestion, the articles are ready for use with the RAG system:

1. **Query Engine**: Can search for relevant chunks
2. **Response Generator**: Uses retrieved chunks to generate answers
3. **Conversational Interface**: Provides interactive access to the knowledge base

## Verification

To verify successful ingestion:

```python
from src.vector_store.vector_store import VectorStore

# Check number of stored chunks
store = VectorStore()
count = store.count()
print(f"Vector store contains {count} chunks")

# Check available ministries
ministries = store.get_unique_ministries()
print(f"Available ministries: {ministries}")
```

## Notes

- The script uses the `all-MiniLM-L6-v2` model for embeddings (384 dimensions)
- ChromaDB is used for vector storage with cosine similarity
- Content normalization preserves original content for reference
- Paragraph boundaries are respected during chunking
- The vector store persists to disk by default
