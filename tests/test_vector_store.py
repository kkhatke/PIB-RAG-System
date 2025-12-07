"""
Property-based tests for VectorStore module.
Tests embedding storage, retrieval, filtering, and metadata preservation.
"""
import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

from src.vector_store.vector_store import VectorStore, SearchResult
from src.data_ingestion.article_chunker import Chunk
from src.embedding.embedding_generator import EmbeddingGenerator


# Test fixtures
@pytest.fixture
def temp_vector_store():
    """Create a temporary vector store for testing."""
    temp_dir = tempfile.mkdtemp()
    store = VectorStore(persist_directory=temp_dir, collection_name="test_collection")
    yield store
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def embedding_generator():
    """Create an embedding generator for testing."""
    return EmbeddingGenerator()


# Hypothesis strategies
@st.composite
def chunk_strategy(draw):
    """Generate random Chunk objects."""
    article_id = draw(st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    chunk_index = draw(st.integers(min_value=0, max_value=10))
    chunk_id = f"{article_id}_{chunk_index}"
    
    # Generate date in ISO format
    days_ago = draw(st.integers(min_value=0, max_value=365))
    date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
    
    ministries = [
        "Ministry of Health",
        "Ministry of Finance",
        "Ministry of Education",
        "Ministry of Defence",
        "Ministry of Agriculture"
    ]
    ministry = draw(st.sampled_from(ministries))
    
    title = draw(st.text(min_size=10, max_size=100))
    content = draw(st.text(min_size=50, max_size=500))
    
    return Chunk(
        chunk_id=chunk_id,
        article_id=article_id,
        content=content,
        metadata={
            'date': date,
            'ministry': ministry,
            'title': title,
            'chunk_index': chunk_index,
            'total_chunks': draw(st.integers(min_value=chunk_index + 1, max_value=20))
        }
    )


# Property 14: Embedding retrieval consistency
# Feature: pib-rag-system, Property 14: Embedding retrieval consistency
@settings(max_examples=100, deadline=2000)  # Increased deadline for vector store operations
@given(chunks=st.lists(chunk_strategy(), min_size=1, max_size=10, unique_by=lambda c: c.chunk_id))
def test_property_14_embedding_retrieval_consistency(chunks, embedding_generator):
    """
    Property 14: Embedding retrieval consistency
    For any embedding stored in the vector store, it should be retrievable through similarity search.
    Validates: Requirements 5.2
    """
    # Create temporary vector store
    temp_dir = tempfile.mkdtemp()
    try:
        store = VectorStore(persist_directory=temp_dir, collection_name="test_prop14")
        
        # Generate embeddings for chunks
        embeddings = embedding_generator.batch_generate_embeddings([c.content for c in chunks])
        
        # Add chunks to vector store
        store.add_chunks(chunks, embeddings)
        
        # For each chunk, verify it can be retrieved
        for i, chunk in enumerate(chunks):
            # Search using the same embedding
            results = store.similarity_search(
                query_embedding=embeddings[i],
                k=1
            )
            
            # Should retrieve at least one result
            assert len(results) > 0, f"Failed to retrieve chunk {chunk.chunk_id}"
            
            # The top result should be the same chunk (or very similar)
            top_result = results[0]
            assert top_result.chunk.chunk_id == chunk.chunk_id, \
                f"Retrieved wrong chunk: expected {chunk.chunk_id}, got {top_result.chunk.chunk_id}"
            
            # Score should be very high (close to 1.0 for identical embeddings)
            assert top_result.score > 0.99, \
                f"Similarity score too low: {top_result.score}"
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Property 15: Incremental update preservation
# Feature: pib-rag-system, Property 15: Incremental update preservation
@settings(max_examples=100, deadline=2000)  # Increased deadline for vector store operations
@given(
    initial_chunks=st.lists(chunk_strategy(), min_size=1, max_size=5, unique_by=lambda c: c.chunk_id),
    new_chunks=st.lists(chunk_strategy(), min_size=1, max_size=5, unique_by=lambda c: c.chunk_id)
)
def test_property_15_incremental_update_preservation(initial_chunks, new_chunks, embedding_generator):
    """
    Property 15: Incremental update preservation
    For any existing article in the system, adding new articles should not modify or remove 
    the existing article's embedding or metadata.
    Validates: Requirements 5.4
    """
    # Ensure no overlap between initial and new chunks
    new_chunk_ids = {c.chunk_id for c in new_chunks}
    initial_chunks = [c for c in initial_chunks if c.chunk_id not in new_chunk_ids]
    
    if not initial_chunks:
        return  # Skip if no initial chunks after filtering
    
    # Create temporary vector store
    temp_dir = tempfile.mkdtemp()
    try:
        store = VectorStore(persist_directory=temp_dir, collection_name="test_prop15")
        
        # Add initial chunks
        initial_embeddings = embedding_generator.batch_generate_embeddings([c.content for c in initial_chunks])
        store.add_chunks(initial_chunks, initial_embeddings)
        
        # Store initial count
        initial_count = store.count()
        
        # Verify initial chunks are retrievable
        initial_results = {}
        for i, chunk in enumerate(initial_chunks):
            results = store.similarity_search(query_embedding=initial_embeddings[i], k=1)
            assert len(results) > 0
            initial_results[chunk.chunk_id] = results[0]
        
        # Add new chunks
        new_embeddings = embedding_generator.batch_generate_embeddings([c.content for c in new_chunks])
        store.add_chunks(new_chunks, new_embeddings)
        
        # Verify count increased correctly
        final_count = store.count()
        assert final_count == initial_count + len(new_chunks), \
            f"Count mismatch: expected {initial_count + len(new_chunks)}, got {final_count}"
        
        # Verify initial chunks are still retrievable with same metadata
        for i, chunk in enumerate(initial_chunks):
            results = store.similarity_search(query_embedding=initial_embeddings[i], k=1)
            
            assert len(results) > 0, f"Initial chunk {chunk.chunk_id} not found after adding new chunks"
            
            retrieved = results[0]
            assert retrieved.chunk.chunk_id == chunk.chunk_id, \
                f"Retrieved wrong chunk for {chunk.chunk_id}"
            
            # Verify metadata preserved
            assert retrieved.chunk.metadata['date'] == chunk.metadata['date']
            assert retrieved.chunk.metadata['ministry'] == chunk.metadata['ministry']
            assert retrieved.chunk.metadata['title'] == chunk.metadata['title']
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Property 16: Metadata preservation with embeddings
# Feature: pib-rag-system, Property 16: Metadata preservation with embeddings
@settings(max_examples=100, deadline=2000)  # Increased deadline for vector store operations
@given(chunks=st.lists(chunk_strategy(), min_size=1, max_size=10, unique_by=lambda c: c.chunk_id))
def test_property_16_metadata_preservation(chunks, embedding_generator):
    """
    Property 16: Metadata preservation with embeddings
    For any chunk stored in the vector store, retrieving it should return all original 
    metadata fields (date, ministry, title, article_id).
    Validates: Requirements 5.5
    """
    # Create temporary vector store
    temp_dir = tempfile.mkdtemp()
    try:
        store = VectorStore(persist_directory=temp_dir, collection_name="test_prop16")
        
        # Generate embeddings and add chunks
        embeddings = embedding_generator.batch_generate_embeddings([c.content for c in chunks])
        store.add_chunks(chunks, embeddings)
        
        # Retrieve each chunk and verify metadata
        for i, original_chunk in enumerate(chunks):
            results = store.similarity_search(query_embedding=embeddings[i], k=1)
            
            assert len(results) > 0, f"Failed to retrieve chunk {original_chunk.chunk_id}"
            
            retrieved_chunk = results[0].chunk
            
            # Verify all required metadata fields are present and correct
            assert retrieved_chunk.article_id == original_chunk.article_id, \
                f"article_id mismatch for {original_chunk.chunk_id}"
            
            assert 'date' in retrieved_chunk.metadata, "date field missing from metadata"
            assert retrieved_chunk.metadata['date'] == original_chunk.metadata['date'], \
                f"date mismatch for {original_chunk.chunk_id}"
            
            assert 'ministry' in retrieved_chunk.metadata, "ministry field missing from metadata"
            assert retrieved_chunk.metadata['ministry'] == original_chunk.metadata['ministry'], \
                f"ministry mismatch for {original_chunk.chunk_id}"
            
            assert 'title' in retrieved_chunk.metadata, "title field missing from metadata"
            assert retrieved_chunk.metadata['title'] == original_chunk.metadata['title'], \
                f"title mismatch for {original_chunk.chunk_id}"
            
            assert 'chunk_index' in retrieved_chunk.metadata, "chunk_index field missing from metadata"
            assert retrieved_chunk.metadata['chunk_index'] == original_chunk.metadata['chunk_index'], \
                f"chunk_index mismatch for {original_chunk.chunk_id}"
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Property 4: Unique ministry list completeness
# Feature: pib-rag-system, Property 4: Unique ministry list completeness
@settings(max_examples=100, deadline=2000)  # Increased deadline for vector store operations
@given(chunks=st.lists(chunk_strategy(), min_size=1, max_size=20, unique_by=lambda c: c.chunk_id))
def test_property_4_unique_ministry_list_completeness(chunks, embedding_generator):
    """
    Property 4: Unique ministry list completeness
    For any dataset of articles, the list of unique ministries returned by the system 
    should match exactly the set of unique ministry values in the dataset.
    Validates: Requirements 2.3
    """
    # Create temporary vector store
    temp_dir = tempfile.mkdtemp()
    try:
        store = VectorStore(persist_directory=temp_dir, collection_name="test_prop4")
        
        # Generate embeddings and add chunks
        embeddings = embedding_generator.batch_generate_embeddings([c.content for c in chunks])
        store.add_chunks(chunks, embeddings)
        
        # Get expected unique ministries from chunks
        expected_ministries = sorted(set(c.metadata['ministry'] for c in chunks))
        
        # Get unique ministries from vector store
        actual_ministries = store.get_unique_ministries()
        
        # Verify they match exactly
        assert actual_ministries == expected_ministries, \
            f"Ministry list mismatch:\nExpected: {expected_ministries}\nActual: {actual_ministries}"
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
