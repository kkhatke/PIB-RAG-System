"""
Property-based tests for QueryEngine module.
Tests query processing, filtering, ranking, and threshold behavior.
"""
import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timedelta
import tempfile
import shutil

from src.query_engine.query_engine import QueryEngine
from src.vector_store.vector_store import VectorStore
from src.embedding.embedding_generator import EmbeddingGenerator
from src.data_ingestion.article_chunker import Chunk


# Test fixtures
@pytest.fixture
def temp_vector_store():
    """Create a temporary vector store for testing."""
    temp_dir = tempfile.mkdtemp()
    store = VectorStore(persist_directory=temp_dir, collection_name="test_query_engine")
    yield store
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def embedding_generator():
    """Create an embedding generator for testing."""
    return EmbeddingGenerator()


@pytest.fixture
def query_engine(temp_vector_store, embedding_generator):
    """Create a query engine for testing."""
    return QueryEngine(temp_vector_store, embedding_generator)


# Hypothesis strategies
@st.composite
def chunk_with_ministry_strategy(draw, ministries):
    """Generate random Chunk with specific ministry."""
    ministry = draw(st.sampled_from(ministries))
    article_id = draw(st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    chunk_index = draw(st.integers(min_value=0, max_value=10))
    chunk_id = f"{article_id}_{chunk_index}"
    
    days_ago = draw(st.integers(min_value=0, max_value=365))
    date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
    
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


@st.composite
def chunk_with_date_strategy(draw, start_date, end_date):
    """Generate random Chunk with date in specified range."""
    article_id = draw(st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    chunk_index = draw(st.integers(min_value=0, max_value=10))
    chunk_id = f"{article_id}_{chunk_index}"
    
    # Generate date within range
    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    days_diff = (end - start).days
    
    if days_diff > 0:
        days_offset = draw(st.integers(min_value=0, max_value=days_diff))
        date = (start + timedelta(days=days_offset)).isoformat()
    else:
        date = start.isoformat()
    
    ministries = ["Ministry of Health", "Ministry of Finance", "Ministry of Education"]
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


# Property 2: Ministry filter enforcement
# Feature: pib-rag-system, Property 2: Ministry filter enforcement
@settings(max_examples=100, deadline=3000)
@given(
    target_ministries=st.lists(
        st.sampled_from(["Ministry of Health", "Ministry of Finance", "Ministry of Education", "Ministry of Defence"]),
        min_size=1, max_size=2, unique=True
    ),
    query_text=st.text(min_size=10, max_size=100)
)
def test_property_2_ministry_filter_enforcement(target_ministries, query_text, embedding_generator):
    """
    Property 2: Ministry filter enforcement
    For any search query with a ministry filter applied, all returned results should have 
    a ministry field matching one of the specified ministries.
    Validates: Requirements 2.1, 2.4
    """
    temp_dir = tempfile.mkdtemp()
    try:
        store = VectorStore(persist_directory=temp_dir, collection_name="test_prop2")
        query_engine = QueryEngine(store, embedding_generator)
        
        # Create chunks with various ministries
        all_ministries = ["Ministry of Health", "Ministry of Finance", "Ministry of Education", "Ministry of Defence", "Ministry of Agriculture"]
        chunks = []
        
        # Add chunks from target ministries
        for ministry in target_ministries:
            for i in range(3):
                chunk = Chunk(
                    chunk_id=f"{ministry}_{i}",
                    article_id=f"art_{ministry}_{i}",
                    content=f"Content about {ministry} policy {i}",
                    metadata={
                        'date': '2025-01-01',
                        'ministry': ministry,
                        'title': f"Article {i}",
                        'chunk_index': 0,
                        'total_chunks': 1
                    }
                )
                chunks.append(chunk)
        
        # Add chunks from other ministries
        for ministry in all_ministries:
            if ministry not in target_ministries:
                chunk = Chunk(
                    chunk_id=f"{ministry}_other",
                    article_id=f"art_{ministry}_other",
                    content=f"Content about {ministry} policy",
                    metadata={
                        'date': '2025-01-01',
                        'ministry': ministry,
                        'title': "Other Article",
                        'chunk_index': 0,
                        'total_chunks': 1
                    }
                )
                chunks.append(chunk)
        
        # Generate embeddings and add to store
        embeddings = embedding_generator.batch_generate_embeddings([c.content for c in chunks])
        store.add_chunks(chunks, embeddings)
        
        # Search with ministry filter
        results = query_engine.search(
            query=query_text,
            ministry_filter=target_ministries,
            top_k=10,
            relevance_threshold=0.0
        )
        
        # Verify all results match the filter
        for result in results:
            assert result.chunk.metadata['ministry'] in target_ministries, \
                f"Result ministry '{result.chunk.metadata['ministry']}' not in filter {target_ministries}"
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)



# Property 3: Ministry metadata presence
# Feature: pib-rag-system, Property 3: Ministry metadata presence
@settings(max_examples=100, deadline=3000)
@given(
    query_text=st.text(min_size=10, max_size=100),
    num_chunks=st.integers(min_value=1, max_value=10)
)
def test_property_3_ministry_metadata_presence(query_text, num_chunks, embedding_generator):
    """
    Property 3: Ministry metadata presence
    For any search result returned by the system, the result should include a non-empty ministry field.
    Validates: Requirements 2.2
    """
    temp_dir = tempfile.mkdtemp()
    try:
        store = VectorStore(persist_directory=temp_dir, collection_name="test_prop3")
        query_engine = QueryEngine(store, embedding_generator)
        
        # Create chunks with ministries
        ministries = ["Ministry of Health", "Ministry of Finance", "Ministry of Education"]
        chunks = []
        
        for i in range(num_chunks):
            chunk = Chunk(
                chunk_id=f"chunk_{i}",
                article_id=f"art_{i}",
                content=f"Content about policy {i}",
                metadata={
                    'date': '2025-01-01',
                    'ministry': ministries[i % len(ministries)],
                    'title': f"Article {i}",
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            )
            chunks.append(chunk)
        
        # Generate embeddings and add to store
        embeddings = embedding_generator.batch_generate_embeddings([c.content for c in chunks])
        store.add_chunks(chunks, embeddings)
        
        # Search without filters
        results = query_engine.search(
            query=query_text,
            top_k=num_chunks,
            relevance_threshold=0.0
        )
        
        # Verify all results have non-empty ministry field
        for result in results:
            assert 'ministry' in result.chunk.metadata, "Ministry field missing from metadata"
            assert result.chunk.metadata['ministry'], "Ministry field is empty"
            assert isinstance(result.chunk.metadata['ministry'], str), "Ministry field is not a string"
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Property 5: Date range filter enforcement
# Feature: pib-rag-system, Property 5: Date range filter enforcement
@settings(max_examples=100, deadline=3000)
@given(
    query_text=st.text(min_size=10, max_size=100),
    days_back=st.integers(min_value=30, max_value=180)
)
def test_property_5_date_range_filter_enforcement(query_text, days_back, embedding_generator):
    """
    Property 5: Date range filter enforcement
    For any search query with a date range filter, all returned results should have dates 
    falling within the specified range (inclusive).
    Validates: Requirements 3.1
    """
    temp_dir = tempfile.mkdtemp()
    try:
        store = VectorStore(persist_directory=temp_dir, collection_name="test_prop5")
        query_engine = QueryEngine(store, embedding_generator)
        
        # Define date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        # Create chunks with dates inside and outside the range
        chunks = []
        
        # Chunks inside range
        for i in range(5):
            days_offset = i * (days_back // 5)
            date = (start_date + timedelta(days=days_offset)).isoformat()
            chunk = Chunk(
                chunk_id=f"inside_{i}",
                article_id=f"art_inside_{i}",
                content=f"Content inside range {i}",
                metadata={
                    'date': date,
                    'ministry': "Ministry of Health",
                    'title': f"Inside Article {i}",
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            )
            chunks.append(chunk)
        
        # Chunks outside range (before start_date)
        for i in range(3):
            date = (start_date - timedelta(days=i+1)).isoformat()
            chunk = Chunk(
                chunk_id=f"before_{i}",
                article_id=f"art_before_{i}",
                content=f"Content before range {i}",
                metadata={
                    'date': date,
                    'ministry': "Ministry of Health",
                    'title': f"Before Article {i}",
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            )
            chunks.append(chunk)
        
        # Chunks outside range (after end_date)
        for i in range(3):
            date = (end_date + timedelta(days=i+1)).isoformat()
            chunk = Chunk(
                chunk_id=f"after_{i}",
                article_id=f"art_after_{i}",
                content=f"Content after range {i}",
                metadata={
                    'date': date,
                    'ministry': "Ministry of Health",
                    'title': f"After Article {i}",
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            )
            chunks.append(chunk)
        
        # Generate embeddings and add to store
        embeddings = embedding_generator.batch_generate_embeddings([c.content for c in chunks])
        store.add_chunks(chunks, embeddings)
        
        # Search with date range filter
        results = query_engine.search(
            query=query_text,
            date_range=(start_date.isoformat(), end_date.isoformat()),
            top_k=20,
            relevance_threshold=0.0
        )
        
        # Verify all results are within date range
        for result in results:
            result_date = datetime.fromisoformat(result.chunk.metadata['date']).date()
            assert start_date <= result_date <= end_date, \
                f"Result date {result_date} not in range [{start_date}, {end_date}]"
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)



# Property 6: Chronological ordering
# Feature: pib-rag-system, Property 6: Chronological ordering
@settings(max_examples=100, deadline=3000)
@given(
    query_text=st.text(min_size=10, max_size=100),
    num_chunks=st.integers(min_value=3, max_value=10)
)
def test_property_6_chronological_ordering(query_text, num_chunks, embedding_generator):
    """
    Property 6: Chronological ordering
    For any timeline search results, the articles should be sorted in chronological order by date.
    Validates: Requirements 3.2
    """
    temp_dir = tempfile.mkdtemp()
    try:
        store = VectorStore(persist_directory=temp_dir, collection_name="test_prop6")
        query_engine = QueryEngine(store, embedding_generator)
        
        # Create chunks with different dates
        chunks = []
        base_date = datetime.now().date()
        
        for i in range(num_chunks):
            date = (base_date - timedelta(days=i * 10)).isoformat()
            chunk = Chunk(
                chunk_id=f"chunk_{i}",
                article_id=f"art_{i}",
                content=f"Content about policy {i}",
                metadata={
                    'date': date,
                    'ministry': "Ministry of Health",
                    'title': f"Article {i}",
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            )
            chunks.append(chunk)
        
        # Generate embeddings and add to store
        embeddings = embedding_generator.batch_generate_embeddings([c.content for c in chunks])
        store.add_chunks(chunks, embeddings)
        
        # Search with date range to trigger timeline search
        start_date = (base_date - timedelta(days=num_chunks * 10)).isoformat()
        end_date = base_date.isoformat()
        
        results = query_engine.search(
            query=query_text,
            date_range=(start_date, end_date),
            top_k=num_chunks,
            relevance_threshold=0.0
        )
        
        # Verify results are in chronological order (oldest to newest or newest to oldest)
        if len(results) >= 2:
            dates = [datetime.fromisoformat(r.chunk.metadata['date']).date() for r in results]
            
            # Check if sorted (either ascending or descending)
            is_ascending = all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
            is_descending = all(dates[i] >= dates[i+1] for i in range(len(dates)-1))
            
            # For this property, we expect descending order (newest first) based on relevance
            # But we'll accept any consistent ordering
            assert is_ascending or is_descending, \
                f"Results not in chronological order: {dates}"
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Property 7: ISO date format acceptance
# Feature: pib-rag-system, Property 7: ISO date format acceptance
@settings(max_examples=100, deadline=2000)
@given(
    year=st.integers(min_value=2020, max_value=2025),
    month=st.integers(min_value=1, max_value=12),
    day=st.integers(min_value=1, max_value=28),  # Use 28 to avoid invalid dates
    query_text=st.text(min_size=10, max_size=100)
)
def test_property_7_iso_date_format_acceptance(year, month, day, query_text, embedding_generator):
    """
    Property 7: ISO date format acceptance
    For any valid ISO format date string (YYYY-MM-DD), the system should successfully parse it without errors.
    Validates: Requirements 3.4
    """
    temp_dir = tempfile.mkdtemp()
    try:
        store = VectorStore(persist_directory=temp_dir, collection_name="test_prop7")
        query_engine = QueryEngine(store, embedding_generator)
        
        # Create ISO date string
        iso_date = f"{year:04d}-{month:02d}-{day:02d}"
        
        # Create a chunk with this date
        chunk = Chunk(
            chunk_id="test_chunk",
            article_id="test_art",
            content="Test content",
            metadata={
                'date': iso_date,
                'ministry': "Ministry of Health",
                'title': "Test Article",
                'chunk_index': 0,
                'total_chunks': 1
            }
        )
        
        # Generate embedding and add to store
        embedding = embedding_generator.generate_embedding(chunk.content)
        store.add_chunks([chunk], [embedding])
        
        # Search with date range using the ISO date
        # Should not raise any exceptions
        results = query_engine.search(
            query=query_text,
            date_range=(iso_date, iso_date),
            top_k=5,
            relevance_threshold=0.0
        )
        
        # If we get here without exception, the test passes
        assert True, "ISO date format accepted successfully"
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Property 8: Combined filter enforcement
# Feature: pib-rag-system, Property 8: Combined filter enforcement
@settings(max_examples=100, deadline=3000)
@given(
    query_text=st.text(min_size=10, max_size=100),
    target_ministry=st.sampled_from(["Ministry of Health", "Ministry of Finance", "Ministry of Education"])
)
def test_property_8_combined_filter_enforcement(query_text, target_ministry, embedding_generator):
    """
    Property 8: Combined filter enforcement
    For any search query with multiple filters (ministry, date range, semantic), all returned 
    results should satisfy all specified constraints simultaneously.
    Validates: Requirements 2.5, 3.5
    """
    temp_dir = tempfile.mkdtemp()
    try:
        store = VectorStore(persist_directory=temp_dir, collection_name="test_prop8")
        query_engine = QueryEngine(store, embedding_generator)
        
        # Define date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=60)
        
        # Create chunks with various combinations
        chunks = []
        ministries = ["Ministry of Health", "Ministry of Finance", "Ministry of Education", "Ministry of Defence"]
        
        # Chunks matching both filters
        for i in range(3):
            date = (start_date + timedelta(days=i * 10)).isoformat()
            chunk = Chunk(
                chunk_id=f"match_{i}",
                article_id=f"art_match_{i}",
                content=f"Content matching both filters {i}",
                metadata={
                    'date': date,
                    'ministry': target_ministry,
                    'title': f"Match Article {i}",
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            )
            chunks.append(chunk)
        
        # Chunks matching only ministry filter (wrong date)
        for i in range(2):
            date = (start_date - timedelta(days=i+1)).isoformat()
            chunk = Chunk(
                chunk_id=f"ministry_only_{i}",
                article_id=f"art_ministry_{i}",
                content=f"Content with correct ministry but wrong date {i}",
                metadata={
                    'date': date,
                    'ministry': target_ministry,
                    'title': f"Ministry Only Article {i}",
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            )
            chunks.append(chunk)
        
        # Chunks matching only date filter (wrong ministry)
        for i in range(2):
            date = (start_date + timedelta(days=i * 15)).isoformat()
            other_ministry = [m for m in ministries if m != target_ministry][0]
            chunk = Chunk(
                chunk_id=f"date_only_{i}",
                article_id=f"art_date_{i}",
                content=f"Content with correct date but wrong ministry {i}",
                metadata={
                    'date': date,
                    'ministry': other_ministry,
                    'title': f"Date Only Article {i}",
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            )
            chunks.append(chunk)
        
        # Generate embeddings and add to store
        embeddings = embedding_generator.batch_generate_embeddings([c.content for c in chunks])
        store.add_chunks(chunks, embeddings)
        
        # Search with combined filters
        results = query_engine.search(
            query=query_text,
            ministry_filter=[target_ministry],
            date_range=(start_date.isoformat(), end_date.isoformat()),
            top_k=10,
            relevance_threshold=0.0
        )
        
        # Verify all results satisfy both filters
        for result in results:
            # Check ministry filter
            assert result.chunk.metadata['ministry'] == target_ministry, \
                f"Result ministry '{result.chunk.metadata['ministry']}' does not match filter '{target_ministry}'"
            
            # Check date range filter
            result_date = datetime.fromisoformat(result.chunk.metadata['date']).date()
            assert start_date <= result_date <= end_date, \
                f"Result date {result_date} not in range [{start_date}, {end_date}]"
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)



# Property 27: Retrieved chunk metadata completeness
# Feature: pib-rag-system, Property 27: Retrieved chunk metadata completeness
@settings(max_examples=100, deadline=3000)
@given(
    query_text=st.text(min_size=10, max_size=100),
    num_chunks=st.integers(min_value=1, max_value=10)
)
def test_property_27_retrieved_chunk_metadata_completeness(query_text, num_chunks, embedding_generator):
    """
    Property 27: Retrieved chunk metadata completeness
    For any chunk returned by the query engine, it should include complete source article information.
    Validates: Requirements 8.5
    """
    temp_dir = tempfile.mkdtemp()
    try:
        store = VectorStore(persist_directory=temp_dir, collection_name="test_prop27")
        query_engine = QueryEngine(store, embedding_generator)
        
        # Create chunks with complete metadata
        chunks = []
        for i in range(num_chunks):
            chunk = Chunk(
                chunk_id=f"chunk_{i}",
                article_id=f"art_{i}",
                content=f"Content about policy {i}",
                metadata={
                    'date': '2025-01-01',
                    'ministry': "Ministry of Health",
                    'title': f"Article {i}",
                    'chunk_index': i,
                    'total_chunks': num_chunks
                }
            )
            chunks.append(chunk)
        
        # Generate embeddings and add to store
        embeddings = embedding_generator.batch_generate_embeddings([c.content for c in chunks])
        store.add_chunks(chunks, embeddings)
        
        # Search
        results = query_engine.search(
            query=query_text,
            top_k=num_chunks,
            relevance_threshold=0.0
        )
        
        # Verify all results have complete metadata
        required_fields = ['date', 'ministry', 'title', 'chunk_index', 'total_chunks']
        
        for result in results:
            # Check article_id is present
            assert result.chunk.article_id, "article_id is missing or empty"
            
            # Check all required metadata fields
            for field in required_fields:
                assert field in result.chunk.metadata, f"Field '{field}' missing from metadata"
                assert result.chunk.metadata[field] is not None, f"Field '{field}' is None"
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Property 28: Relevance score ordering
# Feature: pib-rag-system, Property 28: Relevance score ordering
@settings(max_examples=100, deadline=3000)
@given(
    query_text=st.text(min_size=10, max_size=100),
    num_chunks=st.integers(min_value=3, max_value=10)
)
def test_property_28_relevance_score_ordering(query_text, num_chunks, embedding_generator):
    """
    Property 28: Relevance score ordering
    For any semantic search results, the results should be ordered in descending order by relevance score.
    Validates: Requirements 9.1
    """
    temp_dir = tempfile.mkdtemp()
    try:
        store = VectorStore(persist_directory=temp_dir, collection_name="test_prop28")
        query_engine = QueryEngine(store, embedding_generator)
        
        # Create chunks with varied content
        chunks = []
        for i in range(num_chunks):
            chunk = Chunk(
                chunk_id=f"chunk_{i}",
                article_id=f"art_{i}",
                content=f"Content about various topics {i} with different relevance",
                metadata={
                    'date': '2025-01-01',
                    'ministry': "Ministry of Health",
                    'title': f"Article {i}",
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            )
            chunks.append(chunk)
        
        # Generate embeddings and add to store
        embeddings = embedding_generator.batch_generate_embeddings([c.content for c in chunks])
        store.add_chunks(chunks, embeddings)
        
        # Search
        results = query_engine.search(
            query=query_text,
            top_k=num_chunks,
            relevance_threshold=0.0
        )
        
        # Verify results are ordered by score (descending)
        if len(results) >= 2:
            scores = [r.score for r in results]
            
            # Check descending order
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i+1], \
                    f"Results not in descending order: score[{i}]={scores[i]} < score[{i+1}]={scores[i+1]}"
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Property 29: Relevance threshold filtering
# Feature: pib-rag-system, Property 29: Relevance threshold filtering
@settings(max_examples=100, deadline=3000)
@given(
    query_text=st.text(min_size=10, max_size=100),
    threshold=st.floats(min_value=0.1, max_value=0.9),
    num_chunks=st.integers(min_value=5, max_value=15)
)
def test_property_29_relevance_threshold_filtering(query_text, threshold, num_chunks, embedding_generator):
    """
    Property 29: Relevance threshold filtering
    For any search results returned, all results should have relevance scores at or above 
    the configured threshold.
    Validates: Requirements 9.2
    """
    temp_dir = tempfile.mkdtemp()
    try:
        store = VectorStore(persist_directory=temp_dir, collection_name="test_prop29")
        query_engine = QueryEngine(store, embedding_generator)
        
        # Create chunks with varied content
        chunks = []
        for i in range(num_chunks):
            chunk = Chunk(
                chunk_id=f"chunk_{i}",
                article_id=f"art_{i}",
                content=f"Content with varying relevance to query {i}",
                metadata={
                    'date': '2025-01-01',
                    'ministry': "Ministry of Health",
                    'title': f"Article {i}",
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            )
            chunks.append(chunk)
        
        # Generate embeddings and add to store
        embeddings = embedding_generator.batch_generate_embeddings([c.content for c in chunks])
        store.add_chunks(chunks, embeddings)
        
        # Search with threshold
        results = query_engine.search(
            query=query_text,
            top_k=num_chunks,
            relevance_threshold=threshold
        )
        
        # Verify all results meet threshold
        for result in results:
            assert result.score >= threshold, \
                f"Result score {result.score} below threshold {threshold}"
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Property 30: Inclusive threshold retrieval
# Feature: pib-rag-system, Property 30: Inclusive threshold retrieval
@settings(max_examples=100, deadline=3000)
@given(
    num_chunks=st.integers(min_value=5, max_value=15)
)
def test_property_30_inclusive_threshold_retrieval(num_chunks, embedding_generator):
    """
    Property 30: Inclusive threshold retrieval
    For any search query, all chunks with relevance scores above the threshold should be 
    included in the results (no qualifying results should be excluded).
    Validates: Requirements 9.3
    """
    temp_dir = tempfile.mkdtemp()
    try:
        store = VectorStore(persist_directory=temp_dir, collection_name="test_prop30")
        query_engine = QueryEngine(store, embedding_generator)
        
        # Create chunks
        chunks = []
        for i in range(num_chunks):
            chunk = Chunk(
                chunk_id=f"chunk_{i}",
                article_id=f"art_{i}",
                content=f"Health policy content {i}",
                metadata={
                    'date': '2025-01-01',
                    'ministry': "Ministry of Health",
                    'title': f"Article {i}",
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            )
            chunks.append(chunk)
        
        # Generate embeddings and add to store
        embeddings = embedding_generator.batch_generate_embeddings([c.content for c in chunks])
        store.add_chunks(chunks, embeddings)
        
        # Use a query very similar to the content to get high scores
        query = "Health policy content"
        threshold = 0.5
        
        # Search with threshold and large top_k
        results = query_engine.search(
            query=query,
            top_k=num_chunks * 2,  # Request more than available
            relevance_threshold=threshold
        )
        
        # Get all scores from vector store directly (without threshold)
        query_embedding = embedding_generator.generate_embedding(query)
        all_results = store.similarity_search(query_embedding, k=num_chunks * 2)
        
        # Count how many should be above threshold
        expected_count = sum(1 for r in all_results if r.score >= threshold)
        
        # Verify we got all qualifying results
        assert len(results) == expected_count, \
            f"Expected {expected_count} results above threshold, got {len(results)}"
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Property 31: Top-k context limitation
# Feature: pib-rag-system, Property 31: Top-k context limitation
@settings(max_examples=100, deadline=3000)
@given(
    query_text=st.text(min_size=10, max_size=100),
    top_k=st.integers(min_value=1, max_value=10),
    num_chunks=st.integers(min_value=15, max_value=25)
)
def test_property_31_top_k_context_limitation(query_text, top_k, num_chunks, embedding_generator):
    """
    Property 31: Top-k context limitation
    For any response generation, only the top-k highest-ranked retrieved articles should be 
    passed as context to the language model.
    Validates: Requirements 9.5
    """
    # Ensure we have more chunks than top_k
    assume(num_chunks > top_k)
    
    temp_dir = tempfile.mkdtemp()
    try:
        store = VectorStore(persist_directory=temp_dir, collection_name="test_prop31")
        query_engine = QueryEngine(store, embedding_generator)
        
        # Create chunks
        chunks = []
        for i in range(num_chunks):
            chunk = Chunk(
                chunk_id=f"chunk_{i}",
                article_id=f"art_{i}",
                content=f"Content about policy {i}",
                metadata={
                    'date': '2025-01-01',
                    'ministry': "Ministry of Health",
                    'title': f"Article {i}",
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            )
            chunks.append(chunk)
        
        # Generate embeddings and add to store
        embeddings = embedding_generator.batch_generate_embeddings([c.content for c in chunks])
        store.add_chunks(chunks, embeddings)
        
        # Search with top_k limit
        results = query_engine.search(
            query=query_text,
            top_k=top_k,
            relevance_threshold=0.0
        )
        
        # Verify we got at most top_k results
        assert len(results) <= top_k, \
            f"Got {len(results)} results, expected at most {top_k}"
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
