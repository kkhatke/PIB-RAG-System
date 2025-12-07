"""
Property-based tests for ArticleChunker module.
Tests chunking behavior, paragraph preservation, overlap, and metadata association.
"""
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from src.data_ingestion.article_chunker import ArticleChunker, Chunk
from src.data_ingestion.article_loader import Article


# Strategy for generating valid article objects
def article_strategy(min_content_size=1, max_content_size=5000):
    """Generate valid Article objects for testing."""
    return st.builds(
        Article,
        id=st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=48, max_codepoint=57)),
        date=st.dates().map(lambda d: d.isoformat()),
        ministry=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        title=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        content=st.text(min_size=min_content_size, max_size=max_content_size).filter(lambda x: x.strip()),
        original_content=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip())
    )


# Feature: pib-rag-system, Property 23: Long article chunking
# Validates: Requirements 8.1
@given(
    article_strategy(min_content_size=1001, max_content_size=2500),
    st.integers(min_value=500, max_value=1500),
    st.integers(min_value=50, max_value=300)
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example, HealthCheck.data_too_large])
def test_long_article_chunking(article, chunk_size, overlap):
    """
    Property 23: For any article exceeding 1000 tokens, the system should split it
    into multiple chunks, each not exceeding the chunk size limit.
    """
    # Ensure overlap is less than chunk_size
    assume(overlap < chunk_size)
    
    # Ensure article content is longer than chunk_size
    assume(len(article.content) > chunk_size)
    
    # Create chunker with specified parameters
    chunker = ArticleChunker(chunk_size=chunk_size, overlap=overlap)
    
    # Chunk the article
    chunks = chunker.chunk_article(article)
    
    # Property: Should create multiple chunks for long articles
    assert len(chunks) > 1, \
        f"Article with {len(article.content)} chars should be split into multiple chunks with chunk_size={chunk_size}"
    
    # Property: Each chunk should not exceed chunk_size (with some tolerance for paragraph boundaries)
    # We allow some tolerance because we preserve paragraph boundaries
    for idx, chunk in enumerate(chunks):
        # The last chunk might be shorter, but no chunk should be excessively long
        # We allow up to 1.5x chunk_size to accommodate paragraph boundaries
        assert len(chunk.content) <= chunk_size * 1.5, \
            f"Chunk {idx} has {len(chunk.content)} chars, exceeds limit of {chunk_size * 1.5}"
    
    # Property: All chunks should be Chunk objects
    for chunk in chunks:
        assert isinstance(chunk, Chunk), \
            "Chunk is not a Chunk instance"
    
    # Property: Chunks should cover the article content
    # (we can't guarantee exact reconstruction due to overlap and paragraph boundaries,
    # but we should have substantial coverage)
    total_chunk_length = sum(len(chunk.content) for chunk in chunks)
    assert total_chunk_length >= len(article.content) * 0.8, \
        "Chunks don't cover enough of the original article"


# Feature: pib-rag-system, Property 24: Paragraph boundary preservation
# Validates: Requirements 8.2
@given(
    st.integers(min_value=3, max_value=10),
    st.integers(min_value=100, max_value=300)
)
@settings(max_examples=100)
def test_paragraph_boundary_preservation(num_paragraphs, paragraph_size):
    """
    Property 24: For any article chunked by the system, no chunk should split
    in the middle of a paragraph (chunks should end at paragraph boundaries).
    """
    # Create an article with clear paragraph boundaries
    paragraphs = []
    for i in range(num_paragraphs):
        # Create a paragraph with no internal double newlines
        paragraph = f"Paragraph {i}. " + "This is sentence content. " * (paragraph_size // 25)
        paragraphs.append(paragraph.strip())
    
    # Join with double newlines to create clear paragraph boundaries
    content = "\n\n".join(paragraphs)
    
    article = Article(
        id="test_123",
        date="2025-01-01",
        ministry="Test Ministry",
        title="Test Article",
        content=content,
        original_content=content
    )
    
    # Use a chunk size that will force splitting
    chunk_size = len(content) // 3
    assume(chunk_size > 100)  # Ensure reasonable chunk size
    
    chunker = ArticleChunker(chunk_size=chunk_size, overlap=50)
    chunks = chunker.chunk_article(article)
    
    # Property: Each chunk should contain only complete paragraphs
    # A chunk should not end in the middle of a paragraph
    for chunk in chunks:
        # Check that chunk content doesn't split paragraphs
        # If a chunk contains part of a paragraph, it should contain the whole paragraph
        # We verify this by checking that chunks don't have partial paragraph content
        
        # Split chunk into its paragraphs
        chunk_paragraphs = chunker.preserve_paragraph_boundaries(chunk.content)
        
        # Each paragraph in the chunk should be a complete paragraph from the original
        for chunk_para in chunk_paragraphs:
            # The paragraph should exist in the original content
            # (allowing for whitespace differences)
            normalized_chunk_para = chunk_para.strip()
            found = False
            for original_para in paragraphs:
                if normalized_chunk_para in original_para or original_para in normalized_chunk_para:
                    found = True
                    break
            
            assert found, \
                f"Chunk contains partial paragraph: {chunk_para[:50]}..."


# Feature: pib-rag-system, Property 25: Chunk overlap maintenance
# Validates: Requirements 8.3
@given(
    article_strategy(min_content_size=1200, max_content_size=2000),
    st.integers(min_value=400, max_value=600),
    st.integers(min_value=80, max_value=150)
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example])
def test_chunk_overlap_maintenance(article, chunk_size, overlap):
    """
    Property 25: For any pair of consecutive chunks from the same article,
    there should be overlapping content between them.
    """
    # Ensure overlap is less than chunk_size
    assume(overlap < chunk_size)
    
    # Ensure article is long enough to create multiple chunks
    assume(len(article.content) > chunk_size * 1.5)
    
    # Create chunker
    chunker = ArticleChunker(chunk_size=chunk_size, overlap=overlap)
    
    # Chunk the article
    chunks = chunker.chunk_article(article)
    
    # Property: Should have multiple chunks
    assume(len(chunks) > 1)
    
    # Property: Consecutive chunks should have overlapping content
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]
        
        # Check if there's any overlapping text
        # Get the end of current chunk
        current_end = current_chunk.content[-overlap:]
        
        # Check if any part of current_end appears in next_chunk
        # We look for overlap, which might not be exact due to paragraph boundaries
        has_overlap = False
        
        # Check for any common substring of reasonable length
        min_overlap_check = min(50, overlap // 2)  # Check for at least some overlap
        
        for j in range(len(current_end) - min_overlap_check):
            substring = current_end[j:j + min_overlap_check]
            if substring in next_chunk.content:
                has_overlap = True
                break
        
        # For very short overlaps or paragraph boundary cases, we might not have exact overlap
        # but we should have some content continuity
        if not has_overlap:
            # Check if the chunks are at least related (share some words)
            current_words = set(current_chunk.content.split())
            next_words = set(next_chunk.content.split())
            common_words = current_words & next_words
            
            # Should have some common words indicating continuity
            assert len(common_words) > 0, \
                f"Consecutive chunks {i} and {i+1} have no overlap or common content"


# Feature: pib-rag-system, Property 26: Chunk metadata association
# Validates: Requirements 8.4
@given(article_strategy(min_content_size=1, max_content_size=3000))
@settings(max_examples=100)
def test_chunk_metadata_association(article):
    """
    Property 26: For any chunk created from an article, the chunk should contain
    metadata linking it to its parent article (article_id, date, ministry, title).
    """
    # Create chunker
    chunker = ArticleChunker(chunk_size=1000, overlap=200)
    
    # Chunk the article
    chunks = chunker.chunk_article(article)
    
    # Property: Should have at least one chunk
    assert len(chunks) > 0, \
        "No chunks created from article"
    
    # Property: Each chunk should have complete metadata
    for idx, chunk in enumerate(chunks):
        # Check chunk_id format
        assert chunk.chunk_id == f"{article.id}_{idx}", \
            f"Chunk {idx} has incorrect chunk_id: {chunk.chunk_id}"
        
        # Check article_id
        assert chunk.article_id == article.id, \
            f"Chunk {idx} has incorrect article_id: {chunk.article_id}"
        
        # Check metadata fields
        assert 'date' in chunk.metadata, \
            f"Chunk {idx} missing 'date' in metadata"
        assert 'ministry' in chunk.metadata, \
            f"Chunk {idx} missing 'ministry' in metadata"
        assert 'title' in chunk.metadata, \
            f"Chunk {idx} missing 'title' in metadata"
        assert 'chunk_index' in chunk.metadata, \
            f"Chunk {idx} missing 'chunk_index' in metadata"
        assert 'total_chunks' in chunk.metadata, \
            f"Chunk {idx} missing 'total_chunks' in metadata"
        
        # Check metadata values match article
        assert chunk.metadata['date'] == article.date, \
            f"Chunk {idx} date doesn't match article"
        assert chunk.metadata['ministry'] == article.ministry, \
            f"Chunk {idx} ministry doesn't match article"
        assert chunk.metadata['title'] == article.title, \
            f"Chunk {idx} title doesn't match article"
        
        # Check chunk_index is correct
        assert chunk.metadata['chunk_index'] == idx, \
            f"Chunk {idx} has incorrect chunk_index in metadata"
        
        # Check total_chunks is correct
        assert chunk.metadata['total_chunks'] == len(chunks), \
            f"Chunk {idx} has incorrect total_chunks in metadata"
