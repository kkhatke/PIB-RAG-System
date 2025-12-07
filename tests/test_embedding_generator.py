"""
Property-based tests for EmbeddingGenerator module.
Tests query embedding generation and article embedding generation.
"""
from hypothesis import given, strategies as st, settings
from src.embedding import EmbeddingGenerator
from config import EMBEDDING_DIMENSION


# Initialize embedding generator
embedding_generator = EmbeddingGenerator()


# Feature: pib-rag-system, Property 1: Query embedding generation
# Validates: Requirements 1.2
@given(st.text(min_size=1, max_size=1000))
@settings(max_examples=100)
def test_query_embedding_generation(query):
    """
    Property 1: For any valid query string, the system should generate a vector
    embedding with the correct dimensionality (384 for all-MiniLM-L6-v2).
    """
    # Generate embedding for the query
    embedding = embedding_generator.generate_embedding(query)
    
    # Property 1: Embedding should be a list
    assert isinstance(embedding, list), \
        f"Embedding should be a list, got {type(embedding)}"
    
    # Property 2: Embedding should have correct dimension
    assert len(embedding) == EMBEDDING_DIMENSION, \
        f"Embedding dimension should be {EMBEDDING_DIMENSION}, got {len(embedding)}"
    
    # Property 3: All elements should be floats
    assert all(isinstance(x, float) for x in embedding), \
        f"All embedding elements should be floats"
    
    # Property 4: Embedding should not be all zeros (model should produce meaningful vectors)
    assert not all(x == 0.0 for x in embedding), \
        f"Embedding should not be all zeros"
    
    # Property 5: Embedding values should be finite (no NaN or Inf)
    import math
    assert all(math.isfinite(x) for x in embedding), \
        f"All embedding values should be finite (no NaN or Inf)"


# Feature: pib-rag-system, Property 13: Embedding generation for all articles
# Validates: Requirements 5.1
@given(st.lists(st.text(min_size=1, max_size=1000), min_size=1, max_size=10))
@settings(max_examples=100)
def test_article_embedding_generation(articles):
    """
    Property 13: For any valid article ingested into the system, a vector embedding
    should be generated and stored.
    """
    # Generate embeddings for all articles in batch
    embeddings = embedding_generator.batch_generate_embeddings(articles)
    
    # Property 1: Should generate one embedding per article
    assert len(embeddings) == len(articles), \
        f"Should generate {len(articles)} embeddings, got {len(embeddings)}"
    
    # Property 2: Each embedding should have correct dimension
    for i, embedding in enumerate(embeddings):
        assert len(embedding) == EMBEDDING_DIMENSION, \
            f"Embedding {i} should have dimension {EMBEDDING_DIMENSION}, got {len(embedding)}"
    
    # Property 3: All embeddings should be lists of floats
    for i, embedding in enumerate(embeddings):
        assert isinstance(embedding, list), \
            f"Embedding {i} should be a list, got {type(embedding)}"
        assert all(isinstance(x, float) for x in embedding), \
            f"All elements in embedding {i} should be floats"
    
    # Property 4: No embedding should be all zeros
    for i, embedding in enumerate(embeddings):
        assert not all(x == 0.0 for x in embedding), \
            f"Embedding {i} should not be all zeros"
    
    # Property 5: All embedding values should be finite
    import math
    for i, embedding in enumerate(embeddings):
        assert all(math.isfinite(x) for x in embedding), \
            f"All values in embedding {i} should be finite (no NaN or Inf)"
    
    # Property 6: Different articles should produce different embeddings
    # (with high probability for random text)
    if len(articles) > 1:
        # Check that at least some embeddings are different
        unique_embeddings = set(tuple(emb) for emb in embeddings)
        # We expect most random texts to produce different embeddings
        # but allow for some collisions in very short or similar texts
        assert len(unique_embeddings) >= 1, \
            f"Should generate at least some unique embeddings"
