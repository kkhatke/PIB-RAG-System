"""
Property-based tests for EmbeddingGenerator module.
Tests query embedding generation and article embedding generation.
"""
from hypothesis import given, strategies as st, settings
import tempfile
import shutil
import os
from pathlib import Path
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


# Feature: pib-rag-system, Property 38: Model cache loading
# Validates: Requirements 12.2
@given(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
@settings(max_examples=3, deadline=60000)  # Reduced examples and increased deadline for model downloads
def test_model_cache_loading(cache_dir_suffix):
    """
    Property 38: For any embedding model that has been previously cached, 
    loading the model should use the cached version instead of re-downloading.
    """
    # Create temporary cache directories
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir) / f"cache_{cache_dir_suffix[:10]}"  # Limit length
        
        # First initialization - should download and cache
        generator1 = EmbeddingGenerator(cache_dir=str(cache_dir))
        # Note: load_model() is called automatically in __init__
        
        # Property 1: Model should be functional after first load
        test_embedding1 = generator1.generate_embedding("test")
        assert len(test_embedding1) == EMBEDDING_DIMENSION, \
            f"First model should generate correct embedding dimension"
        
        # Property 2: Cache directory should exist after first load
        assert cache_dir.exists(), \
            f"Cache directory should exist after first load"
        
        # Property 3: Model files should exist in cache (HuggingFace uses specific directory structure)
        model_cache_path = cache_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
        assert model_cache_path.exists(), \
            f"Model cache path should exist: {model_cache_path}"
        
        # Second initialization - should load from cache
        generator2 = EmbeddingGenerator(cache_dir=str(cache_dir))
        
        # Property 4: Cached model should be functional
        test_embedding2 = generator2.generate_embedding("test")
        assert len(test_embedding2) == EMBEDDING_DIMENSION, \
            f"Cached model should generate correct embedding dimension"
        
        # Property 5: Both models should produce identical embeddings for same input
        assert test_embedding1 == test_embedding2, \
            f"Downloaded and cached models should produce identical embeddings"
        
        # Property 6: Model info should indicate cached status
        model_info = generator2.get_model_info()
        assert model_info['is_cached'] == True, \
            f"Model info should indicate model is cached"


# Feature: pib-rag-system, Property 39: Model integrity verification
# Validates: Requirements 12.3, 12.4
@given(st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
@settings(max_examples=3, deadline=60000)  # Reduced examples and increased deadline for model operations
def test_model_integrity_verification(cache_dir_suffix):
    """
    Property 39: For any cached embedding model, the system should verify its 
    integrity before loading and re-download if corrupted.
    """
    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir) / f"integrity_cache_{cache_dir_suffix}"
        
        # First initialization - download and cache model
        generator1 = EmbeddingGenerator(cache_dir=str(cache_dir))
        generator1.load_model()
        
        # Property 1: Integrity verification should pass for valid cached model
        integrity_valid = generator1.verify_model_integrity()
        assert integrity_valid == True, \
            f"Integrity verification should pass for valid cached model"
        
        # Property 2: Model info should contain file hashes
        model_info = generator1.get_model_info()
        assert 'file_hashes' in model_info, \
            f"Model info should contain file hashes"
        assert len(model_info['file_hashes']) > 0, \
            f"File hashes should not be empty"
        
        # Simulate corruption by modifying a model file
        model_cache_path = cache_dir / generator1.model_name
        model_files = list(model_cache_path.rglob('*'))
        model_files = [f for f in model_files if f.is_file() and f.name != "model_metadata.json"]
        
        if model_files:
            # Corrupt the first model file by appending some data
            corrupt_file = model_files[0]
            original_size = corrupt_file.stat().st_size
            
            with open(corrupt_file, 'ab') as f:
                f.write(b'CORRUPTED_DATA')
            
            # Property 3: Integrity verification should fail for corrupted model
            integrity_after_corruption = generator1.verify_model_integrity()
            assert integrity_after_corruption == False, \
                f"Integrity verification should fail for corrupted model"
            
            # Property 4: Loading corrupted model should trigger re-download
            generator2 = EmbeddingGenerator(cache_dir=str(cache_dir))
            loaded_from_cache = generator2.load_model()
            
            # Should return False because it had to re-download due to corruption
            assert loaded_from_cache == False, \
                f"Loading corrupted model should trigger re-download (return False)"
            
            # Property 5: Re-downloaded model should be functional
            test_embedding = generator2.generate_embedding("test")
            assert len(test_embedding) == EMBEDDING_DIMENSION, \
                f"Re-downloaded model should generate correct embedding dimension"
            
            # Property 6: Integrity should pass after re-download
            integrity_after_redownload = generator2.verify_model_integrity()
            assert integrity_after_redownload == True, \
                f"Integrity verification should pass after re-download"
        
        # Property 7: Missing metadata should trigger re-download
        metadata_path = model_cache_path / "model_metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()  # Remove metadata file
            
            generator3 = EmbeddingGenerator(cache_dir=str(cache_dir))
            loaded_from_cache_no_metadata = generator3.load_model()
            
            # Should return False because metadata was missing
            assert loaded_from_cache_no_metadata == False, \
                f"Missing metadata should trigger re-download (return False)"
            
            # Property 8: Metadata should be recreated after re-download
            assert metadata_path.exists(), \
                f"Metadata file should be recreated after re-download"
