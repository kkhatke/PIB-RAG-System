"""
Property-based tests for ArticleLoader module.
Tests JSON parsing, field extraction, validation, deduplication, and content preservation.
"""
import json
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
from src.data_ingestion.article_loader import ArticleLoader, Article


# Initialize loader
loader = ArticleLoader()


# Strategy for generating valid article dictionaries
def article_dict_strategy():
    """Generate valid article dictionaries for testing."""
    return st.fixed_dictionaries({
        'id': st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=48, max_codepoint=57)),  # digits
        'date': st.dates().map(lambda d: d.isoformat()),
        'ministry': st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),  # Ensure not just whitespace
        'title': st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),  # Ensure not just whitespace
        'content': st.text(min_size=1, max_size=1000).filter(lambda x: x.strip())  # Ensure not just whitespace
    })


# Feature: pib-rag-system, Property 19: JSON parsing correctness
# Validates: Requirements 7.1
@given(st.lists(article_dict_strategy(), min_size=1, max_size=20))
@settings(max_examples=100)
def test_json_parsing_correctness(articles_data):
    """
    Property 19: For any valid JSON file containing PIB articles in the expected format,
    the system should successfully parse and extract all articles.
    """
    # Filter out articles that would be invalid after normalization
    # (e.g., content that is only whitespace becomes empty after normalization)
    valid_input_count = sum(
        1 for article in articles_data
        if all(str(article.get(field, '')).strip() for field in ['id', 'date', 'ministry', 'title', 'content'])
    )
    
    # Skip test if no valid articles
    assume(valid_input_count > 0)
    
    # Create a temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(articles_data, f)
        temp_path = f.name
    
    try:
        # Create a fresh loader for each test
        test_loader = ArticleLoader()
        
        # Load articles from the JSON file
        loaded_articles = test_loader.load_articles(temp_path)
        
        # Property: Should load at least some articles if input had valid ones
        assert len(loaded_articles) > 0, \
            "Failed to parse any articles from valid JSON"
        
        # Property: Number of loaded articles should not exceed input
        assert len(loaded_articles) <= len(articles_data), \
            f"Loaded more articles ({len(loaded_articles)}) than in input ({len(articles_data)})"
        
        # Property: Each loaded article should have all required fields
        for article in loaded_articles:
            assert isinstance(article, Article), \
                "Loaded item is not an Article instance"
            assert article.id, "Article missing id"
            assert article.date, "Article missing date"
            assert article.ministry, "Article missing ministry"
            assert article.title, "Article missing title"
            assert article.content, "Article missing content"
            assert article.original_content, "Article missing original_content"
    
    finally:
        # Clean up temporary file
        Path(temp_path).unlink(missing_ok=True)


# Feature: pib-rag-system, Property 20: Required field extraction
# Validates: Requirements 7.2
@given(article_dict_strategy())
@settings(max_examples=100)
def test_required_field_extraction(article_data):
    """
    Property 20: For any valid article parsed from JSON, all required fields
    (id, date, ministry, title, content) should be extracted and non-empty.
    """
    # Create a temporary JSON file with single article
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump([article_data], f)
        temp_path = f.name
    
    try:
        # Create a fresh loader for each test
        test_loader = ArticleLoader()
        
        # Load the article
        loaded_articles = test_loader.load_articles(temp_path)
        
        # Property: Should load exactly one article
        assert len(loaded_articles) == 1, \
            f"Expected 1 article, got {len(loaded_articles)}"
        
        article = loaded_articles[0]
        
        # Property: All required fields should be present and non-empty
        assert article.id and article.id.strip(), \
            "Article id is empty"
        assert article.date and article.date.strip(), \
            "Article date is empty"
        assert article.ministry and article.ministry.strip(), \
            "Article ministry is empty"
        assert article.title and article.title.strip(), \
            "Article title is empty"
        assert article.content and article.content.strip(), \
            "Article content is empty"
        
        # Property: Fields should match input data (after normalization for content)
        assert article.id == str(article_data['id']), \
            "Article id doesn't match input"
        assert article.date == str(article_data['date']), \
            "Article date doesn't match input"
        assert article.ministry == str(article_data['ministry']), \
            "Article ministry doesn't match input"
        assert article.title == str(article_data['title']), \
            "Article title doesn't match input"
    
    finally:
        # Clean up temporary file
        Path(temp_path).unlink(missing_ok=True)


# Feature: pib-rag-system, Property 21: Invalid article handling
# Validates: Requirements 7.3
@given(
    st.lists(article_dict_strategy(), min_size=2, max_size=10),
    st.integers(min_value=0, max_value=9)
)
@settings(max_examples=100)
def test_invalid_article_handling(valid_articles, invalid_index):
    """
    Property 21: For any article with missing or invalid content, the system should
    skip it and continue processing remaining articles without crashing.
    """
    # Ensure we have enough articles
    assume(len(valid_articles) > invalid_index)
    
    # Create a copy and invalidate one article by removing a required field
    articles_data = valid_articles.copy()
    invalid_article = articles_data[invalid_index].copy()
    
    # Remove a random required field
    field_to_remove = 'content'  # Always remove content to make it invalid
    del invalid_article[field_to_remove]
    articles_data[invalid_index] = invalid_article
    
    # Create a temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(articles_data, f)
        temp_path = f.name
    
    try:
        # Create a fresh loader for each test
        test_loader = ArticleLoader()
        
        # Load articles - should not crash
        loaded_articles = test_loader.load_articles(temp_path)
        
        # Property: Should load fewer articles than input (invalid one skipped)
        assert len(loaded_articles) < len(articles_data), \
            "Invalid article was not skipped"
        
        # Property: Should load at least some valid articles (at least 1 less than input)
        # Note: We expect exactly 1 invalid article, so loaded should be input - 1
        # But accounting for potential duplicates in the generated data
        assert len(loaded_articles) <= len(articles_data) - 1, \
            "Did not skip the invalid article"
        
        # Property: All loaded articles should be valid
        for article in loaded_articles:
            assert isinstance(article, Article), \
                "Loaded item is not an Article instance"
            assert article.content and article.content.strip(), \
                "Loaded article has empty content"
    
    finally:
        # Clean up temporary file
        Path(temp_path).unlink(missing_ok=True)


# Feature: pib-rag-system, Property 22: Deduplication by article ID
# Validates: Requirements 7.5
@given(
    article_dict_strategy(),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=100)
def test_deduplication_by_article_id(article_data, duplicate_count):
    """
    Property 22: For any set of articles containing duplicates with the same article ID,
    the final stored set should contain only one instance of each unique article ID.
    """
    # Create multiple copies of the same article
    articles_data = [article_data.copy() for _ in range(duplicate_count)]
    
    # Create a temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(articles_data, f)
        temp_path = f.name
    
    try:
        # Create a fresh loader for each test
        test_loader = ArticleLoader()
        
        # Load articles
        loaded_articles = test_loader.load_articles(temp_path)
        
        # Property: Should load exactly one article (duplicates removed)
        assert len(loaded_articles) == 1, \
            f"Expected 1 unique article, got {len(loaded_articles)}"
        
        # Property: The loaded article should have the correct ID
        assert loaded_articles[0].id == str(article_data['id']), \
            "Loaded article has wrong ID"
        
        # Test the deduplicate_articles method directly as well
        # Create duplicate Article objects
        duplicate_articles = []
        for _ in range(duplicate_count):
            test_loader2 = ArticleLoader()
            articles = test_loader2.load_articles(temp_path)
            if articles:
                duplicate_articles.extend(articles)
        
        if duplicate_articles:
            # Use a fresh loader for deduplication
            test_loader3 = ArticleLoader()
            deduplicated = test_loader3.deduplicate_articles(duplicate_articles)
            
            # Property: Deduplication should result in unique IDs only
            unique_ids = set(article.id for article in deduplicated)
            assert len(deduplicated) == len(unique_ids), \
                "Deduplicated list still contains duplicate IDs"
    
    finally:
        # Clean up temporary file
        Path(temp_path).unlink(missing_ok=True)


# Feature: pib-rag-system, Property 37: Original content preservation
# Validates: Requirements 11.5
@given(article_dict_strategy())
@settings(max_examples=100)
def test_original_content_preservation(article_data):
    """
    Property 37: For any article ingested into the system, both the normalized content
    and the original content should be stored and retrievable.
    """
    # Create a temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump([article_data], f)
        temp_path = f.name
    
    try:
        # Create a fresh loader for each test
        test_loader = ArticleLoader()
        
        # Load the article
        loaded_articles = test_loader.load_articles(temp_path)
        
        # Property: Should load exactly one article
        assert len(loaded_articles) == 1, \
            f"Expected 1 article, got {len(loaded_articles)}"
        
        article = loaded_articles[0]
        
        # Property: Original content should be preserved
        assert article.original_content == article_data['content'], \
            "Original content was not preserved"
        
        # Property: Both normalized and original content should exist
        assert article.content is not None, \
            "Normalized content is None"
        assert article.original_content is not None, \
            "Original content is None"
        
        # Property: Original content should not be empty if input wasn't empty
        if article_data['content'].strip():
            assert article.original_content.strip(), \
                "Original content is empty when input was not"
    
    finally:
        # Clean up temporary file
        Path(temp_path).unlink(missing_ok=True)
