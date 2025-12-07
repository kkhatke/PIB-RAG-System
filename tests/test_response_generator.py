"""
Property-based tests for ResponseGenerator module.
Tests citation completeness, field completeness, and context passing.
"""
import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from src.response_generation.response_generator import (
    ResponseGenerator,
    Citation,
    Response
)
from src.vector_store.vector_store import SearchResult
from src.data_ingestion.article_chunker import Chunk


# Test data strategies
@st.composite
def chunk_strategy(draw):
    """Generate random Chunk objects."""
    article_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    chunk_id = f"{article_id}_{draw(st.integers(min_value=0, max_value=10))}"
    content = draw(st.text(min_size=10, max_size=500))
    
    metadata = {
        'date': draw(st.dates(min_value=__import__('datetime').date(2020, 1, 1), max_value=__import__('datetime').date(2025, 12, 31))).isoformat(),
        'ministry': draw(st.text(min_size=5, max_size=50)),
        'title': draw(st.text(min_size=5, max_size=100)),
        'chunk_index': draw(st.integers(min_value=0, max_value=10))
    }
    
    return Chunk(
        chunk_id=chunk_id,
        article_id=article_id,
        content=content,
        metadata=metadata
    )


@st.composite
def search_result_strategy(draw):
    """Generate random SearchResult objects."""
    chunk = draw(chunk_strategy())
    score = draw(st.floats(min_value=0.0, max_value=1.0))
    
    return SearchResult(chunk=chunk, score=score)


@st.composite
def search_results_list_strategy(draw):
    """Generate list of SearchResult objects."""
    num_results = draw(st.integers(min_value=1, max_value=10))
    results = [draw(search_result_strategy()) for _ in range(num_results)]
    return results


# Property 9: Citation completeness
# Feature: pib-rag-system, Property 9: Citation completeness
@settings(max_examples=100, deadline=None)
@given(search_results=search_results_list_strategy())
def test_citation_completeness(search_results):
    """
    Property 9: Citation completeness
    For any generated response, the number of citations should equal the number of 
    unique source articles used, and each citation should reference a retrieved article.
    
    Validates: Requirements 4.1, 4.4
    """
    # Create mock ResponseGenerator (we only test extract_citations method)
    with patch('src.response_generation.response_generator.Ollama'):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        generator.ollama_base_url = "http://localhost:11434"
        generator.model = "llama3.2"
        generator.timeout = 120
        
        # Extract citations
        citations = generator.extract_citations(search_results)
        
        # Get unique article IDs from search results
        unique_article_ids = set()
        for result in search_results:
            unique_article_ids.add(result.chunk.article_id)
        
        # Property: Number of citations equals number of unique articles
        assert len(citations) == len(unique_article_ids), \
            f"Expected {len(unique_article_ids)} citations, got {len(citations)}"
        
        # Property: Each citation references a retrieved article
        citation_article_ids = {citation.article_id for citation in citations}
        assert citation_article_ids == unique_article_ids, \
            "Citation article IDs don't match unique article IDs from search results"
        
        # Property: No duplicate citations
        assert len(citations) == len(citation_article_ids), \
            "Found duplicate citations"


# Property 10: Citation field completeness
# Feature: pib-rag-system, Property 10: Citation field completeness
@settings(max_examples=100, deadline=None)
@given(search_results=search_results_list_strategy())
def test_citation_field_completeness(search_results):
    """
    Property 10: Citation field completeness
    For any citation in a response, it should contain all required fields: 
    article_id, date, ministry, and title.
    
    Validates: Requirements 4.2
    """
    # Create mock ResponseGenerator
    with patch('src.response_generation.response_generator.Ollama'):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        generator.ollama_base_url = "http://localhost:11434"
        generator.model = "llama3.2"
        generator.timeout = 120
        
        # Extract citations
        citations = generator.extract_citations(search_results)
        
        # Property: Each citation has all required fields
        for citation in citations:
            assert hasattr(citation, 'article_id'), "Citation missing article_id"
            assert hasattr(citation, 'date'), "Citation missing date"
            assert hasattr(citation, 'ministry'), "Citation missing ministry"
            assert hasattr(citation, 'title'), "Citation missing title"
            assert hasattr(citation, 'relevance_score'), "Citation missing relevance_score"
            
            # Property: Fields are not None
            assert citation.article_id is not None, "Citation article_id is None"
            assert citation.date is not None, "Citation date is None"
            assert citation.ministry is not None, "Citation ministry is None"
            assert citation.title is not None, "Citation title is None"
            assert citation.relevance_score is not None, "Citation relevance_score is None"
            
            # Property: String fields are not empty
            assert isinstance(citation.article_id, str) and citation.article_id, \
                "Citation article_id is empty"
            assert isinstance(citation.date, str) and citation.date, \
                "Citation date is empty"
            assert isinstance(citation.ministry, str) and citation.ministry, \
                "Citation ministry is empty"
            assert isinstance(citation.title, str) and citation.title, \
                "Citation title is empty"


# Property 11: Article retrieval by ID
# Feature: pib-rag-system, Property 11: Article retrieval by ID
@settings(max_examples=100, deadline=None)
@given(search_results=search_results_list_strategy())
def test_article_retrieval_by_id(search_results):
    """
    Property 11: Article retrieval by ID
    For any valid article ID in the system, requesting the full article should 
    return the complete article content with all metadata.
    
    Validates: Requirements 4.3
    
    Note: This tests that citations preserve article IDs correctly so they can be
    used for retrieval. The actual retrieval mechanism would be in the vector store.
    """
    # Create mock ResponseGenerator
    with patch('src.response_generation.response_generator.Ollama'):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        generator.ollama_base_url = "http://localhost:11434"
        generator.model = "llama3.2"
        generator.timeout = 120
        
        # Extract citations
        citations = generator.extract_citations(search_results)
        
        # Property: Each citation's article_id can be traced back to a search result
        for citation in citations:
            # Find the corresponding search result
            found = False
            for result in search_results:
                if result.chunk.article_id == citation.article_id:
                    found = True
                    # Property: Citation metadata matches source chunk metadata
                    assert citation.date == result.chunk.metadata.get('date', 'Unknown'), \
                        f"Citation date doesn't match source for article {citation.article_id}"
                    assert citation.ministry == result.chunk.metadata.get('ministry', 'Unknown'), \
                        f"Citation ministry doesn't match source for article {citation.article_id}"
                    assert citation.title == result.chunk.metadata.get('title', 'Unknown'), \
                        f"Citation title doesn't match source for article {citation.article_id}"
                    break
            
            assert found, f"Citation article_id {citation.article_id} not found in search results"


# Property 12: Citation format consistency
# Feature: pib-rag-system, Property 12: Citation format consistency
@settings(max_examples=100, deadline=None)
@given(search_results=search_results_list_strategy())
def test_citation_format_consistency(search_results):
    """
    Property 12: Citation format consistency
    For any set of citations in a response, all citations should follow the same 
    formatting pattern.
    
    Validates: Requirements 4.5
    """
    # Create mock ResponseGenerator
    with patch('src.response_generation.response_generator.Ollama'):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        generator.ollama_base_url = "http://localhost:11434"
        generator.model = "llama3.2"
        generator.timeout = 120
        
        # Extract citations
        citations = generator.extract_citations(search_results)
        
        if len(citations) < 2:
            # Need at least 2 citations to test consistency
            return
        
        # Property: All citations have the same structure (same fields)
        first_citation = citations[0]
        first_fields = set(vars(first_citation).keys())
        
        for citation in citations[1:]:
            citation_fields = set(vars(citation).keys())
            assert citation_fields == first_fields, \
                "Citations have inconsistent field structure"
        
        # Property: All citations have the same field types
        for citation in citations[1:]:
            assert type(citation.article_id) == type(first_citation.article_id), \
                "Inconsistent article_id type across citations"
            assert type(citation.date) == type(first_citation.date), \
                "Inconsistent date type across citations"
            assert type(citation.ministry) == type(first_citation.ministry), \
                "Inconsistent ministry type across citations"
            assert type(citation.title) == type(first_citation.title), \
                "Inconsistent title type across citations"
            assert type(citation.relevance_score) == type(first_citation.relevance_score), \
                "Inconsistent relevance_score type across citations"


# Property 32: Context passing to LLM
# Feature: pib-rag-system, Property 32: Context passing to LLM
@settings(max_examples=100, deadline=None)
@given(search_results=search_results_list_strategy())
def test_context_passing_to_llm(search_results):
    """
    Property 32: Context passing to LLM
    For any response generation with retrieved articles, the articles should be 
    included in the context passed to the language model.
    
    Validates: Requirements 10.1
    """
    # Create mock ResponseGenerator
    with patch('src.response_generation.response_generator.Ollama'):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        generator.ollama_base_url = "http://localhost:11434"
        generator.model = "llama3.2"
        generator.timeout = 120
        
        # Format context
        context = generator.format_context(search_results)
        
        # Property: Context is not empty when search results exist
        assert context, "Context should not be empty when search results exist"
        assert isinstance(context, str), "Context should be a string"
        
        # Property: Context includes information from all search results
        for idx, result in enumerate(search_results, 1):
            chunk = result.chunk
            metadata = chunk.metadata
            
            # Check that key information from each result is in the context
            # At minimum, the article ID should be present
            assert chunk.article_id in context, \
                f"Article ID {chunk.article_id} not found in context"
            
            # Check that content is included
            # (may be truncated, so check for a substring)
            content_sample = chunk.content[:50] if len(chunk.content) > 50 else chunk.content
            if content_sample.strip():  # Only check if there's actual content
                # The content might be formatted differently, so we check for presence
                # of at least some words from the content
                words = content_sample.split()[:5]  # Check first 5 words
                found_words = sum(1 for word in words if len(word) > 3 and word in context)
                assert found_words > 0, \
                    f"No content from chunk {chunk.chunk_id} found in context"
        
        # Property: Context includes metadata
        for result in search_results:
            metadata = result.chunk.metadata
            # At least one of the metadata fields should be present
            ministry = metadata.get('ministry', '')
            title = metadata.get('title', '')
            
            # Check if ministry or title appears in context
            has_metadata = (ministry and ministry in context) or (title and title in context)
            assert has_metadata, \
                f"No metadata from article {result.chunk.article_id} found in context"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
