#!/usr/bin/env python3
"""
Integration tests for PIB RAG System.
Tests end-to-end flow from query to response with actual embeddings and retrieval.
"""
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.data_ingestion.article_loader import ArticleLoader, Article
from src.data_ingestion.article_chunker import ArticleChunker
from src.data_ingestion.content_normalizer import ContentNormalizer
from src.embedding.embedding_generator import EmbeddingGenerator
from src.vector_store.vector_store import VectorStore
from src.query_engine.query_engine import QueryEngine
from src.response_generation.response_generator import ResponseGenerator, Response, Citation
from src.interface.conversational_interface import ConversationalInterface


# Sample PIB articles for testing
SAMPLE_ARTICLES = [
    {
        "id": "test_001",
        "date": "2024-01-15",
        "ministry": "Ministry of Health and Family Welfare",
        "title": "New Healthcare Initiative Launched",
        "content": "The Ministry of Health and Family Welfare announced a new healthcare initiative today. The program aims to provide free medical checkups to rural populations. This initiative will cover over 10,000 villages across the country. The program includes preventive care, diagnostic services, and treatment for common ailments. Mobile health units will be deployed to reach remote areas."
    },
    {
        "id": "test_002",
        "date": "2024-02-20",
        "ministry": "Ministry of Education",
        "title": "Digital Education Program Expansion",
        "content": "The Ministry of Education has expanded its digital education program to include 5,000 more schools. The program provides tablets and internet connectivity to students in underserved areas. Teachers will receive training on digital teaching methods. The initiative aims to bridge the digital divide in education. Special focus will be given to STEM subjects."
    },
    {
        "id": "test_003",
        "date": "2024-03-10",
        "ministry": "Ministry of Health and Family Welfare",
        "title": "Vaccination Drive Update",
        "content": "The nationwide vaccination drive has reached a new milestone with over 100 million doses administered. The Ministry of Health and Family Welfare reports high participation rates across all age groups. Special vaccination camps are being organized in remote areas. The drive includes vaccines for multiple diseases including COVID-19, measles, and polio."
    },
    {
        "id": "test_004",
        "date": "2024-04-05",
        "ministry": "Ministry of Environment, Forest and Climate Change",
        "title": "Green Energy Initiative",
        "content": "A new green energy initiative was announced to promote solar and wind power. The Ministry of Environment, Forest and Climate Change will provide subsidies for renewable energy projects. The goal is to achieve 50% renewable energy by 2030. This includes rooftop solar installations and large-scale wind farms."
    }
]


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_vector_store_dir():
    """Create a temporary directory for vector store."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_articles_file(temp_data_dir):
    """Create a JSON file with sample articles."""
    articles_file = temp_data_dir / "sample_articles.json"
    with open(articles_file, 'w', encoding='utf-8') as f:
        json.dump(SAMPLE_ARTICLES, f, indent=2)
    return articles_file


@pytest.fixture
def embedding_generator():
    """Create an embedding generator instance."""
    return EmbeddingGenerator()


@pytest.fixture
def vector_store(temp_vector_store_dir):
    """Create a vector store instance with temporary directory."""
    return VectorStore(persist_directory=str(temp_vector_store_dir))


@pytest.fixture
def populated_vector_store(vector_store, embedding_generator, sample_articles_file):
    """Create a vector store populated with sample articles."""
    # Load articles
    loader = ArticleLoader()
    articles = loader.load_articles(str(sample_articles_file))
    
    # Chunk articles
    chunker = ArticleChunker()
    all_chunks = []
    for article in articles:
        chunks = chunker.chunk_article(article)
        all_chunks.extend(chunks)
    
    # Generate embeddings
    chunk_texts = [chunk.content for chunk in all_chunks]
    embeddings = embedding_generator.batch_generate_embeddings(chunk_texts)
    
    # Add to vector store
    vector_store.add_chunks(all_chunks, embeddings)
    
    return vector_store


class TestEndToEndFlow:
    """Test complete end-to-end query-to-response flow."""
    
    def test_complete_query_flow_with_real_embeddings(
        self, 
        populated_vector_store, 
        embedding_generator
    ):
        """
        Test complete flow: query -> embedding -> retrieval -> response.
        Uses real embeddings and retrieval.
        """
        # Create query engine
        query_engine = QueryEngine(
            vector_store=populated_vector_store,
            embedding_generator=embedding_generator
        )
        
        # Mock response generator to avoid Ollama dependency
        with patch('src.response_generation.response_generator.Ollama') as mock_ollama:
            mock_llm = Mock()
            mock_llm.invoke.return_value = "Based on the provided context, the Ministry of Health and Family Welfare has launched several healthcare initiatives including free medical checkups in rural areas and a nationwide vaccination drive."
            mock_ollama.return_value = mock_llm
            
            response_generator = ResponseGenerator()
            
            # Create conversational interface
            interface = ConversationalInterface(
                query_engine=query_engine,
                response_generator=response_generator
            )
            
            # Process a query
            query = "What healthcare initiatives have been announced?"
            response = interface.process_message(query)
            
            # Verify response structure
            assert isinstance(response, Response)
            assert response.answer is not None
            assert len(response.answer) > 0
            assert isinstance(response.citations, list)
            assert len(response.citations) > 0
            
            # Verify citations contain expected fields
            for citation in response.citations:
                assert isinstance(citation, Citation)
                assert citation.article_id is not None
                assert citation.date is not None
                assert citation.ministry is not None
                assert citation.title is not None
    
    def test_ministry_filter_integration(
        self,
        populated_vector_store,
        embedding_generator
    ):
        """Test ministry filtering in complete flow."""
        query_engine = QueryEngine(
            vector_store=populated_vector_store,
            embedding_generator=embedding_generator
        )
        
        with patch('src.response_generation.response_generator.Ollama') as mock_ollama:
            mock_llm = Mock()
            mock_llm.invoke.return_value = "The Ministry of Health has announced healthcare initiatives."
            mock_ollama.return_value = mock_llm
            
            response_generator = ResponseGenerator()
            interface = ConversationalInterface(
                query_engine=query_engine,
                response_generator=response_generator
            )
            
            # Apply ministry filter
            interface.handle_ministry_filter(["Ministry of Health and Family Welfare"])
            
            # Process query
            response = interface.process_message("What initiatives were announced?")
            
            # Verify all citations are from the filtered ministry
            for citation in response.citations:
                assert citation.ministry == "Ministry of Health and Family Welfare"
    
    def test_date_range_filter_integration(
        self,
        populated_vector_store,
        embedding_generator
    ):
        """Test date range filtering in complete flow."""
        query_engine = QueryEngine(
            vector_store=populated_vector_store,
            embedding_generator=embedding_generator
        )
        
        with patch('src.response_generation.response_generator.Ollama') as mock_ollama:
            mock_llm = Mock()
            mock_llm.invoke.return_value = "Recent announcements include healthcare and education initiatives."
            mock_ollama.return_value = mock_llm
            
            response_generator = ResponseGenerator()
            interface = ConversationalInterface(
                query_engine=query_engine,
                response_generator=response_generator
            )
            
            # Apply date filter (January to February 2024)
            interface.handle_date_filter("2024-01-01", "2024-02-28")
            
            # Process query
            response = interface.process_message("What were the announcements?")
            
            # Verify all citations are within date range
            for citation in response.citations:
                citation_date = datetime.strptime(citation.date, "%Y-%m-%d")
                assert datetime(2024, 1, 1) <= citation_date <= datetime(2024, 2, 28)
    
    def test_conversation_history_integration(
        self,
        populated_vector_store,
        embedding_generator
    ):
        """Test conversation history maintenance."""
        query_engine = QueryEngine(
            vector_store=populated_vector_store,
            embedding_generator=embedding_generator
        )
        
        with patch('src.response_generation.response_generator.Ollama') as mock_ollama:
            mock_llm = Mock()
            mock_llm.invoke.return_value = "Test response"
            mock_ollama.return_value = mock_llm
            
            response_generator = ResponseGenerator()
            interface = ConversationalInterface(
                query_engine=query_engine,
                response_generator=response_generator
            )
            
            # Process multiple queries
            interface.process_message("What are healthcare initiatives?")
            interface.process_message("Tell me more about vaccination.")
            interface.process_message("What about education programs?")
            
            # Verify conversation history
            assert len(interface.conversation_history) == 6  # 3 queries + 3 responses
            
            # Verify history structure
            assert interface.conversation_history[0].role == "user"
            assert interface.conversation_history[1].role == "assistant"


class TestDataIngestionIntegration:
    """Test data ingestion pipeline integration."""
    
    def test_complete_ingestion_pipeline(
        self,
        sample_articles_file,
        vector_store,
        embedding_generator
    ):
        """Test complete data ingestion: load -> normalize -> chunk -> embed -> store."""
        # Load articles
        loader = ArticleLoader()
        articles = loader.load_articles(str(sample_articles_file))
        
        assert len(articles) == 4
        
        # Normalize content
        normalizer = ContentNormalizer()
        for article in articles:
            article.content = normalizer.normalize_content(article.content)
        
        # Chunk articles
        chunker = ArticleChunker()
        all_chunks = []
        for article in articles:
            chunks = chunker.chunk_article(article)
            all_chunks.extend(chunks)
        
        assert len(all_chunks) >= 4  # At least one chunk per article
        
        # Generate embeddings
        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = embedding_generator.batch_generate_embeddings(chunk_texts)
        
        assert len(embeddings) == len(all_chunks)
        assert all(len(emb) == 384 for emb in embeddings)
        
        # Store in vector store
        vector_store.add_chunks(all_chunks, embeddings)
        
        # Verify storage
        assert vector_store.count() == len(all_chunks)
        
        # Verify retrieval
        query_embedding = embedding_generator.generate_embedding("healthcare")
        results = vector_store.similarity_search(query_embedding, k=3)
        
        assert len(results) > 0
        assert all(hasattr(r, 'chunk') and hasattr(r, 'score') for r in results)
    
    def test_incremental_ingestion(
        self,
        temp_data_dir,
        vector_store,
        embedding_generator
    ):
        """Test incremental addition of articles."""
        # First batch
        batch1 = SAMPLE_ARTICLES[:2]
        file1 = temp_data_dir / "batch1.json"
        with open(file1, 'w', encoding='utf-8') as f:
            json.dump(batch1, f)
        
        loader = ArticleLoader()
        chunker = ArticleChunker()
        
        articles1 = loader.load_articles(str(file1))
        chunks1 = []
        for article in articles1:
            chunks1.extend(chunker.chunk_article(article))
        
        embeddings1 = embedding_generator.batch_generate_embeddings(
            [c.content for c in chunks1]
        )
        vector_store.add_chunks(chunks1, embeddings1)
        
        count_after_batch1 = vector_store.count()
        
        # Second batch
        batch2 = SAMPLE_ARTICLES[2:]
        file2 = temp_data_dir / "batch2.json"
        with open(file2, 'w', encoding='utf-8') as f:
            json.dump(batch2, f)
        
        articles2 = loader.load_articles(str(file2))
        chunks2 = []
        for article in articles2:
            chunks2.extend(chunker.chunk_article(article))
        
        embeddings2 = embedding_generator.batch_generate_embeddings(
            [c.content for c in chunks2]
        )
        vector_store.add_chunks(chunks2, embeddings2)
        
        count_after_batch2 = vector_store.count()
        
        # Verify incremental addition
        assert count_after_batch2 > count_after_batch1
        assert count_after_batch2 == count_after_batch1 + len(chunks2)


class TestErrorHandlingIntegration:
    """Test error handling scenarios in integration."""
    
    def test_ollama_not_running_error(self, populated_vector_store, embedding_generator):
        """Test handling when Ollama is not running."""
        query_engine = QueryEngine(
            vector_store=populated_vector_store,
            embedding_generator=embedding_generator
        )
        
        # Mock Ollama to raise connection error
        with patch('src.response_generation.response_generator.Ollama') as mock_ollama:
            mock_ollama.side_effect = RuntimeError("Connection refused")
            
            with pytest.raises(RuntimeError) as exc_info:
                response_generator = ResponseGenerator()
            
            assert "connection" in str(exc_info.value).lower() or "refused" in str(exc_info.value).lower()
    
    def test_model_not_found_error(self, populated_vector_store, embedding_generator):
        """Test handling when Ollama model is not found."""
        query_engine = QueryEngine(
            vector_store=populated_vector_store,
            embedding_generator=embedding_generator
        )
        
        # Mock Ollama to raise model not found error
        with patch('src.response_generation.response_generator.Ollama') as mock_ollama:
            mock_ollama.side_effect = RuntimeError("model 'llama3.2' not found")
            
            with pytest.raises(RuntimeError) as exc_info:
                response_generator = ResponseGenerator()
            
            assert "not found" in str(exc_info.value).lower()
    
    def test_empty_vector_store_query(self, vector_store, embedding_generator):
        """Test querying an empty vector store."""
        query_engine = QueryEngine(
            vector_store=vector_store,
            embedding_generator=embedding_generator
        )
        
        # Query empty store
        results = query_engine.search("test query", top_k=5)
        
        # Should return empty results, not crash
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_invalid_article_data_handling(self, temp_data_dir):
        """Test handling of invalid article data."""
        # Create file with invalid articles
        invalid_articles = [
            {"id": "valid_001", "date": "2024-01-01", "ministry": "Test", "title": "Valid", "content": "Content"},
            {"id": "invalid_001", "date": "2024-01-01"},  # Missing fields
            {"id": "invalid_002", "ministry": "Test", "title": "No content"},  # Missing content
        ]
        
        invalid_file = temp_data_dir / "invalid.json"
        with open(invalid_file, 'w', encoding='utf-8') as f:
            json.dump(invalid_articles, f)
        
        loader = ArticleLoader()
        articles = loader.load_articles(str(invalid_file))
        
        # Should only load valid articles
        assert len(articles) == 1
        assert articles[0].id == "valid_001"
    
    def test_connection_timeout_handling(self, populated_vector_store, embedding_generator):
        """Test handling of connection timeout."""
        query_engine = QueryEngine(
            vector_store=populated_vector_store,
            embedding_generator=embedding_generator
        )
        
        with patch('src.response_generation.response_generator.Ollama') as mock_ollama:
            mock_llm = Mock()
            mock_llm.invoke.side_effect = TimeoutError("Request timeout")
            mock_ollama.return_value = mock_llm
            
            response_generator = ResponseGenerator()
            interface = ConversationalInterface(
                query_engine=query_engine,
                response_generator=response_generator
            )
            
            # Should handle timeout gracefully
            with pytest.raises((TimeoutError, RuntimeError)):
                interface.process_message("test query")
    
    def test_ollama_connection_refused_detailed(self, populated_vector_store, embedding_generator):
        """Test detailed error message when Ollama connection is refused."""
        with patch('src.response_generation.response_generator.Ollama') as mock_ollama:
            mock_ollama.side_effect = Exception("Connection refused: [Errno 111] Connection refused")
            
            with pytest.raises(RuntimeError) as exc_info:
                ResponseGenerator()
            
            error_msg = str(exc_info.value)
            # Should provide helpful error message
            assert "ollama" in error_msg.lower() or "connection" in error_msg.lower()
    
    def test_ollama_timeout_error(self, populated_vector_store, embedding_generator):
        """Test handling of Ollama timeout during initialization."""
        with patch('src.response_generation.response_generator.Ollama') as mock_ollama:
            mock_ollama.side_effect = TimeoutError("Connection timeout after 120 seconds")
            
            with pytest.raises(RuntimeError) as exc_info:
                ResponseGenerator()
            
            assert "timeout" in str(exc_info.value).lower()
    
    def test_malformed_json_handling(self, temp_data_dir):
        """Test handling of malformed JSON files."""
        malformed_file = temp_data_dir / "malformed.json"
        with open(malformed_file, 'w', encoding='utf-8') as f:
            f.write("{invalid json content")
        
        loader = ArticleLoader()
        articles = loader.load_articles(str(malformed_file))
        
        # Should return empty list, not crash
        assert isinstance(articles, list)
        assert len(articles) == 0
    
    def test_empty_query_handling(self, populated_vector_store, embedding_generator):
        """Test handling of empty query."""
        query_engine = QueryEngine(
            vector_store=populated_vector_store,
            embedding_generator=embedding_generator
        )
        
        with patch('src.response_generation.response_generator.Ollama') as mock_ollama:
            mock_llm = Mock()
            mock_llm.invoke.return_value = "I need a question to answer."
            mock_ollama.return_value = mock_llm
            
            response_generator = ResponseGenerator()
            
            # Empty query should be handled
            with pytest.raises(ValueError):
                response_generator.generate_response("", [], None)


class TestSystemIntegrationWithMocks:
    """Test system integration with mocked Ollama for CI/CD."""
    
    def test_full_system_with_mocked_ollama(
        self,
        populated_vector_store,
        embedding_generator
    ):
        """Test full system with mocked Ollama responses."""
        query_engine = QueryEngine(
            vector_store=populated_vector_store,
            embedding_generator=embedding_generator
        )
        
        # Mock Ollama with realistic responses
        with patch('src.response_generation.response_generator.Ollama') as mock_ollama:
            mock_llm = Mock()
            
            def mock_invoke(prompt):
                # Simulate realistic LLM response based on prompt
                if "healthcare" in prompt.lower():
                    return "The Ministry of Health and Family Welfare has announced several healthcare initiatives including free medical checkups in rural areas and a nationwide vaccination drive that has reached over 100 million doses."
                elif "education" in prompt.lower():
                    return "The Ministry of Education has expanded its digital education program to include 5,000 more schools, providing tablets and internet connectivity to students."
                else:
                    return "Based on the provided context, I can provide information about government initiatives."
            
            mock_llm.invoke.side_effect = mock_invoke
            mock_ollama.return_value = mock_llm
            
            response_generator = ResponseGenerator()
            interface = ConversationalInterface(
                query_engine=query_engine,
                response_generator=response_generator
            )
            
            # Test healthcare query
            response1 = interface.process_message("What healthcare initiatives were announced?")
            assert "healthcare" in response1.answer.lower()
            assert len(response1.citations) > 0
            
            # Test education query
            response2 = interface.process_message("Tell me about education programs")
            assert "education" in response2.answer.lower()
            assert len(response2.citations) > 0
    
    def test_multiple_queries_with_filters(
        self,
        populated_vector_store,
        embedding_generator
    ):
        """Test multiple queries with different filters."""
        query_engine = QueryEngine(
            vector_store=populated_vector_store,
            embedding_generator=embedding_generator
        )
        
        with patch('src.response_generation.response_generator.Ollama') as mock_ollama:
            mock_llm = Mock()
            mock_llm.invoke.return_value = "Test response"
            mock_ollama.return_value = mock_llm
            
            response_generator = ResponseGenerator()
            interface = ConversationalInterface(
                query_engine=query_engine,
                response_generator=response_generator
            )
            
            # Query 1: No filters
            response1 = interface.process_message("What are the initiatives?")
            citations1_count = len(response1.citations)
            
            # Query 2: With ministry filter
            interface.handle_ministry_filter(["Ministry of Health and Family Welfare"])
            response2 = interface.process_message("What are the initiatives?")
            
            # Verify filtering worked
            for citation in response2.citations:
                assert citation.ministry == "Ministry of Health and Family Welfare"
            
            # Query 3: Clear filter and apply date filter
            interface.handle_ministry_filter(None)
            interface.handle_date_filter("2024-01-01", "2024-02-28")
            response3 = interface.process_message("What are the initiatives?")
            
            # Verify date filtering
            for citation in response3.citations:
                date = datetime.strptime(citation.date, "%Y-%m-%d")
                assert datetime(2024, 1, 1) <= date <= datetime(2024, 2, 28)


class TestRetrievalQuality:
    """Test retrieval quality and relevance."""
    
    def test_semantic_search_relevance(
        self,
        populated_vector_store,
        embedding_generator
    ):
        """Test that semantic search returns relevant results."""
        query_engine = QueryEngine(
            vector_store=populated_vector_store,
            embedding_generator=embedding_generator
        )
        
        # Query about healthcare
        results = query_engine.search("healthcare medical checkups", top_k=3)
        
        # Verify results are relevant
        assert len(results) > 0
        
        # Check that healthcare-related articles are in results
        healthcare_found = any(
            "health" in result.chunk.metadata.get("ministry", "").lower()
            for result in results
        )
        assert healthcare_found
    
    def test_relevance_threshold_filtering(
        self,
        populated_vector_store,
        embedding_generator
    ):
        """Test that relevance threshold filters low-quality results."""
        query_engine = QueryEngine(
            vector_store=populated_vector_store,
            embedding_generator=embedding_generator
        )
        
        # Query with high threshold
        results_high = query_engine.search(
            "healthcare initiatives",
            top_k=10,
            relevance_threshold=0.7
        )
        
        # Query with low threshold
        results_low = query_engine.search(
            "healthcare initiatives",
            top_k=10,
            relevance_threshold=0.3
        )
        
        # High threshold should return fewer results
        assert len(results_high) <= len(results_low)
        
        # All results should meet threshold
        for result in results_high:
            assert result.score >= 0.7
    
    def test_top_k_limitation(
        self,
        populated_vector_store,
        embedding_generator
    ):
        """Test that top-k limits number of results."""
        query_engine = QueryEngine(
            vector_store=populated_vector_store,
            embedding_generator=embedding_generator
        )
        
        # Query with different k values
        results_k2 = query_engine.search("government initiatives", top_k=2)
        results_k5 = query_engine.search("government initiatives", top_k=5)
        
        # Verify k limitation
        assert len(results_k2) <= 2
        assert len(results_k5) <= 5
        
        # Results should be ordered by relevance
        if len(results_k5) > 1:
            for i in range(len(results_k5) - 1):
                assert results_k5[i].score >= results_k5[i + 1].score


class TestWithActualOllama:
    """
    Tests that use actual Ollama when available.
    These tests are skipped if Ollama is not running.
    """
    
    @pytest.fixture
    def check_ollama_available(self):
        """Check if Ollama is available and skip test if not."""
        try:
            from langchain_community.llms import Ollama
            import config
            
            llm = Ollama(
                base_url=config.OLLAMA_BASE_URL,
                model=config.OLLAMA_MODEL,
                timeout=5
            )
            # Try a simple test
            llm.invoke("test")
            return True
        except Exception:
            pytest.skip("Ollama not available - skipping test")
    
    def test_real_ollama_query_flow(
        self,
        check_ollama_available,
        populated_vector_store,
        embedding_generator
    ):
        """Test complete flow with actual Ollama LLM."""
        query_engine = QueryEngine(
            vector_store=populated_vector_store,
            embedding_generator=embedding_generator
        )
        
        # Use real ResponseGenerator (no mocking)
        response_generator = ResponseGenerator()
        interface = ConversationalInterface(
            query_engine=query_engine,
            response_generator=response_generator
        )
        
        # Process a real query
        response = interface.process_message("What healthcare initiatives were announced?")
        
        # Verify response structure
        assert isinstance(response, Response)
        assert response.answer is not None
        assert len(response.answer) > 0
        assert isinstance(response.citations, list)
        assert len(response.citations) > 0
        
        # Verify answer is coherent (has reasonable length)
        assert len(response.answer.split()) > 5
    
    def test_real_ollama_with_filters(
        self,
        check_ollama_available,
        populated_vector_store,
        embedding_generator
    ):
        """Test filtering with actual Ollama."""
        query_engine = QueryEngine(
            vector_store=populated_vector_store,
            embedding_generator=embedding_generator
        )
        
        response_generator = ResponseGenerator()
        interface = ConversationalInterface(
            query_engine=query_engine,
            response_generator=response_generator
        )
        
        # Apply ministry filter
        interface.handle_ministry_filter(["Ministry of Health and Family Welfare"])
        
        # Process query
        response = interface.process_message("What initiatives were announced?")
        
        # Verify all citations are from filtered ministry
        for citation in response.citations:
            assert citation.ministry == "Ministry of Health and Family Welfare"
    
    def test_real_ollama_conversation_context(
        self,
        check_ollama_available,
        populated_vector_store,
        embedding_generator
    ):
        """Test conversation context with actual Ollama."""
        query_engine = QueryEngine(
            vector_store=populated_vector_store,
            embedding_generator=embedding_generator
        )
        
        response_generator = ResponseGenerator()
        interface = ConversationalInterface(
            query_engine=query_engine,
            response_generator=response_generator
        )
        
        # First query
        response1 = interface.process_message("What are healthcare initiatives?")
        assert len(response1.answer) > 0
        
        # Follow-up query (should use context)
        response2 = interface.process_message("Tell me more about the vaccination program")
        assert len(response2.answer) > 0
        
        # Verify conversation history
        assert len(interface.conversation_history) == 4  # 2 queries + 2 responses


class TestCICDMockScenarios:
    """
    Tests specifically designed for CI/CD environments with mocked Ollama.
    These tests ensure the system works in automated testing environments.
    """
    
    def test_cicd_complete_flow_mocked(
        self,
        populated_vector_store,
        embedding_generator
    ):
        """Test complete flow with mocked Ollama for CI/CD."""
        query_engine = QueryEngine(
            vector_store=populated_vector_store,
            embedding_generator=embedding_generator
        )
        
        with patch('src.response_generation.response_generator.Ollama') as mock_ollama:
            # Setup realistic mock
            mock_llm = Mock()
            
            def realistic_response(prompt):
                if "healthcare" in prompt.lower():
                    return "The Ministry of Health and Family Welfare has launched healthcare initiatives including free medical checkups and vaccination programs."
                elif "education" in prompt.lower():
                    return "The Ministry of Education has expanded digital education programs to 5,000 schools."
                else:
                    return "Based on the provided context, I can provide information about government initiatives."
            
            mock_llm.invoke.side_effect = realistic_response
            mock_ollama.return_value = mock_llm
            
            response_generator = ResponseGenerator()
            interface = ConversationalInterface(
                query_engine=query_engine,
                response_generator=response_generator
            )
            
            # Test multiple queries
            queries = [
                "What healthcare initiatives were announced?",
                "Tell me about education programs",
                "What are the recent government announcements?"
            ]
            
            for query in queries:
                response = interface.process_message(query)
                
                # Verify response structure
                assert isinstance(response, Response)
                assert len(response.answer) > 0
                assert len(response.citations) > 0
                
                # Verify citations have required fields
                for citation in response.citations:
                    assert citation.article_id is not None
                    assert citation.date is not None
                    assert citation.ministry is not None
                    assert citation.title is not None
    
    def test_cicd_error_recovery(
        self,
        populated_vector_store,
        embedding_generator
    ):
        """Test error recovery in CI/CD environment."""
        query_engine = QueryEngine(
            vector_store=populated_vector_store,
            embedding_generator=embedding_generator
        )
        
        with patch('src.response_generation.response_generator.Ollama') as mock_ollama:
            mock_llm = Mock()
            
            # First call fails, second succeeds
            call_count = [0]
            
            def flaky_response(prompt):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise Exception("Temporary error")
                return "Response after retry"
            
            mock_llm.invoke.side_effect = flaky_response
            mock_ollama.return_value = mock_llm
            
            response_generator = ResponseGenerator()
            interface = ConversationalInterface(
                query_engine=query_engine,
                response_generator=response_generator
            )
            
            # Should succeed after retry
            response = interface.process_message("test query")
            assert response.answer == "Response after retry"
    
    def test_cicd_batch_processing(
        self,
        temp_data_dir,
        vector_store,
        embedding_generator
    ):
        """Test batch processing of articles in CI/CD."""
        # Create multiple batches of articles
        batches = [
            SAMPLE_ARTICLES[:2],
            SAMPLE_ARTICLES[2:]
        ]
        
        loader = ArticleLoader()
        chunker = ArticleChunker()
        
        total_chunks = 0
        
        for idx, batch in enumerate(batches):
            # Write batch to file
            batch_file = temp_data_dir / f"batch_{idx}.json"
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch, f)
            
            # Load and process
            articles = loader.load_articles(str(batch_file))
            chunks = []
            for article in articles:
                chunks.extend(chunker.chunk_article(article))
            
            # Generate embeddings and store
            embeddings = embedding_generator.batch_generate_embeddings(
                [c.content for c in chunks]
            )
            vector_store.add_chunks(chunks, embeddings)
            
            total_chunks += len(chunks)
        
        # Verify all chunks were stored
        assert vector_store.count() == total_chunks
        
        # Verify retrieval works
        query_embedding = embedding_generator.generate_embedding("government")
        results = vector_store.similarity_search(query_embedding, k=5)
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
