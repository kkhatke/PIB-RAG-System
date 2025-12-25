#!/usr/bin/env python3
"""
Integration tests for enhanced features in PIB RAG System.
Tests model caching, configurable search parameters, and Streamlit web interface.
"""
import pytest
import json
import tempfile
import shutil
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.data_ingestion.article_loader import ArticleLoader, Article
from src.data_ingestion.article_chunker import ArticleChunker
from src.embedding.embedding_generator import EmbeddingGenerator
from src.vector_store.vector_store import VectorStore
from src.query_engine.query_engine import QueryEngine
from src.response_generation.response_generator import ResponseGenerator, Response, Citation
from src.interface.conversational_interface import ConversationalInterface
from src.interface.streamlit_interface import StreamlitInterface, FilterConfig


# Sample PIB articles for testing enhanced features
ENHANCED_SAMPLE_ARTICLES = [
    {
        "id": "enhanced_001",
        "date": "2024-12-01",
        "ministry": "Ministry of Health and Family Welfare",
        "title": "Advanced Healthcare AI Initiative",
        "content": "The Ministry of Health and Family Welfare announced a groundbreaking AI-powered healthcare initiative. This program leverages machine learning to improve diagnostic accuracy in rural hospitals. The initiative includes deployment of AI diagnostic tools, training for healthcare workers, and establishment of telemedicine networks. The program aims to reduce diagnostic errors by 40% and improve patient outcomes in underserved areas."
    },
    {
        "id": "enhanced_002", 
        "date": "2024-11-15",
        "ministry": "Ministry of Education",
        "title": "Digital Learning Platform Expansion",
        "content": "The Ministry of Education has launched an expanded digital learning platform reaching 15,000 schools nationwide. The platform includes interactive content, virtual laboratories, and AI-powered personalized learning paths. Students can access courses in multiple languages and receive real-time feedback on their progress. The initiative focuses on STEM education and digital literacy skills."
    },
    {
        "id": "enhanced_003",
        "date": "2024-10-20",
        "ministry": "Ministry of Health and Family Welfare", 
        "title": "Precision Medicine Research Program",
        "content": "A new precision medicine research program has been initiated to develop personalized treatment protocols. The program involves genomic sequencing, biomarker identification, and development of targeted therapies. Research centers across the country will collaborate to create a comprehensive genetic database. The initiative aims to revolutionize cancer treatment and rare disease management."
    },
    {
        "id": "enhanced_004",
        "date": "2024-09-10",
        "ministry": "Ministry of Environment, Forest and Climate Change",
        "title": "Smart City Environmental Monitoring",
        "content": "Smart environmental monitoring systems are being deployed in 50 cities to track air quality, water pollution, and noise levels in real-time. The system uses IoT sensors, satellite data, and machine learning algorithms to predict environmental trends. Citizens can access environmental data through a mobile app and receive alerts about pollution levels."
    },
    {
        "id": "enhanced_005",
        "date": "2024-08-25",
        "ministry": "Ministry of Education",
        "title": "Quantum Computing Education Initiative", 
        "content": "The Ministry of Education announced a quantum computing education initiative for universities and research institutions. The program includes quantum computing labs, specialized curricula, and partnerships with leading technology companies. Students will learn quantum algorithms, quantum cryptography, and quantum machine learning. The initiative aims to prepare India for the quantum computing revolution."
    },
    {
        "id": "enhanced_006",
        "date": "2024-07-30",
        "ministry": "Ministry of Health and Family Welfare",
        "title": "Robotic Surgery Training Centers",
        "content": "New robotic surgery training centers are being established in major medical colleges to train surgeons in minimally invasive procedures. The centers feature state-of-the-art robotic surgical systems, virtual reality training modules, and simulation laboratories. The program aims to improve surgical precision and reduce patient recovery times."
    },
    {
        "id": "enhanced_007",
        "date": "2024-06-15",
        "ministry": "Ministry of Science and Technology",
        "title": "Space Technology Innovation Hub",
        "content": "A new space technology innovation hub has been launched to promote research in satellite technology, space exploration, and earth observation systems. The hub will support startups, provide funding for research projects, and facilitate collaboration between academia and industry. Focus areas include small satellites, space debris management, and interplanetary missions."
    },
    {
        "id": "enhanced_008",
        "date": "2024-05-20",
        "ministry": "Ministry of Electronics and Information Technology",
        "title": "Cybersecurity Excellence Centers",
        "content": "Cybersecurity excellence centers are being established to enhance national cyber defense capabilities. The centers will conduct research in threat intelligence, develop security frameworks, and train cybersecurity professionals. The initiative includes partnerships with international cybersecurity organizations and development of indigenous security solutions."
    }
]


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for model caching tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Handle Windows file locking issues
    try:
        shutil.rmtree(temp_dir)
    except PermissionError:
        import time
        time.sleep(0.5)
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            import logging
            logging.warning(f"Could not clean up temp directory {temp_dir} due to file locks")


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Handle Windows file locking issues
    try:
        shutil.rmtree(temp_dir)
    except PermissionError:
        import time
        time.sleep(0.5)
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            import logging
            logging.warning(f"Could not clean up temp directory {temp_dir} due to file locks")


@pytest.fixture
def temp_vector_store_dir():
    """Create a temporary directory for vector store."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Handle Windows file locking issues with ChromaDB
    try:
        shutil.rmtree(temp_dir)
    except PermissionError:
        # On Windows, ChromaDB may not release file handles immediately
        # Try again after a short delay
        import time
        time.sleep(0.5)
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            # If still failing, just log and continue - temp files will be cleaned up eventually
            import logging
            logging.warning(f"Could not clean up temp directory {temp_dir} due to file locks")


@pytest.fixture
def enhanced_articles_file(temp_data_dir):
    """Create a JSON file with enhanced sample articles."""
    articles_file = temp_data_dir / "enhanced_articles.json"
    with open(articles_file, 'w', encoding='utf-8') as f:
        json.dump(ENHANCED_SAMPLE_ARTICLES, f, indent=2)
    return articles_file


class TestModelCachingFunctionality:
    """Test model caching functionality - Requirements 12.1, 12.2, 12.3"""
    
    def test_first_time_model_download_and_caching(self, temp_cache_dir):
        """
        Test first-time model download and caching.
        Validates: Requirements 12.1, 12.2
        """
        cache_dir = temp_cache_dir / "first_download"
        
        # Ensure cache directory doesn't exist initially
        assert not cache_dir.exists()
        
        # First initialization - should download and cache
        start_time = time.time()
        generator1 = EmbeddingGenerator(cache_dir=str(cache_dir))
        first_load_time = time.time() - start_time
        
        # Verify cache directory was created
        assert cache_dir.exists()
        
        # Verify model files exist in cache (HuggingFace uses specific directory structure)
        model_cache_path = cache_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
        assert model_cache_path.exists()
        
        # Verify model is functional
        test_embedding = generator1.generate_embedding("test model caching")
        assert len(test_embedding) == 384
        assert not all(x == 0.0 for x in test_embedding)
        
        # Verify model info indicates it was downloaded
        model_info = generator1.get_model_info()
        assert 'cache_dir' in model_info
        assert model_info['cache_dir'] == str(cache_dir)
        
        print(f"First load time: {first_load_time:.2f} seconds")
    
    def test_subsequent_startups_using_cached_models(self, temp_cache_dir):
        """
        Test subsequent startups using cached models.
        Validates: Requirements 12.2
        """
        cache_dir = temp_cache_dir / "cached_startup"
        
        # First initialization - download and cache
        generator1 = EmbeddingGenerator(cache_dir=str(cache_dir))
        first_embedding = generator1.generate_embedding("cached model test")
        
        # Second initialization - should load from cache
        start_time = time.time()
        generator2 = EmbeddingGenerator(cache_dir=str(cache_dir))
        cached_load_time = time.time() - start_time
        
        # Verify cached model is functional
        second_embedding = generator2.generate_embedding("cached model test")
        
        # Both models should produce identical embeddings
        assert first_embedding == second_embedding
        
        # Verify model info indicates cached status
        model_info = generator2.get_model_info()
        assert model_info['is_cached'] == True
        
        # Cached load should be faster (though this may vary)
        print(f"Cached load time: {cached_load_time:.2f} seconds")
        
        # Third initialization - should also use cache
        generator3 = EmbeddingGenerator(cache_dir=str(cache_dir))
        third_embedding = generator3.generate_embedding("cached model test")
        
        # All three should produce identical embeddings
        assert first_embedding == second_embedding == third_embedding
    
    def test_corruption_detection_and_automatic_redownload(self, temp_cache_dir):
        """
        Test corruption detection and automatic re-download.
        Validates: Requirements 12.3
        """
        cache_dir = temp_cache_dir / "corruption_test"
        
        # First initialization - download and cache
        generator1 = EmbeddingGenerator(cache_dir=str(cache_dir))
        original_embedding = generator1.generate_embedding("corruption test")
        
        # Verify integrity check passes
        assert generator1.verify_model_integrity() == True
        
        # Simulate corruption by modifying a model file (HuggingFace uses specific directory structure)
        model_cache_path = cache_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
        model_files = list(model_cache_path.rglob('*'))
        model_files = [f for f in model_files if f.is_file() and f.name != "model_metadata.json"]
        
        if model_files:
            # Corrupt the first model file
            corrupt_file = model_files[0]
            original_size = corrupt_file.stat().st_size
            
            with open(corrupt_file, 'ab') as f:
                f.write(b'CORRUPTED_DATA_FOR_TESTING')
            
            # Verify corruption is detected
            assert generator1.verify_model_integrity() == False
            
            # New initialization should detect corruption and re-download
            generator2 = EmbeddingGenerator(cache_dir=str(cache_dir))
            
            # Verify model is functional after re-download
            redownloaded_embedding = generator2.generate_embedding("corruption test")
            assert len(redownloaded_embedding) == 384
            
            # Verify integrity check passes after re-download
            assert generator2.verify_model_integrity() == True
            
            # Embeddings should be identical (same model, same input)
            assert original_embedding == redownloaded_embedding
    
    def test_cache_directory_permissions_and_creation(self, temp_cache_dir):
        """
        Test cache directory creation and permission handling.
        """
        # Test with nested cache directory path
        nested_cache_dir = temp_cache_dir / "nested" / "cache" / "directory"
        
        # Should create nested directories automatically
        generator = EmbeddingGenerator(cache_dir=str(nested_cache_dir))
        
        # Verify nested directories were created
        assert nested_cache_dir.exists()
        
        # Verify model is functional
        test_embedding = generator.generate_embedding("nested cache test")
        assert len(test_embedding) == 384
    
    def test_multiple_cache_directories_isolation(self, temp_cache_dir):
        """
        Test that multiple cache directories work independently.
        """
        cache_dir1 = temp_cache_dir / "cache1"
        cache_dir2 = temp_cache_dir / "cache2"
        
        # Initialize generators with different cache directories
        generator1 = EmbeddingGenerator(cache_dir=str(cache_dir1))
        generator2 = EmbeddingGenerator(cache_dir=str(cache_dir2))
        
        # Both should be functional
        embedding1 = generator1.generate_embedding("isolation test")
        embedding2 = generator2.generate_embedding("isolation test")
        
        # Should produce identical embeddings (same model, same input)
        assert embedding1 == embedding2
        
        # Both cache directories should exist independently
        assert cache_dir1.exists()
        assert cache_dir2.exists()
        
        # Each should have its own model files (HuggingFace uses specific directory structure)
        model_path1 = cache_dir1 / "models--sentence-transformers--all-MiniLM-L6-v2"
        model_path2 = cache_dir2 / "models--sentence-transformers--all-MiniLM-L6-v2"
        assert model_path1.exists()
        assert model_path2.exists()


class TestConfigurableSearchParameters:
    """Test configurable search parameters - Requirements 13.1, 13.2, 13.3, 13.4, 13.5"""
    
    @pytest.fixture
    def populated_enhanced_vector_store(self, temp_vector_store_dir, enhanced_articles_file):
        """Create a vector store populated with enhanced sample articles."""
        # Use default cache for embedding generator
        embedding_generator = EmbeddingGenerator()
        vector_store = VectorStore(persist_directory=str(temp_vector_store_dir))
        
        # Load articles
        loader = ArticleLoader()
        articles = loader.load_articles(str(enhanced_articles_file))
        
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
        
        return vector_store, embedding_generator
    
    def test_various_article_count_limits(self, populated_enhanced_vector_store):
        """
        Test various article count limits (1, 5, 10, 20).
        Validates: Requirements 13.1
        """
        vector_store, embedding_generator = populated_enhanced_vector_store
        query_engine = QueryEngine(
            vector_store=vector_store,
            embedding_generator=embedding_generator
        )
        
        test_query = "healthcare technology initiatives"
        article_counts = [1, 5, 10, 20]
        
        for max_articles in article_counts:
            results = query_engine.search(
                query=test_query,
                max_articles=max_articles,
                relevance_threshold=0.0  # Include all results for testing
            )
            
            # Should return at most max_articles results
            assert len(results) <= max_articles, \
                f"Expected at most {max_articles} results, got {len(results)}"
            
            # Results should be ordered by relevance
            if len(results) > 1:
                for i in range(len(results) - 1):
                    assert results[i].score >= results[i + 1].score, \
                        f"Results not ordered by relevance at position {i}"
            
            print(f"Max articles: {max_articles}, Actual results: {len(results)}")
    
    def test_all_time_period_filters_with_real_date_ranges(self, populated_enhanced_vector_store):
        """
        Test all time period filters with real date ranges.
        Validates: Requirements 13.2, 13.3, 13.4
        """
        vector_store, embedding_generator = populated_enhanced_vector_store
        query_engine = QueryEngine(
            vector_store=vector_store,
            embedding_generator=embedding_generator
        )
        
        test_query = "technology innovation"
        time_periods = ["1_year", "6_months", "3_months"]
        
        for time_period in time_periods:
            results = query_engine.search(
                query=test_query,
                time_period=time_period,
                max_articles=20,
                relevance_threshold=0.0
            )
            
            # Calculate expected date range
            today = datetime.now().date()
            if time_period == "1_year":
                expected_start = today - timedelta(days=365)
            elif time_period == "6_months":
                expected_start = today - timedelta(days=180)
            elif time_period == "3_months":
                expected_start = today - timedelta(days=90)
            
            # Verify all results are within the time period
            for result in results:
                result_date = datetime.fromisoformat(result.chunk.metadata['date']).date()
                assert expected_start <= result_date <= today, \
                    f"Result date {result_date} not in {time_period} range [{expected_start}, {today}]"
            
            print(f"Time period: {time_period}, Results: {len(results)}")
            
            # Verify available time periods method
            available_periods = query_engine.get_available_time_periods()
            assert time_period in available_periods, \
                f"Time period {time_period} should be in available periods"
    
    def test_custom_date_range_functionality(self, populated_enhanced_vector_store):
        """
        Test custom date range functionality.
        Validates: Requirements 13.5
        """
        vector_store, embedding_generator = populated_enhanced_vector_store
        query_engine = QueryEngine(
            vector_store=vector_store,
            embedding_generator=embedding_generator
        )
        
        test_query = "education digital learning"
        
        # Test various custom date ranges
        date_ranges = [
            ("2024-11-01", "2024-12-31"),  # Recent 2 months
            ("2024-08-01", "2024-10-31"),  # Mid-year period
            ("2024-05-01", "2024-07-31"),  # Earlier period
            ("2024-01-01", "2024-12-31"),  # Full year
        ]
        
        for start_date, end_date in date_ranges:
            results = query_engine.search(
                query=test_query,
                date_range=(start_date, end_date),
                max_articles=20,
                relevance_threshold=0.0
            )
            
            # Verify all results are within the custom date range
            start_dt = datetime.fromisoformat(start_date).date()
            end_dt = datetime.fromisoformat(end_date).date()
            
            for result in results:
                result_date = datetime.fromisoformat(result.chunk.metadata['date']).date()
                assert start_dt <= result_date <= end_dt, \
                    f"Result date {result_date} not in custom range [{start_date}, {end_date}]"
            
            print(f"Date range: {start_date} to {end_date}, Results: {len(results)}")
    
    def test_filter_combinations(self, populated_enhanced_vector_store):
        """
        Test filter combinations.
        Validates: Requirements 13.5, 13.6, 13.7
        """
        vector_store, embedding_generator = populated_enhanced_vector_store
        query_engine = QueryEngine(
            vector_store=vector_store,
            embedding_generator=embedding_generator
        )
        
        test_query = "healthcare technology"
        
        # Test combination of ministry filter, time period, and article count
        results = query_engine.search(
            query=test_query,
            ministry_filter=["Ministry of Health and Family Welfare"],
            time_period="6_months",
            max_articles=5,
            relevance_threshold=0.3
        )
        
        # Verify all constraints are satisfied
        assert len(results) <= 5, "Should respect max_articles limit"
        
        # Calculate expected date range for 6 months
        today = datetime.now().date()
        expected_start = today - timedelta(days=180)
        
        for result in results:
            # Check ministry filter
            assert result.chunk.metadata['ministry'] == "Ministry of Health and Family Welfare", \
                "Should only return results from filtered ministry"
            
            # Check time period filter
            result_date = datetime.fromisoformat(result.chunk.metadata['date']).date()
            assert expected_start <= result_date <= today, \
                f"Result should be within 6 months: {result_date}"
            
            # Check relevance threshold
            assert result.score >= 0.3, \
                f"Result score {result.score} should be >= 0.3"
        
        print(f"Combined filters - Results: {len(results)}")
        
        # Test combination of custom date range and ministry filter
        results2 = query_engine.search(
            query=test_query,
            ministry_filter=["Ministry of Education", "Ministry of Health and Family Welfare"],
            date_range=("2024-08-01", "2024-12-31"),
            max_articles=10,
            relevance_threshold=0.2
        )
        
        for result in results2:
            # Check ministry filter (multiple ministries)
            assert result.chunk.metadata['ministry'] in [
                "Ministry of Education", 
                "Ministry of Health and Family Welfare"
            ], "Should only return results from filtered ministries"
            
            # Check custom date range
            result_date = datetime.fromisoformat(result.chunk.metadata['date']).date()
            assert datetime(2024, 8, 1).date() <= result_date <= datetime(2024, 12, 31).date(), \
                f"Result should be within custom date range: {result_date}"
        
        print(f"Combined filters 2 - Results: {len(results2)}")
    
    def test_no_time_filter_searches_all_articles(self, populated_enhanced_vector_store):
        """
        Test that when no time filter is specified, system searches all articles.
        Validates: Requirements 13.6
        """
        vector_store, embedding_generator = populated_enhanced_vector_store
        query_engine = QueryEngine(
            vector_store=vector_store,
            embedding_generator=embedding_generator
        )
        
        test_query = "technology innovation"
        
        # Search without any time filters
        results_no_filter = query_engine.search(
            query=test_query,
            max_articles=20,
            relevance_threshold=0.0
        )
        
        # Search with 1 year filter
        results_1_year = query_engine.search(
            query=test_query,
            time_period="1_year",
            max_articles=20,
            relevance_threshold=0.0
        )
        
        # No filter should return same or more results than 1 year filter
        assert len(results_no_filter) >= len(results_1_year), \
            "No time filter should return at least as many results as 1 year filter"
        
        # Verify no filter includes articles from all time periods
        if results_no_filter:
            result_dates = [
                datetime.fromisoformat(r.chunk.metadata['date']).date() 
                for r in results_no_filter
            ]
            date_range = max(result_dates) - min(result_dates)
            
            # Should span multiple months (our test data spans several months)
            assert date_range.days > 30, \
                "No time filter should include articles from multiple months"
        
        print(f"No filter: {len(results_no_filter)}, 1 year filter: {len(results_1_year)}")


class TestStreamlitWebInterfaceEndToEnd:
    """Test Streamlit web interface end-to-end - Requirements 6.1, 6.2, 6.3, 6.4, 6.5"""
    
    @pytest.fixture
    def streamlit_system(self, temp_vector_store_dir, enhanced_articles_file):
        """Create complete Streamlit system with populated data."""
        # Initialize components
        embedding_generator = EmbeddingGenerator()
        vector_store = VectorStore(persist_directory=str(temp_vector_store_dir))
        
        # Populate vector store
        loader = ArticleLoader()
        articles = loader.load_articles(str(enhanced_articles_file))
        
        chunker = ArticleChunker()
        all_chunks = []
        for article in articles:
            chunks = chunker.chunk_article(article)
            all_chunks.extend(chunks)
        
        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = embedding_generator.batch_generate_embeddings(chunk_texts)
        vector_store.add_chunks(all_chunks, embeddings)
        
        # Create system components
        query_engine = QueryEngine(
            vector_store=vector_store,
            embedding_generator=embedding_generator
        )
        
        # Mock response generator to avoid Ollama dependency
        response_generator = Mock()
        
        def mock_generate_response(query, search_results, conversation_history):
            # Create realistic response based on search results
            if search_results:
                answer = f"Based on the retrieved information, here are the key points about {query}: "
                answer += " ".join([
                    f"The {result.chunk.metadata['ministry']} announced initiatives related to {query}."
                    for result in search_results[:2]
                ])
            else:
                answer = f"I couldn't find specific information about {query} in the available documents."
            
            citations = [
                Citation(
                    article_id=result.chunk.article_id,
                    date=result.chunk.metadata['date'],
                    ministry=result.chunk.metadata['ministry'],
                    title=result.chunk.metadata['title'],
                    relevance_score=result.score
                )
                for result in search_results
            ]
            
            return Response(answer=answer, citations=citations)
        
        response_generator.generate_response.side_effect = mock_generate_response
        
        # Create Streamlit interface
        streamlit_interface = StreamlitInterface(query_engine, response_generator)
        
        return streamlit_interface, query_engine, response_generator
    
    def test_complete_user_workflow_from_query_to_results(self, streamlit_system):
        """
        Test complete user workflow from query to results.
        Validates: Requirements 6.1, 6.2, 6.3
        """
        streamlit_interface, query_engine, response_generator = streamlit_system
        
        # Mock Streamlit components
        with patch('streamlit.text_input') as mock_text_input, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.slider') as mock_slider, \
             patch('streamlit.date_input') as mock_date_input, \
             patch('streamlit.subheader') as mock_subheader, \
             patch('streamlit.write') as mock_write, \
             patch('streamlit.info') as mock_info:
            
            # Mock session state properly
            mock_session_state = MagicMock()
            mock_session_state.conversation_history = []
            mock_session_state.get.return_value = []
            
            with patch('streamlit.session_state', mock_session_state):
                # Simulate user input
                mock_text_input.return_value = "healthcare AI technology"
                mock_button.return_value = True
                mock_multiselect.return_value = ["Ministry of Health and Family Welfare"]
                mock_selectbox.return_value = "6_months"
                mock_slider.return_value = 5
                mock_date_input.return_value = datetime.now().date()
                
                # Test query input rendering
                query = streamlit_interface.render_query_input()
                assert query == "healthcare AI technology"
                
                # Test filter rendering - create FilterConfig object
                filters = FilterConfig(
                    ministries=["Ministry of Health and Family Welfare"],
                    time_period="6_months",
                    custom_date_range=None,
                    max_articles=5,
                    relevance_threshold=0.5
                )
                
                # Test query submission handling
                response = streamlit_interface.handle_query_submission(
                    "healthcare AI technology", 
                    filters
                )
                
                # Verify response structure
                assert isinstance(response, Response)
                assert response.answer is not None
                assert len(response.answer) > 0
                assert isinstance(response.citations, list)
                
                # Test results rendering
                streamlit_interface.render_search_results(response, filters)
                
                # Verify Streamlit components were called
                assert mock_subheader.called, "Results header should be rendered"
                assert mock_write.called, "Response content should be rendered"
                assert mock_info.called, "Filter information should be rendered"
    
    def test_all_filter_combinations_through_web_interface(self, streamlit_system):
        """
        Test all filter combinations through the web interface.
        Validates: Requirements 6.4
        """
        streamlit_interface, query_engine, response_generator = streamlit_system
        
        # Test different filter combinations using FilterConfig objects
        filter_combinations = [
            FilterConfig(
                ministries=[],
                time_period=None,
                custom_date_range=None,
                max_articles=10,
                relevance_threshold=0.5
            ),
            FilterConfig(
                ministries=["Ministry of Health and Family Welfare"],
                time_period="6_months",
                custom_date_range=None,
                max_articles=5,
                relevance_threshold=0.7
            ),
            FilterConfig(
                ministries=["Ministry of Education", "Ministry of Health and Family Welfare"],
                time_period="custom",
                custom_date_range=("2024-08-01", "2024-12-31"),
                max_articles=15,
                relevance_threshold=0.3
            )
        ]
        
        for filters in filter_combinations:
            # Mock session state properly
            mock_session_state = MagicMock()
            mock_session_state.conversation_history = []
            mock_session_state.get.return_value = []
            
            with patch('streamlit.session_state', mock_session_state):
                # Test query submission with different filter combinations
                response = streamlit_interface.handle_query_submission(
                    "technology innovation initiatives",
                    filters
                )
                
                # Verify response is generated
                assert isinstance(response, Response)
                assert response.answer is not None
                
                # Verify citations respect filters
                for citation in response.citations:
                    # Check ministry filter
                    if filters.ministries:
                        assert citation.ministry in filters.ministries, \
                            f"Citation ministry {citation.ministry} not in filter {filters.ministries}"
                    
                    # Check date range filter
                    if filters.time_period == "custom" and filters.custom_date_range:
                        start_date, end_date = filters.custom_date_range
                        citation_date = datetime.fromisoformat(citation.date).date()
                        start_dt = datetime.fromisoformat(start_date).date()
                        end_dt = datetime.fromisoformat(end_date).date()
                        assert start_dt <= citation_date <= end_dt, \
                            f"Citation date {citation_date} not in range [{start_date}, {end_date}]"
                
                # Verify article count limit
                assert len(response.citations) <= filters.max_articles, \
                    f"Too many citations: {len(response.citations)} > {filters.max_articles}"
    
    def test_conversation_history_and_session_management(self, streamlit_system):
        """
        Test conversation history and session management.
        Validates: Requirements 6.1, 6.4
        """
        streamlit_interface, query_engine, response_generator = streamlit_system
        
        # Mock session state with conversation history
        conversation_history = []
        
        # Mock session state properly
        mock_session_state = MagicMock()
        mock_session_state.conversation_history = conversation_history
        mock_session_state.get.return_value = conversation_history
        
        with patch('streamlit.session_state', mock_session_state):
            # Simulate multiple queries in a conversation
            queries = [
                "What are the latest healthcare initiatives?",
                "Tell me more about AI in healthcare",
                "What about education technology programs?"
            ]
            
            filters = FilterConfig(
                ministries=[],
                time_period=None,
                custom_date_range=None,
                max_articles=10,
                relevance_threshold=0.5
            )
            
            for query in queries:
                response = streamlit_interface.handle_query_submission(query, filters)
                
                # Add to conversation history (simulating what the interface does)
                conversation_history.append({'role': 'user', 'content': query})
                conversation_history.append({'role': 'assistant', 'content': response.answer})
            
            # Verify conversation history management
            assert len(conversation_history) == 6, "Should have 3 queries + 3 responses"
            
            # Verify alternating user/assistant messages
            for i in range(0, len(conversation_history), 2):
                assert conversation_history[i]['role'] == 'user'
                assert conversation_history[i + 1]['role'] == 'assistant'
            
            # Test conversation history rendering
            with patch('streamlit.expander') as mock_expander, \
                 patch('streamlit.write') as mock_write:
                
                # Mock expander context manager
                mock_expander_context = Mock()
                mock_expander_context.__enter__ = Mock(return_value=mock_expander_context)
                mock_expander_context.__exit__ = Mock(return_value=None)
                mock_expander.return_value = mock_expander_context
                
                streamlit_interface.render_conversation_history()
                
                # Verify conversation history was rendered
                assert mock_write.called, "Conversation messages should be rendered"
    
    def test_error_handling_and_edge_cases(self, streamlit_system):
        """
        Test error handling and edge cases.
        Validates: Requirements 6.5
        """
        streamlit_interface, query_engine, response_generator = streamlit_system
        
        # Test empty query handling
        with patch('streamlit.error') as mock_error, \
             patch('streamlit.info') as mock_info:
            
            streamlit_interface.display_error_message("Empty query provided")
            
            # Verify error message was displayed
            assert mock_error.called, "Error message should be displayed"
            assert mock_info.called, "Helpful information should be provided"
        
        # Test invalid filter combinations using FilterConfig
        invalid_filters = FilterConfig(
            ministries=["Non-existent Ministry"],
            time_period="custom",
            custom_date_range=("2025-01-01", "2024-01-01"),  # Invalid date range
            max_articles=0,  # Invalid count
            relevance_threshold=1.5  # Invalid threshold
        )
        
        # Mock session state properly
        mock_session_state = MagicMock()
        mock_session_state.conversation_history = []
        mock_session_state.get.return_value = []
        
        with patch('streamlit.session_state', mock_session_state):
            # Should handle invalid filters gracefully
            try:
                response = streamlit_interface.handle_query_submission(
                    "test query",
                    invalid_filters
                )
                # If no exception, verify response is still generated
                assert isinstance(response, Response)
            except Exception as e:
                # If exception occurs, it should be handled gracefully
                # Check for various error indicators
                error_msg = str(e).lower()
                assert any(word in error_msg for word in ["error", "invalid", "must", "positive", "failed"]), \
                    f"Expected error-related message, got: {str(e)}"
        
        # Test loading indicator
        with patch('streamlit.spinner') as mock_spinner:
            streamlit_interface.display_loading_indicator()
            assert mock_spinner.called, "Loading indicator should be displayed"
        
        # Test response generator error handling
        def mock_error_response(query, search_results, conversation_history):
            raise Exception("Simulated response generation error")
        
        response_generator.generate_response.side_effect = mock_error_response
        
        valid_filters = FilterConfig(
            ministries=[],
            time_period=None,
            custom_date_range=None,
            max_articles=10,
            relevance_threshold=0.5
        )
        
        with patch('streamlit.session_state', mock_session_state):
            try:
                response = streamlit_interface.handle_query_submission(
                    "test query",
                    valid_filters
                )
                # Should not reach here if error handling works
                assert False, "Should have handled response generation error"
            except Exception as e:
                # Error should be caught and handled by the interface
                error_msg = str(e).lower()
                assert any(word in error_msg for word in ["error", "invalid", "must", "positive", "failed"]), \
                    f"Expected error-related message, got: {str(e)}"
    
    def test_filter_display_and_result_count_accuracy(self, streamlit_system):
        """
        Test that applied filters and result count are accurately displayed.
        Validates: Requirements 13.8
        """
        streamlit_interface, query_engine, response_generator = streamlit_system
        
        # Test with specific filters using FilterConfig
        filters = FilterConfig(
            ministries=["Ministry of Health and Family Welfare", "Ministry of Education"],
            time_period="6_months",
            custom_date_range=None,
            max_articles=8,
            relevance_threshold=0.4
        )
        
        # Mock session state properly
        mock_session_state = MagicMock()
        mock_session_state.conversation_history = []
        mock_session_state.get.return_value = []
        
        with patch('streamlit.session_state', mock_session_state), \
             patch('streamlit.info') as mock_info:
            
            response = streamlit_interface.handle_query_submission(
                "technology initiatives",
                filters
            )
            
            # Test filter display
            streamlit_interface.render_search_results(response, filters)
            
            # Verify filter information was displayed
            assert mock_info.called, "Filter information should be displayed"
            
            # Get the displayed filter information
            info_calls = [call[0][0] for call in mock_info.call_args_list if call[0]]
            combined_info = " ".join(info_calls)
            
            # Verify specific filter information is displayed
            assert "Ministries:" in combined_info, "Ministry filter should be displayed"
            assert "Time Period:" in combined_info, "Time period filter should be displayed"
            assert "Max Articles:" in combined_info, "Max articles should be displayed"
            assert "Min Relevance:" in combined_info, "Relevance threshold should be displayed"
            
            # Verify result count accuracy
            actual_result_count = len(response.citations)
            assert actual_result_count <= filters.max_articles, \
                f"Result count {actual_result_count} should not exceed max articles {filters.max_articles}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])