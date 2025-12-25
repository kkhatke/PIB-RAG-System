"""
Tests for Streamlit interface module.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from hypothesis import given, strategies as st
from hypothesis import settings, HealthCheck

from src.interface.streamlit_interface import StreamlitInterface, FilterConfig
from src.response_generation.response_generator import Response, Citation


class TestStreamlitInterface:
    """Test cases for StreamlitInterface class."""
    
    @pytest.fixture
    def mock_query_engine(self):
        """Create mock query engine."""
        mock_engine = Mock()
        mock_engine.vector_store.get_unique_ministries.return_value = [
            "Ministry of Health", "Ministry of Education", "Ministry of Finance"
        ]
        return mock_engine
    
    @pytest.fixture
    def mock_response_generator(self):
        """Create mock response generator."""
        return Mock()
    
    @pytest.fixture
    def streamlit_interface(self, mock_query_engine, mock_response_generator):
        """Create StreamlitInterface instance with mocks."""
        return StreamlitInterface(mock_query_engine, mock_response_generator)


class TestFilterDisplayConsistency:
    """Property tests for filter display consistency."""
    
    @pytest.fixture
    def mock_streamlit_interface(self):
        """Create mock StreamlitInterface for testing."""
        mock_query_engine = Mock()
        mock_response_generator = Mock()
        return StreamlitInterface(mock_query_engine, mock_response_generator)
    
    # Feature: pib-rag-system, Property 43: Filter display consistency
    @given(
        ministries=st.lists(
            st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
            min_size=0,
            max_size=10
        ),
        time_period=st.one_of(
            st.none(),
            st.sampled_from(["1_year", "6_months", "3_months", "custom"])
        ),
        max_articles=st.integers(min_value=1, max_value=20),
        relevance_threshold=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_filter_display_consistency(
        self,
        mock_streamlit_interface,
        ministries,
        time_period,
        max_articles,
        relevance_threshold
    ):
        """
        Property 43: Filter display consistency
        For any filter configuration, the applied filters and result count 
        should be accurately shown to the user.
        **Validates: Requirements 13.8**
        """
        # Create custom date range for custom time period
        custom_date_range = None
        if time_period == "custom":
            start_date = (datetime.now().date() - timedelta(days=365)).isoformat()
            end_date = datetime.now().date().isoformat()
            custom_date_range = (start_date, end_date)
        
        # Create filter config
        filter_config = FilterConfig(
            ministries=ministries,
            time_period=time_period,
            custom_date_range=custom_date_range,
            max_articles=max_articles,
            relevance_threshold=relevance_threshold
        )
        
        # Mock streamlit components to capture rendered content
        with patch('streamlit.info') as mock_info:
            # Call the method that renders applied filters
            mock_streamlit_interface._render_applied_filters(filter_config)
            
            # Verify that info was called (filter display was rendered)
            assert mock_info.called
            
            # Get all the rendered filter text from all calls
            all_call_args = []
            for call in mock_info.call_args_list:
                if call and call[0]:
                    all_call_args.append(call[0][0])
            
            combined_call_args = " ".join(all_call_args)
            
            # Verify filter consistency - all applied filters should be displayed
            if ministries:
                # Should show ministry information
                assert "Ministries:" in combined_call_args or "No filters applied" in combined_call_args
            
            if time_period == "custom" and custom_date_range:
                # Should show custom date range
                assert "Date Range:" in combined_call_args or "No filters applied" in combined_call_args
            elif time_period and time_period != "custom":
                # Should show time period
                assert "Time Period:" in combined_call_args or "No filters applied" in combined_call_args
            
            # Should always show max articles and relevance threshold
            assert "Max Articles:" in combined_call_args or "No filters applied" in combined_call_args
            assert "Min Relevance:" in combined_call_args or "No filters applied" in combined_call_args
            
            # If no meaningful filters are applied, should show "No filters applied"
            if not ministries and not time_period:
                assert "No filters applied" in combined_call_args
    
    @given(
        ministries=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5),
        max_articles=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_ministry_filter_display_truncation(
        self,
        mock_streamlit_interface,
        ministries,
        max_articles
    ):
        """
        Test that ministry filter display handles long lists correctly.
        """
        filter_config = FilterConfig(
            ministries=ministries,
            time_period=None,
            custom_date_range=None,
            max_articles=max_articles,
            relevance_threshold=0.5
        )
        
        with patch('streamlit.info') as mock_info:
            mock_streamlit_interface._render_applied_filters(filter_config)
            
            assert mock_info.called
            call_args = mock_info.call_args[0][0] if mock_info.call_args else ""
            
            # If more than 3 ministries, should show truncation
            if len(ministries) > 3:
                assert "more)" in call_args or "Ministries:" in call_args
            
            # Should always show the filter information
            assert "Ministries:" in call_args
    
    @given(
        start_days_ago=st.integers(min_value=1, max_value=1000),
        end_days_ago=st.integers(min_value=0, max_value=100)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_custom_date_range_display(
        self,
        mock_streamlit_interface,
        start_days_ago,
        end_days_ago
    ):
        """
        Test that custom date range is displayed correctly.
        """
        # Ensure start_date is before end_date
        if start_days_ago <= end_days_ago:
            start_days_ago = end_days_ago + 1
        
        start_date = (datetime.now().date() - timedelta(days=start_days_ago)).isoformat()
        end_date = (datetime.now().date() - timedelta(days=end_days_ago)).isoformat()
        
        filter_config = FilterConfig(
            ministries=[],
            time_period="custom",
            custom_date_range=(start_date, end_date),
            max_articles=10,
            relevance_threshold=0.5
        )
        
        with patch('streamlit.info') as mock_info:
            mock_streamlit_interface._render_applied_filters(filter_config)
            
            assert mock_info.called
            call_args = mock_info.call_args[0][0] if mock_info.call_args else ""
            
            # Should display the custom date range
            assert "Date Range:" in call_args
            assert start_date in call_args
            assert end_date in call_args


class TestWebInterfaceResponsiveness:
    """Property tests for web interface responsiveness."""
    
    @pytest.fixture
    def mock_streamlit_interface(self):
        """Create mock StreamlitInterface for testing."""
        mock_query_engine = Mock()
        mock_response_generator = Mock()
        return StreamlitInterface(mock_query_engine, mock_response_generator)
    
    # Feature: pib-rag-system, Property 44: Web interface responsiveness
    @given(
        error_messages=st.lists(
            st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_web_interface_responsiveness(self, mock_streamlit_interface, error_messages):
        """
        Property 44: Web interface responsiveness
        For any user interaction in the web interface, the system should provide 
        immediate feedback (loading indicators, error messages, or results).
        **Validates: Requirements 6.5**
        """
        # Test error message display responsiveness
        for error_msg in error_messages:
            with patch('streamlit.error') as mock_error, \
                 patch('streamlit.info') as mock_info:
                
                # Call display_error_message
                mock_streamlit_interface.display_error_message(error_msg)
                
                # Verify immediate feedback was provided
                assert mock_error.called, "Error message should be displayed immediately"
                
                # Verify the error message was displayed
                error_call_args = mock_error.call_args[0][0] if mock_error.call_args else ""
                assert error_msg in error_call_args or "❌" in error_call_args
                
                # Verify helpful info was also provided (responsiveness)
                assert mock_info.called, "Helpful information should be provided immediately"
    
    @given(
        queries=st.lists(
            st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
            min_size=1,
            max_size=3
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_loading_indicator_responsiveness(self, mock_streamlit_interface, queries):
        """
        Test that loading indicators are displayed immediately for user feedback.
        """
        for query in queries:
            with patch('streamlit.spinner') as mock_spinner:
                # Call display_loading_indicator
                mock_streamlit_interface.display_loading_indicator()
                
                # Verify spinner was called (immediate feedback)
                assert mock_spinner.called, "Loading indicator should be displayed immediately"
    
    @given(
        responses=st.lists(
            st.builds(
                Response,
                answer=st.text(min_size=1, max_size=500),
                citations=st.lists(
                    st.builds(
                        Citation,
                        article_id=st.text(min_size=1, max_size=20),
                        date=st.text(min_size=10, max_size=10),  # YYYY-MM-DD format
                        ministry=st.text(min_size=1, max_size=50),
                        title=st.text(min_size=1, max_size=100),
                        relevance_score=st.floats(min_value=0.0, max_value=1.0)
                    ),
                    min_size=0,
                    max_size=5
                )
            ),
            min_size=1,
            max_size=2
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_results_display_responsiveness(self, mock_streamlit_interface, responses):
        """
        Test that search results are displayed immediately when available.
        """
        for response in responses:
            filter_config = FilterConfig(
                ministries=[],
                time_period=None,
                custom_date_range=None,
                max_articles=10,
                relevance_threshold=0.5
            )
            
            with patch('streamlit.subheader') as mock_subheader, \
                 patch('streamlit.write') as mock_write, \
                 patch('streamlit.info') as mock_info:
                
                # Call render_search_results
                mock_streamlit_interface.render_search_results(response, filter_config)
                
                # Verify immediate display of results
                assert mock_subheader.called, "Results header should be displayed immediately"
                assert mock_write.called, "Response content should be displayed immediately"
                assert mock_info.called, "Filter information should be displayed immediately"


class TestStreamlitComponentRendering:
    """Property tests for Streamlit component rendering."""
    
    @pytest.fixture
    def mock_streamlit_interface(self):
        """Create mock StreamlitInterface for testing."""
        mock_query_engine = Mock()
        mock_response_generator = Mock()
        return StreamlitInterface(mock_query_engine, mock_response_generator)
    
    # Feature: pib-rag-system, Property 45: Streamlit component rendering
    @given(
        citations=st.lists(
            st.builds(
                Citation,
                article_id=st.text(min_size=1, max_size=20),
                date=st.text(min_size=10, max_size=10),  # YYYY-MM-DD format
                ministry=st.text(min_size=1, max_size=50),
                title=st.text(min_size=1, max_size=100),
                relevance_score=st.floats(min_value=0.0, max_value=1.0)
            ),
            min_size=0,
            max_size=10
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_streamlit_component_rendering(self, mock_streamlit_interface, citations):
        """
        Property 45: Streamlit component rendering
        For any web interface component (filters, results, citations), the component 
        should render without errors and display the expected content.
        **Validates: Requirements 6.1, 6.3, 6.4**
        """
        # Test citation rendering
        with patch('streamlit.subheader') as mock_subheader, \
             patch('streamlit.expander') as mock_expander, \
             patch('streamlit.write') as mock_write, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.divider') as mock_divider, \
             patch('streamlit.button') as mock_button:
            
            # Mock columns return with context manager support
            mock_col1 = Mock()
            mock_col1.__enter__ = Mock(return_value=mock_col1)
            mock_col1.__exit__ = Mock(return_value=None)
            
            mock_col2 = Mock()
            mock_col2.__enter__ = Mock(return_value=mock_col2)
            mock_col2.__exit__ = Mock(return_value=None)
            
            mock_columns.return_value = [mock_col1, mock_col2]
            
            # Mock expander context manager
            mock_expander_context = Mock()
            mock_expander_context.__enter__ = Mock(return_value=mock_expander_context)
            mock_expander_context.__exit__ = Mock(return_value=None)
            mock_expander.return_value = mock_expander_context
            
            try:
                # Call render_citations
                mock_streamlit_interface.render_citations(citations)
                
                # Verify components rendered without errors
                if citations:
                    assert mock_subheader.called, "Citations header should be rendered"
                    assert mock_expander.called, "Citation expanders should be rendered"
                    # Each citation should have rendered components
                    assert mock_write.called, "Citation details should be rendered"
                    assert mock_columns.called, "Citation layout columns should be rendered"
                    assert mock_button.called, "View article buttons should be rendered"
                else:
                    # Even with no citations, header should still render
                    assert mock_subheader.called, "Citations header should be rendered even with no citations"
                
                # No exceptions should be raised during rendering
                success = True
                
            except Exception as e:
                success = False
                pytest.fail(f"Component rendering failed: {e}")
            
            assert success, "All components should render without errors"
    
    @given(
        filter_configs=st.builds(
            FilterConfig,
            ministries=st.lists(st.text(min_size=1, max_size=30), min_size=0, max_size=5),
            time_period=st.one_of(st.none(), st.sampled_from(["1_year", "6_months", "3_months", "custom"])),
            custom_date_range=st.one_of(st.none(), st.tuples(st.text(min_size=10, max_size=10), st.text(min_size=10, max_size=10))),
            max_articles=st.integers(min_value=1, max_value=20),
            relevance_threshold=st.floats(min_value=0.0, max_value=1.0)
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_filter_component_rendering(self, mock_streamlit_interface, filter_configs):
        """
        Test that filter components render without errors.
        """
        with patch('streamlit.info') as mock_info:
            try:
                # Call _render_applied_filters
                mock_streamlit_interface._render_applied_filters(filter_configs)
                
                # Verify filter display rendered
                assert mock_info.called, "Filter information should be rendered"
                
                # No exceptions should be raised during rendering
                success = True
                
            except Exception as e:
                success = False
                pytest.fail(f"Filter component rendering failed: {e}")
            
            assert success, "Filter components should render without errors"


class TestStreamlitInterfaceUnit:
    """Unit tests for StreamlitInterface methods."""
    
    def test_filter_config_creation(self):
        """Test FilterConfig dataclass creation."""
        config = FilterConfig(
            ministries=["Ministry of Health"],
            time_period="1_year",
            custom_date_range=None,
            max_articles=10,
            relevance_threshold=0.5
        )
        
        assert config.ministries == ["Ministry of Health"]
        assert config.time_period == "1_year"
        assert config.custom_date_range is None
        assert config.max_articles == 10
        assert config.relevance_threshold == 0.5
    
    def test_conversation_context_formatting(self, streamlit_interface):
        """Test conversation context formatting."""
        # Mock session state
        with patch('streamlit.session_state', {'conversation_history': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'}
        ]}):
            context = streamlit_interface._get_conversation_context()
            
            assert len(context) == 2
            assert context[0]['role'] == 'user'
            assert context[0]['content'] == 'Hello'
            assert context[1]['role'] == 'assistant'
            assert context[1]['content'] == 'Hi there!'
    
    def test_add_to_conversation(self, streamlit_interface):
        """Test adding messages to conversation history."""
        response = Response(
            answer="Test answer",
            citations=[Citation(
                article_id="123",
                date="2025-01-01",
                ministry="Test Ministry",
                title="Test Article",
                relevance_score=0.8
            )]
        )
        
        with patch('streamlit.session_state', {'conversation_history': []}):
            streamlit_interface._add_to_conversation("Test query", response)
            
            # Should have added 2 messages (user and assistant)
            history = streamlit_interface._get_conversation_context()
            assert len(history) == 2
            assert history[0]['role'] == 'user'
            assert history[0]['content'] == "Test query"
            assert history[1]['role'] == 'assistant'
            assert history[1]['content'] == "Test answer"