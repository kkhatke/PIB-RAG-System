"""
Streamlit web interface module for PIB RAG System.
Provides modern web-based user interface for the RAG system.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timedelta
import streamlit as st

from src.query_engine.query_engine import QueryEngine
from src.response_generation.response_generator import ResponseGenerator, Response, Citation
from src.interface.conversational_interface import Message


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """
    Configuration for user filter preferences.
    
    Attributes:
        ministries: List of selected ministry names
        time_period: Selected time period filter ("1_year", "6_months", "3_months", "custom")
        custom_date_range: Custom date range tuple (start_date, end_date)
        max_articles: Maximum number of articles to retrieve
        relevance_threshold: Minimum relevance score threshold
    """
    ministries: List[str]
    time_period: Optional[str]
    custom_date_range: Optional[Tuple[str, str]]
    max_articles: int
    relevance_threshold: float


class StreamlitInterface:
    """
    Streamlit-based web interface for the PIB RAG system.
    Provides interactive UI for querying and displaying results.
    """
    
    def __init__(self, query_engine: QueryEngine, response_generator: ResponseGenerator):
        """
        Initialize the Streamlit interface.
        
        Args:
            query_engine: QueryEngine instance for search functionality
            response_generator: ResponseGenerator for generating responses
        """
        self.query_engine = query_engine
        self.response_generator = response_generator
        
        # Initialize session state
        self._initialize_session_state()
        
        logger.info("StreamlitInterface initialized")
    
    def render_sidebar_filters(self) -> FilterConfig:
        """
        Render sidebar filters for ministry, date, and article count controls.
        
        Returns:
            FilterConfig object with current filter settings
        """
        st.sidebar.header("🔍 Search Filters")
        
        # Ministry filter
        st.sidebar.subheader("Ministry Filter")
        
        # Get available ministries (cached)
        available_ministries = self._get_available_ministries()
        
        # Ministry selection
        selected_ministries = st.sidebar.multiselect(
            "Select Ministries",
            options=available_ministries,
            default=st.session_state.get('selected_ministries', []),
            help="Filter results by specific government ministries"
        )
        
        # Update session state
        st.session_state.selected_ministries = selected_ministries
        
        # Date/Time filter
        st.sidebar.subheader("Time Period Filter")
        
        time_period_options = {
            "All Time": None,
            "Last 1 Year": "1_year",
            "Last 6 Months": "6_months", 
            "Last 3 Months": "3_months",
            "Custom Range": "custom"
        }
        
        selected_time_option = st.sidebar.selectbox(
            "Select Time Period",
            options=list(time_period_options.keys()),
            index=0,
            help="Filter results by publication date"
        )
        
        time_period = time_period_options[selected_time_option]
        custom_date_range = None
        
        # Custom date range inputs
        if time_period == "custom":
            st.sidebar.write("**Custom Date Range:**")
            
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now().date() - timedelta(days=365),
                    help="Start date for custom range"
                )
            
            with col2:
                end_date = st.date_input(
                    "End Date", 
                    value=datetime.now().date(),
                    help="End date for custom range"
                )
            
            # Validate date range
            if start_date <= end_date:
                custom_date_range = (start_date.isoformat(), end_date.isoformat())
            else:
                st.sidebar.error("Start date must be before end date")
                custom_date_range = None
        
        # Article count filter
        st.sidebar.subheader("Result Limits")
        
        max_articles = st.sidebar.slider(
            "Maximum Articles",
            min_value=1,
            max_value=20,
            value=st.session_state.get('max_articles', 10),
            help="Maximum number of articles to retrieve"
        )
        
        st.session_state.max_articles = max_articles
        
        # Relevance threshold
        relevance_threshold = st.sidebar.slider(
            "Relevance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('relevance_threshold', 0.5),
            step=0.1,
            help="Minimum relevance score for results"
        )
        
        st.session_state.relevance_threshold = relevance_threshold
        
        # Clear filters button
        if st.sidebar.button("🗑️ Clear All Filters"):
            self._clear_filters()
            st.rerun()
        
        # Create and return FilterConfig
        return FilterConfig(
            ministries=selected_ministries,
            time_period=time_period,
            custom_date_range=custom_date_range,
            max_articles=max_articles,
            relevance_threshold=relevance_threshold
        )
    
    def _initialize_session_state(self) -> None:
        """
        Initialize Streamlit session state variables.
        """
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        if 'selected_ministries' not in st.session_state:
            st.session_state.selected_ministries = []
        
        if 'max_articles' not in st.session_state:
            st.session_state.max_articles = 10
        
        if 'relevance_threshold' not in st.session_state:
            st.session_state.relevance_threshold = 0.5
        
        if 'available_ministries' not in st.session_state:
            st.session_state.available_ministries = None
    
    def _get_available_ministries(self) -> List[str]:
        """
        Get available ministries from the vector store (cached).
        
        Returns:
            List of available ministry names
        """
        if st.session_state.available_ministries is None:
            try:
                ministries = self.query_engine.vector_store.get_unique_ministries()
                st.session_state.available_ministries = sorted(ministries)
                logger.info(f"Loaded {len(ministries)} available ministries")
            except Exception as e:
                logger.error(f"Failed to load ministries: {e}")
                st.session_state.available_ministries = []
        
        return st.session_state.available_ministries
    
    def _clear_filters(self) -> None:
        """
        Clear all filter settings in session state.
        """
        st.session_state.selected_ministries = []
        st.session_state.max_articles = 10
        st.session_state.relevance_threshold = 0.5
        logger.info("Filters cleared")
    
    def render_query_input(self) -> str:
        """
        Render query input field with search button.
        
        Returns:
            User query string, or empty string if no query submitted
        """
        st.header("🤖 PIB RAG System")
        st.write("Ask questions about Indian government policies and announcements")
        
        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What are the latest healthcare initiatives?",
            help="Ask any question about government policies and announcements",
            key="query_input"
        )
        
        # Search button and submit handling
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            search_clicked = st.button("🔍 Search", type="primary")
        
        with col2:
            clear_clicked = st.button("🗑️ Clear Chat")
        
        # Handle clear chat
        if clear_clicked:
            st.session_state.conversation_history = []
            st.session_state.query_input = ""
            st.rerun()
        
        # Return query if search was clicked and query is not empty
        if search_clicked and query and query.strip():
            return query.strip()
        
        return ""
    
    def handle_query_submission(self, query: str, filters: FilterConfig) -> Response:
        """
        Process user query submission with applied filters.
        
        Args:
            query: User's natural language query
            filters: Current filter configuration
            
        Returns:
            Response object containing answer and citations
            
        Raises:
            ValueError: If query is empty or filters are invalid
            RuntimeError: If query processing fails
        """
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        
        try:
            # Prepare search parameters
            ministry_filter = filters.ministries if filters.ministries else None
            date_range = None
            time_period = None
            
            # Handle time filtering
            if filters.time_period == "custom" and filters.custom_date_range:
                date_range = filters.custom_date_range
            elif filters.time_period and filters.time_period != "custom":
                time_period = filters.time_period
            
            # Perform search
            search_results = self.query_engine.search(
                query=query,
                ministry_filter=ministry_filter,
                date_range=date_range,
                time_period=time_period,
                max_articles=filters.max_articles,
                relevance_threshold=filters.relevance_threshold
            )
            
            # Generate response
            conversation_history = self._get_conversation_context()
            
            response = self.response_generator.generate_response(
                query=query,
                search_results=search_results,
                conversation_history=conversation_history
            )
            
            # Add to conversation history
            self._add_to_conversation(query, response)
            
            logger.info(f"Query processed successfully: '{query[:50]}...'")
            
            return response
            
        except ValueError:
            raise
        except Exception as e:
            error_msg = f"Query processing failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def display_loading_indicator(self) -> None:
        """
        Display loading indicator for user feedback during processing.
        """
        with st.spinner("🔍 Searching articles and generating response..."):
            # This will be used as a context manager in the main app
            pass
    
    def _get_conversation_context(self) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for LLM context.
        
        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        conversation_history = st.session_state.get('conversation_history', [])
        
        # Format for LLM (exclude current query)
        formatted_history = []
        for msg in conversation_history:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                formatted_history.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        return formatted_history[-10:]  # Keep last 10 messages for context
    
    def _add_to_conversation(self, query: str, response: Response) -> None:
        """
        Add query and response to conversation history.
        
        Args:
            query: User query
            response: Generated response
        """
        timestamp = datetime.now().isoformat()
        
        # Add user message
        user_message = {
            'role': 'user',
            'content': query,
            'timestamp': timestamp
        }
        
        # Add assistant message
        assistant_message = {
            'role': 'assistant', 
            'content': response.answer,
            'timestamp': timestamp,
            'citations': response.citations
        }
        
        # Update session state
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        st.session_state.conversation_history.extend([user_message, assistant_message])
        
        # Limit conversation history length
        max_history = 50  # 25 exchanges
        if len(st.session_state.conversation_history) > max_history:
            st.session_state.conversation_history = st.session_state.conversation_history[-max_history:]
    
    def render_search_results(self, response: Response, filters: FilterConfig) -> None:
        """
        Render search results with formatted response and applied filters.
        
        Args:
            response: Response object containing answer and citations
            filters: Applied filter configuration
        """
        # Display applied filters summary
        self._render_applied_filters(filters)
        
        # Display the response
        st.subheader("📝 Response")
        
        # Response content
        st.write(response.answer)
        
        # Display citations
        if response.citations:
            self.render_citations(response.citations)
        else:
            st.info("No sources found for this query.")
        
        # Add some spacing
        st.write("")
    
    def render_citations(self, citations: List[Citation]) -> None:
        """
        Render citations with expandable citation details.
        
        Args:
            citations: List of Citation objects to display
        """
        st.subheader(f"📚 Sources ({len(citations)} articles)")
        
        for idx, citation in enumerate(citations, 1):
            # Create expandable section for each citation
            with st.expander(
                f"**{idx}. {citation.title}**",
                expanded=False
            ):
                # Citation metadata
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Ministry:** {citation.ministry}")
                    st.write(f"**Date:** {citation.date}")
                
                with col2:
                    st.write(f"**Article ID:** {citation.article_id}")
                    st.write(f"**Relevance:** {citation.relevance_score:.3f}")
                
                # Add visual separator
                st.divider()
                
                # Option to view full article (placeholder for now)
                if st.button(f"View Full Article", key=f"view_article_{citation.article_id}"):
                    st.info("Full article viewing feature coming soon!")
    
    def render_conversation_history(self) -> None:
        """
        Render conversation history for displaying chat history.
        Uses Streamlit session state to display conversation history.
        """
        conversation_history = st.session_state.get('conversation_history', [])
        
        if not conversation_history:
            st.info("No conversation history yet. Start by asking a question!")
            return
        
        st.subheader("💬 Conversation History")
        
        # Display messages in chronological order
        for msg in conversation_history:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')
            citations = msg.get('citations', [])
            
            # Format timestamp
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_time = dt.strftime("%H:%M:%S")
            except:
                formatted_time = "Unknown time"
            
            # Display message based on role
            if role == 'user':
                st.write(f"**🧑 You** _{formatted_time}_")
                st.write(content)
            elif role == 'assistant':
                st.write(f"**🤖 Assistant** _{formatted_time}_")
                st.write(content)
                
                # Show citations if available
                if citations:
                    with st.expander(f"Sources ({len(citations)})"):
                        for idx, citation in enumerate(citations, 1):
                            if hasattr(citation, 'title'):
                                # Citation object
                                st.write(f"{idx}. **{citation.title}** ({citation.ministry}, {citation.date})")
                            else:
                                # Citation dict
                                title = citation.get('title', 'Unknown')
                                ministry = citation.get('ministry', 'Unknown')
                                date = citation.get('date', 'Unknown')
                                st.write(f"{idx}. **{title}** ({ministry}, {date})")
            
            st.write("")  # Add spacing between messages
    
    def clear_conversation_history(self) -> None:
        """
        Clear conversation history from session state.
        """
        st.session_state.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_conversation_history(self) -> List[dict]:
        """
        Get conversation history from session state.
        
        Returns:
            List of conversation messages
        """
        return st.session_state.get('conversation_history', [])
    
    def display_error_message(self, error: str) -> None:
        """
        Display user-friendly error message.
        
        Args:
            error: Error message to display
        """
        st.error(f"❌ {error}")
        
        # Provide helpful suggestions based on error type
        error_lower = error.lower()
        
        if "ollama" in error_lower or "connection" in error_lower:
            st.info(
                "💡 **Troubleshooting Tips:**\n"
                "- Make sure Ollama is running: `ollama serve`\n"
                "- Check if the model is available: `ollama list`\n"
                "- Pull the required model: `ollama pull llama3.2`"
            )
        elif "empty" in error_lower or "query" in error_lower:
            st.info("💡 Please enter a valid question to search for information.")
        elif "filter" in error_lower or "date" in error_lower:
            st.info("💡 Please check your filter settings and try again.")
        else:
            st.info("💡 Please try again or contact support if the problem persists.")
    
    def validate_query_input(self, query: str) -> bool:
        """
        Validate user query input.
        
        Args:
            query: User query string to validate
            
        Returns:
            True if query is valid, False otherwise
        """
        if not query or not isinstance(query, str):
            return False
        
        # Check if query is not just whitespace
        if not query.strip():
            return False
        
        # Check minimum length
        if len(query.strip()) < 3:
            return False
        
        return True
    
    def validate_filter_parameters(self, filters: FilterConfig) -> Tuple[bool, Optional[str]]:
        """
        Validate filter parameters for edge cases.
        
        Args:
            filters: FilterConfig object to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Validate max_articles
            if filters.max_articles <= 0 or filters.max_articles > 50:
                return False, "Maximum articles must be between 1 and 50"
            
            # Validate relevance_threshold
            if not (0.0 <= filters.relevance_threshold <= 1.0):
                return False, "Relevance threshold must be between 0.0 and 1.0"
            
            # Validate custom date range if provided
            if filters.time_period == "custom" and filters.custom_date_range:
                start_date, end_date = filters.custom_date_range
                
                # Check date format
                try:
                    start_dt = datetime.fromisoformat(start_date)
                    end_dt = datetime.fromisoformat(end_date)
                    
                    # Check if start is before end
                    if start_dt > end_dt:
                        return False, "Start date must be before or equal to end date"
                    
                    # Check if dates are not too far in the future
                    today = datetime.now().date()
                    if start_dt.date() > today or end_dt.date() > today:
                        return False, "Dates cannot be in the future"
                        
                except ValueError:
                    return False, "Invalid date format. Please use YYYY-MM-DD format"
            
            # Validate ministries
            if filters.ministries:
                for ministry in filters.ministries:
                    if not isinstance(ministry, str) or not ministry.strip():
                        return False, "All ministry names must be non-empty strings"
            
            return True, None
            
        except Exception as e:
            return False, f"Filter validation error: {str(e)}"
    
    def handle_empty_query_case(self) -> None:
        """
        Handle the case when user submits an empty query.
        """
        st.warning("⚠️ Please enter a question to search for information.")
        st.info("💡 Try asking about government policies, announcements, or specific ministries.")
    
    def handle_invalid_date_range_case(self, start_date: str, end_date: str) -> None:
        """
        Handle invalid date range cases.
        
        Args:
            start_date: Start date string
            end_date: End date string
        """
        if start_date > end_date:
            st.error("❌ Start date must be before or equal to end date.")
        else:
            st.error("❌ Invalid date range. Please check your date inputs.")
        
        st.info("💡 Use the date picker to select valid dates, or choose a predefined time period.")
    
    def _render_applied_filters(self, filters: FilterConfig) -> None:
        """
        Render summary of applied filters and result count.
        
        Args:
            filters: Current filter configuration
        """
        filter_parts = []
        
        # Ministry filter
        if filters.ministries:
            ministry_text = ", ".join(filters.ministries[:3])
            if len(filters.ministries) > 3:
                ministry_text += f" (+{len(filters.ministries) - 3} more)"
            filter_parts.append(f"**Ministries:** {ministry_text}")
        
        # Time filter
        if filters.time_period == "custom" and filters.custom_date_range:
            start_date, end_date = filters.custom_date_range
            filter_parts.append(f"**Date Range:** {start_date} to {end_date}")
        elif filters.time_period:
            time_labels = {
                "1_year": "Last 1 Year",
                "6_months": "Last 6 Months", 
                "3_months": "Last 3 Months"
            }
            filter_parts.append(f"**Time Period:** {time_labels.get(filters.time_period, filters.time_period)}")
        
        # Always show article limit and relevance threshold as they affect results
        filter_parts.append(f"**Max Articles:** {filters.max_articles}")
        filter_parts.append(f"**Min Relevance:** {filters.relevance_threshold}")
        
        # Display filters in an info box
        if filters.ministries or filters.time_period:
            # Meaningful filters are applied
            filter_text = " | ".join(filter_parts)
            st.info(f"🔍 **Applied Filters:** {filter_text}")
        else:
            # No meaningful filters applied - just show defaults
            st.info("🔍 **No filters applied** - searching all articles")
            st.info(f"**Search Settings:** Max Articles: {filters.max_articles} | Min Relevance: {filters.relevance_threshold}")