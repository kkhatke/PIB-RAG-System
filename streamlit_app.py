"""
Streamlit main application for PIB RAG System.
Main entry point for the web-based interface.
"""
import logging
import os
import sys
from pathlib import Path
import streamlit as st
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.interface.streamlit_interface import StreamlitInterface, FilterConfig
from src.query_engine.query_engine import QueryEngine
from src.response_generation.response_generator import ResponseGenerator
from src.vector_store.vector_store import VectorStore
from src.embedding.embedding_generator import EmbeddingGenerator
import config
from streamlit_config import streamlit_config


# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def configure_streamlit_page():
    """
    Configure Streamlit page settings and layout.
    """
    # Validate environment before configuring
    validation = streamlit_config.validate_environment()
    
    # Set page configuration using streamlit_config
    page_config = streamlit_config.get_page_config()
    st.set_page_config(**page_config)
    
    # Apply custom CSS with theme configuration
    custom_css = streamlit_config.get_custom_css()
    st.markdown(custom_css, unsafe_allow_html=True)
    
    # Display validation warnings if any
    if validation["warnings"] and streamlit_config.SHOW_DEBUG_INFO:
        for warning in validation["warnings"]:
            st.sidebar.warning(warning)
    
    # Display validation errors if any
    if validation["errors"]:
        for error in validation["errors"]:
            st.error(error)
        if not validation["valid"]:
            st.stop()
    
    return validation


@st.cache_resource
def initialize_rag_system():
    """
    Initialize all RAG system components with caching and graceful degradation.
    This function is cached to avoid reinitializing components on every rerun.
    
    Returns:
        Tuple of (StreamlitInterface, system_status, error_message, fallback_config)
    """
    try:
        logger.info("Initializing RAG system components...")
        
        # Check if graceful degradation is enabled and Ollama is not available
        ollama_available = streamlit_config.is_ollama_available()
        fallback_config = None
        
        if not ollama_available and streamlit_config.ENABLE_GRACEFUL_DEGRADATION:
            fallback_config = streamlit_config.get_fallback_config()
            logger.warning(f"Ollama not available, using fallback mode: {fallback_config['mode']}")
        
        # Initialize embedding generator with model caching
        embedding_generator = EmbeddingGenerator(
            cache_dir=streamlit_config.MODEL_CACHE_DIR
        )
        
        # Initialize vector store
        vector_store = VectorStore(persist_directory=streamlit_config.VECTOR_STORE_DIR)
        
        # Check if vector store has data
        chunk_count = vector_store.count()
        if chunk_count == 0:
            logger.warning("Vector store is empty. Please run data ingestion first.")
            return None, "warning", "Vector store is empty. Please run 'python ingest_articles.py' to load data.", None
        
        logger.info(f"Vector store loaded with {chunk_count} chunks")
        
        # Initialize query engine
        query_engine = QueryEngine(vector_store, embedding_generator)
        
        # Initialize response generator with graceful degradation
        response_generator = None
        if ollama_available:
            try:
                response_generator = ResponseGenerator(
                    ollama_base_url=streamlit_config.OLLAMA_BASE_URL,
                    model=streamlit_config.OLLAMA_MODEL,
                    timeout=streamlit_config.OLLAMA_TIMEOUT
                )
            except Exception as e:
                if streamlit_config.ENABLE_GRACEFUL_DEGRADATION:
                    logger.warning(f"Failed to initialize ResponseGenerator, using fallback: {e}")
                    fallback_config = streamlit_config.get_fallback_config()
                else:
                    raise
        elif not streamlit_config.ENABLE_GRACEFUL_DEGRADATION:
            raise RuntimeError(f"Ollama not available at {streamlit_config.OLLAMA_BASE_URL}")
        
        # Initialize Streamlit interface
        streamlit_interface = StreamlitInterface(query_engine, response_generator)
        
        logger.info("RAG system initialized successfully")
        return streamlit_interface, "success", None, fallback_config
        
    except Exception as e:
        error_msg = f"Failed to initialize RAG system: {str(e)}"
        logger.error(error_msg)
        return None, "error", error_msg, None


def display_system_status(status: str, message: str = None, fallback_config: dict = None):
    """
    Display system initialization status with graceful degradation support.
    
    Args:
        status: Status type ("success", "error", "warning")
        message: Optional status message
        fallback_config: Optional fallback configuration for graceful degradation
    """
    if status == "success":
        if fallback_config:
            # Show fallback mode message
            if fallback_config["mode"] == "demo":
                st.info(fallback_config["message"])
            elif fallback_config["mode"] == "readonly":
                st.warning(fallback_config["message"])
            else:
                st.error(fallback_config["message"])
        else:
            st.success("✅ RAG System initialized successfully!")
            
    elif status == "error":
        st.error(f"❌ System Error: {message}")
        
        # Provide troubleshooting information
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **Common Issues and Solutions:**
            
            1. **Ollama Connection Error:**
               - Make sure Ollama is running: `ollama serve`
               - Check if the model is available: `ollama list`
               - Pull the required model: `ollama pull llama3.2`
            
            2. **Vector Store Empty:**
               - Run data ingestion: `python ingest_articles.py`
               - Make sure PIB articles JSON file exists in the data directory
            
            3. **Model Cache Issues:**
               - Check if model cache directory exists and is writable
               - Clear cache and restart if needed
            
            4. **Environment Variables:**
               - Check `.env` file for correct configuration
               - Verify OLLAMA_BASE_URL and OLLAMA_MODEL settings
            """)
        
        if streamlit_config.SHOW_DETAILED_ERRORS:
            st.error(f"**Detailed Error:** {message}")
        
        st.stop()
        
    elif status == "warning":
        st.warning(f"⚠️ {message}")
        
        with st.expander("📋 Setup Instructions"):
            st.markdown("""
            **To set up the PIB RAG System:**
            
            1. **Prepare Data:**
               ```bash
               python pib_article_scraper.py  # Scrape articles (optional)
               python ingest_articles.py     # Process and index articles
               ```
            
            2. **Start Ollama:**
               ```bash
               ollama serve
               ollama pull llama3.2  # or ollama pull mistral
               ```
            
            3. **Restart the Application:**
               ```bash
               streamlit run streamlit_app.py
               ```
            """)
        
        st.stop()


def display_system_info():
    """
    Display system information and configuration.
    """
    if not streamlit_config.SHOW_SYSTEM_INFO:
        return
    
    with st.sidebar.expander("ℹ️ System Info"):
        env_info = streamlit_config.get_environment_info()
        
        st.write("**Configuration:**")
        st.write(f"- Ollama URL: {env_info['ollama_config']['base_url']}")
        st.write(f"- Model: {env_info['ollama_config']['model']}")
        st.write(f"- Embedding Model: {env_info['model_config']['embedding_model']}")
        st.write(f"- Vector Store: {env_info['storage_config']['vector_store_dir']}")
        
        # Display cache information
        if env_info['model_config']['cache_enabled']:
            st.write(f"- Model Cache: {env_info['model_config']['cache_dir']}")
        
        st.write("**System Status:**")
        st.write(f"- Ollama Available: {'✅' if env_info['ollama_config']['available'] else '❌'}")
        st.write(f"- Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show fallback configuration if applicable
        if env_info['fallback_config']:
            st.write("**Fallback Mode:**")
            st.write(f"- Mode: {env_info['fallback_config']['mode']}")
            st.write(f"- Search Enabled: {'✅' if env_info['fallback_config']['enable_search'] else '❌'}")
            st.write(f"- Response Generation: {'✅' if env_info['fallback_config']['enable_response_generation'] else '❌'}")
        
        # Show debug information if enabled
        if streamlit_config.SHOW_DEBUG_INFO:
            with st.expander("🐛 Debug Info"):
                st.json(env_info)


def handle_fallback_mode(fallback_config: dict, streamlit_interface: StreamlitInterface, query: str, filters: FilterConfig):
    """
    Handle query processing in fallback mode when Ollama is not available.
    
    Args:
        fallback_config: Fallback configuration dictionary
        streamlit_interface: StreamlitInterface instance
        query: User query
        filters: Filter configuration
    """
    if not fallback_config["enable_search"]:
        st.error("Search functionality is disabled in current fallback mode.")
        return
    
    try:
        # Perform search without response generation
        search_results = streamlit_interface.query_engine.search(
            query=query,
            ministry_filter=filters.ministries if filters.ministries else None,
            date_range=filters.custom_date_range if filters.time_period == "custom" else None,
            time_period=filters.time_period if filters.time_period != "custom" else None,
            max_articles=filters.max_articles,
            relevance_threshold=filters.relevance_threshold
        )
        
        if fallback_config["show_demo_responses"]:
            # Generate a demo response
            demo_response = generate_demo_response(query, search_results)
            streamlit_interface.render_search_results(demo_response, filters)
        else:
            # Show search results without generated response
            st.subheader("📝 Search Results")
            st.info("Response generation is not available. Showing search results only.")
            
            if search_results:
                citations = streamlit_interface.response_generator.extract_citations(search_results) if streamlit_interface.response_generator else []
                if not citations:
                    # Extract citations manually if response generator is not available
                    citations = []
                    seen_ids = set()
                    for result in search_results:
                        if result.chunk.article_id not in seen_ids:
                            from src.response_generation.response_generator import Citation
                            citation = Citation(
                                article_id=result.chunk.article_id,
                                date=result.chunk.metadata.get('date', 'Unknown'),
                                ministry=result.chunk.metadata.get('ministry', 'Unknown'),
                                title=result.chunk.metadata.get('title', 'Unknown'),
                                relevance_score=result.score
                            )
                            citations.append(citation)
                            seen_ids.add(result.chunk.article_id)
                
                streamlit_interface.render_citations(citations)
            else:
                st.info("No relevant articles found for your query.")
    
    except Exception as e:
        logger.error(f"Error in fallback mode: {e}")
        streamlit_interface.display_error_message(f"Search failed: {str(e)}")


def generate_demo_response(query: str, search_results) -> 'Response':
    """
    Generate a demo response for fallback mode.
    
    Args:
        query: User query
        search_results: Search results from vector store
        
    Returns:
        Demo Response object
    """
    from src.response_generation.response_generator import Response, Citation
    
    # Create demo answer
    demo_answer = (
        f"**[DEMO MODE]** This is a simulated response for the query: '{query}'\n\n"
        f"Based on the search results, I found {len(search_results)} relevant articles. "
        "However, since Ollama is not available, I cannot generate a proper response. "
        "Please check the citations below for the actual information from the retrieved articles."
    )
    
    # Extract citations
    citations = []
    seen_ids = set()
    for result in search_results:
        if result.chunk.article_id not in seen_ids:
            citation = Citation(
                article_id=result.chunk.article_id,
                date=result.chunk.metadata.get('date', 'Unknown'),
                ministry=result.chunk.metadata.get('ministry', 'Unknown'),
                title=result.chunk.metadata.get('title', 'Unknown'),
                relevance_score=result.score
            )
            citations.append(citation)
            seen_ids.add(result.chunk.article_id)
    
    return Response(answer=demo_answer, citations=citations)


def main():
    """
    Main Streamlit application function.
    """
    # Configure Streamlit page and validate environment
    validation = configure_streamlit_page()
    
    # Display main header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title(f"{streamlit_config.PAGE_ICON} {streamlit_config.PAGE_TITLE}")
    st.markdown("**Intelligent Q&A about Indian Government Policies**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize RAG system with caching
    with st.spinner("🔄 Initializing RAG system..."):
        streamlit_interface, status, error_message, fallback_config = initialize_rag_system()
    
    # Display system status
    display_system_status(status, error_message, fallback_config)
    
    # Display system information in sidebar
    display_system_info()
    
    # Main application logic
    if streamlit_interface:
        # Render sidebar filters
        filters = streamlit_interface.render_sidebar_filters()
        
        # Render query input
        query = streamlit_interface.render_query_input()
        
        # Process query if submitted
        if query and query.strip():
            if streamlit_config.LOG_USER_QUERIES:
                logger.info(f"Processing query: {query[:100]}...")
            
            try:
                # Check if we're in fallback mode
                if fallback_config:
                    handle_fallback_mode(fallback_config, streamlit_interface, query, filters)
                else:
                    # Normal mode - process query with full functionality
                    with st.spinner("🔍 Searching and generating response..."):
                        response = streamlit_interface.handle_query_submission(query, filters)
                    
                    if response:
                        streamlit_interface.render_search_results(response, filters)
                    else:
                        st.info("No relevant articles found for your query. Try adjusting your filters or rephrasing your question.")
            
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                streamlit_interface.display_error_message(f"An error occurred while processing your query: {str(e)}")
        
        # Render conversation history if available
        if hasattr(streamlit_interface, 'render_conversation_history'):
            streamlit_interface.render_conversation_history()
    
    # Footer with additional information
    st.markdown("---")
    st.markdown(
        f"<div class='system-info'>PIB RAG System v1.0 | "
        f"Powered by {streamlit_config.OLLAMA_MODEL} via Ollama</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()