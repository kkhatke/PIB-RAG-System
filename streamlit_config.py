"""
Streamlit-specific configuration for PIB RAG System.
Handles web interface settings, themes, and environment variables.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Load base configuration
import config


class StreamlitConfig:
    """
    Configuration manager for Streamlit-specific settings.
    Handles environment variables, graceful degradation, and web interface settings.
    """
    
    # Page configuration
    PAGE_TITLE = os.getenv("STREAMLIT_PAGE_TITLE", "PIB RAG System")
    PAGE_ICON = os.getenv("STREAMLIT_PAGE_ICON", "🏛️")
    LAYOUT = os.getenv("STREAMLIT_LAYOUT", "wide")
    INITIAL_SIDEBAR_STATE = os.getenv("STREAMLIT_SIDEBAR_STATE", "expanded")
    
    # Theme configuration
    THEME_PRIMARY_COLOR = os.getenv("STREAMLIT_PRIMARY_COLOR", "#007bff")
    THEME_BACKGROUND_COLOR = os.getenv("STREAMLIT_BACKGROUND_COLOR", "#ffffff")
    THEME_SECONDARY_BACKGROUND_COLOR = os.getenv("STREAMLIT_SECONDARY_BG", "#f0f2f6")
    THEME_TEXT_COLOR = os.getenv("STREAMLIT_TEXT_COLOR", "#262730")
    
    # Model cache configuration
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", str(Path("./model_cache").resolve()))
    ENABLE_MODEL_CACHE = os.getenv("ENABLE_MODEL_CACHE", "true").lower() == "true"
    
    # Vector store configuration
    VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", str(config.VECTOR_STORE_DIR))
    
    # Ollama configuration with fallback
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", config.OLLAMA_BASE_URL)
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", config.OLLAMA_MODEL)
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", str(config.OLLAMA_TIMEOUT)))
    
    # Graceful degradation settings
    ENABLE_GRACEFUL_DEGRADATION = os.getenv("ENABLE_GRACEFUL_DEGRADATION", "true").lower() == "true"
    FALLBACK_MODE = os.getenv("FALLBACK_MODE", "demo")  # "demo", "readonly", "disabled"
    
    # Performance settings
    CACHE_TTL = int(os.getenv("STREAMLIT_CACHE_TTL", "3600"))  # 1 hour
    MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", "5"))
    
    # UI settings
    DEFAULT_MAX_ARTICLES = int(os.getenv("DEFAULT_MAX_ARTICLES", "10"))
    DEFAULT_RELEVANCE_THRESHOLD = float(os.getenv("DEFAULT_RELEVANCE_THRESHOLD", "0.5"))
    SHOW_SYSTEM_INFO = os.getenv("SHOW_SYSTEM_INFO", "true").lower() == "true"
    SHOW_DEBUG_INFO = os.getenv("SHOW_DEBUG_INFO", "false").lower() == "true"
    
    # Error handling
    SHOW_DETAILED_ERRORS = os.getenv("SHOW_DETAILED_ERRORS", "false").lower() == "true"
    LOG_USER_QUERIES = os.getenv("LOG_USER_QUERIES", "true").lower() == "true"
    
    @classmethod
    def get_page_config(cls) -> Dict[str, Any]:
        """
        Get Streamlit page configuration dictionary.
        
        Returns:
            Dictionary with page configuration settings
        """
        return {
            "page_title": cls.PAGE_TITLE,
            "page_icon": cls.PAGE_ICON,
            "layout": cls.LAYOUT,
            "initial_sidebar_state": cls.INITIAL_SIDEBAR_STATE,
            "menu_items": {
                'Get Help': None,
                'Report a bug': None,
                'About': f"{cls.PAGE_TITLE} - Intelligent Q&A about Indian Government Policies"
            }
        }
    
    @classmethod
    def get_theme_config(cls) -> Dict[str, str]:
        """
        Get theme configuration for custom CSS.
        
        Returns:
            Dictionary with theme color settings
        """
        return {
            "primary_color": cls.THEME_PRIMARY_COLOR,
            "background_color": cls.THEME_BACKGROUND_COLOR,
            "secondary_background_color": cls.THEME_SECONDARY_BACKGROUND_COLOR,
            "text_color": cls.THEME_TEXT_COLOR
        }
    
    @classmethod
    def get_custom_css(cls) -> str:
        """
        Generate custom CSS based on theme configuration.
        
        Returns:
            CSS string for Streamlit styling
        """
        theme = cls.get_theme_config()
        
        return f"""
        <style>
        .main-header {{
            text-align: center;
            padding: 1rem 0;
            border-bottom: 2px solid {theme['secondary_background_color']};
            margin-bottom: 2rem;
        }}
        
        .status-success {{
            color: #28a745;
            font-weight: bold;
        }}
        
        .status-error {{
            color: #dc3545;
            font-weight: bold;
        }}
        
        .status-warning {{
            color: #ffc107;
            font-weight: bold;
        }}
        
        .citation-box {{
            background-color: {theme['secondary_background_color']};
            border-left: 4px solid {theme['primary_color']};
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0.25rem;
        }}
        
        .filter-summary {{
            background-color: #e3f2fd;
            padding: 0.5rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }}
        
        .system-info {{
            font-size: 0.8em;
            color: #666;
            background-color: {theme['secondary_background_color']};
            padding: 0.5rem;
            border-radius: 0.25rem;
        }}
        
        .conversation-message {{
            margin-bottom: 1rem;
            padding: 0.5rem;
            border-radius: 0.25rem;
        }}
        
        .user-message {{
            background-color: #e3f2fd;
            border-left: 3px solid {theme['primary_color']};
        }}
        
        .assistant-message {{
            background-color: {theme['secondary_background_color']};
            border-left: 3px solid #28a745;
        }}
        
        .error-details {{
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 0.75rem;
            border-radius: 0.25rem;
            margin-top: 0.5rem;
        }}
        </style>
        """
    
    @classmethod
    def is_ollama_available(cls) -> bool:
        """
        Check if Ollama is available and configured.
        
        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            import requests
            response = requests.get(f"{cls.OLLAMA_BASE_URL}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    @classmethod
    def get_fallback_config(cls) -> Dict[str, Any]:
        """
        Get configuration for fallback mode when Ollama is not available.
        
        Returns:
            Dictionary with fallback configuration
        """
        return {
            "mode": cls.FALLBACK_MODE,
            "enable_search": cls.FALLBACK_MODE in ["demo", "readonly"],
            "enable_response_generation": cls.FALLBACK_MODE == "demo",
            "show_demo_responses": cls.FALLBACK_MODE == "demo",
            "message": cls._get_fallback_message()
        }
    
    @classmethod
    def _get_fallback_message(cls) -> str:
        """
        Get appropriate message for fallback mode.
        
        Returns:
            Fallback mode message
        """
        if cls.FALLBACK_MODE == "demo":
            return (
                "⚠️ **Demo Mode**: Ollama is not available. "
                "Search functionality is enabled, but responses are simulated."
            )
        elif cls.FALLBACK_MODE == "readonly":
            return (
                "⚠️ **Read-Only Mode**: Ollama is not available. "
                "You can search articles but response generation is disabled."
            )
        else:
            return (
                "❌ **Service Unavailable**: Ollama is not available. "
                "Please start Ollama and refresh the page."
            )
    
    @classmethod
    def validate_environment(cls) -> Dict[str, Any]:
        """
        Validate environment configuration and return status.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "info": []
        }
        
        # Check model cache directory
        cache_dir = Path(cls.MODEL_CACHE_DIR)
        if not cache_dir.exists():
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                validation_results["info"].append(f"Created model cache directory: {cache_dir}")
            except Exception as e:
                validation_results["errors"].append(f"Cannot create model cache directory: {e}")
                validation_results["valid"] = False
        
        # Check vector store directory
        vector_dir = Path(cls.VECTOR_STORE_DIR)
        if not vector_dir.exists():
            try:
                vector_dir.mkdir(parents=True, exist_ok=True)
                validation_results["info"].append(f"Created vector store directory: {vector_dir}")
            except Exception as e:
                validation_results["errors"].append(f"Cannot create vector store directory: {e}")
                validation_results["valid"] = False
        
        # Check Ollama availability
        if not cls.is_ollama_available():
            if cls.ENABLE_GRACEFUL_DEGRADATION:
                validation_results["warnings"].append(
                    f"Ollama not available at {cls.OLLAMA_BASE_URL}. Running in {cls.FALLBACK_MODE} mode."
                )
            else:
                validation_results["errors"].append(
                    f"Ollama not available at {cls.OLLAMA_BASE_URL} and graceful degradation is disabled."
                )
                validation_results["valid"] = False
        
        # Validate numeric settings
        try:
            if cls.DEFAULT_MAX_ARTICLES <= 0 or cls.DEFAULT_MAX_ARTICLES > 50:
                validation_results["warnings"].append(
                    f"DEFAULT_MAX_ARTICLES ({cls.DEFAULT_MAX_ARTICLES}) should be between 1 and 50"
                )
            
            if not (0.0 <= cls.DEFAULT_RELEVANCE_THRESHOLD <= 1.0):
                validation_results["warnings"].append(
                    f"DEFAULT_RELEVANCE_THRESHOLD ({cls.DEFAULT_RELEVANCE_THRESHOLD}) should be between 0.0 and 1.0"
                )
        except (ValueError, TypeError) as e:
            validation_results["errors"].append(f"Invalid numeric configuration: {e}")
            validation_results["valid"] = False
        
        return validation_results
    
    @classmethod
    def get_environment_info(cls) -> Dict[str, Any]:
        """
        Get comprehensive environment information for debugging.
        
        Returns:
            Dictionary with environment information
        """
        return {
            "streamlit_config": {
                "page_title": cls.PAGE_TITLE,
                "layout": cls.LAYOUT,
                "cache_ttl": cls.CACHE_TTL
            },
            "model_config": {
                "cache_dir": cls.MODEL_CACHE_DIR,
                "cache_enabled": cls.ENABLE_MODEL_CACHE,
                "embedding_model": config.EMBEDDING_MODEL
            },
            "ollama_config": {
                "base_url": cls.OLLAMA_BASE_URL,
                "model": cls.OLLAMA_MODEL,
                "timeout": cls.OLLAMA_TIMEOUT,
                "available": cls.is_ollama_available()
            },
            "storage_config": {
                "vector_store_dir": cls.VECTOR_STORE_DIR,
                "data_dir": str(config.DATA_DIR)
            },
            "ui_config": {
                "default_max_articles": cls.DEFAULT_MAX_ARTICLES,
                "default_relevance_threshold": cls.DEFAULT_RELEVANCE_THRESHOLD,
                "show_system_info": cls.SHOW_SYSTEM_INFO,
                "show_debug_info": cls.SHOW_DEBUG_INFO
            },
            "fallback_config": cls.get_fallback_config() if cls.ENABLE_GRACEFUL_DEGRADATION else None
        }


# Create a global instance for easy access
streamlit_config = StreamlitConfig()