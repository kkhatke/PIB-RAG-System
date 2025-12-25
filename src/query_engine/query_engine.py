"""
Query engine module for PIB RAG System.
Processes user queries and retrieves relevant article chunks with filtering.
"""
import logging
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import re

from src.vector_store.vector_store import VectorStore, SearchResult
from src.embedding.embedding_generator import EmbeddingGenerator


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Query engine that processes user queries and retrieves relevant chunks.
    Integrates vector store and embedding generator with filtering capabilities.
    """
    
    def __init__(self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator):
        """
        Initialize the query engine.
        
        Args:
            vector_store: VectorStore instance for similarity search
            embedding_generator: EmbeddingGenerator for query embeddings
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
    
    def search(
        self,
        query: str,
        ministry_filter: Optional[List[str]] = None,
        date_range: Optional[Tuple[str, str]] = None,
        time_period: Optional[str] = None,
        max_articles: int = 10,
        top_k: int = 5,
        relevance_threshold: float = 0.5
    ) -> List[SearchResult]:
        """
        Search for relevant chunks based on query with optional filters.
        
        Args:
            query: Natural language query string
            ministry_filter: Optional list of ministry names to filter by
            date_range: Optional tuple of (start_date, end_date) in ISO format (YYYY-MM-DD)
            time_period: Optional time period filter ("1_year", "6_months", "3_months")
            max_articles: Maximum number of articles to return (configurable limit)
            top_k: Maximum number of results to return (for backward compatibility)
            relevance_threshold: Minimum relevance score (0.0 to 1.0)
            
        Returns:
            List of SearchResult objects ordered by relevance score (descending)
            
        Raises:
            ValueError: If query is empty or parameters are invalid
            RuntimeError: If search fails
        """
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        
        if max_articles <= 0:
            raise ValueError("max_articles must be positive")
        
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        if not (0.0 <= relevance_threshold <= 1.0):
            raise ValueError("relevance_threshold must be between 0.0 and 1.0")
        
        # Validate time_period if provided
        if time_period is not None:
            valid_periods = self.get_available_time_periods()
            if time_period not in valid_periods:
                raise ValueError(f"time_period must be one of: {valid_periods}")
        
        # Validate filter parameter combinations
        if time_period and date_range:
            logger.warning("Both time_period and date_range provided. time_period takes precedence.")
        
        # Use max_articles as the effective limit (max_articles takes precedence over top_k)
        effective_limit = max_articles
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Build filters dictionary with enhanced combination logic
            filters = {}
            
            if ministry_filter:
                # Validate ministry filter
                if isinstance(ministry_filter, str):
                    ministry_filter = [ministry_filter]
                
                if not isinstance(ministry_filter, list):
                    raise ValueError("ministry_filter must be a string or list of strings")
                
                # Filter out empty strings
                ministry_filter = [m for m in ministry_filter if m and isinstance(m, str)]
                
                if ministry_filter:
                    filters['ministry'] = ministry_filter
            
            # Handle date filtering with proper precedence
            date_filter_applied = False
            
            # Handle time_period filter (takes precedence over date_range if both provided)
            if time_period:
                time_start, time_end = self.calculate_time_period_range(time_period)
                filters['date_range'] = (time_start, time_end)
                date_filter_applied = True
                logger.info(f"Applied time period filter: {time_period} -> {time_start} to {time_end}")
            
            # Handle custom date_range filter (only if time_period not provided)
            elif date_range:
                # Validate date range
                if not isinstance(date_range, tuple) or len(date_range) != 2:
                    raise ValueError("date_range must be a tuple of (start_date, end_date)")
                
                start_date, end_date = date_range
                
                # Validate date format
                self._validate_iso_date(start_date)
                self._validate_iso_date(end_date)
                
                # Ensure start_date <= end_date
                if start_date > end_date:
                    raise ValueError("start_date must be before or equal to end_date")
                
                filters['date_range'] = (start_date, end_date)
                date_filter_applied = True
                logger.info(f"Applied custom date range filter: {start_date} to {end_date}")
            
            # Log filter combination status
            if not date_filter_applied:
                logger.info("No time filter specified - searching across all available articles")
            
            # Validate that we have meaningful filters or query
            if not filters and not query.strip():
                raise ValueError("Either filters or a non-empty query must be provided")
            
            # Perform similarity search with filters
            # Request more results than needed to account for threshold filtering
            search_k = min(effective_limit * 3, 100)  # Get extra results for filtering
            
            results = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                k=search_k,
                filters=filters if filters else None
            )
            
            # Apply relevance threshold filtering
            filtered_results = [
                result for result in results
                if result.score >= relevance_threshold
            ]
            
            # Sort by score (descending) - should already be sorted, but ensure it
            filtered_results.sort(key=lambda x: x.score, reverse=True)
            
            # Limit to effective_limit (max_articles)
            final_results = filtered_results[:effective_limit]
            
            logger.info(
                f"Search completed: query='{query[:50]}...', "
                f"filters={filters}, found {len(final_results)} results"
            )
            
            return final_results
            
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Search failed: {str(e)}") from e
    
    def parse_date_range(self, query: str) -> Optional[Tuple[str, str]]:
        """
        Parse date range from natural language query.
        
        Args:
            query: Natural language query that may contain date references
            
        Returns:
            Tuple of (start_date, end_date) in ISO format, or None if no date found
        """
        query_lower = query.lower()
        today = datetime.now().date()
        
        # Check for "recent" (last 30 days)
        if 'recent' in query_lower or 'recently' in query_lower:
            start_date = (today - timedelta(days=30)).isoformat()
            end_date = today.isoformat()
            return (start_date, end_date)
        
        # Check for "last week"
        if 'last week' in query_lower:
            start_date = (today - timedelta(days=7)).isoformat()
            end_date = today.isoformat()
            return (start_date, end_date)
        
        # Check for "last month"
        if 'last month' in query_lower:
            start_date = (today - timedelta(days=30)).isoformat()
            end_date = today.isoformat()
            return (start_date, end_date)
        
        # Check for "last year"
        if 'last year' in query_lower:
            start_date = (today - timedelta(days=365)).isoformat()
            end_date = today.isoformat()
            return (start_date, end_date)
        
        # Try to find ISO format dates (YYYY-MM-DD)
        iso_date_pattern = r'\b(\d{4}-\d{2}-\d{2})\b'
        matches = re.findall(iso_date_pattern, query)
        
        if len(matches) >= 2:
            # Found two dates, use as range
            dates = sorted(matches[:2])
            return (dates[0], dates[1])
        elif len(matches) == 1:
            # Found one date, use as both start and end
            return (matches[0], matches[0])
        
        return None
    
    def calculate_time_period_range(self, period: str) -> Tuple[str, str]:
        """
        Calculate date range for a given time period.
        
        Args:
            period: Time period string ("1_year", "6_months", "3_months")
            
        Returns:
            Tuple of (start_date, end_date) in ISO format
            
        Raises:
            ValueError: If period is not supported
        """
        today = datetime.now().date()
        
        if period == "1_year":
            start_date = (today - timedelta(days=365)).isoformat()
        elif period == "6_months":
            start_date = (today - timedelta(days=180)).isoformat()
        elif period == "3_months":
            start_date = (today - timedelta(days=90)).isoformat()
        else:
            raise ValueError(f"Unsupported time period: {period}")
        
        end_date = today.isoformat()
        return (start_date, end_date)
    
    def get_available_time_periods(self) -> List[str]:
        """
        Get list of available time period filters.
        
        Returns:
            List of supported time period strings
        """
        return ["1_year", "6_months", "3_months"]
    
    def _validate_iso_date(self, date_str: str) -> None:
        """
        Validate that a string is in ISO date format (YYYY-MM-DD).
        
        Args:
            date_str: Date string to validate
            
        Raises:
            ValueError: If date string is not in valid ISO format
        """
        try:
            datetime.fromisoformat(date_str)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid ISO date format: {date_str}") from e
