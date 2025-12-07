"""
Conversational interface module for PIB RAG System.
Manages user interactions and conversation state.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime

from src.query_engine.query_engine import QueryEngine
from src.response_generation.response_generator import ResponseGenerator, Response


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Message:
    """
    Represents a message in the conversation history.
    
    Attributes:
        role: Role of the message sender ("user" or "assistant")
        content: Content of the message
        timestamp: ISO format timestamp of when the message was created
    """
    role: str
    content: str
    timestamp: str


class ConversationalInterface:
    """
    Manages conversational interactions with the RAG system.
    Maintains conversation history and handles user queries with context.
    """
    
    def __init__(
        self,
        query_engine: QueryEngine,
        response_generator: ResponseGenerator,
        max_history_length: int = 20
    ):
        """
        Initialize the conversational interface.
        
        Args:
            query_engine: QueryEngine instance for retrieving relevant articles
            response_generator: ResponseGenerator for generating responses
            max_history_length: Maximum number of messages to keep in history
        """
        self.query_engine = query_engine
        self.response_generator = response_generator
        self.max_history_length = max_history_length
        
        # Conversation state
        self.conversation_history: List[Message] = []
        self.ministry_filter: Optional[List[str]] = None
        self.date_filter: Optional[Tuple[str, str]] = None
        
        logger.info("ConversationalInterface initialized")
    
    def process_message(
        self,
        user_message: str,
        top_k: int = 5,
        relevance_threshold: float = 0.5
    ) -> Response:
        """
        Process a user message and generate a response.
        
        Args:
            user_message: User's natural language query
            top_k: Maximum number of results to retrieve
            relevance_threshold: Minimum relevance score for results
            
        Returns:
            Response object containing answer and citations
            
        Raises:
            ValueError: If user_message is empty or invalid
            RuntimeError: If processing fails
        """
        if not user_message or not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("User message must be a non-empty string")
        
        try:
            # Add user message to history
            user_msg = Message(
                role="user",
                content=user_message.strip(),
                timestamp=datetime.now().isoformat()
            )
            self.conversation_history.append(user_msg)
            
            # Perform search with current filters
            search_results = self.query_engine.search(
                query=user_message,
                ministry_filter=self.ministry_filter,
                date_range=self.date_filter,
                top_k=top_k,
                relevance_threshold=relevance_threshold
            )
            
            # Generate response with conversation history
            conversation_context = self._format_history_for_llm()
            
            response = self.response_generator.generate_response(
                query=user_message,
                search_results=search_results,
                conversation_history=conversation_context
            )
            
            # Add assistant response to history
            assistant_msg = Message(
                role="assistant",
                content=response.answer,
                timestamp=datetime.now().isoformat()
            )
            self.conversation_history.append(assistant_msg)
            
            # Truncate history if needed
            self._truncate_history()
            
            logger.info(f"Processed message, history length: {len(self.conversation_history)}")
            
            return response
            
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Message processing failed: {str(e)}") from e
    
    def handle_ministry_filter(self, ministries: Optional[List[str]]) -> None:
        """
        Set or clear ministry filter for searches.
        
        Args:
            ministries: List of ministry names to filter by, or None to clear filter
        """
        if ministries is None:
            self.ministry_filter = None
            logger.info("Ministry filter cleared")
        else:
            if isinstance(ministries, str):
                ministries = [ministries]
            
            if not isinstance(ministries, list):
                raise ValueError("ministries must be a string, list of strings, or None")
            
            # Filter out empty strings
            valid_ministries = [m for m in ministries if m and isinstance(m, str)]
            
            if valid_ministries:
                self.ministry_filter = valid_ministries
                logger.info(f"Ministry filter set: {valid_ministries}")
            else:
                self.ministry_filter = None
                logger.info("Ministry filter cleared (no valid ministries)")
    
    def handle_date_filter(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> None:
        """
        Set or clear date range filter for searches.
        
        Args:
            start_date: Start date in ISO format (YYYY-MM-DD), or None to clear
            end_date: End date in ISO format (YYYY-MM-DD), or None to clear
            
        Raises:
            ValueError: If date format is invalid
        """
        if start_date is None and end_date is None:
            self.date_filter = None
            logger.info("Date filter cleared")
        elif start_date and end_date:
            # Validate dates
            self._validate_iso_date(start_date)
            self._validate_iso_date(end_date)
            
            if start_date > end_date:
                raise ValueError("start_date must be before or equal to end_date")
            
            self.date_filter = (start_date, end_date)
            logger.info(f"Date filter set: {start_date} to {end_date}")
        else:
            raise ValueError("Both start_date and end_date must be provided or both None")
    
    def clear_context(self) -> None:
        """
        Clear conversation history and reset filters.
        """
        self.conversation_history.clear()
        self.ministry_filter = None
        self.date_filter = None
        logger.info("Conversation context cleared")
    
    def display_response(self, response: Response) -> str:
        """
        Format response for display to user.
        
        Args:
            response: Response object to format
            
        Returns:
            Formatted string for display
        """
        output_parts = []
        
        # Add answer
        output_parts.append("ANSWER:")
        output_parts.append(response.answer)
        output_parts.append("")
        
        # Add citations if available
        if response.citations:
            output_parts.append("SOURCES:")
            for idx, citation in enumerate(response.citations, 1):
                citation_text = (
                    f"{idx}. {citation.title}\n"
                    f"   Ministry: {citation.ministry}\n"
                    f"   Date: {citation.date}\n"
                    f"   Article ID: {citation.article_id}\n"
                    f"   Relevance: {citation.relevance_score:.3f}"
                )
                output_parts.append(citation_text)
        else:
            output_parts.append("SOURCES: No sources found")
        
        return "\n".join(output_parts)
    
    def get_conversation_history(self) -> List[Message]:
        """
        Get the current conversation history.
        
        Returns:
            List of Message objects in chronological order
        """
        return self.conversation_history.copy()
    
    def _truncate_history(self) -> None:
        """
        Truncate conversation history if it exceeds maximum length.
        Keeps the most recent messages.
        """
        if len(self.conversation_history) > self.max_history_length:
            # Keep only the most recent messages
            num_to_remove = len(self.conversation_history) - self.max_history_length
            self.conversation_history = self.conversation_history[num_to_remove:]
            logger.info(f"Truncated conversation history, removed {num_to_remove} messages")
    
    def _format_history_for_llm(self) -> List[dict]:
        """
        Format conversation history for LLM context.
        
        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        # Exclude the most recent user message (it's already in the query)
        history_to_format = self.conversation_history[:-1] if self.conversation_history else []
        
        formatted_history = []
        for msg in history_to_format:
            formatted_history.append({
                'role': msg.role,
                'content': msg.content
            })
        
        return formatted_history
    
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
