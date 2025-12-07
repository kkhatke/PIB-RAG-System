"""
Response generation module for PIB RAG System.
Generates natural language responses using Ollama LLM with retrieved context.
"""
import logging
import time
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

from src.vector_store.vector_store import SearchResult
import config


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """
    Represents a source citation for a response.
    
    Attributes:
        article_id: Unique identifier of the source article
        date: Publication date of the article
        ministry: Government ministry that published the article
        title: Title of the article
        relevance_score: Relevance score from similarity search
    """
    article_id: str
    date: str
    ministry: str
    title: str
    relevance_score: float


@dataclass
class Response:
    """
    Represents a generated response with citations.
    
    Attributes:
        answer: Generated natural language answer
        citations: List of source citations used in the response
    """
    answer: str
    citations: List[Citation]


class ResponseGenerator:
    """
    Generates natural language responses using Ollama LLM.
    Integrates with LangChain for prompt engineering and response generation.
    """
    
    def __init__(
        self,
        ollama_base_url: str = config.OLLAMA_BASE_URL,
        model: str = config.OLLAMA_MODEL,
        timeout: int = config.OLLAMA_TIMEOUT
    ):
        """
        Initialize the ResponseGenerator with Ollama connection.
        
        Args:
            ollama_base_url: Base URL for Ollama API (default: http://localhost:11434)
            model: Model name to use (default: llama3.2)
            timeout: Request timeout in seconds
            
        Raises:
            RuntimeError: If Ollama connection fails or model is not available
        """
        self.ollama_base_url = ollama_base_url
        self.model = model
        self.timeout = timeout
        
        try:
            # Initialize Ollama LLM
            self.llm = Ollama(
                base_url=ollama_base_url,
                model=model,
                timeout=timeout
            )
            
            # Test connection
            self._test_connection()
            
            logger.info(f"ResponseGenerator initialized with model: {model}")
            
        except Exception as e:
            error_msg = self._format_connection_error(e)
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def generate_response(
        self,
        query: str,
        search_results: List[SearchResult],
        conversation_history: Optional[List[dict]] = None
    ) -> Response:
        """
        Generate a natural language response based on query and retrieved context.
        
        Args:
            query: User's natural language query
            search_results: List of SearchResult objects from vector search
            conversation_history: Optional list of previous messages
            
        Returns:
            Response object containing answer and citations
            
        Raises:
            ValueError: If query is empty or search_results is invalid
            RuntimeError: If response generation fails
        """
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        
        if not isinstance(search_results, list):
            raise ValueError("search_results must be a list")
        
        try:
            # Extract citations from search results
            citations = self.extract_citations(search_results)
            
            # Format context from search results
            context = self.format_context(search_results)
            
            # Build conversation context if available
            conversation_context = ""
            if conversation_history:
                conversation_context = self._format_conversation_history(conversation_history)
            
            # Generate response with retry logic
            answer = self._generate_with_retry(query, context, conversation_context)
            
            # Create and return Response object
            response = Response(
                answer=answer,
                citations=citations
            )
            
            logger.info(f"Generated response with {len(citations)} citations")
            return response
            
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Response generation failed: {str(e)}") from e
    
    def format_context(self, search_results: List[SearchResult]) -> str:
        """
        Format retrieved chunks into context string for LLM.
        
        Args:
            search_results: List of SearchResult objects
            
        Returns:
            Formatted context string
        """
        if not search_results:
            return "No relevant information found."
        
        context_parts = []
        
        for idx, result in enumerate(search_results, 1):
            chunk = result.chunk
            metadata = chunk.metadata
            
            # Format each chunk with metadata
            chunk_text = f"""
Source {idx}:
Ministry: {metadata.get('ministry', 'Unknown')}
Date: {metadata.get('date', 'Unknown')}
Title: {metadata.get('title', 'Unknown')}
Article ID: {chunk.article_id}
Relevance Score: {result.score:.3f}

Content:
{chunk.content}
"""
            context_parts.append(chunk_text.strip())
        
        # Join all chunks with separators
        formatted_context = "\n\n" + "="*80 + "\n\n".join(context_parts)
        
        # Truncate if too long
        max_length = config.MAX_CONTEXT_LENGTH
        if len(formatted_context) > max_length:
            formatted_context = formatted_context[:max_length] + "\n\n[Context truncated due to length...]"
            logger.warning(f"Context truncated to {max_length} characters")
        
        return formatted_context
    
    def extract_citations(self, search_results: List[SearchResult]) -> List[Citation]:
        """
        Extract citations from search results.
        
        Args:
            search_results: List of SearchResult objects
            
        Returns:
            List of Citation objects
        """
        citations = []
        seen_article_ids = set()
        
        for result in search_results:
            chunk = result.chunk
            article_id = chunk.article_id
            
            # Only include each article once
            if article_id not in seen_article_ids:
                metadata = chunk.metadata
                
                citation = Citation(
                    article_id=article_id,
                    date=metadata.get('date', 'Unknown'),
                    ministry=metadata.get('ministry', 'Unknown'),
                    title=metadata.get('title', 'Unknown'),
                    relevance_score=result.score
                )
                
                citations.append(citation)
                seen_article_ids.add(article_id)
        
        return citations
    
    def _generate_with_retry(
        self,
        query: str,
        context: str,
        conversation_context: str = ""
    ) -> str:
        """
        Generate response with exponential backoff retry logic.
        
        Args:
            query: User query
            context: Formatted context from search results
            conversation_context: Formatted conversation history
            
        Returns:
            Generated answer string
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        max_retries = config.MAX_RETRIES
        backoff_factor = config.RETRY_BACKOFF_FACTOR
        
        for attempt in range(max_retries):
            try:
                # Build prompt
                prompt = self._build_prompt(query, context, conversation_context)
                
                # Generate response
                answer = self.llm.invoke(prompt)
                
                # Validate response
                if not answer or not isinstance(answer, str) or not answer.strip():
                    raise ValueError("LLM returned empty response")
                
                return answer.strip()
                
            except Exception as e:
                wait_time = backoff_factor ** attempt
                
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Response generation attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    error_msg = f"Response generation failed after {max_retries} attempts: {str(e)}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
        
        raise RuntimeError("Response generation failed: max retries exceeded")
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        conversation_context: str = ""
    ) -> str:
        """
        Build prompt for LLM with proper instructions.
        
        Args:
            query: User query
            context: Formatted context
            conversation_context: Formatted conversation history
            
        Returns:
            Complete prompt string
        """
        prompt_template = """You are a helpful assistant that answers questions about Indian government policies and announcements based on Press Information Bureau (PIB) articles.

IMPORTANT INSTRUCTIONS:
1. Answer the question ONLY based on the provided context below
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Do NOT make up or hallucinate information
4. Be concise and factual
5. Reference specific sources when making claims
6. If multiple sources support your answer, mention them

{conversation_context}

CONTEXT FROM PIB ARTICLES:
{context}

USER QUESTION:
{query}

ANSWER:"""
        
        # Add conversation context if available
        conv_section = ""
        if conversation_context:
            conv_section = f"\nPREVIOUS CONVERSATION:\n{conversation_context}\n"
        
        prompt = prompt_template.format(
            conversation_context=conv_section,
            context=context,
            query=query
        )
        
        return prompt
    
    def _format_conversation_history(self, conversation_history: List[dict]) -> str:
        """
        Format conversation history for inclusion in prompt.
        
        Args:
            conversation_history: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted conversation string
        """
        if not conversation_history:
            return ""
        
        formatted_messages = []
        for msg in conversation_history[-5:]:  # Only include last 5 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            formatted_messages.append(f"{role.upper()}: {content}")
        
        return "\n".join(formatted_messages)
    
    def _test_connection(self) -> None:
        """
        Test connection to Ollama server.
        
        Raises:
            RuntimeError: If connection test fails
        """
        try:
            # Try a simple test query
            test_response = self.llm.invoke("Hello")
            if not test_response:
                raise RuntimeError("Ollama returned empty response")
            logger.info("Ollama connection test successful")
        except Exception as e:
            raise RuntimeError(f"Ollama connection test failed: {str(e)}") from e
    
    def _format_connection_error(self, error: Exception) -> str:
        """
        Format connection error with helpful troubleshooting information.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Formatted error message with troubleshooting steps
        """
        error_str = str(error).lower()
        
        if "connection" in error_str or "refused" in error_str:
            return (
                f"Failed to connect to Ollama at {self.ollama_base_url}. "
                "Please ensure Ollama is running. "
                "Start Ollama with: 'ollama serve' or check if it's running as a service."
            )
        elif "model" in error_str or "not found" in error_str:
            return (
                f"Model '{self.model}' not found. "
                f"Please pull the model first: 'ollama pull {self.model}'"
            )
        elif "timeout" in error_str:
            return (
                f"Connection to Ollama timed out after {self.timeout} seconds. "
                "The server may be overloaded or not responding."
            )
        else:
            return f"Ollama initialization failed: {str(error)}"
