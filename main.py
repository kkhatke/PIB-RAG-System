#!/usr/bin/env python3
"""
Main application entry point for PIB RAG System.
Provides command-line interface for querying government policy information.
"""
import sys
import logging
from typing import Optional
from pathlib import Path

from src.data_ingestion.article_loader import ArticleLoader
from src.data_ingestion.article_chunker import ArticleChunker
from src.data_ingestion.content_normalizer import ContentNormalizer
from src.embedding.embedding_generator import EmbeddingGenerator
from src.vector_store.vector_store import VectorStore
from src.query_engine.query_engine import QueryEngine
from src.response_generation.response_generator import ResponseGenerator
from src.interface.conversational_interface import ConversationalInterface
import config


# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PIBRAGSystem:
    """
    Main PIB RAG System application.
    Manages initialization and lifecycle of all components.
    """
    
    def __init__(self):
        """Initialize the PIB RAG System."""
        self.embedding_generator: Optional[EmbeddingGenerator] = None
        self.vector_store: Optional[VectorStore] = None
        self.query_engine: Optional[QueryEngine] = None
        self.response_generator: Optional[ResponseGenerator] = None
        self.interface: Optional[ConversationalInterface] = None
        self.initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            print("Initializing PIB RAG System...")
            print("=" * 80)
            
            # Initialize embedding generator
            print("Loading embedding model...")
            self.embedding_generator = EmbeddingGenerator()
            print(f"✓ Embedding model loaded: {config.EMBEDDING_MODEL}")
            
            # Initialize vector store
            print("Connecting to vector store...")
            self.vector_store = VectorStore()
            chunk_count = self.vector_store.count()
            print(f"✓ Vector store connected: {chunk_count} chunks available")
            
            # Initialize query engine
            print("Initializing query engine...")
            self.query_engine = QueryEngine(
                vector_store=self.vector_store,
                embedding_generator=self.embedding_generator
            )
            print("✓ Query engine initialized")
            
            # Initialize response generator (with Ollama connection check)
            print("Connecting to Ollama LLM...")
            print(f"  URL: {config.OLLAMA_BASE_URL}")
            print(f"  Model: {config.OLLAMA_MODEL}")
            
            self.response_generator = ResponseGenerator()
            print("✓ Ollama LLM connected")
            
            # Initialize conversational interface
            print("Initializing conversational interface...")
            self.interface = ConversationalInterface(
                query_engine=self.query_engine,
                response_generator=self.response_generator
            )
            print("✓ Conversational interface ready")
            
            print("=" * 80)
            print("✓ System initialization complete!")
            print()
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"\n✗ Initialization failed: {str(e)}")
            print()
            self._print_troubleshooting_help(e)
            return False
    
    def run(self):
        """Run the interactive command-line interface."""
        if not self.initialized:
            print("Error: System not initialized. Please run initialize() first.")
            return
        
        self._print_welcome()
        self._print_help()
        
        while True:
            try:
                # Get user input
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\nGoodbye!")
                    break
                
                elif user_input.lower() in ['help', 'h', '?']:
                    self._print_help()
                    continue
                
                elif user_input.lower() in ['clear', 'reset']:
                    self.interface.clear_context()
                    print("✓ Conversation context cleared")
                    continue
                
                elif user_input.lower().startswith('filter ministry'):
                    self._handle_ministry_filter_command(user_input)
                    continue
                
                elif user_input.lower().startswith('filter date'):
                    self._handle_date_filter_command(user_input)
                    continue
                
                elif user_input.lower() == 'filter clear':
                    self.interface.handle_ministry_filter(None)
                    self.interface.handle_date_filter(None, None)
                    print("✓ All filters cleared")
                    continue
                
                elif user_input.lower() == 'ministries':
                    self._list_ministries()
                    continue
                
                elif user_input.lower() == 'status':
                    self._print_status()
                    continue
                
                # Process as query
                self._process_query(user_input)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit or continue querying.")
                continue
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\n✗ Error: {str(e)}")
                logger.exception("Error processing user input")
    
    def _process_query(self, query: str):
        """
        Process a user query and display the response.
        
        Args:
            query: User's natural language query
        """
        try:
            print("\nSearching...")
            
            # Process the message
            response = self.interface.process_message(query)
            
            # Display the response
            print("\n" + "=" * 80)
            formatted_response = self.interface.display_response(response)
            print(formatted_response)
            print("=" * 80)
            
        except Exception as e:
            print(f"\n✗ Query processing failed: {str(e)}")
            logger.exception("Error processing query")
    
    def _handle_ministry_filter_command(self, command: str):
        """
        Handle ministry filter command.
        
        Args:
            command: Command string starting with 'filter ministry'
        """
        try:
            # Parse ministry names from command
            # Format: "filter ministry <ministry1>, <ministry2>, ..."
            parts = command.split('filter ministry', 1)
            if len(parts) < 2 or not parts[1].strip():
                print("Usage: filter ministry <ministry_name> [, <ministry_name2>, ...]")
                print("Example: filter ministry Ministry of Health and Family Welfare")
                return
            
            ministry_str = parts[1].strip()
            ministries = [m.strip() for m in ministry_str.split(',')]
            
            self.interface.handle_ministry_filter(ministries)
            print(f"✓ Ministry filter set: {', '.join(ministries)}")
            
        except Exception as e:
            print(f"✗ Failed to set ministry filter: {str(e)}")
    
    def _handle_date_filter_command(self, command: str):
        """
        Handle date filter command.
        
        Args:
            command: Command string starting with 'filter date'
        """
        try:
            # Parse dates from command
            # Format: "filter date YYYY-MM-DD to YYYY-MM-DD"
            parts = command.split('filter date', 1)
            if len(parts) < 2 or not parts[1].strip():
                print("Usage: filter date YYYY-MM-DD to YYYY-MM-DD")
                print("Example: filter date 2024-01-01 to 2024-12-31")
                return
            
            date_str = parts[1].strip()
            
            if ' to ' not in date_str.lower():
                print("Invalid format. Use: filter date YYYY-MM-DD to YYYY-MM-DD")
                return
            
            date_parts = date_str.lower().split(' to ')
            if len(date_parts) != 2:
                print("Invalid format. Use: filter date YYYY-MM-DD to YYYY-MM-DD")
                return
            
            start_date = date_parts[0].strip()
            end_date = date_parts[1].strip()
            
            self.interface.handle_date_filter(start_date, end_date)
            print(f"✓ Date filter set: {start_date} to {end_date}")
            
        except Exception as e:
            print(f"✗ Failed to set date filter: {str(e)}")
    
    def _list_ministries(self):
        """List all available ministries in the vector store."""
        try:
            ministries = self.vector_store.get_unique_ministries()
            
            if not ministries:
                print("No ministries found in the database.")
                return
            
            print(f"\nAvailable Ministries ({len(ministries)}):")
            print("-" * 80)
            for idx, ministry in enumerate(ministries, 1):
                print(f"{idx}. {ministry}")
            print("-" * 80)
            
        except Exception as e:
            print(f"✗ Failed to list ministries: {str(e)}")
    
    def _print_status(self):
        """Print current system status and active filters."""
        print("\nSystem Status:")
        print("-" * 80)
        print(f"Vector Store: {self.vector_store.count()} chunks")
        print(f"Embedding Model: {config.EMBEDDING_MODEL}")
        print(f"LLM Model: {config.OLLAMA_MODEL}")
        print(f"Conversation History: {len(self.interface.conversation_history)} messages")
        
        # Print active filters
        if self.interface.ministry_filter:
            print(f"Ministry Filter: {', '.join(self.interface.ministry_filter)}")
        else:
            print("Ministry Filter: None")
        
        if self.interface.date_filter:
            start, end = self.interface.date_filter
            print(f"Date Filter: {start} to {end}")
        else:
            print("Date Filter: None")
        
        print("-" * 80)
    
    def _print_welcome(self):
        """Print welcome message."""
        print("\n" + "=" * 80)
        print("PIB RAG System - Government Policy Information Assistant")
        print("=" * 80)
        print("\nAsk questions about Indian government policies and announcements.")
        print("All answers are based on Press Information Bureau (PIB) articles.")
        print()
    
    def _print_help(self):
        """Print help information."""
        print("\nAvailable Commands:")
        print("-" * 80)
        print("  <your question>          - Ask a question about government policies")
        print("  help, h, ?               - Show this help message")
        print("  ministries               - List all available ministries")
        print("  status                   - Show system status and active filters")
        print()
        print("  filter ministry <name>   - Filter results by ministry")
        print("                             Example: filter ministry Ministry of Health")
        print("  filter date <start> to <end>  - Filter by date range (YYYY-MM-DD)")
        print("                             Example: filter date 2024-01-01 to 2024-12-31")
        print("  filter clear             - Clear all filters")
        print()
        print("  clear, reset             - Clear conversation history")
        print("  exit, quit, q            - Exit the application")
        print("-" * 80)
    
    def _print_troubleshooting_help(self, error: Exception):
        """
        Print troubleshooting help based on the error.
        
        Args:
            error: The exception that occurred
        """
        error_str = str(error).lower()
        
        print("Troubleshooting:")
        print("-" * 80)
        
        if "ollama" in error_str or "connection" in error_str:
            print("Ollama Connection Issue:")
            print("  1. Ensure Ollama is installed:")
            print("     Visit: https://ollama.ai/download")
            print()
            print("  2. Start Ollama server:")
            print("     $ ollama serve")
            print()
            print("  3. Pull the required model:")
            print(f"     $ ollama pull {config.OLLAMA_MODEL}")
            print()
            print("  4. Verify Ollama is running:")
            print("     $ ollama list")
            print()
        
        elif "model" in error_str and "not found" in error_str:
            print("Model Not Found:")
            print(f"  Pull the required model:")
            print(f"     $ ollama pull {config.OLLAMA_MODEL}")
            print()
            print("  Available models: llama3.2, mistral, llama2")
            print()
        
        elif "embedding" in error_str:
            print("Embedding Model Issue:")
            print("  The sentence-transformers model may need to be downloaded.")
            print("  This should happen automatically on first run.")
            print("  Ensure you have internet connection.")
            print()
        
        elif "vector store" in error_str or "chroma" in error_str:
            print("Vector Store Issue:")
            print("  The vector store may be empty or corrupted.")
            print("  Run the data ingestion script to populate it:")
            print("     $ python ingest_articles.py")
            print()
        
        print("-" * 80)


def main():
    """Main entry point for the application."""
    # Create and initialize system
    system = PIBRAGSystem()
    
    if not system.initialize():
        print("\nFailed to initialize system. Please fix the issues above and try again.")
        sys.exit(1)
    
    # Run interactive interface
    try:
        system.run()
    except Exception as e:
        logger.exception("Fatal error in main loop")
        print(f"\n✗ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
