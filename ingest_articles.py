"""
Data ingestion script for PIB RAG System.
Loads PIB articles from JSON, processes them, generates embeddings, and stores in vector database.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List
import time

from src.data_ingestion.article_loader import ArticleLoader, Article
from src.data_ingestion.content_normalizer import ContentNormalizer
from src.data_ingestion.article_chunker import ArticleChunker, Chunk
from src.embedding.embedding_generator import EmbeddingGenerator
from src.vector_store.vector_store import VectorStore
from config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ingestion.log')
    ]
)
logger = logging.getLogger(__name__)


class ArticleIngestionPipeline:
    """
    Pipeline for ingesting PIB articles into the RAG system.
    Handles loading, normalization, chunking, embedding, and storage.
    """
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        batch_size: int = 32
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            chunk_size: Maximum size for article chunks
            chunk_overlap: Overlap between consecutive chunks
            batch_size: Number of chunks to process in each batch
        """
        self.batch_size = batch_size
        
        logger.info("Initializing ingestion pipeline components...")
        
        # Initialize components
        self.normalizer = ContentNormalizer()
        self.loader = ArticleLoader(normalizer=self.normalizer)
        self.chunker = ArticleChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        
        logger.info("Pipeline initialization complete")
    
    def ingest_articles(self, filepath: str, reset_store: bool = False) -> dict:
        """
        Ingest articles from a JSON file into the vector store.
        
        Args:
            filepath: Path to JSON file containing articles
            reset_store: If True, clear existing data before ingestion
            
        Returns:
            Dictionary with ingestion statistics
        """
        start_time = time.time()
        stats = {
            'articles_loaded': 0,
            'articles_processed': 0,
            'chunks_created': 0,
            'chunks_stored': 0,
            'errors': 0
        }
        
        try:
            # Reset vector store if requested
            if reset_store:
                logger.info("Resetting vector store...")
                self.vector_store.reset()
            
            # Load articles
            logger.info(f"Loading articles from {filepath}...")
            articles = self.loader.load_articles(filepath)
            stats['articles_loaded'] = len(articles)
            
            if not articles:
                logger.warning("No articles loaded from file")
                return stats
            
            # Deduplicate articles
            logger.info("Deduplicating articles...")
            articles = self.loader.deduplicate_articles(articles)
            
            # Process articles in batches
            logger.info(f"Processing {len(articles)} articles...")
            all_chunks = []
            
            for i, article in enumerate(articles, 1):
                try:
                    # Chunk the article
                    chunks = self.chunker.chunk_article(article)
                    all_chunks.extend(chunks)
                    stats['articles_processed'] += 1
                    stats['chunks_created'] += len(chunks)
                    
                    # Log progress
                    if i % 100 == 0:
                        logger.info(f"Processed {i}/{len(articles)} articles, created {len(all_chunks)} chunks")
                    
                except Exception as e:
                    logger.error(f"Error processing article {article.id}: {e}")
                    stats['errors'] += 1
                    continue
            
            logger.info(f"Created {len(all_chunks)} chunks from {stats['articles_processed']} articles")
            
            # Generate embeddings and store in batches
            logger.info("Generating embeddings and storing in vector database...")
            self._process_chunks_in_batches(all_chunks, stats)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            logger.info(f"Ingestion complete in {elapsed_time:.2f} seconds")
            
            # Log final statistics
            self._log_statistics(stats, elapsed_time)
            
            return stats
            
        except Exception as e:
            logger.error(f"Fatal error during ingestion: {e}")
            stats['errors'] += 1
            raise
    
    def _process_chunks_in_batches(self, chunks: List[Chunk], stats: dict) -> None:
        """
        Process chunks in batches to generate embeddings and store them.
        
        Args:
            chunks: List of all chunks to process
            stats: Statistics dictionary to update
        """
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[batch_idx:batch_idx + self.batch_size]
            batch_num = (batch_idx // self.batch_size) + 1
            
            try:
                # Extract chunk contents for embedding
                chunk_texts = [chunk.content for chunk in batch_chunks]
                
                # Generate embeddings in batch
                logger.info(f"Generating embeddings for batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...")
                embeddings = self.embedding_generator.batch_generate_embeddings(chunk_texts)
                
                # Store in vector database
                logger.info(f"Storing batch {batch_num}/{total_batches} in vector database...")
                self.vector_store.add_chunks(batch_chunks, embeddings)
                
                stats['chunks_stored'] += len(batch_chunks)
                
                logger.info(f"Batch {batch_num}/{total_batches} complete. Total stored: {stats['chunks_stored']}/{len(chunks)}")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                stats['errors'] += 1
                # Continue with next batch instead of failing completely
                continue
    
    def _log_statistics(self, stats: dict, elapsed_time: float) -> None:
        """
        Log final ingestion statistics.
        
        Args:
            stats: Statistics dictionary
            elapsed_time: Total elapsed time in seconds
        """
        logger.info("=" * 60)
        logger.info("INGESTION STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Articles loaded:     {stats['articles_loaded']}")
        logger.info(f"Articles processed:  {stats['articles_processed']}")
        logger.info(f"Chunks created:      {stats['chunks_created']}")
        logger.info(f"Chunks stored:       {stats['chunks_stored']}")
        logger.info(f"Errors encountered:  {stats['errors']}")
        logger.info(f"Total time:          {elapsed_time:.2f} seconds")
        
        if stats['chunks_stored'] > 0:
            avg_time_per_chunk = elapsed_time / stats['chunks_stored']
            logger.info(f"Avg time per chunk:  {avg_time_per_chunk:.3f} seconds")
        
        logger.info("=" * 60)
        
        # Verify vector store count
        stored_count = self.vector_store.count()
        logger.info(f"Vector store contains {stored_count} chunks")
        
        if stored_count != stats['chunks_stored']:
            logger.warning(
                f"Mismatch: Expected {stats['chunks_stored']} chunks, "
                f"but vector store contains {stored_count}"
            )


def main():
    """Main entry point for the ingestion script."""
    parser = argparse.ArgumentParser(
        description='Ingest PIB articles into the RAG system vector database'
    )
    parser.add_argument(
        'filepath',
        type=str,
        nargs='?',
        default=str(DATA_DIR / 'all_articles.json'),
        help='Path to JSON file containing articles (default: data/all_articles.json)'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset vector store before ingestion (clears existing data)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=CHUNK_SIZE,
        help=f'Maximum chunk size in characters (default: {CHUNK_SIZE})'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=CHUNK_OVERLAP,
        help=f'Overlap between chunks in characters (default: {CHUNK_OVERLAP})'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Number of chunks to process in each batch (default: 32)'
    )
    
    args = parser.parse_args()
    
    # Validate file exists
    filepath = Path(args.filepath)
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        sys.exit(1)
    
    # Validate chunk parameters
    if args.chunk_size <= 0:
        logger.error("Chunk size must be positive")
        sys.exit(1)
    
    if args.chunk_overlap < 0:
        logger.error("Chunk overlap cannot be negative")
        sys.exit(1)
    
    if args.chunk_overlap >= args.chunk_size:
        logger.error("Chunk overlap must be less than chunk size")
        sys.exit(1)
    
    if args.batch_size <= 0:
        logger.error("Batch size must be positive")
        sys.exit(1)
    
    # Display configuration
    logger.info("=" * 60)
    logger.info("PIB RAG SYSTEM - ARTICLE INGESTION")
    logger.info("=" * 60)
    logger.info(f"Input file:      {filepath}")
    logger.info(f"Reset store:     {args.reset}")
    logger.info(f"Chunk size:      {args.chunk_size}")
    logger.info(f"Chunk overlap:   {args.chunk_overlap}")
    logger.info(f"Batch size:      {args.batch_size}")
    logger.info("=" * 60)
    
    # Create pipeline and run ingestion
    try:
        pipeline = ArticleIngestionPipeline(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size
        )
        
        stats = pipeline.ingest_articles(
            filepath=str(filepath),
            reset_store=args.reset
        )
        
        # Exit with appropriate code
        if stats['errors'] > 0:
            logger.warning(f"Ingestion completed with {stats['errors']} errors")
            sys.exit(1)
        else:
            logger.info("Ingestion completed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Ingestion interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
