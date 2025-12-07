"""
Article chunking module for PIB RAG System.
Handles splitting long articles into semantically coherent chunks with overlap.
"""
import logging
from dataclasses import dataclass
from typing import List, Dict, Any
import re

from src.data_ingestion.article_loader import Article


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    Represents a chunk of an article with metadata.
    
    Attributes:
        chunk_id: Unique identifier for the chunk (format: article_id_index)
        article_id: ID of the parent article
        content: Text content of the chunk
        metadata: Dictionary containing parent article metadata and chunk info
    """
    chunk_id: str
    article_id: str
    content: str
    metadata: Dict[str, Any]


class ArticleChunker:
    """
    Chunks long articles into smaller pieces while preserving semantic coherence.
    Respects paragraph boundaries and maintains overlap between consecutive chunks.
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the ArticleChunker.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            overlap: Number of characters to overlap between consecutive chunks
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_article(self, article: Article) -> List[Chunk]:
        """
        Split an article into chunks while preserving paragraph boundaries.
        
        Args:
            article: Article object to chunk
            
        Returns:
            List of Chunk objects
        """
        # Get paragraphs from the article content
        paragraphs = self.preserve_paragraph_boundaries(article.content)
        
        # If content is short enough, return as single chunk
        if len(article.content) <= self.chunk_size:
            chunk = Chunk(
                chunk_id=f"{article.id}_0",
                article_id=article.id,
                content=article.content,
                metadata={
                    'date': article.date,
                    'ministry': article.ministry,
                    'title': article.title,
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            )
            return [chunk]
        
        # Build chunks from paragraphs
        chunks = []
        current_chunk_text = ""
        chunk_index = 0
        paragraph_index = 0
        
        while paragraph_index < len(paragraphs):
            paragraph = paragraphs[paragraph_index]
            
            # Calculate what the new chunk size would be if we add this paragraph
            separator_length = 2 if current_chunk_text else 0  # "\n\n" if not first paragraph
            new_chunk_length = len(current_chunk_text) + separator_length + len(paragraph)
            
            # If adding this paragraph would exceed chunk_size, start a new chunk
            if current_chunk_text and new_chunk_length > self.chunk_size:
                # Save current chunk
                chunks.append(current_chunk_text)
                
                # Start new chunk with overlap
                # Find overlap text from the end of current chunk
                overlap_text = self._get_overlap_text(current_chunk_text, paragraphs, paragraph_index)
                current_chunk_text = overlap_text
                chunk_index += 1
                
                # Recalculate for the new chunk with overlap
                separator_length = 2 if current_chunk_text else 0
                new_chunk_length = len(current_chunk_text) + separator_length + len(paragraph)
                
                # If even with overlap the paragraph is too large, we need to split it further
                # This handles the case where overlap + paragraph > 1.5 * chunk_size
                if new_chunk_length > self.chunk_size * 1.5:
                    # The paragraph itself is too large even with minimal overlap
                    # We need to reduce the overlap or split the paragraph
                    # For now, let's just use less overlap
                    max_allowed_overlap = int(self.chunk_size * 0.5) - separator_length
                    if len(overlap_text) > max_allowed_overlap:
                        overlap_text = overlap_text[-max_allowed_overlap:] if max_allowed_overlap > 0 else ""
                        current_chunk_text = overlap_text
            
            # Add paragraph to current chunk
            if current_chunk_text:
                current_chunk_text += "\n\n" + paragraph
            else:
                current_chunk_text = paragraph
            
            paragraph_index += 1
        
        # Add the last chunk if it has content
        if current_chunk_text:
            chunks.append(current_chunk_text)
        
        # Create Chunk objects
        total_chunks = len(chunks)
        chunk_objects = []
        
        for idx, chunk_text in enumerate(chunks):
            chunk = Chunk(
                chunk_id=f"{article.id}_{idx}",
                article_id=article.id,
                content=chunk_text,
                metadata={
                    'date': article.date,
                    'ministry': article.ministry,
                    'title': article.title,
                    'chunk_index': idx,
                    'total_chunks': total_chunks
                }
            )
            chunk_objects.append(chunk)
        
        logger.info(f"Chunked article {article.id} into {total_chunks} chunks")
        return chunk_objects
    
    def preserve_paragraph_boundaries(self, text: str) -> List[str]:
        """
        Split text into paragraphs, preserving paragraph structure.
        
        Args:
            text: Text to split into paragraphs
            
        Returns:
            List of paragraph strings
        """
        # Split on double newlines or more (paragraph boundaries)
        # This regex splits on 2 or more newlines, preserving paragraph structure
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filter out empty paragraphs and strip whitespace
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # If we have paragraphs that are too long, split them further
        result_paragraphs = []
        for para in paragraphs:
            if len(para) > self.chunk_size:
                # Split long paragraph into smaller pieces
                # Use chunk_size as the maximum size for each piece
                # Use a step size that accounts for overlap to prevent chunks from being too large
                # When we add overlap, the chunk size should not exceed chunk_size * 1.5
                # So each piece should be at most chunk_size
                for i in range(0, len(para), self.chunk_size):
                    piece = para[i:i + self.chunk_size]
                    result_paragraphs.append(piece)
            else:
                result_paragraphs.append(para)
        
        return result_paragraphs
    
    def _get_overlap_text(self, current_chunk: str, paragraphs: List[str], next_paragraph_idx: int) -> str:
        """
        Get overlap text from the end of the current chunk.
        
        Args:
            current_chunk: The current chunk text
            paragraphs: List of all paragraphs
            next_paragraph_idx: Index of the next paragraph to be added
            
        Returns:
            Text to use as overlap for the next chunk
        """
        # Get the last overlap characters from current chunk
        if len(current_chunk) <= self.overlap:
            return current_chunk
        
        # Calculate how much overlap we can actually use
        # We need to ensure that overlap + next content doesn't exceed 1.5x chunk_size
        # Since next content can be up to chunk_size, overlap should be at most 0.5x chunk_size
        max_safe_overlap = min(self.overlap, self.chunk_size // 2)
        
        # Try to find a paragraph boundary within the overlap region
        overlap_start = len(current_chunk) - max_safe_overlap
        overlap_text = current_chunk[overlap_start:]
        
        # Try to start at a paragraph boundary if possible
        # Look for double newline in the overlap region
        paragraph_boundary = overlap_text.find("\n\n")
        if paragraph_boundary != -1:
            # Start from the paragraph boundary
            overlap_text = overlap_text[paragraph_boundary + 2:]
        
        return overlap_text
