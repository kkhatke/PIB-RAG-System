"""
Embedding generation module for PIB RAG System.
Generates vector embeddings for text using sentence-transformers.
"""
from typing import List, Union
from sentence_transformers import SentenceTransformer
import numpy as np
from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION


class EmbeddingGenerator:
    """
    Generates vector embeddings for text using sentence-transformers.
    Uses all-MiniLM-L6-v2 model by default (384 dimensions).
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedding generator with specified model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            
        Raises:
            RuntimeError: If model fails to load
            ValueError: If model produces embeddings with unexpected dimensions
        """
        self.model_name = model_name
        self.expected_dimension = EMBEDDING_DIMENSION
        
        try:
            self.model = SentenceTransformer(model_name)
            
            # Validate embedding dimension
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            actual_dimension = len(test_embedding)
            
            if actual_dimension != self.expected_dimension:
                raise ValueError(
                    f"Model {model_name} produces embeddings of dimension {actual_dimension}, "
                    f"but expected {self.expected_dimension}"
                )
                
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}': {str(e)}"
            ) from e
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a vector embedding for a single text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            ValueError: If text is empty or None
            RuntimeError: If embedding generation fails
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Validate dimension
            if len(embedding) != self.expected_dimension:
                raise RuntimeError(
                    f"Generated embedding has dimension {len(embedding)}, "
                    f"expected {self.expected_dimension}"
                )
            
            # Convert to list of floats
            return embedding.tolist()
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate embedding for text: {str(e)}"
            ) from e
    
    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch (more efficient).
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors, one per input text
            
        Raises:
            ValueError: If texts is empty or contains invalid entries
            RuntimeError: If batch embedding generation fails
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Texts must be a non-empty list")
        
        if not all(isinstance(t, str) and t for t in texts):
            raise ValueError("All texts must be non-empty strings")
        
        try:
            # Generate embeddings in batch
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            # Validate dimensions
            for i, embedding in enumerate(embeddings):
                if len(embedding) != self.expected_dimension:
                    raise RuntimeError(
                        f"Embedding {i} has dimension {len(embedding)}, "
                        f"expected {self.expected_dimension}"
                    )
            
            # Convert to list of lists
            return [emb.tolist() for emb in embeddings]
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate batch embeddings: {str(e)}"
            ) from e
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this generator.
        
        Returns:
            Integer dimension of embedding vectors
        """
        return self.expected_dimension
