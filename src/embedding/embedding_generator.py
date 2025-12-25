"""
Embedding generation module for PIB RAG System.
Generates vector embeddings for text using sentence-transformers.
"""
from typing import List, Union, Optional, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import hashlib
import json
import logging
from pathlib import Path
from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION


class EmbeddingGenerator:
    """
    Generates vector embeddings for text using sentence-transformers.
    Uses all-MiniLM-L6-v2 model by default (384 dimensions).
    Supports local model caching to avoid re-downloading models.
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, cache_dir: str = "./model_cache"):
        """
        Initialize the embedding generator with specified model and cache directory.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            cache_dir: Directory to cache downloaded models
            
        Raises:
            RuntimeError: If model fails to load
            ValueError: If model produces embeddings with unexpected dimensions
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.expected_dimension = EMBEDDING_DIMENSION
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the model (from cache if available)
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the embedding model, checking cache first before downloading.
        Includes automatic re-download for corrupted models.
        
        Returns:
            True if loaded from cache, False if downloaded
            
        Raises:
            RuntimeError: If model fails to load after multiple attempts
        """
        # Find the actual model directory (sentence-transformers uses specific naming)
        model_base_path = self.cache_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
        metadata_path = model_base_path / "model_metadata.json"
        
        # Check if model is cached and valid
        if model_base_path.exists():
            try:
                # Verify model integrity if metadata exists
                if metadata_path.exists() and self.verify_model_integrity():
                    self.logger.info(f"Loading model {self.model_name} from cache")
                    # Load using the model name, sentence-transformers will find it in cache
                    self.model = SentenceTransformer(self.model_name, cache_folder=str(self.cache_dir))
                    self._validate_model_dimension()
                    self.logger.info(f"Successfully loaded cached model {self.model_name}")
                    return True
                else:
                    self.logger.warning(f"Cached model {self.model_name} failed integrity check or missing metadata")
                    self._handle_corrupted_model()
            except Exception as e:
                self.logger.warning(f"Failed to load cached model: {e}")
                self._handle_corrupted_model()
        
        # Download and cache the model
        return self._download_and_cache_model()
    
    def _handle_corrupted_model(self):
        """Handle corrupted model by cleaning up and preparing for re-download."""
        self.logger.info(f"Handling corrupted model {self.model_name}")
        self._cleanup_corrupted_model()
        self.logger.info(f"Cleaned up corrupted model, will re-download")
    
    def _download_and_cache_model(self) -> bool:
        """
        Download and cache the model with retry logic.
        
        Returns:
            False (indicating model was downloaded, not loaded from cache)
            
        Raises:
            RuntimeError: If download fails after retries
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.logger.info(f"Downloading model {self.model_name} (attempt {retry_count + 1}/{max_retries})")
                
                # Check available disk space before download
                self._check_disk_space()
                
                # Download model to cache
                self.model = SentenceTransformer(self.model_name, cache_folder=str(self.cache_dir))
                self._validate_model_dimension()
                self._save_model_metadata()
                
                self.logger.info(f"Successfully downloaded and cached model {self.model_name}")
                return False
                
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Download attempt {retry_count} failed: {e}")
                
                if retry_count < max_retries:
                    self.logger.info(f"Retrying download in 2 seconds...")
                    import time
                    time.sleep(2)
                    # Clean up any partial download
                    self._cleanup_corrupted_model()
                else:
                    raise RuntimeError(
                        f"Failed to download model '{self.model_name}' after {max_retries} attempts: {str(e)}"
                    ) from e
    
    def _check_disk_space(self):
        """Check if there's sufficient disk space for model download."""
        try:
            import shutil
            free_space = shutil.disk_usage(self.cache_dir).free
            # Require at least 1GB free space (models are typically 100-500MB)
            required_space = 1024 * 1024 * 1024  # 1GB in bytes
            
            if free_space < required_space:
                raise RuntimeError(
                    f"Insufficient disk space for model download. "
                    f"Available: {free_space / (1024**3):.2f}GB, Required: {required_space / (1024**3):.2f}GB"
                )
        except Exception as e:
            self.logger.warning(f"Could not check disk space: {e}")
    
    def _cleanup_corrupted_model(self):
        """Remove corrupted model files from cache with enhanced logging."""
        # Find the actual model directory
        model_base_path = self.cache_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
        if not model_base_path.exists():
            model_base_path = self.cache_dir / self.model_name
            
        if model_base_path.exists():
            import shutil
            try:
                self.logger.info(f"Removing corrupted model cache: {model_base_path}")
                shutil.rmtree(model_base_path)
                self.logger.info(f"Successfully cleaned up corrupted model cache")
            except Exception as e:
                self.logger.error(f"Failed to cleanup corrupted model: {e}")
                raise RuntimeError(f"Could not clean up corrupted model cache: {e}") from e
    
    def verify_model_integrity(self) -> bool:
        """
        Verify the integrity of a cached model using file hashes.
        
        Returns:
            True if model integrity is verified, False otherwise
        """
        # Find the actual model directory
        model_base_path = self.cache_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
        if not model_base_path.exists():
            model_base_path = self.cache_dir / self.model_name
            
        metadata_path = model_base_path / "model_metadata.json"
        
        if not metadata_path.exists():
            return False
        
        try:
            # Load stored metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            stored_hashes = metadata.get('file_hashes', {})
            
            # Verify each file's hash
            for file_path, stored_hash in stored_hashes.items():
                full_path = model_base_path / file_path
                if not full_path.exists():
                    self.logger.warning(f"Missing model file: {full_path}")
                    return False
                
                current_hash = self._calculate_file_hash(full_path)
                if current_hash != stored_hash:
                    self.logger.warning(f"Hash mismatch for {full_path}: expected {stored_hash}, got {current_hash}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying model integrity: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model and cache status.
        
        Returns:
            Dictionary containing model metadata and cache information
        """
        # Find the actual model directory
        model_base_path = self.cache_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
        if not model_base_path.exists():
            model_base_path = self.cache_dir / self.model_name
            
        metadata_path = model_base_path / "model_metadata.json"
        
        info = {
            'model_name': self.model_name,
            'cache_dir': str(self.cache_dir),
            'is_cached': model_base_path.exists(),
            'cache_path': str(model_base_path),
            'expected_dimension': self.expected_dimension
        }
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                info.update(metadata)
            except Exception as e:
                info['metadata_error'] = str(e)
        
        return info
    
    def _validate_model_dimension(self):
        """Validate that the model produces embeddings with expected dimensions."""
        test_embedding = self.model.encode("test", convert_to_numpy=True)
        actual_dimension = len(test_embedding)
        
        if actual_dimension != self.expected_dimension:
            raise ValueError(
                f"Model {self.model_name} produces embeddings of dimension {actual_dimension}, "
                f"but expected {self.expected_dimension}"
            )
    
    def _save_model_metadata(self):
        """Save metadata about the cached model including file hashes."""
        # Find the actual model directory (sentence-transformers creates nested structure)
        model_base_path = self.cache_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
        if not model_base_path.exists():
            # Fallback to direct model name path
            model_base_path = self.cache_dir / self.model_name
        
        if not model_base_path.exists():
            self.logger.warning(f"Could not find model directory for metadata saving")
            return
            
        metadata_path = model_base_path / "model_metadata.json"
        
        try:
            # Calculate hashes for all model files
            file_hashes = {}
            for file_path in model_base_path.rglob('*'):
                if file_path.is_file() and file_path.name != "model_metadata.json":
                    relative_path = file_path.relative_to(model_base_path)
                    file_hashes[str(relative_path)] = self._calculate_file_hash(file_path)
            
            metadata = {
                'model_name': self.model_name,
                'cache_timestamp': str(model_base_path.stat().st_mtime),
                'expected_dimension': self.expected_dimension,
                'file_hashes': file_hashes
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to save model metadata: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
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
