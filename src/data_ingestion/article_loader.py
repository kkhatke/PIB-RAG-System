"""
Article loading and validation module for PIB RAG System.
Handles JSON parsing, validation, deduplication, and content normalization.
"""
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Set
from pathlib import Path

from src.data_ingestion.content_normalizer import ContentNormalizer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Article:
    """
    Represents a PIB article with all required fields.
    
    Attributes:
        id: Unique article identifier
        date: Publication date in YYYY-MM-DD format
        ministry: Government ministry that published the article
        title: Article title
        content: Normalized article content
        original_content: Original article content before normalization
    """
    id: str
    date: str
    ministry: str
    title: str
    content: str
    original_content: str


class ArticleLoader:
    """
    Loads and validates PIB articles from JSON files.
    Handles parsing, validation, deduplication, and content normalization.
    """
    
    def __init__(self, normalizer: Optional[ContentNormalizer] = None):
        """
        Initialize the ArticleLoader.
        
        Args:
            normalizer: ContentNormalizer instance for content normalization.
                       If None, creates a new instance.
        """
        self.normalizer = normalizer or ContentNormalizer()
        self.seen_ids: Set[str] = set()
    
    def load_articles(self, filepath: str) -> List[Article]:
        """
        Load articles from a JSON file.
        
        Args:
            filepath: Path to the JSON file containing articles
            
        Returns:
            List of validated Article objects
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        try:
            file_path = Path(filepath)
            if not file_path.exists():
                logger.error(f"File not found: {filepath}")
                raise FileNotFoundError(f"File not found: {filepath}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.error(f"Invalid JSON format in {filepath}: expected a list")
                return []
            
            articles = []
            for idx, article_data in enumerate(data):
                if not isinstance(article_data, dict):
                    logger.warning(f"Skipping non-dict item at index {idx}")
                    continue
                
                if self.validate_article(article_data):
                    article = self._create_article(article_data)
                    if article:
                        articles.append(article)
                else:
                    logger.warning(f"Skipping invalid article at index {idx}: {article_data.get('id', 'unknown')}")
            
            logger.info(f"Loaded {len(articles)} valid articles from {filepath}")
            return articles
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading articles from {filepath}: {e}")
            raise
    
    def validate_article(self, article: dict) -> bool:
        """
        Validate that an article has all required fields.
        
        Args:
            article: Dictionary containing article data
            
        Returns:
            True if article is valid, False otherwise
        """
        required_fields = ['id', 'date', 'ministry', 'title', 'content']
        
        # Check all required fields exist
        for field in required_fields:
            if field not in article:
                logger.warning(f"Article missing required field: {field}")
                return False
        
        # Check that required fields are not empty
        for field in required_fields:
            value = article[field]
            if value is None or (isinstance(value, str) and not value.strip()):
                logger.warning(f"Article has empty {field}: {article.get('id', 'unknown')}")
                return False
        
        return True
    
    def deduplicate_articles(self, articles: List[Article]) -> List[Article]:
        """
        Remove duplicate articles based on article ID.
        
        Args:
            articles: List of articles to deduplicate
            
        Returns:
            List of unique articles (first occurrence kept)
        """
        unique_articles = []
        seen_ids = set()
        
        for article in articles:
            if article.id not in seen_ids:
                unique_articles.append(article)
                seen_ids.add(article.id)
            else:
                logger.info(f"Skipping duplicate article: {article.id}")
        
        logger.info(f"Deduplicated {len(articles)} articles to {len(unique_articles)} unique articles")
        return unique_articles
    
    def _create_article(self, article_data: dict) -> Optional[Article]:
        """
        Create an Article object from dictionary data.
        Normalizes content while preserving original.
        
        Args:
            article_data: Dictionary containing article fields
            
        Returns:
            Article object or None if creation fails
        """
        try:
            article_id = str(article_data['id'])
            
            # Check for duplicates
            if article_id in self.seen_ids:
                logger.info(f"Skipping duplicate article during loading: {article_id}")
                return None
            
            original_content = article_data['content']
            normalized_content = self.normalizer.normalize_content(original_content)
            
            article = Article(
                id=article_id,
                date=str(article_data['date']),
                ministry=str(article_data['ministry']),
                title=str(article_data['title']),
                content=normalized_content,
                original_content=original_content
            )
            
            self.seen_ids.add(article_id)
            return article
            
        except Exception as e:
            logger.error(f"Error creating article: {e}")
            return None
