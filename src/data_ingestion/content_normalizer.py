"""
Content normalization module for PIB RAG System.
Handles whitespace, HTML entity, and Unicode normalization while preserving paragraph structure.
"""
import html
import re
from typing import List


class ContentNormalizer:
    """Normalizes article content for consistent processing and embedding generation."""
    
    def normalize_content(self, text: str) -> str:
        """
        Apply all normalization steps to the content.
        
        Args:
            text: Raw article content
            
        Returns:
            Normalized content with consistent formatting
        """
        if not text:
            return text
            
        # Apply normalization steps in order
        text = self.normalize_unicode(text)
        text = self.decode_html_entities(text)
        text = self.remove_excessive_whitespace(text)
        
        return text
    
    def remove_excessive_whitespace(self, text: str) -> str:
        """
        Remove excessive whitespace while preserving paragraph structure.
        - Multiple spaces/tabs become single space
        - Multiple newlines become double newline (paragraph break)
        
        Args:
            text: Text with potentially excessive whitespace
            
        Returns:
            Text with normalized whitespace
        """
        if not text:
            return text
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Normalize line breaks: 3+ newlines become 2 (paragraph break)
        text = re.sub(r'\n\n+', '\n\n', text)
        
        # Remove spaces at the beginning and end of lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        # Final trim
        text = text.strip()
        
        return text
    
    def decode_html_entities(self, text: str) -> str:
        """
        Decode HTML entities to standard characters.
        Examples: &amp; -> &, &quot; -> ", &#39; -> '
        
        Args:
            text: Text containing HTML entities
            
        Returns:
            Text with decoded entities
        """
        if not text:
            return text
            
        return html.unescape(text)
    
    def normalize_unicode(self, text: str) -> str:
        """
        Ensure text is valid UTF-8 encoded Unicode.
        
        Args:
            text: Text to normalize
            
        Returns:
            UTF-8 normalized text
        """
        if not text:
            return text
        
        # Ensure the text is properly encoded as UTF-8
        # If it's already a string, encode and decode to ensure consistency
        if isinstance(text, str):
            # Normalize Unicode to NFC form (canonical composition)
            import unicodedata
            text = unicodedata.normalize('NFC', text)
        
        return text
    
    def preserve_paragraph_structure(self, text: str) -> str:
        """
        Ensure paragraph structure is preserved during normalization.
        This is primarily handled by remove_excessive_whitespace.
        
        Args:
            text: Text to check
            
        Returns:
            Text with preserved paragraph structure
        """
        # This method is mainly for validation/verification
        # The actual preservation happens in remove_excessive_whitespace
        return text
    
    def count_paragraphs(self, text: str) -> int:
        """
        Count the number of paragraphs in text.
        Paragraphs are separated by blank lines (double newlines).
        
        Args:
            text: Text to analyze
            
        Returns:
            Number of paragraphs
        """
        if not text or not text.strip():
            return 0
        
        # Split by double newlines and count non-empty segments
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return len(paragraphs)
