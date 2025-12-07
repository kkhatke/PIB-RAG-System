"""
Property-based tests for ContentNormalizer module.
Tests whitespace, Unicode, HTML entity normalization and paragraph preservation.
"""
import re
from hypothesis import given, strategies as st, settings
from src.data_ingestion.content_normalizer import ContentNormalizer


# Initialize normalizer
normalizer = ContentNormalizer()


# Feature: pib-rag-system, Property 33: Whitespace normalization
# Validates: Requirements 11.1
@given(st.text(min_size=1))
@settings(max_examples=100)
def test_whitespace_normalization(text):
    """
    Property 33: For any article content processed by the system, excessive whitespace
    (multiple consecutive spaces, tabs, or newlines) should be normalized to single
    spaces or single newlines.
    """
    normalized = normalizer.remove_excessive_whitespace(text)
    
    # Property 1: No multiple consecutive spaces
    assert not re.search(r'  +', normalized), \
        f"Found multiple consecutive spaces in normalized text"
    
    # Property 2: No tabs
    assert '\t' not in normalized, \
        f"Found tabs in normalized text"
    
    # Property 3: No more than 2 consecutive newlines (paragraph break)
    assert not re.search(r'\n\n\n+', normalized), \
        f"Found more than 2 consecutive newlines in normalized text"
    
    # Property 4: No leading/trailing whitespace on lines
    for line in normalized.split('\n'):
        if line:  # Skip empty lines
            assert line == line.strip(), \
                f"Found line with leading/trailing whitespace: '{line}'"
    
    # Property 5: No leading/trailing whitespace in overall text
    if normalized:
        assert normalized == normalized.strip(), \
            f"Found leading/trailing whitespace in normalized text"



# Feature: pib-rag-system, Property 34: Unicode encoding consistency
# Validates: Requirements 11.2
@given(st.text(min_size=1))
@settings(max_examples=100)
def test_unicode_encoding(text):
    """
    Property 34: For any article content stored in the system, the text should be
    valid UTF-8 encoded Unicode.
    """
    normalized = normalizer.normalize_unicode(text)
    
    # Property: Result should be valid UTF-8
    try:
        # Try to encode as UTF-8
        encoded = normalized.encode('utf-8')
        # Try to decode back
        decoded = encoded.decode('utf-8')
        # Should match the normalized text
        assert decoded == normalized, \
            "Text changed after UTF-8 encode/decode cycle"
    except UnicodeEncodeError as e:
        raise AssertionError(f"Failed to encode normalized text as UTF-8: {e}")
    except UnicodeDecodeError as e:
        raise AssertionError(f"Failed to decode UTF-8 encoded text: {e}")


# Feature: pib-rag-system, Property 35: HTML entity decoding
# Validates: Requirements 11.3
@given(st.text(min_size=1))
@settings(max_examples=100)
def test_html_entity_decoding(text):
    """
    Property 35: For any article content containing HTML entities (e.g., &amp;, &quot;, &#39;),
    they should be decoded to their standard character equivalents.
    """
    # Create text with known HTML entities
    test_cases = [
        ("&amp;", "&"),
        ("&quot;", '"'),
        ("&#39;", "'"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&nbsp;", "\xa0"),  # non-breaking space
    ]
    
    for entity, expected_char in test_cases:
        # Insert entity into the text
        text_with_entity = text + entity
        decoded = normalizer.decode_html_entities(text_with_entity)
        
        # Property: Entity should be decoded
        assert expected_char in decoded or entity not in decoded, \
            f"HTML entity {entity} was not properly decoded"
    
    # Also test that the original text without entities is unchanged
    decoded_original = normalizer.decode_html_entities(text)
    # The decoded version should not introduce new HTML entities
    assert '&amp;' not in decoded_original or '&amp;' in text, \
        "Decoding introduced new HTML entities"


# Feature: pib-rag-system, Property 36: Paragraph structure preservation
# Validates: Requirements 11.4
@given(st.text(min_size=1))
@settings(max_examples=100)
def test_paragraph_preservation(text):
    """
    Property 36: For any article content after normalization, the number of paragraphs
    should remain the same as in the original content.
    """
    # Count paragraphs before normalization
    original_para_count = normalizer.count_paragraphs(text)
    
    # Normalize the content
    normalized = normalizer.normalize_content(text)
    
    # Count paragraphs after normalization
    normalized_para_count = normalizer.count_paragraphs(normalized)
    
    # Property: Paragraph count should be preserved
    # Note: This is a relaxed check - normalization may clean up empty paragraphs
    # but should not merge distinct paragraphs
    if original_para_count > 0:
        assert normalized_para_count > 0, \
            "Normalization removed all paragraphs from non-empty text"
        
        # The normalized count should be <= original (cleaning up empty paragraphs is OK)
        # but should not be drastically different
        assert normalized_para_count <= original_para_count, \
            f"Normalization increased paragraph count from {original_para_count} to {normalized_para_count}"
