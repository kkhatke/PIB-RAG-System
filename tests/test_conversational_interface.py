"""
Property-based tests for ConversationalInterface module.
Tests conversation history maintenance and truncation behavior.
"""
import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.interface.conversational_interface import (
    ConversationalInterface,
    Message
)
from src.query_engine.query_engine import QueryEngine
from src.response_generation.response_generator import (
    ResponseGenerator,
    Response,
    Citation
)
from src.vector_store.vector_store import SearchResult
from src.data_ingestion.article_chunker import Chunk


# Test data strategies
@st.composite
def message_content_strategy(draw):
    """Generate random message content."""
    # Generate text that won't be all whitespace
    text = draw(st.text(min_size=1, max_size=200, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'),
        min_codepoint=ord('A')
    )))
    # Ensure it's not empty after stripping
    if not text.strip():
        text = "test"
    return text


@st.composite
def message_sequence_strategy(draw):
    """Generate a sequence of user messages."""
    num_messages = draw(st.integers(min_value=1, max_value=15))
    messages = [draw(message_content_strategy()) for _ in range(num_messages)]
    return messages


# Property 17: Conversation history maintenance
# Feature: pib-rag-system, Property 17: Conversation history maintenance
@settings(max_examples=100, deadline=None)
@given(messages=message_sequence_strategy())
def test_conversation_history_maintenance(messages):
    """
    Property 17: Conversation history maintenance
    For any sequence of messages sent in a conversation, the conversation history 
    should contain all messages in the order they were sent.
    
    Validates: Requirements 6.1
    """
    # Create mock dependencies
    mock_query_engine = Mock(spec=QueryEngine)
    mock_response_generator = Mock(spec=ResponseGenerator)
    
    # Create a simple chunk for mock search results
    mock_chunk = Chunk(
        chunk_id="test_chunk_1",
        article_id="test_article_1",
        content="Test content",
        metadata={
            'date': '2025-01-01',
            'ministry': 'Test Ministry',
            'title': 'Test Title',
            'chunk_index': 0
        }
    )
    
    # Mock search results
    mock_search_result = SearchResult(chunk=mock_chunk, score=0.9)
    mock_query_engine.search.return_value = [mock_search_result]
    
    # Mock response
    mock_citation = Citation(
        article_id="test_article_1",
        date="2025-01-01",
        ministry="Test Ministry",
        title="Test Title",
        relevance_score=0.9
    )
    mock_response = Response(
        answer="Test answer",
        citations=[mock_citation]
    )
    mock_response_generator.generate_response.return_value = mock_response
    
    # Create conversational interface with high max_history_length
    # to ensure no truncation during this test
    interface = ConversationalInterface(
        query_engine=mock_query_engine,
        response_generator=mock_response_generator,
        max_history_length=100
    )
    
    # Process all messages
    for message in messages:
        interface.process_message(message)
    
    # Get conversation history
    history = interface.get_conversation_history()
    
    # Property: History contains all messages (user + assistant pairs)
    expected_length = len(messages) * 2  # Each user message gets an assistant response
    assert len(history) == expected_length, \
        f"Expected {expected_length} messages in history, got {len(history)}"
    
    # Property: Messages are in chronological order
    for i in range(len(history) - 1):
        current_time = datetime.fromisoformat(history[i].timestamp)
        next_time = datetime.fromisoformat(history[i + 1].timestamp)
        assert current_time <= next_time, \
            f"Messages not in chronological order at index {i}"
    
    # Property: User and assistant messages alternate
    for i in range(0, len(history), 2):
        assert history[i].role == "user", \
            f"Expected user message at index {i}, got {history[i].role}"
        if i + 1 < len(history):
            assert history[i + 1].role == "assistant", \
                f"Expected assistant message at index {i+1}, got {history[i+1].role}"
    
    # Property: User messages match the input messages
    user_messages = [msg for msg in history if msg.role == "user"]
    assert len(user_messages) == len(messages), \
        f"Expected {len(messages)} user messages, got {len(user_messages)}"
    
    for i, (user_msg, original_msg) in enumerate(zip(user_messages, messages)):
        assert user_msg.content == original_msg, \
            f"User message {i} content doesn't match: expected '{original_msg}', got '{user_msg.content}'"
    
    # Property: All messages have timestamps
    for msg in history:
        assert msg.timestamp, "Message missing timestamp"
        # Validate timestamp format
        try:
            datetime.fromisoformat(msg.timestamp)
        except ValueError:
            pytest.fail(f"Invalid timestamp format: {msg.timestamp}")


# Property 18: History truncation behavior
# Feature: pib-rag-system, Property 18: History truncation behavior
@settings(max_examples=100, deadline=None)
@given(
    messages=st.lists(
        message_content_strategy(),
        min_size=5,
        max_size=30
    ),
    max_history=st.integers(min_value=2, max_value=10)
)
def test_history_truncation_behavior(messages, max_history):
    """
    Property 18: History truncation behavior
    For any conversation where history exceeds the maximum threshold, the total 
    number of messages should be reduced while preserving recent messages.
    
    Validates: Requirements 6.5
    """
    # Ensure we have enough messages to trigger truncation
    assume(len(messages) * 2 > max_history)  # *2 because each user message gets a response
    
    # Create mock dependencies
    mock_query_engine = Mock(spec=QueryEngine)
    mock_response_generator = Mock(spec=ResponseGenerator)
    
    # Create a simple chunk for mock search results
    mock_chunk = Chunk(
        chunk_id="test_chunk_1",
        article_id="test_article_1",
        content="Test content",
        metadata={
            'date': '2025-01-01',
            'ministry': 'Test Ministry',
            'title': 'Test Title',
            'chunk_index': 0
        }
    )
    
    # Mock search results
    mock_search_result = SearchResult(chunk=mock_chunk, score=0.9)
    mock_query_engine.search.return_value = [mock_search_result]
    
    # Mock response
    mock_citation = Citation(
        article_id="test_article_1",
        date="2025-01-01",
        ministry="Test Ministry",
        title="Test Title",
        relevance_score=0.9
    )
    mock_response = Response(
        answer="Test answer",
        citations=[mock_citation]
    )
    mock_response_generator.generate_response.return_value = mock_response
    
    # Create conversational interface with specified max_history_length
    interface = ConversationalInterface(
        query_engine=mock_query_engine,
        response_generator=mock_response_generator,
        max_history_length=max_history
    )
    
    # Process all messages
    for message in messages:
        interface.process_message(message)
    
    # Get conversation history
    history = interface.get_conversation_history()
    
    # Property: History length does not exceed maximum
    assert len(history) <= max_history, \
        f"History length {len(history)} exceeds maximum {max_history}"
    
    # Property: If truncation occurred, most recent messages are preserved
    if len(messages) * 2 > max_history:
        # Truncation should have occurred
        # The last message should be the most recent assistant response
        assert history[-1].role == "assistant", \
            "Last message should be assistant response"
        
        # The second-to-last message should be the most recent user message
        if len(history) >= 2:
            assert history[-2].role == "user", \
                "Second-to-last message should be user message"
        
        # The most recent user message should match the last input message
        user_messages_in_history = [msg for msg in history if msg.role == "user"]
        if user_messages_in_history:
            last_user_msg = user_messages_in_history[-1]
            assert last_user_msg.content == messages[-1], \
                "Most recent user message not preserved after truncation"
    
    # Property: Messages remain in chronological order after truncation
    for i in range(len(history) - 1):
        current_time = datetime.fromisoformat(history[i].timestamp)
        next_time = datetime.fromisoformat(history[i + 1].timestamp)
        assert current_time <= next_time, \
            f"Messages not in chronological order after truncation at index {i}"
    
    # Property: User and assistant messages still alternate after truncation
    # Note: After truncation, we might not start with "user" if we removed an odd number
    # But the alternating pattern should still hold
    if len(history) >= 2:
        for i in range(len(history) - 1):
            current_role = history[i].role
            next_role = history[i + 1].role
            # Roles should alternate
            assert current_role != next_role, \
                f"Roles should alternate, but found {current_role} followed by {next_role} at index {i}"


# Additional unit tests for edge cases
def test_clear_context():
    """Test that clear_context properly resets all state."""
    mock_query_engine = Mock(spec=QueryEngine)
    mock_response_generator = Mock(spec=ResponseGenerator)
    
    interface = ConversationalInterface(
        query_engine=mock_query_engine,
        response_generator=mock_response_generator
    )
    
    # Set some state
    interface.ministry_filter = ["Ministry of Health"]
    interface.date_filter = ("2025-01-01", "2025-12-31")
    interface.conversation_history.append(
        Message(role="user", content="test", timestamp=datetime.now().isoformat())
    )
    
    # Clear context
    interface.clear_context()
    
    # Verify all state is cleared
    assert len(interface.conversation_history) == 0
    assert interface.ministry_filter is None
    assert interface.date_filter is None


def test_empty_message_raises_error():
    """Test that empty messages raise ValueError."""
    mock_query_engine = Mock(spec=QueryEngine)
    mock_response_generator = Mock(spec=ResponseGenerator)
    
    interface = ConversationalInterface(
        query_engine=mock_query_engine,
        response_generator=mock_response_generator
    )
    
    with pytest.raises(ValueError, match="non-empty string"):
        interface.process_message("")
    
    with pytest.raises(ValueError, match="non-empty string"):
        interface.process_message("   ")


def test_ministry_filter_handling():
    """Test ministry filter setting and clearing."""
    mock_query_engine = Mock(spec=QueryEngine)
    mock_response_generator = Mock(spec=ResponseGenerator)
    
    interface = ConversationalInterface(
        query_engine=mock_query_engine,
        response_generator=mock_response_generator
    )
    
    # Set filter with list
    interface.handle_ministry_filter(["Ministry A", "Ministry B"])
    assert interface.ministry_filter == ["Ministry A", "Ministry B"]
    
    # Set filter with single string
    interface.handle_ministry_filter("Ministry C")
    assert interface.ministry_filter == ["Ministry C"]
    
    # Clear filter
    interface.handle_ministry_filter(None)
    assert interface.ministry_filter is None


def test_date_filter_handling():
    """Test date filter setting and clearing."""
    mock_query_engine = Mock(spec=QueryEngine)
    mock_response_generator = Mock(spec=ResponseGenerator)
    
    interface = ConversationalInterface(
        query_engine=mock_query_engine,
        response_generator=mock_response_generator
    )
    
    # Set valid date range
    interface.handle_date_filter("2025-01-01", "2025-12-31")
    assert interface.date_filter == ("2025-01-01", "2025-12-31")
    
    # Clear filter
    interface.handle_date_filter(None, None)
    assert interface.date_filter is None
    
    # Test invalid date range (start > end)
    with pytest.raises(ValueError, match="start_date must be before"):
        interface.handle_date_filter("2025-12-31", "2025-01-01")
    
    # Test invalid date format
    with pytest.raises(ValueError, match="Invalid ISO date format"):
        interface.handle_date_filter("2025-13-01", "2025-12-31")


def test_display_response():
    """Test response formatting for display."""
    mock_query_engine = Mock(spec=QueryEngine)
    mock_response_generator = Mock(spec=ResponseGenerator)
    
    interface = ConversationalInterface(
        query_engine=mock_query_engine,
        response_generator=mock_response_generator
    )
    
    # Create test response
    citations = [
        Citation(
            article_id="123",
            date="2025-01-01",
            ministry="Test Ministry",
            title="Test Article",
            relevance_score=0.95
        )
    ]
    response = Response(
        answer="This is a test answer.",
        citations=citations
    )
    
    # Format response
    formatted = interface.display_response(response)
    
    # Verify formatting
    assert "ANSWER:" in formatted
    assert "This is a test answer." in formatted
    assert "SOURCES:" in formatted
    assert "Test Article" in formatted
    assert "Test Ministry" in formatted
    assert "123" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
