"""
Buffer Module Test Suite (OLD)
======================

This module contains comprehensive unit tests for the Buffer implementation,
which is a critical component of the event handling system. The tests verify
both the Stack and Buffer classes' functionality, ensuring reliable event
management.

Test Categories:
--------------
1. Stack Tests:
   - Initialization
   - Push operations
   - Size limits
   - FIFO behavior

2. Buffer Tests:
   - Priority handling
   - Event ordering
   - Error conditions
   - Buffer operations

Test Coverage:
------------
- Basic functionality
- Edge cases
- Error handling
- Performance constraints
"""

# Standard library imports
import pytest
import sys
import os

# Add parent directory to Python path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.buffer import Stack, Buffer
from config import BUFFER_CONFIG


# --- Test Fixtures ---
@pytest.fixture
def empty_stack():
    """
    Provides a fresh Stack instance for each test.
    
    Returns:
        Stack: A new, empty Stack instance
    """
    return Stack()

@pytest.fixture
def empty_buffer():
    """
    Provides a fresh Buffer instance for each test.
    
    Returns:
        Buffer: A new, empty Buffer instance with initialized priority buckets
    """
    return Buffer()


# --- Stack Tests ---
def test_stack_initialization(empty_stack):
    """
    Verify Stack initialization.
    
    Tests:
        - Empty stack creation
        - Correct bucket size configuration
    """
    assert empty_stack.stack == []
    assert empty_stack.bucket_size == BUFFER_CONFIG["BUCKET_SIZE"]

def test_stack_push(empty_stack):
    """
    Verify Stack push operations.
    
    Tests:
        - Single item addition
        - Item integrity after push
        - Stack size after push
    """
    test_event = {"priority": 1, "event": "test"}
    empty_stack.push(test_event)
    assert len(empty_stack.stack) == 1
    assert empty_stack.stack[0] == test_event

def test_stack_size_limit(empty_stack):
    """
    Verify Stack size constraints.
    
    Tests:
        - Maximum size enforcement
        - FIFO behavior when full
        - Oldest item removal
    """
    bucket_size = BUFFER_CONFIG["BUCKET_SIZE"]
    # Fill stack beyond its capacity
    for i in range(bucket_size + 2):
        empty_stack.push({"priority": 1, "event": f"test_{i}"})
    
    assert len(empty_stack.stack) == bucket_size
    # Verify FIFO behavior - first items should be removed
    assert empty_stack.stack[0]["event"] == f"test_{2}"

def test_stack_repr(empty_stack):
    """
    Verify Stack string representation.
    
    Tests:
        - String conversion format
        - Representation accuracy
    """
    test_event = {"priority": 1, "event": "test"}
    empty_stack.push(test_event)
    assert str(empty_stack) == str([test_event])


# --- Buffer Tests ---
def test_buffer_initialization(empty_buffer):
    """
    Verify Buffer initialization.
    
    Tests:
        - Priority bucket creation
        - Bucket type verification
        - Buffer size configuration
    """
    assert len(empty_buffer.buffer) == BUFFER_CONFIG["BUFFER_SIZE"]
    assert all(isinstance(bucket, Stack) for bucket in empty_buffer.buffer)

def test_buffer_add_priority(empty_buffer):
    """
    Verify priority-based event addition.
    
    Tests:
        - High priority event handling
        - Medium priority event handling
        - Priority bucket assignment
    """
    test_event = {"priority": 1, "event": "high_priority"}
    empty_buffer.add(test_event)
    assert len(empty_buffer.buffer[0].stack) == 1
    
    test_event2 = {"priority": 2, "event": "medium_priority"}
    empty_buffer.add(test_event2)
    assert len(empty_buffer.buffer[1].stack) == 1

def test_buffer_invalid_priority(empty_buffer):
    """
    Verify invalid priority handling.
    
    Tests:
        - Below minimum priority
        - Above maximum priority
        - Error message accuracy
    """
    with pytest.raises(Exception) as exc_info:
        empty_buffer.add({"priority": 0, "event": "invalid"})
    assert "interupt" in str(exc_info.value).lower()
    
    with pytest.raises(Exception) as exc_info:
        empty_buffer.add({"priority": BUFFER_CONFIG["BUFFER_SIZE"] + 1, "event": "invalid"})
    assert "out of scope" in str(exc_info.value).lower()

def test_buffer_next_empty(empty_buffer):
    """
    Verify empty buffer behavior.
    
    Tests:
        - Empty buffer next() returns None
        - No exceptions on empty buffer
    """
    assert empty_buffer.next() is None

def test_buffer_next_priority_order(empty_buffer):
    """
    Verify priority-based event retrieval.
    
    Tests:
        - High priority first
        - Medium priority second
        - Low priority last
        - Correct event ordering
    """
    # Add events with different priorities
    events = [
        {"priority": 2, "event": "medium"},
        {"priority": 1, "event": "high"},
        {"priority": 3, "event": "low"}
    ]
    for event in events:
        empty_buffer.add(event)
    
    # Should get high priority first
    next_event = empty_buffer.next()
    assert next_event["event"] == "high"
    
    # Then medium priority
    next_event = empty_buffer.next()
    assert next_event["event"] == "medium"
    
    # Finally low priority
    next_event = empty_buffer.next()
    assert next_event["event"] == "low"

def test_buffer_clear(empty_buffer):
    """
    Verify buffer clearing functionality.
    
    Tests:
        - Complete buffer clearing
        - Empty bucket verification
        - Post-clear operations
    """
    # Add some events
    events = [
        {"priority": 1, "event": "test1"},
        {"priority": 2, "event": "test2"}
    ]
    for event in events:
        empty_buffer.add(event)
        
    # Clear the buffer
    empty_buffer.clear()
    
    # Verify all buckets are empty
    assert all(len(bucket.stack) == 0 for bucket in empty_buffer.buffer)
    assert empty_buffer.next() is None