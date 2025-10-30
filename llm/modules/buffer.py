"""
Event Buffer Module
==================

This module implements a priority-based event buffer for the table football commentary system.
It manages game events using a dynamic scoring system that considers multiple factors to
determine which events should be commented on and in what order.

Key Features:
-----------
1. Dynamic Event Scoring:
   - Priority-based ranking
   - Time decay for event relevance
   - Confidence and likeliness weighting
   - Automatic buffer size management

2. Event Management:
   - FIFO queue with priority override
   - Automatic cleanup of stale events
   - Interest score recalculation
   - Buffer overflow protection

Configuration:
------------
Buffer size and timing parameters are controlled via BUFFER_CONFIG in the main config file.
"""

# Standard library imports for system operations and timing
import sys
import os
import time

# Add parent directory to Python path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BUFFER_CONFIG

# Buffer configuration constants
BUFFER_SIZE = BUFFER_CONFIG["BUFFER_SIZE"]    # Maximum number of events in buffer
TIME_OFFSET = BUFFER_CONFIG["TIME_OFFSET"]    # Time window (ms) for event relevance


class Buffer:
    """
    Priority-based event buffer with dynamic interest scoring.

    The buffer maintains events based on their calculated interest score, which
    considers multiple factors including priority, likeliness, confidence, and
    time decay. Events are automatically removed when they become stale or when
    the buffer reaches capacity.

    Attributes:
    ----------
    buffer : list
        List of events with their associated metadata and scores
    size : int
        Maximum buffer capacity from BUFFER_CONFIG
    start_time : float
        Timestamp of the first event, used for relative timing
    """

    def __init__(self):
        """
        Initializes an empty event buffer with configured capacity.
        Sets up internal storage and timing tracking.
        """
        self.buffer = []              # Event storage list
        self.size = BUFFER_SIZE       # Maximum capacity
        self.start_time = 0           # Track first event time


    def __repr__(self):
        """
        Creates a human-readable representation of the buffer contents.
        
        Returns:
        -------
        str
            Formatted string showing events by priority level
        """
        buffer_repr = []

        for i in range(self.size):
            bucket = self.buffer[i]
            if bucket.stack:
                bucket_content = ', '.join(map(str, bucket.stack))
            else: 
                bucket_content = "Empty"
            buffer_repr.append(f"Priority {i + 1}: {bucket_content}")

        return "\n".join(buffer_repr)


    def clear(self):
        """
        Resets the buffer to empty state.
        Useful when handling interrupts or system resets.
        """
        self.buffer = []


    def add(self, event):
        """
        Adds a new event to the buffer with interest score calculation.
        
        The method:
        1. Validates required event fields
        2. Calculates initial interest score
        3. Manages buffer overflow by removing least interesting event
        4. Adds new event to appropriate position

        Args:
        ----
        event : dict
            Event dictionary containing required fields:
            - priority: Importance level (higher = less important)
            - likeliness: How common the event is (0.0-1.0)
            - confidence: Detection confidence (0.0-1.0)
            - time: Event timestamp

        Raises:
        ------
        Exception
            If event is missing required fields or has invalid priority
        """
        # Validate required event fields
        if not all(k in event for k in ["priority", "likeliness", "confidence", "time"]):
            raise Exception("Event missing required fields: priority, likeliness, confidence, or time.", event)

        # Initialize start time for first event
        if self.start_time is None and event.get("time") is not None:
            self.start_time = event['time'] * 1000

        # Validate priority (0 is reserved for interrupts)
        if event["priority"] == 0:
            raise Exception(f"Bucket ID is 0! You should throw an interupt here rather then buffering the event.")

        # Calculate initial interest score
        event["interest_score"] = self.calculate_interest_score(
            event["priority"], event["likeliness"], event["confidence"], event["time"]
        )

        # Manage buffer overflow
        if len(self.buffer) >= BUFFER_SIZE:
            least_interesting_event = min(self.buffer, key=lambda e: e["interest_score"])
            self.buffer.remove(least_interesting_event)

        self.buffer.append(event)
    

    def next(self):
        """
        Retrieves the most interesting event from the buffer.
        
        The method:
        1. Removes outdated events based on TIME_OFFSET
        2. Recalculates interest scores for remaining events
        3. Returns and removes the highest-scoring event

        Returns:
        -------
        dict
            The event with the highest current interest score

        Raises:
        ------
        Exception
            If buffer is empty or error occurs during processing
        """
        current_time = time.time() * 1000

        # Validate buffer state
        if not self.buffer:
            raise Exception("Buffer is empty.")

        try:
            # Remove events older than TIME_OFFSET
            self.buffer = [event for event in self.buffer if (event["time"] * 1000 + current_time) >= (self.start_time + current_time - TIME_OFFSET)]
        except Exception as e:
            raise Exception(f"Error removing outdated events: {e}")

        # Update interest scores based on current time
        for event in self.buffer:
            event["interest_score"] = self.calculate_interest_score(
                event["priority"], event["likeliness"], event["confidence"], event["time"]
            )

        # Extract and return highest-scoring event
        best_event = max(self.buffer, key=lambda e: e["interest_score"])
        self.buffer.remove(best_event)
        return best_event

    @staticmethod
    def calculate_interest_score(priority, likeliness, confidence, event_time):
        """
        Calculates a dynamic interest score for event prioritization.
        
        The score considers:
        1. Priority level (inverse relationship)
        2. Event likeliness (rarer = more interesting)
        3. Detection confidence
        4. Time decay based on event age

        Args:
        ----
        priority : int
            Event priority level (lower = more important)
        likeliness : float
            How common the event is (0.0-1.0)
        confidence : float
            Detection confidence (0.0-1.0)
        event_time : float
            Event timestamp

        Returns:
        -------
        float
            Calculated interest score (0.0-100.0)
        """
        # Calculate time-based decay factor
        current_time = time.time()
        time_decay = max(0, 1 - (current_time - event_time) / TIME_OFFSET)

        # Convert priority to weight (inverse relationship)
        priority_weight = 1 / (priority + 1)

        # Combine factors into final score
        interest_score = priority_weight * (1 - likeliness) * confidence * 100 * time_decay
        return max(0, interest_score)  # Ensure non-negative