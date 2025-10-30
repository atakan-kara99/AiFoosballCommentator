"""
Commentary Output Module
======================

This module manages the presentation of LLM-generated commentary for the table football system.
It simulates natural speech patterns by controlling the timing and flow of text output,
with support for interruptions and sentence completion.

Key Features:
-----------
1. Natural Speech Simulation:
   - Configurable words-per-minute rate
   - Word-by-word output with timing control

2. Interruption Management:
   - Graceful handling of commentary interruptions
   - Optional sentence completion
   - Clean transition between outputs

3. Output Control:
   - Flexible output targeting (console/frontend)
   - Configurable end tokens
   - Speed adjustment during runtime

Configuration:
------------
Speech parameters are controlled via COMMENTATOR_CONFIG in the main config file.
"""

# Standard library imports for timing and text processing
import time
import sys
import re
import os

# Add parent directory to Python path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import GENERAL_SETTINGS, COMMENTATOR_CONFIG


# Global configuration constants
VERSION     = GENERAL_SETTINGS["VERSION"]       # System version for logging
DEBUG_MODE  = GENERAL_SETTINGS["DEBUG_MODE"]    # Debug flag for verbose output

# Speech timing and formatting parameters
WORDS_PER_MINUTE = COMMENTATOR_CONFIG["WORDS_PER_MINUTE"]  # Default speech rate
END_TOKEN        = COMMENTATOR_CONFIG["END_TOKEN"]         # Text completion marker


class Commentator:
    """
    Manages the presentation of commentary text with natural speech simulation.

    This class controls the output of commentary text, simulating natural speech
    patterns through timed word-by-word presentation. It supports interruption
    handling and dynamic speed adjustment.

    Attributes:
    ----------
    wpm : int
        Current words per minute rate
    word_delay : float
        Calculated delay between words in seconds
    is_speaking : bool
        Flag indicating if commentary is in progress
    interrupted : bool
        Flag indicating if current speech was interrupted
    finish_sentence : bool
        Flag to complete current sentence despite interruption
    push_func : callable
        Function to handle text output (console/frontend)
    """

    def __init__(self, words_per_minute=WORDS_PER_MINUTE, push_func=None):
        """
        Initialize the commentator with speech settings.

        Args:
        ----
        words_per_minute : int
            Speaking speed in words per minute
        push_func : callable
            Function to handle text output, defaults to console
        """
        self.wpm = words_per_minute
        self.word_delay = 60.0 / words_per_minute  # Convert WPM to seconds/word
        self.is_speaking = False                   # Track speech state
        self.interrupted = False                   # Track interruptions
        self.finish_sentence = False               # Control sentence completion
        self.push_func = push_func or (lambda x: sys.stdout.write(x))  # Output handler
        
    def _split_into_words(self, text):
        """
        Split text into words while preserving punctuation and spacing.

        Uses regex to split text into tokens that include:
        - Words with attached punctuation
        - Whitespace between words
        - Special characters and symbols

        Args:
        ----
        text : str
            Input text to split

        Returns:
        -------
        list
            List of word and whitespace tokens
        """
        return re.findall(r'\S+|\s+', text)
        
    def _is_sentence_end(self, word):
        """
        Check if a word marks the end of a sentence.

        Detects standard sentence-ending punctuation (., !, ?)
        at the end of a word, ignoring internal punctuation.

        Args:
        ----
        word : str
            Word to check for sentence-ending punctuation

        Returns:
        -------
        bool
            True if word ends a sentence, False otherwise
        """
        return bool(re.search(r'[.!?]$', word.strip()))
        
    def speak(self, text, end=END_TOKEN):
        """
        Output text with natural speech timing.

        Processes text word by word, adding appropriate delays
        to simulate natural speech. Handles interruptions and
        can optionally complete the current sentence.

        Args:
        ----
        text : str
            Text to be spoken
        end : str
            Token to append after text completion
        """
        # Handle ongoing speech
        if self.is_speaking:
            self.interrupt()
            time.sleep(0.1)  # Ensure clean transition
            
        # Initialize speech state
        self.is_speaking = True
        self.interrupted = False
        words = self._split_into_words(text)
        
        try:
            # Process each word
            for i, word in enumerate(words):
                # Check for interruption
                if self.interrupted and not self.finish_sentence:
                    self.push_func("... ")  # Mark interruption
                    break
                elif self.interrupted and self._is_sentence_end(word):
                    self.push_func(word)    # Complete sentence
                    break
                    
                # Output word and add timing delay
                self.push_func(word)
                if not word.isspace():
                    time.sleep(self.word_delay)
            
            # Add end token if speech completed
            if not self.interrupted:
                self.push_func(end)
        finally:
            # Reset speech state
            self.is_speaking = False
            self.interrupted = False
            self.finish_sentence = False
    
    def interrupt(self, finish_sentence=False):
        """
        Interrupt ongoing speech output.

        Provides control over interruption behavior, optionally
        allowing the current sentence to complete before stopping.

        Args:
        ----
        finish_sentence : bool
            If True, complete current sentence before stopping
        """
        if self.is_speaking:
            self.interrupted = True
            self.finish_sentence = finish_sentence
    
    def set_speed(self, words_per_minute):
        """
        Adjust the speech output speed.

        Updates the words per minute rate and recalculates
        the word delay timing.

        Args:
        ----
        words_per_minute : int
            New speaking speed in words per minute
        """
        self.wpm = words_per_minute
        self.word_delay = 60.0 / words_per_minute  # Update timing
