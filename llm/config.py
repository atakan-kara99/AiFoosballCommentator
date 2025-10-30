"""
Configuration Module
==================

This module centralizes all configuration settings for the LLM part of the table football commentary system.
It defines parameters for event processing, commentary generation, and output management.

Configuration Groups:
------------------
1. General Settings: Basic application configuration
2. Buffer Config: Event queue management settings
3. Event Cleaner Config: Event filtering and classification parameters
4. Prompt Generator Config: LLM prompt templates and constraints
5. LLM Interface Config: Language model parameters and API settings
6. Commentator Config: Output timing and statistics settings

Note: Adjust these values based on the deployment environment (development, testing, production).
"""

# Application-wide settings controlling basic behavior and identification
GENERAL_SETTINGS = {
    "APP_NAME":     "Master Project 24/25",  # Project identifier
    "VERSION":      "1.0.0",                 # Semantic versioning
    "DEBUG_MODE":   True,                    # Enables detailed logging and debugging features
}


# Settings for the event buffer that manages incoming game events
BUFFER_CONFIG = {
    "BUFFER_SIZE":  10,     # Maximum number of events held in buffer before overflow
    "TIME_OFFSET":  3000,   # Time window (ms) before events are considered stale and removed
}


# Thresholds for event filtering and classification
EVENT_CLEANER_CONFIG = {
    "CONFIDENCE_TH": 0.2,   # Minimum confidence threshold for accepting events

    # Event frequency classification thresholds (0.0 to 1.0)
    "LIKELINESS_TH_EXCEPTIONIAL":    0.2,   # Very rare events (e.g., special moves)
    "LIKELINESS_TH_RARE":            0.4,   # Uncommon events (e.g., long shots)
    "LIKELINESS_TH_UNCOMMON":        0.6,   # Regular but notable events
    "LIKELINESS_TH_COMMON":          1.0    # Frequent events (e.g., basic passes)
}


# Templates and constraints for generating LLM prompts
PROMPT_GENERATOR_CONFIG = {
    # Base role definitions for different commentary types
    "TEMPLATE_ROLE":        "You are commentating a table soccer match. Reply with exactly one sentence (4 to 8 words) for ",
    "TEMPLATE_INTERUPT":    "You are commentating a table soccer match. Reply with exactly one sentence (6 to 12 words) for ",
    
    # Style constraints to ensure consistent output
    "TEMPLATE_DONTS":       " Give only one sentence. No multiple sentences! No names! Be direct and punchy. No partial phrases.",
    "TEMPLATE_DONTS_LONG":  " Give only one sentence. No multiple sentences! No names! Be exciting and punchy. No partial phrases.",

    # Template for statistics-based commentary
    "TEMPLATE_STATISTIC":   "You are commentating a table soccer match. Reply with exactly one phrase (4 to 8 words) for the following statistic: {statistic}. No multiple sentences! Focus on the numbers! No names. Be direct and punchy.",

    # Event-specific templates with placeholders
    "TEMPLATE_NONE":        "a {event}",                                      # Basic event
    "TEMPLATE_SINGLE":      "a {event} by the {player}",                      # Single player event
    "TEMPLATE_DOUBLE":      "a {event} from the {player} to the {playerSec}", # Two player event
    "TEMPLATE_TEAM":        "a {event} by team {team}",                       # Team event
}


# Configuration for the LLaMA model interface
LLM_INTERFACE_CONFIG = {
    "DEFAULT_MODEL_URL": "meta-llama/Llama-3.2-3B-Instruct",     # Model identifier
    "API_KEY": None,                                             # HuggingFace API token (set via environment variable)
    "DEFAULT_STATISTICS_PATH": "live_resources/statistics.json", # Path to statistics file

    # Generation parameters controlling output quality and consistency
    "MAX_LENGTH":           50,      # Maximum total sequence length
    "MAX_NEW_TOKENS":       30,      # Maximum new tokens to generate
    "TEMPERATURE":          0.7,     # Randomness in generation (0.0-1.0)
    "TOP_K":                30,      # Number of highest probability tokens to consider
    "TOP_P":                0.7,     # Cumulative probability threshold for tokens
    "EARLY_STOPPING":       True,    # Stop at first complete sentence
    "RETURN_FULL_TEXT":     False,   # Exclude prompt from output
    "TRUNCATION":           True,    # Enable input truncation if needed
}


# Settings controlling commentary timing and statistics generation
COMMENTATOR_CONFIG = {
    "WORDS_PER_MINUTE":     200,     # Speech rate for timing calculations
    "END_TOKEN":            "\n",    # Token marking end of commentary
    "STATISTICS_TIME":      5000,    # Delay (ms) after events before statistics
    "STATISTICS_INTERVAL":  10000,   # Minimum time (ms) between statistics
    "STATISTICS_TYPES":     ["score", "ball_posession"]  # Available statistics types
}


# --- Helper Functions for Configuration ---
def print_config():
    """
    Prints the current configuration settings for debugging purposes.
    Useful for verifying settings during development and troubleshooting.
    """
    print("App Name:            ", GENERAL_SETTINGS["APP_NAME"])
    print("Version:             ", GENERAL_SETTINGS["VERSION"])
    print("Debug Mode:          ", GENERAL_SETTINGS["DEBUG_MODE"])