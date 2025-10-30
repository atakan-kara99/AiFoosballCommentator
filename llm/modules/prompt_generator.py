"""
Prompt Generation Module
=====================

This module manages the creation and formatting of prompts for the language model.
It provides a flexible system for generating contextually appropriate prompts
based on game events, statistics, and interruptions.

Key Features:
-----------
1. Template Management:
   - Role-based templates
   - Event-specific formatting
   - Interrupt handling templates

2. Context Validation:
   - Required field checking
   - Data type validation
   - Error handling

3. Event Processing:
   - Single and multi-player events
   - Team-wide actions
   - Statistical summaries

Configuration:
------------
Templates and formatting rules are controlled via PROMPT_GENERATOR_CONFIG
in the main config file.
"""

# Standard library imports
import sys
import os

# Add parent directory to Python path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import GENERAL_SETTINGS, PROMPT_GENERATOR_CONFIG
from typing import Dict


# System configuration
VERSION     = GENERAL_SETTINGS["VERSION"]    # System version for logging
DEBUG_MODE  = GENERAL_SETTINGS["DEBUG_MODE"] # Debug flag for verbose output

# Template configurations
TEMPLATE_ROLE       = PROMPT_GENERATOR_CONFIG["TEMPLATE_ROLE"]        # Base role template
TEMPLATE_INTERUPT   = PROMPT_GENERATOR_CONFIG["TEMPLATE_INTERUPT"]    # Interrupt handling
TEMPLATE_DONTS      = PROMPT_GENERATOR_CONFIG["TEMPLATE_DONTS"]       # Basic restrictions
TEMPLATE_DONTS_LONG = PROMPT_GENERATOR_CONFIG["TEMPLATE_DONTS_LONG"]  # Extended restrictions
TEMPLATE_STATISTIC  = PROMPT_GENERATOR_CONFIG["TEMPLATE_STATISTIC"]   # Statistics format


def validate_context(context: Dict[str, str], required_keys: list) -> None:
    """
    Validate required fields in event context.

    Ensures that all necessary information is present in the context
    before generating a prompt. This prevents incomplete or invalid
    prompts from being generated.

    Args:
        context (Dict[str, str]): Event context to validate
        required_keys (list): List of keys that must be present

    Raises:
        ValueError: If any required keys are missing from context
    """
    missing_keys = [key for key in required_keys if key not in context]
    if missing_keys:
        raise ValueError(f"Missing required keys in context: {', '.join(missing_keys)}")


def dict_to_context(json_dict: Dict[str,str]) -> Dict[str,str]:
    """
    Convert event data to prompt context.

    Processes raw event data into a standardized context format
    for prompt generation. Handles player identification and
    team information.

    Args:
        json_dict (Dict[str,str]): Raw event data dictionary

    Returns:
        Dict[str,str]: Formatted context for prompt generation

    Notes:
        - Extracts primary and secondary players if present
        - Includes event type, rarity, and team information
        - Handles cases with varying numbers of involved players
    """
    # Extract player information
    player_list = json_dict["involved_players"]
    player      = player_list[0] if len(player_list) > 0 else ""
    playerSec   = player_list[1] if len(player_list) > 1 else ""

    # Create standardized context
    return {
        "event":        json_dict["event"],
        "rarity":       json_dict["rarity"],
        "team":         json_dict["team_id"],
        "player":       player,
        "playerSec":    playerSec,
    }


def match_context(context: Dict[str, str]) -> str:
    """
    Select appropriate template for event context.

    Analyzes the event type and selects the most appropriate
    template for generating the prompt. Handles various game
    situations and player combinations.

    Template Categories:
        1. No Player Events:
           - Game starts
           - Environmental events

        2. Single Player Events:
           - Shots and attempts
           - Individual actions
           - Scoring events

        3. Multi-Player Events:
           - Passes
           - Interactions

        4. Team Events:
           - Collective actions
           - Strategic plays

    Args:
        context (Dict[str, str]): Event context with type and player information

    Returns:
        str: Formatted template string for the event

    Raises:
        ValueError: If event type is not recognized
    """
    event = context.get("event", "").lower()

    if event:
        match str(event):
            # No player involved
            case "throw in" | "throw-in":
                context["event"] = "start of a round"
                return PROMPT_GENERATOR_CONFIG["TEMPLATE_NONE"].format(**context) + TEMPLATE_DONTS

            # Single player involved
            case "shot":
                context["event"] = "unsuccessfull shot"
                return PROMPT_GENERATOR_CONFIG["TEMPLATE_SINGLE"].format(**context) + TEMPLATE_DONTS

            case "dribble":
                context["event"] = "dribbling"
                return PROMPT_GENERATOR_CONFIG["TEMPLATE_SINGLE"].format(**context) + TEMPLATE_DONTS
            
            case "block":
                context["event"] = "defense"
                return PROMPT_GENERATOR_CONFIG["TEMPLATE_SINGLE"].format(**context) + TEMPLATE_DONTS
            
            case "goal":
                return PROMPT_GENERATOR_CONFIG["TEMPLATE_SINGLE"].format(**context) + TEMPLATE_DONTS_LONG

            case "goalshot" | "barrier shot" | "goalkeeper shot" | "hit on goalpost" | "edge shot" | "midfield shot" | "pull shot" | "pin shot" | "deflect ball":
                return PROMPT_GENERATOR_CONFIG["TEMPLATE_SINGLE"].format(**context) + TEMPLATE_DONTS

            # Double player involved
            case "wall_pass" | "through_pass" | "cross pass" | "barrier pass" | "steep pass" | "edge pass":
                return PROMPT_GENERATOR_CONFIG["TEMPLATE_DOUBLE"].format(**context) + TEMPLATE_DONTS
            
            # Team event
            case "barrier dribbling":
                return PROMPT_GENERATOR_CONFIG["TEMPLATE_TEAM"].format(**context) + TEMPLATE_DONTS

            # Handle unknown events
            case _:
                raise ValueError(f'Unknown event: "{event}"')


def generate_prompt(case: str, json_dict: Dict[str,str]) -> str:
    """
    Generate formatted prompt for language model.

    Creates a complete prompt by combining templates and context
    based on the event type and case. Handles different commentary
    styles and ensures all required information is present.

    Args:
        case (str): Type of prompt to generate:
            - "event": Regular game event
            - "statistics": Game statistics
            - "interupt": Interrupt current commentary
        json_dict (Dict[str,str]): Event data and context

    Returns:
        str: Complete formatted prompt

    Raises:
        ValueError: If case is not recognized or context validation fails
    """
    output_string = ""

    match case:
        case "event":
            # Standard game event
            context = dict_to_context(json_dict)
            required_keys = ["rarity", "event", "player", "team"]
            validate_context(context, required_keys)
            output_string = TEMPLATE_ROLE + match_context(context)

        case "statistics":
            # Statistical summary
            required_keys = ["event", "type", "statistic"]
            validate_context(json_dict, required_keys)
            output_string = TEMPLATE_STATISTIC.format(**json_dict)

        case "interupt":
            # Commentary interruption
            context = dict_to_context(json_dict)
            required_keys = ["rarity", "event", "player", "team"]
            validate_context(context, required_keys)
            output_string = TEMPLATE_INTERUPT + match_context(context)

        case _:
            raise ValueError(f"Unknown case: {case}")

    return output_string