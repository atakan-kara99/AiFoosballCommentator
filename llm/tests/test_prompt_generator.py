"""
Prompt Generator Test Suite
========================

This module provides comprehensive unit tests for the Prompt Generator,
which is responsible for creating well-structured prompts for the LLM
based on game events and statistics.

Test Categories:
-------------
1. Context Validation:
   - Required field checking
   - Data type verification
   - Error handling

2. Context Conversion:
   - Event data transformation
   - Player handling (single/multiple)
   - Team statistics

3. Template Selection:
   - Event type matching
   - Template formatting
   - Special cases

4. Prompt Generation:
   - Event prompts
   - Interrupt handling
   - Statistical analysis
   - Edge cases
"""

import pytest
import sys
import os
import json

# Add parent directory to Python path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import prompt generator components
from modules.prompt_generator import (
    validate_context,
    dict_to_context,
    match_context,
    generate_prompt
)
from config import PROMPT_GENERATOR_CONFIG


# Test Fixtures
@pytest.fixture
def sample_event_data():
    """
    Provides sample game event data for testing.
    
    Returns:
        dict: A dictionary containing a typical game event with:
            - event type (goal)
            - rarity level
            - team identifier
            - involved players
    """
    return {
        "event": "goal",
        "rarity": "rare",
        "team_id": "team_a",
        "involved_players": ["player1", "player2"]
    }

@pytest.fixture
def sample_statistic_data():
    """
    Provides sample statistical data for testing.
    
    Returns:
        dict: A dictionary containing player statistics with:
            - event type (statistic)
            - player identifier
            - team identifier
            - statistic type
    """
    return {
        "event": "statistic",
        "player": "player1",
        "team_id": "team_a",
        "statistic": "goals_scored"
    }


# Context Validation Tests
def test_validate_context_success():
    """
    Tests successful validation of a complete context dictionary.
    Verifies that no exception is raised when all required keys are present.
    """
    context = {"event": "goal", "rarity": "rare", "player": "player1", "team": "team_a"}
    required_keys = ["event", "rarity", "player", "team"]
    validate_context(context, required_keys)  # Should not raise an exception

def test_validate_context_missing_keys():
    """
    Tests validation failure when required keys are missing.
    Verifies that appropriate error message includes missing key names.
    """
    context = {"event": "goal", "rarity": "rare"}
    required_keys = ["event", "rarity", "player", "team"]
    with pytest.raises(ValueError) as exc_info:
        validate_context(context, required_keys)
    assert "Missing required keys" in str(exc_info.value)
    assert "player, team" in str(exc_info.value)


# Context Conversion Tests
def test_dict_to_context_single_player(sample_event_data):
    """
    Tests context creation with a single player event.
    Verifies primary player assignment and empty secondary player.
    """
    sample_event_data["involved_players"] = ["player1"]
    context = dict_to_context(sample_event_data)
    assert context["player"] == "player1"
    assert context["playerSec"] == ""

def test_dict_to_context_two_players(sample_event_data):
    """
    Tests context creation with a two-player event.
    Verifies correct assignment of primary and secondary players.
    """
    context = dict_to_context(sample_event_data)
    assert context["player"] == "player1"
    assert context["playerSec"] == "player2"

def test_dict_to_context_no_players(sample_event_data):
    """
    Tests context creation with no players involved.
    Verifies empty string assignments for both player fields.
    """
    sample_event_data["involved_players"] = []
    context = dict_to_context(sample_event_data)
    assert context["player"] == ""
    assert context["playerSec"] == ""


# Template Matching Tests
@pytest.mark.parametrize("event,expected_template", [
    ("throw in", "TEMPLATE_NONE"),
    ("shot", "TEMPLATE_SINGLE"),
    ("dribble", "TEMPLATE_SINGLE"),
    ("cross pass", "TEMPLATE_DOUBLE"),
    ("barrier dribbling", "TEMPLATE_TEAM")
])
def test_match_context_templates(event, expected_template):
    """
    Tests template selection for various event types.
    
    Args:
        event (str): The type of game event
        expected_template (str): The expected template identifier
    
    Verifies that each event type matches its designated template
    and the template can be properly formatted with context data.
    """
    context = {
        "event": event,
        "player": "player1",
        "playerSec": "",
        "team": "team_a",
        "rarity": "rare"
    }
    result = match_context(context)
    assert PROMPT_GENERATOR_CONFIG[expected_template].format(**context) in result

def test_match_context_unknown_event():
    """
    Tests error handling for unknown event types.
    Verifies appropriate error message for unrecognized events.
    """
    context = {"event": "unknown_event", "player": "player1", "team": "team_a"}
    with pytest.raises(ValueError) as exc_info:
        match_context(context)
    assert "Unknown event" in str(exc_info.value)


# Prompt Generation Tests
def test_generate_prompt_event(sample_event_data):
    """
    Tests generation of event-based prompts.
    Verifies inclusion of role template and event details.
    """
    prompt = generate_prompt("event", sample_event_data)
    assert PROMPT_GENERATOR_CONFIG["TEMPLATE_ROLE"] in prompt
    assert sample_event_data["event"] in prompt.lower()

def test_generate_prompt_interrupt(sample_event_data):
    """
    Tests generation of interrupt prompts.
    Verifies inclusion of interrupt template and event details.
    """
    prompt = generate_prompt("interupt", sample_event_data)
    assert PROMPT_GENERATOR_CONFIG["TEMPLATE_INTERUPT"] in prompt
    assert sample_event_data["event"] in prompt.lower()

def test_generate_prompt_statistic(sample_statistic_data):
    """
    Tests generation of statistical prompts.
    Verifies correct formatting of statistical template with data.
    """
    sample_statistic_data["team"] = sample_statistic_data["team_id"]
    prompt = generate_prompt("statistic", sample_statistic_data)
    assert PROMPT_GENERATOR_CONFIG["TEMPLATE_STATISTIC"].format(**sample_statistic_data) == prompt

def test_generate_prompt_invalid_case():
    """
    Tests error handling for invalid prompt types.
    Verifies appropriate error message for unknown cases.
    """
    with pytest.raises(ValueError) as exc_info:
        generate_prompt("invalid_case", {})
    assert "Unknown case" in str(exc_info.value)


# Edge Case Tests
def test_generate_prompt_empty_players(sample_event_data):
    """
    Tests prompt generation with no players involved.
    Verifies graceful handling of empty player lists.
    """
    sample_event_data["involved_players"] = []
    prompt = generate_prompt("event", sample_event_data)
    assert prompt  # Verify prompt is generated without player names

@pytest.mark.parametrize("event_type", [
    "goalshot", "barrier shot", "goalkeeper shot", 
    "hit on goalpost", "edge shot", "midfield shot", 
    "pull shot", "pin shot", "deflect ball"
])
def test_generate_prompt_special_shots(sample_event_data):
    """
    Tests prompt generation for various special shot types.
    
    Args:
        event_type (str): The specific type of shot event
    
    Verifies that all shot variants use the single-player template
    and are properly formatted with context data.
    """
    sample_event_data["event"] = event_type
    prompt = generate_prompt("event", sample_event_data)
    assert PROMPT_GENERATOR_CONFIG["TEMPLATE_SINGLE"].format(**dict_to_context(sample_event_data)) in prompt