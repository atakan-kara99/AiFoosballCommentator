"""
Event Processing and Normalization Module
======================================

This module processes and normalizes raw game events from the table football system.
It handles event classification, priority assignment, and data cleaning to ensure
consistent event formatting for the commentary system.

Key Features:
-----------
1. Event Classification:
   - Priority-based event categorization
   - Confidence threshold filtering
   - Rarity assessment based on event likeliness

2. Data Normalization:
   - Player ID to position mapping
   - Team and position standardization
   - Event type validation

3. Interrupt Management:
   - High-priority event detection
   - Commentary flow control

Configuration:
------------
Event parameters are controlled via EVENT_CLEANER_CONFIG in the main config file:
- Confidence thresholds
- Likeliness thresholds for rarity classification
- Buffer size for event management
"""

# Standard library imports for type hints and system path
import sys
import os
from typing import Dict
import time

# Add parent directory to Python path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import GENERAL_SETTINGS, EVENT_CLEANER_CONFIG, BUFFER_CONFIG


# System version identifier
VERSION = GENERAL_SETTINGS["VERSION"]

# Event classification thresholds
CONFIDENCE_TH = EVENT_CLEANER_CONFIG["CONFIDENCE_TH"]           # Minimum confidence for event validity

# Rarity classification thresholds (ascending order)
THRESHOLD_EXCEPTIONIAL  = EVENT_CLEANER_CONFIG["LIKELINESS_TH_EXCEPTIONIAL"]  # Most rare events
THRESHOLD_RARE          = EVENT_CLEANER_CONFIG["LIKELINESS_TH_RARE"]          # Rare events
THRESHOLD_UNCOMMON      = EVENT_CLEANER_CONFIG["LIKELINESS_TH_UNCOMMON"]      # Uncommon events
THRESHOLD_COMMON        = EVENT_CLEANER_CONFIG["LIKELINESS_TH_COMMON"]        # Common events

# Buffer configuration
BUFFER_SIZE = BUFFER_CONFIG["BUFFER_SIZE"]                     # Maximum events in buffer


"""
Event Parameter Reference:
------------------------
confidence : float (0.0 to 1.0)
    Certainty level of event detection
    0.0 = uncertain, 1.0 = certain

likeliness : float (0.0 to 1.0)
    Frequency/probability of event occurrence
    0.0 = very rare, 1.0 = very common

priority : int (0 to 3)
    Event importance level
    0 = highest (e.g., goals)
    3 = lowest (e.g., routine passes)

involved_players : list
    Players participating in event
    Format: [position_id, team_id]
"""


def replace_item(player_id: str) -> str:
    """
    Convert player identification to standardized position name.

    Maps raw player IDs (combining position and team) to human-readable
    position names for commentary. Supports both detailed positions
    (e.g., 'LCM' = Left Center Midfield) and simplified positions
    (e.g., 'M' = Midfield).

    Args:
    ----
    player_id : str
        Raw player identifier from the game system

    Returns:
    -------
    str
        Standardized position or team name
        If ID unknown, returns 'unknown'

    Position Key:
    -----------
    - GK/K  = Goalkeeper
    - RB/LB = Defense (Right/Left Back)
    - M*    = Midfield variants
    - *F    = Forward/Attacker variants
    - *0    = Black team
    - *1    = White team
    """
    switch = {
        # Detailed position mapping
        'GK': 'Goalkeeper',
        'RB': 'Defense',
        'LB': 'Defense',
        'LM': 'Midfield',
        'LCM': 'Midfield',
        'CM': 'Midfield',
        'RCM': 'Midfield',
        'RM': 'Midfield',
        'LF': 'Attacker',
        'CF': 'Attacker',
        'RF': 'Attacker',

        # Simplified position mapping
        'K': 'Goalkeeper',
        'B': 'Defense',
        'M': 'Midfield',
        'F': 'Attacker',

        # Black team positions
        'K0': 'black Team',
        'B0': 'black Team',
        'M0': 'black Team',
        'F0': 'black Team',

        # White team positions
        'K1': 'white Team',
        'B1': 'white Team',
        'M1': 'white Team',
        'F1': 'white Team',
        
        # Detailed black team positions
        'GK0': 'Goalkeeper',
        'RB0': 'Defense',
        'LB0': 'Defense',
        'LM0': 'Midfield',
        'LCM0': 'Midfield',
        'CM0': 'Midfield',
        'RCM0': 'Midfield',
        'RM0': 'Midfield',
        'LF0': 'Attacker',
        'CF0': 'Attacker',
        'RF0': 'Attacker',

        # Detailed white team positions
        'GK1': 'Goalkeeper',
        'RB1': 'Defense',
        'LB1': 'Defense',
        'LM1': 'Midfield',
        'LCM1': 'Midfield',
        'CM1': 'Midfield',
        'RCM1': 'Midfield',
        'RM1': 'Midfield',
        'LF1': 'Attacker',
        'CF1': 'Attacker',
        'RF1': 'Attacker',

        'default':'unknown'              
    }
    
    player_name = switch.get(player_id, switch['default'])
    return player_name


def add_prio_entry(event: Dict[str, str]) -> Dict[str,str]:
    """
    Assign priority level to game events.

    Analyzes event type and assigns a priority level (0-3) based on
    event significance. Priority affects commentary timing and
    interruption behavior.

    Priority Levels:
    --------------
    0 (Highest): Goals
    1: Shots and direct scoring attempts
    2: Strategic plays (crosses, blocks)
    3 (Lowest): Basic moves and passes

    Args:
    ----
    event : Dict[str, str]
        Game event data including event type

    Returns:
    -------
    Dict[str, str]
        Event data with added priority field

    Raises:
    ------
    ValueError
        If event type is not recognized
    """
    # LEVEL 1: Base priority by event type
    match event["event"]:
            # Priority 0: Game-changing events
            case "goal":
                event.update({'priority': 0})

            # Priority 1: Scoring attempts
            case "goalshot" | "barrier shot" | "goalkeeper shot" | "hit on goalpost" | "edge shot" | "midfield shot" | "pull shot" | "pin shot" | "shot" | "throw in" | "throw-in" | "throw_in":
                event.update({'priority': 1})

            # Priority 2: Strategic plays
            case "through_pass" | "cross pass" | "barrier pass" | "block" | "deflect ball" | "dribble":
                event.update({'priority': 2})
            
            # Priority 3: Basic moves
            case "barrier pass" | "steep pass" | "edge pass" | "wall_pass":
                event.update({'priority': 3})

            # Handle unknown events
            case _:
                print(f"Unknown event TEST: {event['event']}")
                event.update({'priority': BUFFER_SIZE})


    # Note: Additional priority adjustments based on confidence
    # and likeliness are currently disabled
    '''
    # LEVEL 2
    if event['confidence'] < CONFIDENCE_TH and event['priority'] != BUFFER_SIZE:
        event['priority'] += 1


    # LEVEL 3
    if event['likeliness'] <= THRESHOLD_EXCEPTIONIAL and event['priority'] != 0: 
        event['priority'] -= 1
    elif event['likeliness'] > THRESHOLD_COMMON and event['priority'] != BUFFER_SIZE: 
        event['priority'] +=1
    '''

    return event


def clean(event: Dict[str, str]) -> Dict[str, str]:
    """
    Normalize and enhance event data for commentary.

    Processes raw event data to:
    1. Convert player IDs to readable positions
    2. Classify event rarity based on likeliness
    3. Standardize event format

    Args:
    ----
    event : Dict[str, str]
        Raw event data to process

    Returns:
    -------
    Dict[str, str]
        Cleaned and enhanced event data

    Rarity Classification:
    -------------------
    - exceptional: ≤ THRESHOLD_EXCEPTIONIAL
    - rare: ≤ THRESHOLD_RARE
    - uncommon: ≤ THRESHOLD_UNCOMMON
    - common: ≤ THRESHOLD_COMMON
    - very common: > THRESHOLD_COMMON
    """
    # Convert player IDs to position names
    updated_list = []
    for item in event['involved_players']:
        cur_item = replace_item(item)
        updated_list.append(cur_item)
    event['involved_players'] = updated_list

    # Classify event rarity based on likeliness thresholds
    if event['likeliness'] <= THRESHOLD_EXCEPTIONIAL:
        event.update({'rarity': 'exceptional'})
    elif event['likeliness'] <= THRESHOLD_RARE:
        event.update({'rarity': 'rare'})
    elif event['likeliness'] <= THRESHOLD_UNCOMMON:
        event.update({'rarity': 'uncommon'})
    elif event['likeliness'] <= THRESHOLD_COMMON:
        event.update({'rarity': 'common'})
    else:
        event.update({'rarity': 'very common'})

    return event


def checkInterupt(event: Dict[str, str]) -> bool:
    """
    Determine if an event should interrupt current commentary.

    Checks if the event is significant enough to interrupt
    ongoing commentary. Currently, only highest priority
    events (priority = 0) trigger interrupts.

    Args:
    ----
    event : Dict[str, str]
        Event to evaluate for interruption

    Returns:
    -------
    bool
        True if commentary should be interrupted
        False if commentary should continue
    """
    if event['priority'] == 0:
        return True
    else:
        return False