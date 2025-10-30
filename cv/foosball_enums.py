"""
This module defines enumerations for player positions, special game events, and team identification.

Enums:
- PlayerID: Represents different player positions on the field, including goalkeepers, defenders, midfielders, and forwards.
- SpecialEvent: Defines key game events such as goals, throw-ins, and match start.
- TeamID: Identifies teams as either LEFT (0) or RIGHT (1).

Additionally, lists `player_enum_values` and `team_enum_values` store the string and numeric values of PlayerID and TeamID enums for easy access.
"""

from enum import Enum
from typing import List

class PlayerID(Enum):
    """
    Enumeration representing various player positions in a football (soccer) game.
    """
    GOAL_KEEPER = "GK"
    LEFT_BACK = "LB"
    RIGHT_BACK = "RB"
    LEFT_MIDFIELD = "LM"
    RIGHT_MIDFIELD = "RM"
    LEFT_CENTRAL_MIDFIELD = "LCM"
    RIGHT_CENTRAL_MIDFIELD = "RCM"
    CENTRAL_MIDFIELD = "CM"
    LEFT_FORWARD = "LF"
    RIGHT_FORWARD = "RF"
    CENTRAL_FORWARD = "CF"
    WALL = "WALL"  # Represents an event when the ball hits the wall.

# List of all player ID values for quick reference.
player_enum_values: List[str] = [p.value for p in PlayerID]


class SpecialEvent(Enum):
    """
    Enumeration representing special game events.
    """
    GOAL_0 = "goal_0"
    GOAL_1 = "goal_1"
    THROW_IN = "throw_in"
    START = "start"

# List of all event ID values for quick reference.
event_enum_values: List[int] = [e.value for e in SpecialEvent]

class TeamID(Enum):
    """
    Enumeration representing team identifiers.
    """
    LEFT = 0
    RIGHT = 1

# List of all team ID values for quick reference.
team_enum_values: List[int] = [t.value for t in TeamID]
