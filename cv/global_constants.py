"""
Global constants for video processing and foosball analysis.

This module defines various constants related to:
  - Frame timing and processing.
  - Real-life measurements of the playing field, ball, rods, and goals.
  - Player positions on rods.
  - Cropping and tolerance settings for image processing.

These constants are used throughout the application to convert between real-world units
and pixel measurements, as well as to configure event detection thresholds.
"""

from typing import List, Tuple

from cv.foosball_enums import PlayerID, TeamID

# Frame-related constants
FPS: int = 57
FRAME_TIME_SEC: float = 1.0 / FPS  # Duration of one frame in seconds (used for speed calculations)

# Waiting period (in frames) for goal or error events (100 ms).
GOAL_WAIT_FRAMES: int = int(0.100 * FPS)

# Real-life field measurements (in millimeters)
REAL_FIELD_WIDTH: int = 1200  # Field width in mm
REAL_FIELD_HEIGHT: int = 690  # Field height in mm
FIELD_GEOMETRIC_MEAN: float = (REAL_FIELD_WIDTH * REAL_FIELD_HEIGHT) ** 0.5  # Geometric mean of field dimensions (for scaling)

# Real-life ball measurements (in millimeters)
BALL_RADIUS_MIN: int = 15  # Minimum ball radius in mm
BALL_RADIUS_MAX: int = 19  # Maximum ball radius in mm
BALL_MIN_K: float = BALL_RADIUS_MIN / FIELD_GEOMETRIC_MEAN  # Scaling factor for minimum ball radius
BALL_MAX_K: float = BALL_RADIUS_MAX / FIELD_GEOMETRIC_MEAN  # Scaling factor for maximum ball radius

# Real-life rod measurements (in millimeters)
ROD_SPACING_MIN: int = 127  # Minimum spacing between rods in mm
ROD_SPACING_MAX: int = 152  # Maximum spacing between rods in mm
ROD_SPACING_MIN_K: float = ROD_SPACING_MIN / REAL_FIELD_WIDTH  # Scaling factor for minimum rod spacing
ROD_SPACING_MAX_K: float = ROD_SPACING_MAX / REAL_FIELD_WIDTH  # Scaling factor for maximum rod spacing

# Real-life goal measurements (in millimeters)
GOAL_WIDTH_MIN: int = 200  # Minimum goal width in mm
GOAL_WIDTH_MAX: int = 250  # Maximum goal width in mm
GOAL_WIDTH_MIN_K: float = GOAL_WIDTH_MIN / REAL_FIELD_HEIGHT  # Scaling factor for minimum goal width
GOAL_WIDTH_MAX_K: float = GOAL_WIDTH_MAX / REAL_FIELD_HEIGHT  # Scaling factor for maximum goal width

# Player positioning on rods.
# The structure represents the arrangement of players on the field:
# - Each sublist corresponds to a rod (ordered from top to bottom).
# - Each tuple contains (player_id, team_id) for a player.
#   TeamID values indicate the side (e.g., left or right).
PLAYER_RODS: List[List[Tuple[str, int]]] = [
    [(PlayerID.GOAL_KEEPER.value, TeamID.LEFT.value)],
    [(PlayerID.LEFT_BACK.value, TeamID.LEFT.value), (PlayerID.RIGHT_BACK.value, TeamID.LEFT.value)],
    [(PlayerID.RIGHT_FORWARD.value, TeamID.RIGHT.value), (PlayerID.CENTRAL_FORWARD.value, TeamID.RIGHT.value), (PlayerID.LEFT_FORWARD.value, TeamID.RIGHT.value)],
    [
        (PlayerID.LEFT_MIDFIELD.value, TeamID.LEFT.value),
        (PlayerID.LEFT_CENTRAL_MIDFIELD.value, TeamID.LEFT.value),
        (PlayerID.CENTRAL_MIDFIELD.value, TeamID.LEFT.value),
        (PlayerID.RIGHT_CENTRAL_MIDFIELD.value, TeamID.LEFT.value),
        (PlayerID.RIGHT_MIDFIELD.value, TeamID.LEFT.value)
    ],
    [
        (PlayerID.RIGHT_MIDFIELD.value, TeamID.RIGHT.value),
        (PlayerID.RIGHT_MIDFIELD.value, TeamID.RIGHT.value),
        (PlayerID.CENTRAL_MIDFIELD.value, TeamID.RIGHT.value),
        (PlayerID.LEFT_CENTRAL_MIDFIELD.value, TeamID.RIGHT.value),
        (PlayerID.LEFT_MIDFIELD.value, TeamID.RIGHT.value)
    ],
    [(PlayerID.LEFT_FORWARD.value, TeamID.LEFT.value), (PlayerID.CENTRAL_FORWARD.value, TeamID.LEFT.value), (PlayerID.RIGHT_FORWARD.value, TeamID.LEFT.value)],
    [(PlayerID.RIGHT_BACK.value, TeamID.RIGHT.value), (PlayerID.LEFT_BACK.value, TeamID.RIGHT.value)],
    [(PlayerID.GOAL_KEEPER.value, TeamID.RIGHT.value)]
]

# padding for cropped frame
# needed for goal and throwin detection to see field barrier
CF_PADDING_X = 40
CF_PADDING_Y = 20
