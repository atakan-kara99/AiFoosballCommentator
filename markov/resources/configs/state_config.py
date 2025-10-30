GameStatesConfig = {
    # Number of equaly spaced sections for each wall direction. X -> horizontal (long), Y -> vertical (short)
    'WALL_SECTIONS': {'X': 7, 'Y': 4},

    # Number of equaly spaced sections for each wall section depending on wall part
    'WALL_ANGLES': 5,

    # Number of equaly spaced sections for each figure row
    'FIGURE_SECTIONS': {'GK': 4, 'B': 6, 'M': 6, 'F': 6},

    # Number of equaly spaced angle sections for each figure row section
    'FIGURE_ANGLES': {'GK': 8, 'B': 8, 'M': 8, 'F': 8},

    # Speeds are ordered from slow to fast and the values are the maximum speed for the category
    'SPEEDS': {None: 0.01, 'medium': 4.0, 'fast': float('inf')},
}
