# make import from sibling directory work
import sys, os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from configs.state_config import GameStatesConfig as c

TOTAL_GAME_STATES = (   # player states
    (2 * ( c["FIGURE_SECTIONS"]["GK"] * (c["FIGURE_ANGLES"]["GK"] * (len(c["SPEEDS"]) - 1) + 1)  # sections*(angles*2speeds +noAngleAndSpeedState)
         + c["FIGURE_SECTIONS"]["B"] * (c["FIGURE_ANGLES"]["B"] * (len(c["SPEEDS"]) - 1) + 1)
         + c["FIGURE_SECTIONS"]["M"] * (c["FIGURE_ANGLES"]["M"] * (len(c["SPEEDS"]) - 1) + 1)
         + c["FIGURE_SECTIONS"]["F"] * (c["FIGURE_ANGLES"]["F"] * (len(c["SPEEDS"]) - 1) + 1)
         )
    ) 
    +   # wall states
    (2 * (len(c["SPEEDS"]) - 1) * ( c["WALL_SECTIONS"]["X"] * c["WALL_ANGLES"]          # 2 sides each, 2 speed options each
                + c["WALL_SECTIONS"]["Y"] * c["WALL_ANGLES"]
                )
    )
    +   # throw_in states
    2           # 2 teams
    +   # goal states
    2
)

GAME_HISTORY_DEQUE_LEN = 3          # determines number of states in the past that are remembered for event detection
RECONSTRUCT_HISTORY_LEN = 6         # maximum number of states analysed from a reconstructed history

TOUCHES_TO_UPDATE = 30          # number of touches (that are not a goal) before updating the persistent stats

PAUSE_BETWEEN_ERRORS = 500      # only forward consecutive errors if last error was this many milliseconds ago

PAUSE_BETWEEN_SAME_PLAYER_TOUCHES = 1500    # only forward consecutive touches by the same player if there was enough time between them

DELAY_BEFORE_SENDING_LAST_TOUCH = 1500       # delay before sending the last touch after the last touch was detected
