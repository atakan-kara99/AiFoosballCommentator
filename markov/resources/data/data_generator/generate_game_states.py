'''File to generate game_states.txt, a list of all game states, 
enumerated with their index used in the Markov pipeline processes.
It can be used to gain an overview of all states and their ordering regarding to the index, 
as well as for testing and validation purposes.'''
# make import from sibling directory work
import sys, os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')))

from configs.state_config import GameStatesConfig as c


# throwins and goals are special events and do not contain speed and direction information
SPECIAL_EVENTS = ["throw_in", "goal"]
TEAMS = [0,1]
WALLS = ["left", "upper", "right", "lower"]

def build_all_states() -> list[str]:
    '''Returns a list of strings, one for each state of the Markov GameState model.'''
    states = []
    for t in TEAMS:
        for row in c["FIGURE_SECTIONS"].keys():
            for section in range(c["FIGURE_SECTIONS"][row]):
                # add state with no angle direction and speed
                states.append(f"team-{t} {row}_{section} angle-None speed-None")
                for angle in range(c["FIGURE_ANGLES"][row]):
                    for speed in list(c["SPEEDS"].keys())[1:]:  # cut off none speed
                        states.append(f"team-{t} {row}_{section} angle-{angle} speed-{speed}")
        # before starting with the next team, add special events for that team
        for e in SPECIAL_EVENTS:
            states.append(f"team-{t} {e}")
    # after team related states, append wall contacts
    wall_sect = 0
    for i, wall in enumerate(WALLS):
        if i % 2 == 0:  # left, right
            o = "Y"
        else: # upper, lower
            o = "X"
        for section in range(c["WALL_SECTIONS"][o]):
            for angle in range(c["WALL_ANGLES"]):
                for speed in list(c["SPEEDS"].keys())[1:]:  # cut off none speed
                    states.append(f"wall-{wall} section-{wall_sect} angle-{angle} speed-{speed}")
            wall_sect += 1
        
    return states


file_path = "game_states_detailed.txt"
all_states = build_all_states()
# print number of states
print(len(all_states))
with open(file_path, "w") as file:
    for i, state in enumerate(all_states):
        file.write(f"{i}: {state}\n")