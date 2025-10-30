# make import from sibling directory work
import sys, os
sys.path.append(os.path.abspath('..'))

from src.model.entities.state import State
from src.model.markov_model import MarkovModel

def train_synthetic_data(states_config, training_game):
    pass
    ##### Training #####
    # load states
    name_state_dict = {}
    for index, name in enumerate(open(states_config).read().splitlines()):
        name_state_dict[name] = State(index)
    # since we later want to get names for given states, store the inverse dict as well
    state_name_dict = {v: k for k, v in name_state_dict.items()}
    # load training data
    training_data = []
    for state_name in open(training_game).read().splitlines():
        # lookup State matching the state_name stored in the file and add it to training data list
        training_data.append(name_state_dict[state_name])

    model = MarkovModel(len(name_state_dict))
    print(training_data[0])
    model.train(training_data)      # train on sequence of States
    # print("Transition matrix:\n", model.tm)
    
    ##### Predict #####
    current_state_name = "GK0_1 0 slow"
    current_state = name_state_dict[current_state_name]
    # predict the next states
    next_states = model.generate_states(current_state, 3)
    # to print them, we would like the names of the predicted states
    next_states_names = [state_name_dict[s] for s in next_states]
    print(f"Probable next states from '{current_state_name}':", next_states_names)

    # mc.visualize('out.svg') """


if __name__ == "__main__":
    states_config = '../resources/configs/game_states_detailed_old.txt'
    training_game = '../resources/data/synthetic/ball_touches_sequence_goals.txt'

    train_synthetic_data(states_config, training_game)
