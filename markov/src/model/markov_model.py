# make import from sibling directory work
import sys, os
sys.path.append(os.path.abspath('..'))

import numpy as np
import networkx as nx

from .entities.state import State, GameState
from .graph_plotter import MarkovGraphVisual

from .vislibgraph import add_link, add_node, reset 



'''This class contains the information about transition counts and probabilities 
between different states of a markov chain/graph.'''
class TransitionMatrix:
    def __init__(self, amount_states: int):
        self.size = amount_states                                       # dimension of the matrix
        self.transition_counts = np.zeros((self.size, self.size))       # counts of the different transitions
        self.transition_probs = np.zeros((self.size, self.size))        # probabilities of the different transitions
        self.transition_times_sum = np.zeros((self.size, self.size))    # sum of all transition times between states
        self.transition_times_avg = np.zeros((self.size, self.size))    # average transition times between states
        
    def add_transition(self, source:GameState, target:GameState):
        '''Updates the transition count for the given transition.\n
        Intended for life game model updates to visualize the game graph.'''
        self.transition_counts[source.to_index()][target.to_index()] += 1
        # notify server
        add_link(source.to_graph_string(), target.to_graph_string())

    def train(self, game_data: list[int], timestamps: list[int]):
        '''This function accepts a list of indices, which describe the states in a game sequence. 
        This sequence is used to train the probabilities of transitions between states by counting transitions 
        and dividing it through total transitions for each state.\n
        Additionally, the average transition times are learned using the timestamps sequence.\n
        The function can be called multiple times to allow iterative and cumulative training over time.'''

        reset() # Reset the remote graph visualization on the vis server. If no connection is possible it will turn off the vis function.

        for i in range(len(game_data) - 1):
            current_state = game_data[i]
            next_state = game_data[i + 1]
            time_delta = timestamps[i+1]-timestamps[i]
            # adjust transition counts
            self.transition_counts[current_state][next_state] += 1
            # sum up total transition times
            self.transition_times_sum[current_state][next_state] += time_delta
            # Send to graph server.
            add_link(current_state, next_state)


        # after processing the entire sequence, update the transition probabilities
        self.transition_probs = np.nan_to_num(self.transition_counts / self.transition_counts.sum(axis=1, keepdims=True))
        # ... and average times
        self.transition_times_avg = np.nan_to_num(self.transition_times_sum / self.transition_counts)

    def __str__(self):
        '''Returns a printable version of the transition matrix showing the transition probabilities.'''
        return str(self.transition_probs)


'''This class implements the main model to describe the dynamics of a table foosball game.
It contains a transition matrix holding the transition probabilities between states.
The model can be trained on game sequences, and queried in case predictions need to be made.'''
class MarkovModel:
    def __init__(self, amount_states: int):
        self.tm = TransitionMatrix(amount_states)   # the underlying transition matrix containing transition probability data
        
    def add_transition(self, source:GameState, destination:GameState):
        '''Updates the transition count in the underlying transition matrix.\n
        Intended for life game model updates to track the transition counts for visualization.'''
        self.tm.add_transition(source, destination)
                
    def train(self, sequence: list, timestamps: list[int]):
        '''This function accepts a list of States or indices, which describe a game sequence. This sequence is used to train the
        probabilities of transitions between states by counting transitions and dividing it through total transitions
        for each state.\n
        In case of the list of indices, these indices describe the states already, just in a light-weight form.
        Training by using the list of indices is advised, as the passed parameters become way smaller and training is faster.
        The function can be called multiple times to allow iterative and cumulative training over time.\n
        Additionally, a list of timestamps is given, where a timestamp is given for each state, indicating when a touch occured
        that lead to this state. The timestamp sequence is used to simultaniously train the average transition times between states.'''
        # check if list contains indices or States
        if len(sequence) >= 2:      # otherwise no training of transitions can be performed
            if type(sequence[0]) == int:    # already index list
                self.tm.train(sequence, timestamps)
            elif issubclass(type(sequence[0]), State):  # State list
                # pass on list of indices (convert each state to its index)
                self.tm.train([state.to_index() for state in sequence], timestamps)

    def visualize(self, path=None):
        '''Visualizes the Markov graph with transition probabilities in a graphic.'''
        mgv = MarkovGraphVisual(self.tm.transition_probs, list(range(self.tm.size)))
        mgv.draw(path)
        
    def get_transition_probability(self, source:int, destination:int) -> float:
        '''Looks up the transition probability between two states, given by their indices (`source`, `destination`). '''
        return self.tm.transition_probs[source][destination]

    def most_probable_path(self, source:int, destination:int, exclude_states:set[int]=set()) -> tuple[list[int], list[float]]:
        '''Calculates the shortest path between two given node indices of the graph without passing through
        any state with an index in `exclude_states` using dijkstra.\n
        Additionally, returns a list of the probabilities that of the transition used in each step.'''
        trans_counts = self.tm.transition_counts.copy()
        # remember which states will stay in the pruned matrix
        states = [i for i in range(len(self.tm.transition_counts)) if i not in exclude_states]
        # check for well-defined arguments
        if source not in states or destination not in states:
            raise ValueError("source and/or destination is not a legal state index")
        
        # get pruned transition counts
        trans_counts = np.delete(trans_counts, list(exclude_states), axis=0)
        trans_counts = np.delete(trans_counts, list(exclude_states), axis=1)
        
        # calculate new transition probabilities
        trans_probs = np.nan_to_num(trans_counts / trans_counts.sum(axis=1, keepdims=True))
        # add small epsilon to avoid log(0) in upcoming log application
        trans_probs = np.add(trans_probs, 1e-15)
        # now, calculate the transition weights using log with base 1/2
        # this makes probable transactions have weight close to 0
        trans_weights = np.emath.logn(0.5, trans_probs)
        # in case above epsilon made a probability >1, adjust negative log result to 0
        trans_weights[trans_weights<0] = 0
        
        # now, use the weights to set up a graph for shortest path search
        G:nx.DiGraph = nx.from_numpy_array(trans_weights, create_using=nx.DiGraph(), nodelist=states)
        shortest_path = nx.dijkstra_path(G, source, destination)
        # get probabilities from adjusted trans_probs matrix
        probs = [1] # first state is certain
        src = states.index(shortest_path[0])
        dest = None
        for state in shortest_path[1:]:
            dest = states.index(state)
            probs.append(trans_probs[src][dest])
            # update src for next iteration
            src = dest
            
        return shortest_path, probs
        

    ##### Early development functions, probably not used later! #####

    def predict_next_state(self, current_state: State, weighted_random_choice=False) -> State:
        '''This function receives a current state and predicts the next state based on the probabilities 
        in the transition matrix. If weighted_random_choice is True, this is non-deterministic and does a weighted 
        random choice between all transitions with probability >0. Otherwise, it deterministicly chooses the most 
        likey transition (highest probability).'''
        if weighted_random_choice:
            return self.weighted_random_choice(current_state)
        else:
            return self.most_probable_choice(current_state)

    def most_probable_choice(self, current_state: State) -> State:
        '''Helper function of "next_state()" that chooses the most probable transition.'''
        next_index = np.argmax(self.tm.transition_probs[current_state.to_index()])
        return State(next_index)

    def weighted_random_choice(self, current_state: State) -> State:
        '''Helper function of "next_state()" that chooses a transition weighted-random based on transition probabilities.'''
        next_index = np.random.choice(
            range(self.tm.size),
            p=self.tm.transition_probs[current_state.to_index()]
        )
        return State(next_index)

    def generate_states(self, current_state: State, n=3, weighted_random_choice=False) -> list[State]:
        '''Generates the next <n> states from a given starting state.'''
        future_states = []
        for _ in range(n):
            next_state = self.predict_next_state(current_state, weighted_random_choice)
            future_states.append(next_state)
            current_state = next_state
        return future_states
