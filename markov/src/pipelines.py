# make import from sibling directory work
import sys, os

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import pickle
import json
import torch
import numpy
from scipy.spatial.distance import cosine
from collections import deque
from multiprocessing import Queue
import time

from vislib import kpi
from .model.vislibgraph import reset
from markov.src.model.event_recognizer import EventRecognizer
from .model.autoencoder import Autoencoder, train_autoencoder
from .model.entities.event import Event, check_for_event
from .model.entities.state import GameState, GameStateConfig
from resources.configs.constants import *
from .model.entities.touch import Touch, Error, loadTrainingData, processDict
from .model.markov_model import MarkovModel
from .statistics.game_statistics import GameStatistics
from .error_handling.error_handler import ErrorHandler
from resources.configs import state_config

'''This class models an abstract pipeline and is refined within the concrete classes TrainingPipeline and LivePipeline.
This abstract class models common functionality between the two, like loading a model.'''
class Pipeline:
    def __init__(self, path_to_autoencoder:str, path_to_model:str, log_func=print, verbose:bool=True):
        self.path_to_autoencoder = path_to_autoencoder  # path where the autoencoder is or will be
        self.path_to_model = path_to_model              # path where the model is or will be stored
        self.log_func = log_func                        # function to log execution details if verbose is enabled
        self.verbose = verbose                          # if True, logs information about the steps during execution
        self.model : MarkovModel = None                 # the model that will be loaded
        self.autoencoder : Autoencoder = None           # the autoencoder that will be used for embedding generation
        self.steps : list[function] = []                # different pipeline steps here
        
    def execute(self):
        '''Executes the pipeline by running the methods for the respective steps.'''
        if self.verbose:
            self.log_func("Starting execution of pipeline steps.")
        for step in self.steps:
            step()

    def load_autoencoder(self):
        '''Loads the autoencoder from the path self.path_to_autoencoder, or creates a fresh, empty autoencoder if there is no autoencoder stored in that path.'''
        if self.verbose:
            self.log_func(f"Loading autoencoder from {self.path_to_autoencoder}.")
        try:
            # try to open existing autoencoder
            with open(self.path_to_autoencoder, "rb") as f:
                self.autoencoder = pickle.load(f)
        except (FileNotFoundError, pickle.PickleError) as e:
            # if no autoencoder exists yet, create a new one
            self.autoencoder = Autoencoder(3 * 13, 2)  # 3 * 13 is the size of the concetenated touch vector
            if self.verbose:
                self.log_func(f"No autoencoder was found in the specified path, therefore a new autoencoder was created.")

    def store_autoencoder(self):
        '''Stores the autoencoder persistently in a file with path self.path_to_autoencoder. 
        If it existed before, overwrite the old version.'''
        if self.verbose:
            self.log_func(f"Storing the updated autoencoder back to {self.path_to_autoencoder}.")
            
        os.makedirs(os.path.dirname(self.path_to_autoencoder), exist_ok=True)
        with open(self.path_to_autoencoder, "wb") as f:
            pickle.dump(self.autoencoder, f, pickle.HIGHEST_PROTOCOL)
        
    def load_model(self):
        '''Loads model from path self.path_to_model, or creates a fresh, empty model if there is no model stored in that path.'''
        if self.verbose:
            self.log_func(f"Loading model from {self.path_to_model}.")
        try:
            # try to open existing model
            with open(self.path_to_model, "rb") as f:
                self.model = pickle.load(f)
        except (FileNotFoundError, pickle.PickleError) as e:
            # if no model exists yet, create a new one
            self.model = MarkovModel(TOTAL_GAME_STATES)
            if self.verbose:
                self.log_func(f"No model was found in the specified path, therefore a new model was created.")
    
    def store_model(self):
        '''Stores the model persistently in a file with path self.path_to_model. 
        If it existed before, overwrite the old version.'''
        if self.verbose:
            self.log_func(f"Storing the updated model back to {self.path_to_model}.")

        os.makedirs(os.path.dirname(self.path_to_model), exist_ok=True)
        with open(self.path_to_model, "wb") as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)


    
'''This class models the pipeline during the training of the model by abstracting the different steps of the pipeline
to allow customization of the single step for modular development and flexible adaptions to single parts.'''
class TrainingPipeline(Pipeline):
    def __init__(self, path_to_autoencoder, path_to_model, path_to_training_data, log_func=print, verbose=True):
        super().__init__(path_to_autoencoder, path_to_model, log_func, verbose)
        self.path_to_training_data = path_to_training_data  # json with touch sequence
        self.training_data : list[Touch] = []               # sequence of touches of the game
        self.steps = [self.load_model,
                      self.receive_touches, 
                      self.train_model,
                      self.store_model,
                      self.load_autoencoder,
                      self.train_autoencoder, 
                      self.store_autoencoder]
        
    def receive_touches(self):
        '''Receives the raw JSON training data and stores the corresponding touches into self.training_data.
        This can be configured to happen in different ways, e.g. by listening to a network socket and collecting all inputs,
        or by reading in artificial training data from a generator. These different ways allow for training from different
        kind of data during the training phase.'''
        # load data fault-proof in case there is no valid json at the given path
        try:
            self.training_data = loadTrainingData(self.path_to_training_data)
        except Exception as e:
            self.log_func(e)
            return
                
        if self.verbose:
            self.log_func(f"Loaded {len(self.training_data)} touches.")

    def train_autoencoder(self):
        # Step 1: Prepare data as sequences of three consecutive touches
        touch_vectors = [touch.toGameState().to_normalized_vector() for touch in self.training_data]
        
        # Ensure there are enough touches to create sequences
        if len(touch_vectors) < 3:
            self.log_func("Not enough touches to train the autoencoder (need at least 3).")
            return

        # Create concatenated sequences of three consecutive touches
        concatenated_vectors = []
        for i in range(len(touch_vectors) - 2):  # Stop at len - 2 to avoid out-of-bounds
            sequence = numpy.concatenate([touch_vectors[i], touch_vectors[i + 1], touch_vectors[i + 2]])
            concatenated_vectors.append(sequence)
    
        
        # Convert to a PyTorch tensor
        concatenated_vectors = numpy.array(concatenated_vectors)
        concatenated_tensor = torch.tensor(concatenated_vectors, dtype=torch.float32)

        # Step 2: Initialize or load the autoencoder
        input_size = concatenated_tensor.shape[1]  # 3 * touch_vector_size
        embedding_size = 2  # Adjust as needed for the dimensionality of the embedding
        if self.autoencoder is None:
            self.autoencoder = Autoencoder(input_size, embedding_size)

        # Step 3: Train the autoencoder
        if self.verbose:
            self.log_func(f"Training the autoencoder on {len(concatenated_vectors)} sequences of 3 touches.")
        
        train_autoencoder(self.autoencoder, concatenated_tensor)
    
    def train_model(self):
        '''Updates the model using the new training data.'''
        if self.verbose:
            self.log_func(f"Starting updating the model with the new training data ({len(self.training_data)} touches).")
            
        # transform each touch in the training data to its corresponding GameState object, ...
        # ... then train model using the light-weight variant by only passing list of indices instead of list of GameState objects
        index_sequence = [touch.toGameState().to_index() for touch in self.training_data]
        timestamps = [touch.time for touch in self.training_data]
        self.model.train(index_sequence, timestamps)



'''This class models the pipeline during the live usage of the model by abstracting the different steps of the pipeline
to allow customization of the single step for modular development and flexible adaptions to single parts.'''
class LivePipeline(Pipeline):
    def __init__(self, path_to_autoencoder, path_to_model, path_to_statistics, sub_queue:Queue, pub_queue:Queue, log_func, verbose=True, autoencoder_training=False):
        super().__init__(path_to_autoencoder, path_to_model, log_func, verbose)
        self.statistics = GameStatistics(path_to_statistics)
        self.game_graph = MarkovModel(TOTAL_GAME_STATES)
        self.autoencoder_training = autoencoder_training
        self.error_handler = None
        self.sub_queue = sub_queue
        self.pub_queue = pub_queue
        if self.autoencoder_training:
            self.steps = [self.load_autoencoder,
                        self.initialize_event_recognizer,
                        self.load_model, 
                        self.loop,
                        self.store_autoencoder,
                        self.store_model]
        else:
            self.steps = [self.load_autoencoder,
                        self.initialize_event_recognizer,
                        self.load_model, 
                        self.loop,
                        self.store_model]

        reset() # Reset the visual graph on webserver.

        self.config = GameStateConfig(state_config.GameStatesConfig)
        self.labeled_touch_sequences = [
            'markov/resources/configs/labeled_sequences/wall_pass_1.json',
            'markov/resources/configs/labeled_sequences/wall_pass_2.json',
            'markov/resources/configs/labeled_sequences/through_pass_1.json',
            'markov/resources/configs/labeled_sequences/through_pass_2.json'
        ]
        self.similarity_threshold = 0.01

    def initialize_event_recognizer(self):
        # Initialize EventRecognizer
        self.event_recognizer = EventRecognizer(self.autoencoder, self.labeled_touch_sequences, 
                                                self.similarity_threshold, self.log_func, self.verbose)
        
    def loop(self):
        '''This function represents the loop that continuously runs during the live phase.
        It is only ended after receiving a termination signal from the subscription queue.'''
        game_active = True              # tracks transmission end
        self.error_handler = ErrorHandler(self.model, self.log_func)
        
        # store latest states, times and associated players, newest one left end
        game_history = deque(maxlen=GAME_HISTORY_DEQUE_LEN)

        # Autoencoder training components
        criterion = torch.nn.MSELoss()  # Loss function
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)  # Optimizer

        if self.verbose:
            self.log_func("Starting loop to receive and handle input.")
        while game_active:
            touch, error = self.receive_touch_or_error()
            # track time for each loop
            start_time = time.time()
            # check if transmission has ended
            if not (touch or error):
                game_active = False
                continue
            
            # different handling dependent on the received object
            if touch:
                self.statistics.update(touch)
                # find state corresponding to the touch and append it to the state_history
                current_state = touch.toGameState()
                timestamp = touch.time
                kpi('current state', current_state)

                # build game graph
                if len(game_history) > 0:
                    last_state : GameState = game_history[0][0]
                    self.game_graph.add_transition(last_state, current_state)
                
                # check if this touch immidiately follows a sequence of errors
                if self.error_handler.reconstruct_required:
                    # if so, use the Markov model and the current state data to reconstruct the error
                    reconstruct_history, confidence_values = self.error_handler.reconstruct_history((current_state, timestamp))
                    # if the reconstructed history is too long for event detection..
                    if len(reconstruct_history) > RECONSTRUCT_HISTORY_LEN:
                        # examine only last few states
                        reconstruct_history = reconstruct_history[-RECONSTRUCT_HISTORY_LEN:]
                        confidence_values = confidence_values[-RECONSTRUCT_HISTORY_LEN:]
                    # if it is too short for event detection..
                    elif len(reconstruct_history) < 3:
                        # add the previously recorded states to get at least 3 states for event check method
                        reconstruct_history = list(reversed(list(game_history)[:(3-len(reconstruct_history))])) + reconstruct_history
                        # pad the confidence values with 1's, since previous history was certain
                        confidence_values = ([1] * (3-len(reconstruct_history))) + confidence_values
                    
                    # now, iterate the reconstructed history list and check for events for each triple
                    for i in range(len(reconstruct_history)):
                        # Add the history elements to the game history
                        game_history.appendleft(reconstruct_history[i])
                        # only check after 3 new elements were added, otherwise history might be non-continuous
                        if i > 2:
                            event = self.event_recognizer.recognize_event(game_history, optimizer, criterion, self.autoencoder_training)
                            if event:
                                # for an event found in the last reconstructed triplet, we can add the frame number
                                if i == len(reconstruct_history) - 1:
                                    event.frame_no = touch.frame_no
                                # adjust the confidence since the history is uncertain
                                event.confidence = confidence_values[i]
                                # the likeliness depends on the transition probability from the previous to the current state
                                if len(game_history) > 1:
                                    last_state : GameState = list(game_history)[1][0]
                                    event.likeliness = self.model.get_transition_probability(
                                        last_state.to_index(), current_state.to_index())
                                else:
                                    event.likeliness = 0.5  # default value, only relevant for first state in a game
                                self.forward_event(event)
                else:
                    # append the current state to the game history
                    game_history.appendleft((current_state, timestamp))
                        
                    # Use event recognizer to detect possible events
                    event = self.event_recognizer.recognize_event(game_history, optimizer, criterion, self.autoencoder_training)
                    if event:
                        event.frame_no = touch.frame_no
                        # since this event relies on an observed touch, the confidence is 1
                        event.confidence = 1
                        # the likeliness depends on the transition probability from the previous to the current state
                        if len(game_history) > 1:
                            event.likeliness = self.model.get_transition_probability(
                                last_state.to_index(), current_state.to_index())
                        else:
                            event.likeliness = 0.5  # default value, only relevant for first state in a game
                        self.forward_event(event)
                        
                # update error handling variables
                self.error_handler.update_valid(current_state, touch.time)
                
            if error:
                # build error log
                self.error_handler.log(error)
            
            # track time and display as kpi in milliseconds
            loop_time = (time.time() - start_time) * 1000
            kpi("Backend processing time", f"{round(loop_time)} ms")
            
        if self.verbose:
            self.log_func("Game transmission has ended.")
            
    def receive_touch_or_error(self) -> tuple[Touch,Error]:
        '''Receives the JSON object about a touch and transforms it into a touch object or error object, 
        depending on the content of the JSON.
        If the transmission is finished, return (None, None)'''
        curr_touch_log = self.sub_queue.get()
        if curr_touch_log == "FINISHED":
            return None, None
    
        return processDict(curr_touch_log)
    
    def forward_event(self, event: Event):
        '''Called whenever an event was detected in the game.
        Forwards a python dict about the event through transmission channel towards the LLM group to comment on it.'''
        self.pub_queue.put(event.to_dict())
        if self.verbose:
            self.log_func(f"Forwarded event: {event.to_dict()}")
        pass


'''This class is used to model a pseudo-live scenario, where the task of the pipeline is to do everything that is usually done
in a live scenario, besides receiving a live touch feed and forwarding a live event feed. 
Instead, it works with a preset touchlog that contains the entire touch chain that should be processed 
and outputs an entire event log that contains all events that are detected in the touch chain.'''
class PseudoLivePipeLine(LivePipeline):
    def __init__(self, path_to_autoencoder, path_to_model, path_to_statistics, path_to_touchlog, path_to_eventlog, verbose=True):
        super().__init__(path_to_autoencoder, path_to_model, path_to_statistics, sub_queue=None, pub_queue=None, log_func=print, verbose=verbose)
        # overwrite pipeline steps to include a final processing step
        self.steps = [self.load_autoencoder,
                      self.initialize_event_recognizer,
                      self.load_model, 
                      self.loop, 
                      self.store_autoencoder,
                      self.store_model,
                      self.final_steps]
        # load the touchlog that is used for pseudo-live processing
        self.touchlog : list[Touch] = loadTrainingData(path_to_touchlog)
        self.autoencoder_training = False
        self.error_handler = None
        self.touch_index = -1
        self.path_to_eventlog = path_to_eventlog
        self.eventlog : list[Event] = []
        self.config = GameStateConfig(state_config.GameStatesConfig)
        self.labeled_state_sequences = [
        ] 
        self.similarity_threshold = 0.01
    
    def receive_touch_or_error(self) -> tuple[Touch, Error]:
        '''Overwritten function to adjust for the pseudo-live scenario. Returns the next touch in the log.'''
        self.touch_index += 1
        if self.touch_index < len(self.touchlog):
            return self.touchlog[self.touch_index], None    # Touch, no Error
        
        return None, None   # No Touch, No Error; end of touchlog

    def forward_event(self, event: Event):
        '''Overwritten function to adjust for the pseudo-live scenario. Appends the event to the event log 
        that is stored in a file in one step in the end.'''
        if self.verbose:
            self.log_func(f"Forwarded event: {event.to_dict()}")
        self.eventlog.append(event)
        
    def final_steps(self):
        '''Final processing necessary for the pseudo-live scenario. Includes writing the eventlog to its file.'''
        if self.verbose:
            self.log_func(f"Writing the eventlog as a json to its file {self.path_to_eventlog}.")
            
        with open(self.path_to_eventlog, "w") as f:
            json.dump({"events": [event.to_dict() for event in self.eventlog]}, f)
