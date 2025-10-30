import json
import torch
import numpy as np
from scipy.spatial.distance import cosine
from markov.src.model.entities.event import check_for_event, Event
from markov.src.model.entities.touch import Touch


class EventRecognizer:
    '''Class for recognizing events in a game history by comparing the touch data to predefined event embeddings and checking predefined if statements.'''
    def __init__(self, autoencoder, predefined_events, similarity_threshold=0.01, log_func=None, verbose=True):
        self.autoencoder = autoencoder
        self.predefined_event_embeddings = {}
        self.similarity_threshold = similarity_threshold
        self.log_func = log_func
        self.verbose = verbose
        self.labeled_touch_files = predefined_events
        self.precalculate_event_embeddings()

    def recognize_event(self, game_history, optimizer, criterion, autoencoder_training=True) -> Event:
        '''Recognizes an event in the last three states of the game history by comparing the touch data to predefined event embeddings and checking predefined if statements.'''
        event = check_for_event(game_history)
        if event:
            return event
        
        if len(game_history) < 3:
            return None
        
        autoencoder_event = self.apply_autoencoder(game_history, optimizer, criterion, autoencoder_training)
        if autoencoder_event != None and (autoencoder_event.type == "wall_pass" or autoencoder_event.type == "through_pass"):
            # Add the involved players, frame number and time to the event
            autoencoder_event.player_ids = [game_history[0][0].row , game_history[2][0].row]
            autoencoder_event.time = game_history[0][1]
        
        return autoencoder_event

    def apply_autoencoder(self, game_history, optimizer, criterion, autoencoder_training=True) -> Event:
        '''Applies the autoencoder to the last three states in the game history and trains it on the touch data if enabled.'''
        if len(game_history) < 3:
            return None
        
        touches_vector = torch.tensor(
            np.array([state.to_normalized_vector() for (state, _) in list(game_history)[:3]]),
                        dtype=torch.float32
        ).flatten()

        if autoencoder_training:
            optimizer.zero_grad()
            _, reconstructed = self.autoencoder(touches_vector)
            loss = criterion(reconstructed, touches_vector)
            loss.backward()
            optimizer.step()

            if self.verbose:
                self.log_func(f"Training Loss: {loss.item()}")

        with torch.no_grad():
            embedding = self.autoencoder.encoder(touches_vector.unsqueeze(0))
        embedding = embedding.squeeze().numpy()
        
        return self.calculate_event_similarity(embedding)

    def calculate_event_similarity(self, embedding):
        '''Checks if the embedding represents an event by comparing it to predefined event embeddings.'''
        min_distance = float('inf')
        best_match = None

        for event, reference_embedding in self.predefined_event_embeddings.items():
            distance = cosine(embedding, reference_embedding)

            if distance < min_distance:
                min_distance = distance
                best_match = event

        if min_distance < self.similarity_threshold:
            if self.verbose:
                self.log_func(f"Detected event: {best_match.type} with distance {min_distance}")
            
            return best_match

        return None

    def precalculate_event_embeddings(self):
        '''Calculates the embeddings for predefined events and stores them in a dictionary.'''
        # First parse JSON files to get the labeled touch sequences
        self.predefined_event_embeddings = {}
        labeled_touch_sequences = self.parse_json_files()

        for touch_sequence, event in labeled_touch_sequences:
            states = [touch.toGameState() for touch in touch_sequence]
            state_vectors = [state.to_normalized_vector() for state in states]
            concatenated_vectors = np.concatenate(state_vectors)
            concatenated_tensor = torch.tensor(concatenated_vectors, dtype=torch.float32)
            
            with torch.no_grad():
                embedding = self.autoencoder.encoder(concatenated_tensor.unsqueeze(0))
            
            self.predefined_event_embeddings[event] = embedding.squeeze().numpy()

        self.log_func(f"Predefined event embeddings: {[(event.type, embedding) for event, embedding in self.predefined_event_embeddings.items()]}")
        self.log_func(f"Predefined event embeddings calculated: {len(self.predefined_event_embeddings)}")
    
    def parse_json_files(self):
        '''Parses JSON files containing labeled touch sequences and returns them as a list of tuples.'''
        labeled_sequences = []

        for file_path in self.labeled_touch_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Extract touches and event
                touch_data = data.get("touches", [])
                event_data = data.get("event", {})
                
                # Convert touches to objects
                touch_sequence = [Touch(touch) for touch in touch_data]
                event = Event(event_data)
                
                labeled_sequences.append((touch_sequence, event))

        return labeled_sequences