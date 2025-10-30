# make import from sibling directory work
import sys, os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import json
from typing import Optional
from collections import deque
from .state import GameState

class Event:
    '''An event class, which is used to store all informations about a detected event. 
    Also provides methods to check a list of past states for defined events. 
    Events are defined as an occurence of a certain state or a sequence of states in the Markov model.'''
    def __init__(self, data: dict):
        self.time = data["time"]
        self.frame_no = 0           # frame_no added for demo version, might stay for later as well; this is set in the pipeline
        self.type = data["type"]
        self.team_id = data["team_id"]
        self.player_ids = data["player_ids"] if "player_ids" in data else []
        self.confidence = data["confidence"]
        self.likeliness = data["likeliness"]
        
    def to_dict(self) -> dict:
        '''Turns the Event object into a json formatted string that can be passed to the LLM group.'''
        return {
                "time": self.time,
                "frame_no": self.frame_no,
                "event": self.type,
                "team_id": self.team_id,
                "involved_players": self.player_ids,
                "confidence": self.confidence,
                "likeliness": self.likeliness
                } 



def check_for_event(game_history: list[tuple[GameState, int, str]]) -> Optional[Event]:
        '''This method checks the given list of past states and associated players for a defined event.'''
        # unzip the list of tuples first
        past_states = [e[0] for e in game_history]
        timestamps = [e[1] for e in game_history]
        
        # all events require at least one past state, some require more
        if len(game_history) >= 1:  # single state reliant events
            # Throw-in
            event = check_throw_in(past_states, timestamps)
            if event:
                return event
            # Shot
            event = check_shot(past_states, timestamps)
            if event:
                return event
            
        if len(game_history) >= 2:  # multiple states reliant events
            if past_states[0].row == 'WALL':  # if the last state is a wall state, return None
                return None
            # Goal
            event = check_goal(past_states, timestamps)
            if event:
                return event 
            # Block
            event = check_block(past_states, timestamps)
            if event:
                return event
            # Dribble
            event = check_dribble(past_states, timestamps)
            if event:
                return event
            
        return None
    

def check_throw_in(past_states: list[GameState], timestamps: list[int]) -> Optional[Event]:
    '''This method checks the first element of the given list of past states for a throw-in event.'''
    # Check if the last state is a throw-in state
    if past_states[0].throw_in:
        event = Event({
            "time": timestamps[0],
            "type": "throw-in",
            "team_id": past_states[0].team_id,
            "confidence": 1.0,
            "likeliness": 1.0
        })
        
        return event

    return None


def check_shot(past_states: list[GameState], timestamps: list[int]) -> Optional[Event]:
    '''This method checks the first element of the given list of past states and players for a shot event.'''
    if not check_states_completeness(past_states, 0):
        return None

    # Check if the last state is a shot state
    row = past_states[0].row
    angle = past_states[0].angle
    speed = past_states[0].speed

    # Shots are classified as ball touches heading in the direction of the goal, with either high speed or from the forward row and can't be slow
    if (row == "F" or speed == "fast") and angle % 8 <= 1 and speed != None:
        event = Event({
            "time": timestamps[0],
            "type": "shot",
            "team_id": past_states[0].team_id,
            "confidence": 1.0,
            "likeliness": 1.0
        })
        if past_states[0].row is not None:    # if player is not None
            event.player_ids.append(past_states[0].row)
            
        return event

    return None


def check_goal(past_states: list[GameState], timestamps: list[int]) -> Optional[Event]:
    '''This method checks the first element of the given list of past states and players for a goal event.'''
    # Check if the last state is a goal state
    if past_states[0].goal:
        event =  Event({
            "time": timestamps[0],
            "type": "goal",
            "team_id": past_states[0].team_id,
            "confidence": 1.0,
            "likeliness": 1.0
        })

        # Check for the last touch by a player of the scoring team
        for i, state in enumerate(past_states[1:]):
            if state.team_id == past_states[0].team_id:
                if past_states[i+1].row is not None:  # if player is not None
                    event.player_ids.append(past_states[i+1].row) # +1 because i starts at 0 while state iteration starts at 1
                break
        
        return event

    return None


def check_block(past_states: list[GameState], timestamps: list[int]) -> Optional[Event]:
    '''This method checks the first element of the given list of past states and players for a block event.'''
    if not check_states_completeness(past_states, 1) or not check_states_completeness(past_states, 0):
        return None
    # Get infos from the last state
    row = past_states[0].row
    team_id = past_states[0].team_id
    angle = past_states[0].angle


    # Blocks are touches changing the direction of the ball by at least 2 direction groups or more and the previous player touch was by the opposing team
    if past_states[1].team_id != team_id and min(abs(past_states[1].angle - angle), 7 - abs(past_states[1].angle - angle)) >= 2:    
        event = Event({
            "time": timestamps[0],
            "type": "block",
            "team_id": team_id,
            "confidence": 1.0,
            "likeliness": 1.0
        })
        if past_states[0].row is not None:    # if player is not None
            event.player_ids.append(past_states[0].row)
        return event

    return None


def check_dribble(past_states: list[GameState], timestamps: list[int]) -> Optional[Event]:
    '''This method checks the first element of the given list of past states and players for a dribble event.'''
    # Dribbles are concecutive touches by the same player of a team (player must also be not None)
    if not check_states_completeness(past_states, 1) or not check_states_completeness(past_states, 0):
        return None
    if past_states[0].team_id == past_states[1].team_id and past_states[0].row is not None and past_states[1].row == past_states[0].row:
        event = Event({
            "time": timestamps[0],
            "type": "dribble",
            "team_id": past_states[0].team_id,
            "confidence": 1.0,
            "likeliness": 1.0
        })
        if past_states[0].row is not None:    # if player is not None
            event.player_ids.append(past_states[0].row)
        
        return event
    
    return None

def check_states_completeness(game_history: list[GameState], index: int) -> bool:
    '''This method checks if the given list of past states is complete and all states are not None.'''
    # Check if the list is long enough
    if len(game_history) < index:
        return False
    if game_history[index] == None:
        return False
    state = game_history[index]
    # Check all fields for None values
    if state.row == None or state.sector == None or state.team_id == None or state.angle == None or state.speed == None:
        return False
    
    return True
