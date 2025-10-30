# make import from sibling directory work
import sys, os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import json
from .state import GameState
from resources.configs import state_config

'''This class models a touch (of the ball) event by grouping all information 
that is received about it in an object oriented way.
It is used whenever a valid touch was detected by CV and a JSON about it is sent to the backend.'''
class Touch:
    def __init__(self, data: dict):
        self.player = ""                # the player code of the touching player, or "WALL"
        self.team_id = 0                # which team the player belongs to
        self.time = 0                   # time passed since start of the game in milliseconds
        self.frame_no = 0               # the frame where the touch was recorded
        self.position = (0.0, 0.0)      # where on the field the touch happened (ranges 0-1)
        self.speed = 0.0                # how fast the ball was
        self.direction = 0              # degree of ball movement (not yet relative to player orientation)
        
        # the following attributes are usually False, unless there wasn't a real touch but rather a special event
        self.goal = False               # if a goal was shot
        self.throw_in = False           # if the ball just got thrown in
        
        # after initializing all attributes, fill them with details from the given dictionary
        self.readData(data)
    
    def readData(self, data: dict):
        '''This method reads the given dictionary data into the attributes.'''
        self.player = data["player"]
        self.team_id = data["team_id"]
        self.time = data["time"]
        self.frame_no = data["frame_no"]
        self.position = data["position"]
        self.speed = data["speed"]
        self.direction = data["direction"]
        self.goal = data["goal"]
        self.throw_in = data["throw_in"]

    def toGameState(self) -> GameState:
        '''This method converts the touch event into a GameState object.'''
        config = state_config.GameStatesConfig
        state = GameState.from_touch(self, config)

        return state
    
    def to_dict(self) -> dict:
        '''Turns the Touch object into a json formatted string.'''
        return {
                "type": "touch",
                "player": self.player,
                "team_id": self.team_id,
                "time": self.time,
                "frame_no": self.frame_no,
                "position": self.position,
                "speed": self.speed,
                "direction": self.direction,
                "goal": self.goal,
                "throw_in": self.throw_in
                }
        

        
'''This class models an error event by grouping all information 
that is received about it in an object oriented way.
It is used whenever an error was detected by CV and a JSON about it is sent to the backend.'''
class Error:
    def __init__(self, data: dict):
        self.time = 0                           # time passed since start of the game in milliseconds
        self.frame_no = 0                       # frame number of the error
        self.non_involved_players_team0 = {}    # all pairs (player:str,position:tuple[int,int]) of team0 NOT part of the undetected touch
        self.non_involved_players_team1 = {}    # all pairs (player:str,position:tuple[int,int]) of team1 NOT part of the undetected touch
        
        # after initializing all attributes, fill them with details from the given JSON
        self.readData(data)
        
    def readData(self, data: dict):
        '''This method reads the given data into the attributes.'''
        self.time = data["time"]
        self.frame_no = data["frame_no"]
        self.non_involved_players_team0 = data["non_involved_players_team0"]
        self.non_involved_players_team1 = data["non_involved_players_team1"]
    

### FREE FUNCTIONS ###

def processDict(touch_or_error_data: str) -> tuple[Touch, Error]:
    '''This function processes a given JSON received by from CV and returns a Touch and an Error object,
    one of which will always be None.'''
    # Check if the given dictionary has a type flag
    if "type" not in touch_or_error_data:
        return Touch(touch_or_error_data), None #TODO: alway get type flag from cv
        # raise Exception("Received a dictionary that does not have a type flag.")
    if touch_or_error_data["type"] == "error":
        return None, Error(touch_or_error_data)
    elif touch_or_error_data["type"] == "touch":
        return Touch(touch_or_error_data), None
    else:
        raise Exception("Received a dictionary that has type flag other than error or touch.")
    
def loadTrainingData(file_path: str) -> list[Touch]:
    '''Parses a .json file containing multiple touches.'''
    # read the file into a dictionary fault-
    try:
        with open(file_path, "r") as file:
            training_data_dict = json.load(file)
    except Exception as e:
        raise Exception("An error occured while trying to read training data from JSON file.")
    
    # this assumes list of touches in the json are in a json list with key "touches"
    return [Touch(data) for data in training_data_dict["touches"]]
        
