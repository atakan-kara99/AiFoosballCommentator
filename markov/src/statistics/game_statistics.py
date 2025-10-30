# make import from sibling directory work
import sys, os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import json
from ..model.entities.touch import Touch
from markov.resources.configs.constants import *

'''This class collects different kind of statistical data about a game of table football.
It can be updated on every touch event, and has a method to write the statistics to a file in a compact form.'''
class GameStatistics:
    def __init__(self, path):
        self.path = path                    # path of storage file
        self.touches_since_last_update = 0  # count touches and update periodically

        self.score = [0,0]                  # the current game score
        self.touches_by_team = [0,0]        # total ball touches by team, used to calculate ball posession
        self.touches_by_player = {}         # dict that records how often which player has touched the ball
        self.goals_by_player = {}           # dict that records how many goals which player shot
        
        self.prev_touch: Touch = None       # for goals, we need to look up the scoring player in the previous touch
        self.clear_file()                   # clear the statistic file at the start to avoid LLM reading old data
                
    def update(self, touch: Touch):
        '''Performs analytical processing of the last recorded touch and updates the statistics accordingly.
        Also updates the persistent storage of the statistics if it is appropriate (either on important events like goals 
        or there was a long period without an update).'''
        # count the touch
        self.touches_since_last_update += 1
        # goal handling
        if touch.goal:
            self.score[touch.team_id] += 1
            # try to find the scoring player in last touch
            if self.prev_touch:
                player = f"{self.prev_touch.player or ''}{self.prev_touch.team_id or ''}"   # use full player code with team in the dicts
                if player in self.goals_by_player:
                    self.goals_by_player[player] += 1
                else:
                    self.goals_by_player[player] = 1
            self.write_to_file()    # update persistent file instantly
            self.touches_since_last_update = 0
            self.prev_touch = touch # update previous touch before returning
            return
        self.prev_touch = touch # for all other touches, we don't need the previous touch, so update it already
        
        # throw-in handling
        if touch.throw_in:
            return
        
        # standard touch handling
        player = f"{touch.player or ''}{touch.team_id or ''}"   # use full player code with team in the dicts
        if touch.team_id in [0,1]:
            self.touches_by_team[touch.team_id] += 1
        if player in self.touches_by_player:
            self.touches_by_player[player] += 1
        else:
            self.touches_by_player[player] = 1
            
        # check if there were enough touches to trigger the periodic file-write
        if self.touches_since_last_update >= TOUCHES_TO_UPDATE:
            self.write_to_file()
            self.touches_since_last_update = 0
            
    def make_dict(self) -> dict:
        '''Creates the dict of fancy stats that will be written to the persistent json file.'''
        # quick calculation of ball posession
        total_player_touches = sum(self.touches_by_team)
        if total_player_touches > 0:
            posession = [self.touches_by_team[0]/total_player_touches, self.touches_by_team[1]/total_player_touches]
            
        fancy_stats = {
            "score": self.score,
            "goals_by_player": self.goals_by_player,
            "touches_by_player": self.touches_by_player,
            "ball_posession": {
                "team0": posession[0],
                "team1": posession[1]
            }
        }
        
        return fancy_stats
    
    def write_to_file(self):
        '''Writes the statistics to a file in a compact form.'''
        fancy_stats = self.make_dict()
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(fancy_stats, f)
        except Exception:
            pass
        
    def clear_file(self):
        '''Clears the persistent file to avoid old data being accessed.'''
        if os.path.exists(self.path):
            os.remove(self.path)