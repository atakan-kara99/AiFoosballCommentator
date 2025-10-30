from math import ceil
import time as timer

from vislib import kpi
from model.entities.touch import Error
from model.entities.state import State, GameState, GameStateConfig
from model.markov_model import MarkovModel
from resources.configs import state_config, constants


class ErrorHandler():
    '''This class keeps necessary log data and provides functions to reconstruct gameplay sequences
    after a series of error events.'''
    def __init__(self, model: MarkovModel, log_func):
        self.log_func = log_func
        # initialize knowledge parts
        self.knowledge_model = model
        self.log_func("ERROR_HANDLER - Initializing: Building sector to state lookup dict.")
        self.sector_to_states = self._build_sector_to_state_lookup_dict()
        
        # initialize basic variables
        self.last_valid_state : State = None
        self.last_valid_time : int = None
        self.reconstruct_required = False
        self.error_log : list[Error] = []
        self.log_func("ERROR_HANDLER - Initialization finished.")
        # push a kpi for the error log length and recovery time
        kpi("Current error log length", 0)
        kpi("Total reconstructions", 0)
        kpi("Latest error handling time", "None")
        kpi("Average error handling time", 0)
        kpi("Average confidence of reconstructed states", "None")
        self.errors_handled = 0
        self.reconstructed_states = 0
        self.total_time = 0
        self.total_confidence = 0
        
    def update_valid(self, state: State, time: int):
        '''Updates parameters after a valid state was reached without error interference.'''
        self.last_valid_state = state
        self.last_valid_time = time
        
    def log(self, error: Error):
        '''Logs an error and updates remembers necessity to reconstruct it.'''
        self.error_log.append(error)
        kpi("Current error log length", len(self.error_log))
        self.reconstruct_required = True
        self.log_func(f"ERROR_HANDLER - error logged (log contains {len(self.error_log)} elements now)")
        
    def reconstruct_history(self, current_known: tuple[State, int]) -> tuple[list[tuple[GameState, int]], list[float]]:
        '''This function is invoked whenever a touch was detected again after a sequence of errors.
        Based on the last known state and the now detected state, it reconstructs the most likely intermediate history.\n
        To do this, the error log is analysed. The set of all possible states is pruned by cutting all states that relate to
        sectors of the table where permanently visible players were positioned, as this likely means the ball was not in those
        sectors. Then, the most probable path on the pruned states is calculated using the knowledge Markov model to calculate 
        probabilities.\n
        The function returns a list of GameState objects that were part of the most probable history, combined with timestamps 
        for when these GameStates would have approximately appeared in the history.\n
        Additionally, it returns a list of confidence values estimating how likely the reconstruction is what really happened.'''
        self.log_func(f"ERROR_HANDLER - Ball detected after error sequence of length {len(self.error_log)}. Reconstruction of touch log initiated.")
        # track time
        start_time = timer.time()
        current_state, current_time = current_known
        not_involved = {}       # collect all positions of not involved players for each time step
        # fill the dict
        for e in self.error_log:
            for player, pos in e.non_involved_players_team0.items():
                if f"{player}0" in not_involved:
                    not_involved[f"{player}0"] += [pos]
                else:
                    not_involved[f"{player}0"] = [pos]
            for player, pos in e.non_involved_players_team1.items():
                if f"{player}1" in not_involved:
                    not_involved[f"{player}1"] += [pos]
                else:
                    not_involved[f"{player}1"] = [pos]
        
        # now, find all visible playing field sectors by searching the dict for permanently visible players
        visible_sectors = set()
        for player, posis in not_involved.items():
            if len(posis) == len(self.error_log):   # only players that were visible in every error message
                visible_sectors.update(self._to_sectors(player, posis))
        # convert sectors to set of respective states using the precomputed lookup dict
        exclude_states = set()
        for sector in visible_sectors:
            exclude_states.update(self.sector_to_states[sector])
        
        # do not exclude source and dest state accidently
        exclude_states -= {self.last_valid_state.to_index(), current_state.to_index()}
        self.log_func(f"ERROR_HANDLER - {visible_sectors} sectors were visible during error phase, leading to a pruned state set of {constants.TOTAL_GAME_STATES - len(exclude_states)} states.\nFinding shortest path now.")
        path, probs = self.knowledge_model.most_probable_path(self.last_valid_state.to_index(),
                                           current_state.to_index(),
                                           exclude_states=exclude_states)
        # based on the path and the known average transition times, calculate timestamps and confidence values for the states along the path
        if len(path) > 1:
            trans_times = [0]
            time = 0
            confidence_values = [1]
            current_confidence = 1
            for i in range(len(path) - 1):
                avg_time = self.knowledge_model.tm.transition_times_avg[path[i]][path[i+1]]
                time += (avg_time if avg_time != 0 else 1)  # if model has no average time learned for a transition, add a small value
                trans_times.append(time)
                # also calculate confidence values along path
                current_confidence *= probs[i+1]
                confidence_values.append(current_confidence)
            
            # scale transition times to total error timespan
            actual_timespan = current_time - self.last_valid_time
            predicted_timespan = trans_times[-1]
            timestamps = [ceil(t * actual_timespan/predicted_timespan) for t in trans_times]
        else:
            assert len(path)==1, "Should never get empty reconstructed path, but got path with length 0"
            timestamps = [current_time]
            confidence_values = [1]
        
        # reset variables for future error handling
        self.reconstruct_required = False
        self.error_log.clear()
        
        # return state sequence
        self.log_func(f"ERROR_HANDLER - Path of length {len(path)} reconstructed. Final confidence: {confidence_values[-1]}.")
        config = GameStateConfig(state_config.GameStatesConfig)
        state_reconstruction = [GameState(config, index=i) for i in path]
        
        # update kpis
        reconstruction_time = (timer.time() - start_time) * 1000
        self.errors_handled += 1
        self.reconstructed_states += len(path)
        kpi("Current error log length", 0)
        kpi("Total reconstructions", self.errors_handled)
        kpi("Latest error handling time", f"{round(reconstruction_time)} ms")
        self.total_time += reconstruction_time
        kpi("Average error handling time", f"{round(self.total_time / self.errors_handled)} ms")
        self.total_confidence += sum(confidence_values)
        kpi("Average confidence of reconstructed states", f"{(self.total_confidence / self.reconstructed_states):.2f}")
        
        return list(zip(state_reconstruction,timestamps)), confidence_values
    
    def _to_sectors(self, player:str, posis:list[tuple[int,int]]) -> list[tuple[str, int]]:
        '''Get all sectors that belong to a list of player positions, meaning all sectors that
        include at least one position from the given list.\n
        A sector is described by a tuple of (row+team:`str`, section:`int`).'''
        c = GameStateConfig(state_config.GameStatesConfig)
        sectors = []
        for (_,y) in posis:
            if player[-1] == "0":           # team 0
                pass
            elif player[-1] == "1":         # team 1, mirror coords
                y = 1 - y
            else:
                assert False, "player should always have '0' or '1' in last char"
            if y == 1:  # small fix in case position is 1, should never happen tho because player can't be exactly on the rim
                y = 0.9999
            # add the sector
            row = player[:-1] if player[:-1] == "GK" else player[-2] # for GK the row is GK otherwise its only one char (B,M,F)
            rowTeam = f"{row}{player[-1]}"  # dict key uses row and team, not exact player + team
            sectors.append((rowTeam, int(y * c.figure_sections[row])))
        return sectors
        
    def _build_sector_to_state_lookup_dict(self) -> dict:
        '''Precomputes a lookup dictionary to map sectors to all respective states.'''
        sect_to_state = {}
        c = GameStateConfig(state_config.GameStatesConfig)
        # first, add all keys with empty sets
        for team in [0,1]:
            for row in c.figure_sections.keys():
                for fig_sect in range(c.figure_sections[row]):
                    sect_to_state[(f"{row}{team}", fig_sect)] = set()
        # then, go through all states and add their index to the respective sets
        assert constants.TOTAL_GAME_STATES == self.knowledge_model.tm.size, "The total state amount should match the matrix size"
        for i in range(constants.TOTAL_GAME_STATES):
            s = GameState(c, index=i)
            if (s.row == None or s.sector == None or s.team_id == None):
                continue
            sect_to_state[(f"{s.row}{s.team_id}", s.sector)].add(i)
        return sect_to_state