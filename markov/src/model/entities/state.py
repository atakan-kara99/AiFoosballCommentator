'''This class models a state in the Markov model. In its simplest form, states do not need to hold any information.
The only requirement is that there is a fixed order on the states and each state can return its index in this order.'''
import numpy as np
# avoid cyclic imports at runtime, but still use typing
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .touch import Touch

class State:
    def __init__(self, index:int=-1):
        self.index = index
        
    def to_index(self):
        '''Return the index of the state object in the total ordering of states.'''
        return self.index
    
    def __eq__(self, value):
        if not issubclass(type(value), State):
            return False
        return self.index == value.index
    
    def __hash__(self):
        return hash(self.index)


class GameStateConfig:
    def __init__(self, config: dict):
        """
        Initialize the game configuration from a dictionary.
        :param config: Dictionary containing the game parameters.
        """
        self.wall_sections = config['WALL_SECTIONS']
        self.wall_angles = config['WALL_ANGLES']
        self.figure_sections = config['FIGURE_SECTIONS']
        self.figure_angles = config['FIGURE_ANGLES']
        self.speeds = config['SPEEDS']

        # Precompute totals for validation and index mapping
        self.states_per_wall_section = self.wall_angles * (len(self.speeds) - 1) # -1 because no "none" speed for wall touches
        self.wall_states = (
            (self.wall_sections['X'] + self.wall_sections['Y']) * self.states_per_wall_section    
        ) * 2   # 2 horizontal and vertical walls each
        
        self.team_states = sum(
            self.figure_sections[row]
            * (self.figure_angles[row]
                * (len(self.speeds) - 1)
              + 1)  # plus 1 for each section for None angle+speed state
            for row in self.figure_sections
        ) + 2 # 2 special events (throw-in, goal)
        
        self.total_states = self.wall_states + 2 * self.team_states


class GameState(State):
    def __init__(self, config: GameStateConfig, index: int = -1):
        """
        Initialize a GameState object using the provided configuration.
        :param config: GameStateConfig object containing game parameters.
        :param index: Index of the state, -1 for default initialization.
        """
        super().__init__(index)
        self.config = config
        if index == -1:
            self.team_id: int = None
            self.row: str = None
            self.sector: int = None
            self.angle: int = None
            self.speed: str = None
            self.goal = False
            self.throw_in = False
        else:
            self._init_from_index(index)

    def to_index(self) -> int:
        '''Return the index of the state object in the total ordering of states.'''
        # instantly return the index if calculated before
        if self.index != -1:
            return self.index

        config = self.config
        speeds = list(self.config.speeds.keys())[1:]    # cut off none speed

        # Handle goals and throw-ins
        if self.goal:
            self.index = (1 + self.team_id) * config.team_states - 1  # adjust by 1 since index starts at 0
            return self.index
        if self.throw_in:
            self.index = (1 + self.team_id) * config.team_states - 2  # one more here to get to throw-ins
            return self.index
        
        if self.row == "WALL":
            index = 2 * config.team_states  # baseline index for wall touches
            index += self.sector * config.states_per_wall_section   # sector offset
            index += self.angle * len(speeds)   # angle offset
            index += speeds.index(self.speed)   # speed offset
            
            self.index = index
        else: # player state
            index = self.team_id * config.team_states   # offset for team as base index
            # offset per row before self.row
            for row in ['GK', 'B', 'M', 'F']:
                if row == self.row: # do not offset further when row is found
                    break
                index += config.figure_sections[row] * (config.figure_angles[row] * len(speeds) + 1) # +1 for no angle+speed state

            # sector offset
            index += self.sector * (config.figure_angles[self.row] * len(speeds) + 1) # same +1 again
            # angle offset
            if self.angle != None:
                index += self.angle * len(speeds) + 1   # same +1 still
                # speed offset
                index += speeds.index(self.speed)
            
            self.index = index
        return self.index
    
    def _init_from_index(self, index: int):
        '''Initialize a GameState using an index. Sets all instance variables according to the corresponding state attributes.\n
        Since `self.index` is set in `__init__()` already, it does not need to be set itself again.'''
        config = self.config
        speeds = list(self.config.speeds.keys())[1:]    # cut off none speed

        if index >= config.total_states or index < 0:
            raise ValueError("Invalid index, not in range [0, total_states)")
        
        # Determine team (or wall)
        if index >= config.team_states * 2:   # wall related state
            self.team_id, self.row, self.goal, self.throw_in = None, "WALL", False, False
            # transform index to interval [0, total_wall_states)
            index -= config.team_states * 2
            # determine sector, then transform index to [0, states_per_wall_section)
            self.sector = index // config.states_per_wall_section
            index %= config.states_per_wall_section
            # lastly, determine angle and speed
            self.angle = index // len(speeds)
            self.speed = speeds[index % len(speeds)]
            return
        elif index >= config.team_states:  # team 1 related state
            self.team_id = 1
            # adjust index to perform the same calculation as for team 0 related indices for the remaining attributes
            index -= config.team_states
        else:    # team 0 related state
            self.team_id = 0
        # this is only reached for team related states with index in range [0, total_team_states)
        # Still need to set: self.row, self.sector, self.angle, self.speed, self.goal, self.throw_in
        if index == config.team_states - 1: # goal check
            self.row, self.sector, self.angle, self.speed, self.goal, self.throw_in = None, None, None, None, True, False
            return
        if index == config.team_states - 2: # throw-in check
            self.row, self.sector, self.angle, self.speed, self.goal, self.throw_in = None, None, None, None, False, True
            return
        # if it wasn't a special event, calculate remaining attributes
        self.goal, self.throw_in = False, False
        for row in ['GK', 'B', 'M', 'F']:
            row_size = config.figure_sections[row] * (config.figure_angles[row] * len(speeds) + 1) # +1 for no speed/angle state
            if index < row_size:    # row found, index in interval [0, row_size)
                self.row = row
                # determine sector, then transform index to [0, states_per_sector)
                self.sector = index // (config.figure_angles[row] * len(speeds) + 1)    # same +1 as before
                index %= (config.figure_angles[row] * len(speeds) + 1)  # still same +1
                # check for no angle/speed state
                if index == 0:
                    self.angle = None
                    self.speed = None
                    return
                # otherwise, offset by one for the last calculations
                index -= 1
                self.angle = index // len(speeds)
                self.speed = speeds[index % len(speeds)]
                return
            # row not found, adjust index for next row check
            index -= row_size
        # this should never be reached
        raise ValueError("Invalid index")
        
    #--- Methods for calculating the state attributes from a touch ---#

    def calculate_figure_angle_section(self, team_id: int, direction: float, row: str) -> int:
        """Calculate the angle section for a figure touch."""
        # Adjust the direction relative to the team's orientation
        adjusted_direction = (direction + (180 if team_id == 1 else 0)) % 360
        angle_increment = 360 / self.config.figure_angles[row]
        # Shift the breakpoints by half an angle increment to ensure correct categorization
        adjusted_direction = (adjusted_direction + angle_increment / 2) % 360
        angle_index = int(adjusted_direction // angle_increment)
        return angle_index

    def calculate_wall_and_angle_section(self, x: float, y: float, direction: float) -> tuple[int,int]:
        """Calculate the wall sector based on the touch position.\n
            The numbering starts in the bottom-left corner on the Y-wall (x = 0, y = 0)
            and continues clockwise around the field.\n
            Additionally, calculate the angle section for that wall."""
        wall_sections_x = self.config.wall_sections['X']
        wall_sections_y = self.config.wall_sections['Y']

        if direction is None:
            direction = 0
                
        # sector size in degree used for angle sector calculation
        sector_size = 180 / self.config.wall_angles
        # for each close enough wall, double check if the direction fits that wall before returning the section
        if x < 0.1:  # Left Y-wall
            if direction < 90:  # first angle option: (0,90)
                relative_direction = 90 - direction  # Reverse direction to (0,90]
                # Sections increase upwards along the left wall
                return int((1 - y) * wall_sections_y), int(relative_direction // sector_size)
            if direction > 270: # second angle option: (270,360)
                relative_direction = 450 - direction # Reverse+Shift direction to (90,180)
                # Sections increase upwards along the left wall
                return int((1 - y) * wall_sections_y), int(relative_direction // sector_size)
            return int((1 - y) * wall_sections_y), int(self.config.wall_angles / 2)
        if x > 0.9:  # Right Y-wall
            if direction > 90 and direction < 270:
                relative_direction = 270 - direction  # Shift direction to (0,180)
                # Sections increase downwards along the right wall
                return wall_sections_y + wall_sections_x + int(y * wall_sections_y), int(relative_direction // sector_size)
            return wall_sections_y + wall_sections_x + int(y * wall_sections_y), int(self.config.wall_angles / 2)
        if y < 0.1:  # Top X-wall
            if direction > 0 and direction < 180:
                relative_direction = 180 - direction    # Reverse direction to (0,180)
                # Sections increase rightwards along the top wall
                return wall_sections_y + int(x * wall_sections_x), int(relative_direction // sector_size)
            return wall_sections_y + int(x * wall_sections_x), int(self.config.wall_angles / 2)
        if y > 0.9:  # Bottom X-wall
            if direction > 180:
                relative_direction = 360 - direction  # Shift direction to (0,180)
                # Sections increase leftwards along the bottom wall
                return wall_sections_y + wall_sections_x + wall_sections_y + int((1 - x) * wall_sections_x), int(relative_direction // sector_size)
            return wall_sections_y + wall_sections_x + wall_sections_y + int((1 - x) * wall_sections_x), int(self.config.wall_angles / 2)

        # Wall coudn't be determined, send middle section     
        print("Couldn't determine wall section for wall touch. Returning (0, 0)")
        return  0, 0

    def calculate_figure_section(self, y: float, row: str) -> int:
        """Calculate the sector for a touch by a figure of row `row` based on `y` position."""
        if y >= 1:  # touches should not happen on or behind the edges, but incase they do, adjust to max
            y = 0.9999
        if y < 0:
            y = 0
        return int(y * self.config.figure_sections[row])

    def determine_speed_category(self, speed: float) -> str:
        """Determine the speed category based on configured thresholds."""
        # adjust None to 0, should not happen for regular touches though
        if speed is None:
            speed = 0
        
        for category, max_speed in self.config.speeds.items():
            if speed <= max_speed:
                return category
        raise ValueError("Speed does not fit any category.")
    
    def to_normalized_vector(self) -> np.ndarray:
        """Convert the state into a normalized vector for use in the autoencoder."""
        x_wall_sections = self.config.wall_sections['X']
        y_wall_sections = self.config.wall_sections['Y']

        if self.row == "WALL":
            vec = np.array([
                0, 
                0, # 1-Hot encoding for team
                1 if self.sector < y_wall_sections or (self.sector > y_wall_sections + x_wall_sections and self.sector < 2 * y_wall_sections + x_wall_sections) else 0, # 1-Hot encoding for wall type short (y)
                1 if self.sector >= x_wall_sections + 2 * y_wall_sections or (self.sector >= y_wall_sections and self.sector < x_wall_sections + y_wall_sections) else 0, # 1-Hot encoding for wall type long (x)
                0, 0, 0, 0, # 1-Hot encoding for row
                self.normalize_wall_sector(), # sector value normalized
                self.angle / self.config.wall_angles,
                list(self.config.speeds).index(self.speed) / len(self.config.speeds) if self.speed in self.config.speeds else 0, # speed value normalized
                int (self.goal) * 3,  # goal value normalized
                int (self.throw_in) * 1.5 # throw_in value normalized
            ])
        else:
            vec = np.array([
                1 if self.team_id == 0 else 0, # 1-Hot encoding for team 0
                1 if self.team_id == 1 else 0, # 1-Hot encoding for team 1
                0, 0, # 1-Hot encoding for wall type
                1 if self.row == 'GK' else 0, # 1-Hot encoding for row
                1 if self.row == 'B' else 0, # 1-Hot encoding for row
                1 if self.row == 'M' else 0, # 1-Hot encoding for row
                1 if self.row == 'F' else 0, # 1-Hot encoding for row
                -1 if self.sector == None  else self.sector / self.config.figure_sections[self.row if self.row != '' else 'M'], # sector value normalized
                -1 if self.angle == None else self.angle / self.config.figure_angles[self.row if self.row != '' else 'M'],
                -1 if self.speed == None else list(self.config.speeds).index(self.speed) / len(self.config.speeds) if self.speed in self.config.speeds else 0,
                int (self.goal) * 5,
                int (self.throw_in) * 2.5
            ])
        return vec
    
    def normalize_wall_sector(self) -> float:
        """Normalize the wall sector value."""
        # depending on the wall we need to reduce self.sector by the number of sectors in previous walls
        if self.sector < self.config.wall_sections['Y']:
            return self.sector / self.config.wall_sections['Y']
        elif self.sector < self.config.wall_sections['Y'] + self.config.wall_sections['X']:
            return (self.sector - self.config.wall_sections['Y']) / self.config.wall_sections['X']
        elif self.sector < 2 * self.config.wall_sections['Y'] + self.config.wall_sections['X']:
            return (self.sector - self.config.wall_sections['Y'] - self.config.wall_sections['X']) / self.config.wall_sections['Y']
        else:
            return (self.sector - 2 * self.config.wall_sections['Y'] - self.config.wall_sections['X']) / self.config.wall_sections['X']

    @staticmethod
    def player_to_row(player: str) -> str:
        """Convert a player code to a row code."""
        # reconstruct fail cases, those should never happen though if CV sends correct data
        if player == "K":
            return "GK"
        if player == None or player == "":
            return "M"
        if player == "GK":
            return "GK"
        return player[-1]

    @staticmethod
    def from_touch(touch: "Touch", config: dict) -> "GameState":
        """Convert a Touch event into a GameState object."""
        state = GameState(GameStateConfig(config))
        state.team_id = touch.team_id

        # Special cases
        if touch.goal:
            # We need to invert team id for goal state to represent the scoring team
            state.team_id = 1 - state.team_id
            state.goal = True
            return state
        if touch.throw_in:
            state.throw_in = True
            return state

        # Divide into wall or player touches
        if touch.player == "WALL":
            state.row = "WALL"
            state.sector, state.angle = state.calculate_wall_and_angle_section(
                touch.position[0], touch.position[1], touch.direction)
            # Determine speed
            speed = state.determine_speed_category(touch.speed)
            if speed == None:
                # we don't allow None speed for wall touches, so if the ball should ever be this slow for a wall touch..
                # .. set it to the next higher speed category
                state.speed = list(config["SPEEDS"].keys())[1]
            else:
                state.speed = speed
        else:   # player touch
            state.row = GameState.player_to_row(touch.player)
            state.sector = state.calculate_figure_section(
                touch.position[1], state.row)
            state.speed = state.determine_speed_category(touch.speed)
            if state.speed == None: # for none speed, there must be no angle
                state.angle = None
            else:
                state.angle = state.calculate_figure_angle_section(
                    touch.team_id, touch.direction, state.row)
            
        return state
    
    def __str__(self) -> str:
        if self.goal:
            return f"Goal for T{self.team_id}"
        if self.throw_in:
            return f"Throw-in by T{self.team_id}"
        
        return (
            f"T{self.team_id} {self.row}\nS:{self.sector} a:{self.angle} s:{self.speed}"
        )
        
    def to_graph_string(self):
        '''Alternative, shorter string conversion for the graph visualisation.'''
        if self.goal:
            return f"Goal for T{self.team_id}"
        if self.throw_in:
            return f"Throw-in by T{self.team_id}"
        
        if self.team_id != None:
            return f"T{self.team_id} {self.row}"
        
        if self.row != "WALL":
            raise Exception(str(self))
        return f"{self.row}"
        