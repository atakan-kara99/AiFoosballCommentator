import math
from typing import Optional, Tuple, Union, Dict, Any, List

from cv.entities import GameState
from cv.global_constants import CF_PADDING_X, CF_PADDING_Y, GOAL_WAIT_FRAMES
from cv.utils import Utils
from cv.foosball_enums import PlayerID


class TouchLogger:
    """
    Responsible for logging ball touch events and errors during the game.
    Processes game state updates to detect ball touches caused by players or walls,
    and generates corresponding Touch or Error events.
    """
    def __init__(self) -> None:
        # Initialize state variables
        self.frames_ball_not_seen: int = 0
        self.ball_in_game: bool = False
        self.prev_error_list0: Optional[Dict[Any, Any]] = None
        self.prev_error_list1: Optional[Dict[Any, Any]] = None

        # set tresholds
        self.velo_treshold = 0.2
        self.dir_threshold = 5
        self.foot_dist = 60
        self.player_x_tol = 40
        self.player_y_tol = 20
        self.wall_tol = 20
        self.near_threshold = 1.5
        self.dist_weight = 1.5 # factor of previouse compared to current

    def get_config_json(self) -> Dict[str, Union[float, int]]:
        """
        Returns the configuration parameters as a dictionary.
        """
        return {
            "velo_treshold": self.velo_treshold,
            "dir_threshold": self.dir_threshold,
            "foot_dist": self.foot_dist,
            "player_x_tol": self.player_x_tol,
            "player_y_tol": self.player_y_tol,
            "wall_tol": self.wall_tol,
            "near_threshold": self.near_threshold,
            "dist_weight": self.dist_weight
        }

    def check_touch(self, gamestate: GameState, event: str, verbose: bool = False, debug: bool = False) -> Optional[Union['Touch', 'Error']]:
        """
        Evaluates the game state and event string to determine if a ball touch occurred.
        Checks both event-based triggers (goal/throw in) and ball motion (velocity/direction).
        
        Args:
            gamestate (GameState): The current game state.
            event (str): A string event that may indicate a goal or throw in.
            verbose (bool): If True, prints additional details.
            debug (bool): If True, prints debug information.

        Returns:
            Optional[Touch | Error]: A Touch or Error event if detected, otherwise None.
        """
        goal, throw_in, team_id = self.check_event(event)
        self.check_in_game(goal, throw_in)
        if goal or throw_in:
            if debug:
                print("Touch triggered by event.")
            return self.generate_touch(gamestate=gamestate, goal=goal, throw_in=throw_in, team_id=team_id)
        if gamestate.ball:
            # Reset previous error data when the ball is visible
            self.prev_error_list0 = None
            self.prev_error_list1 = None
            if self.ball_in_game:
                self.frames_ball_not_seen = 0
                delta = gamestate.ball.delta
                if delta:
                    if delta.get("velocity") is not None:
                        if abs(delta["velocity"]) > self.velo_treshold:
                            if debug:
                                print(f"Touch triggered by velocity: {abs(delta['velocity'])}")
                            return self.process_touch(gamestate, goal, throw_in, team_id, verbose=verbose, debug=debug)
                    if delta.get("direction") is not None:
                        if abs(delta["direction"]) > self.dir_threshold:
                            if debug:
                                print(f"Touch triggered by direction: {abs(delta['direction'])}")
                            return self.process_touch(gamestate, goal, throw_in, team_id, verbose=verbose, debug=debug)
            return None
        else:
            self.frames_ball_not_seen += 1
            if self.frames_ball_not_seen > GOAL_WAIT_FRAMES and self.ball_in_game:
                return self.generate_error(gamestate, debug=debug)
            else:
                return None

    def check_in_game(self, goal: bool, throw_in: bool) -> None:
        """
        Updates the ball-in-game status based on goal or throw in events.

        Args:
            goal (bool): True if a goal event occurred.
            throw_in (bool): True if a throw in event occurred.
        """
        if self.ball_in_game and goal:
            self.ball_in_game = False
        elif not self.ball_in_game and throw_in:
            self.ball_in_game = True

    def process_touch(self, gamestate: GameState, goal: bool, throw_in: bool, team_id: Optional[int],
                      verbose: bool = False, debug: bool = False) -> Optional['Touch']:
        """
        Processes a touch by verifying if any player's position aligns with the ball's.
        If a valid player is detected, generates and returns a Touch event.
        
        Args:
            gamestate (GameState): The current game state.
            goal (bool): Indicates if the event is a goal.
            throw_in (bool): Indicates if the event is a throw in.
            team_id (Optional[int]): Team identifier.
            verbose (bool): If True, prints additional details.
            debug (bool): If True, prints debug information.
            
        Returns:
            Optional[Touch]: A Touch event if a player is found, else None.
        """
        player, team_id = self.check_player(gamestate, verbose=verbose, debug=debug)
        if player is not None:
            return self.generate_touch(gamestate=gamestate, player=player, team_id=team_id, goal=goal, throw_in=throw_in)
        else:
            return None

    def generate_touch(self, gamestate: GameState, team_id: Optional[int],
                       player: Optional[str] = None, goal: Optional[bool] = None,
                       throw_in: Optional[bool] = None) -> 'Touch':
        """
        Creates a Touch event using the current ball and game state information.
        
        Args:
            gamestate (GameState): The current game state.
            team_id (Optional[int]): The team identifier.
            player (Optional[str]): The player responsible for the touch.
            goal (Optional[bool]): True if the touch event is due to a goal.
            throw_in (Optional[bool]): True if the touch event is due to a throw in.
            
        Returns:
            Touch: The generated touch event.
        """
        if gamestate.ball:
            ball_pos = Utils.coord_to_perc((gamestate.ball.x, gamestate.ball.y), gamestate.frame_shape)
            ball_speed = gamestate.ball.velocity
            ball_direction = gamestate.ball.direction
        else:
            ball_pos, ball_speed, ball_direction = None, None, None

        return Touch(
            player=player,
            team_id=team_id,
            game_time=gamestate.timestamp,
            frame_no=gamestate.frame_no,
            ball_pos=ball_pos,
            speed=ball_speed,
            direction=ball_direction,
            goal=goal,
            throw_in=throw_in
        )

    def generate_error(self, gamestate: GameState, debug: bool = False) -> Optional['Error']:
        """
        Generates an Error event when the players' positions differ from the previous state.
        
        Args:
            gamestate (GameState): The current game state.
            debug (bool): If True, prints debug information.
            
        Returns:
            Optional[Error]: The Error event if a difference is detected, else None.
        """
        if gamestate:
            list_0: Dict[Any, Any] = {}
            list_1: Dict[Any, Any] = {}
            for p in gamestate.players:
                if p.team_id == 0:
                    list_0[p.player_id] = Utils.coord_to_perc((p.x, p.y), gamestate.frame_shape)
                elif p.team_id == 1:
                    list_1[p.player_id] = Utils.coord_to_perc((p.x, p.y), gamestate.frame_shape)
        else:
            list_0, list_1 = {}, {}

        # Generate error only if player positions have changed from the previous error state.
        if list_0 != self.prev_error_list0 or list_1 != self.prev_error_list1:
            if debug:
                print("Player list difference detected for error logging.")
            error_event = Error(
                non_involved_players_team0=list_0,
                non_involved_players_team1=list_1,
                game_time=gamestate.timestamp,
                frame_no=gamestate.frame_no,
            )
            self.prev_error_list0 = list_0
            self.prev_error_list1 = list_1
            return error_event
        else:
            return None

    def check_player(self, gamestate: GameState, verbose: bool = False, debug: bool = False) -> Tuple[Optional[str], Optional[int]]:
        """
        Checks if any player's coordinates (foot position) are close enough to the ball
        to consider that player as having touched the ball. If no player is within reach,
        then checks if the ball is near any wall.

        Args:
            gamestate (GameState): The current game state.
            verbose (bool): If True, prints additional details.
            debug (bool): If True, prints debug information.
            
        Returns:
            Tuple[Optional[str], Optional[int]]:
                - Player ID (or a wall indicator) if in contact, else None.
                - Team ID if a player is detected, otherwise None.
        """
        ball = gamestate.ball
        ball_dx, ball_dy = ball.delta.get('x', 0), ball.delta.get('y', 0)
        ball_speed = ball.velocity

        # Ensure a minimum ball speed to avoid division issues
        if int(ball_speed) < 1:
            ball_speed = 1

        def get_distance(B_x: float, B_y: float, F_x: float, F_y: float) -> float:
            """
            Computes the Euclidean distance from the ball (with applied tolerance offsets)
            to a player's foot position.
            
            Args:
                B_x (float): X-coordinate of the ball.
                B_y (float): Y-coordinate of the ball.
                F_x (float): X-coordinate of the player's foot.
                F_y (float): Y-coordinate of the player's foot.
                
            Returns:
                float: The computed distance.
            """
            return math.sqrt((B_x - F_x - self.player_x_tol) ** 2 + (B_y - F_y - self.player_y_tol) ** 2)

        res_dists: List[Tuple[str, int, float, float, float]] = []

        # Evaluate proximity for each player
        if gamestate.players:
            for player in gamestate.players:
                if all(hasattr(player, attr) for attr in ['x', 'y', 'foot_x', 'foot_y']):
                    if player.x is not None and player.y is not None and player.foot_x is not None and player.foot_y is not None:
                        # Validate foot position based on rotation threshold
                        if abs(player.x - player.foot_x) <= self.foot_dist:
                            dist = get_distance(ball.x, ball.y, player.foot_x, player.foot_y)
                            prev_dist = get_distance(ball.x - ball_dx, ball.y - ball_dy, player.foot_x, player.foot_y)
                            avg_dist = (dist + self.dist_weight * prev_dist) / (self.dist_weight + 1.0)
                            res_dists.append((player.player_id, player.team_id, dist, prev_dist, avg_dist))

        # If no players were eligible, proceed to wall check.
        if not res_dists:
            return None, None

        # Identify the closest player based on weighted average distance
        sorted_res = sorted(res_dists, key=lambda x: x[4])
        closest_player = sorted_res[0]
        # Calculate threshold distance for player contact
        near_threshold_distance = self.near_threshold * ball.r * ball_speed
        in_reach = closest_player[4] <= near_threshold_distance
        if debug:
            print(f"Player {closest_player[0]} (Team {closest_player[1]}) avg distance: {closest_player[4]} vs threshold: {near_threshold_distance} => in reach: {in_reach}")
        if in_reach:
            if debug:
                print(f"Ball touched by player: {closest_player[0]}")
            return closest_player[0], closest_player[1]

        # If no player is close enough, check for wall contact based on frame dimensions.
        fh, fw, _ = gamestate.frame_shape
        if CF_PADDING_X + self.wall_tol * ball.velocity > ball.x:
            if verbose:
                print("Ball near left wall.")
            return PlayerID.WALL.value, None
        elif fw - CF_PADDING_X - self.wall_tol * ball.velocity < ball.x:
            if verbose:
                print("Ball near right wall.")
            return PlayerID.WALL.value, None
        elif CF_PADDING_Y + self.wall_tol * ball.velocity > ball.y:
            if verbose:
                print("Ball near top wall.")
            return PlayerID.WALL.value, None
        elif fh - CF_PADDING_Y - self.wall_tol * ball.velocity < ball.y:
            if verbose:
                print("Ball near bottom wall.")
            return PlayerID.WALL.value, None

        return None, None

    def check_event(self, event: str) -> Tuple[bool, bool, Optional[int]]:
        """
        Parses the event string to determine if it indicates a goal or throw in,
        and assigns the corresponding team identifier.
        
        Args:
            event (str): The event string.
            
        Returns:
            Tuple[bool, bool, Optional[int]]:
                - goal (bool): True if the event indicates a goal.
                - throw_in (bool): True if the event indicates a throw in.
                - team_id (Optional[int]): The team identifier (0 or 1) if applicable.
        """
        event_lower = event.lower()
        if 'goal' in event_lower:
            goal = True
            throw_in = False
            team_id = 0 if 'left' in event_lower else 1
        elif 'throw in' in event_lower:
            goal = False
            throw_in = True
            team_id = 1 if 'top' in event_lower else 0
        else:
            goal = False
            throw_in = False
            team_id = None
        return goal, throw_in, team_id


class Touch:
    """
    Represents a ball touch event in the game.
    """
    def __init__(self, team_id: Optional[int], game_time: int, frame_no: int,
                 ball_pos: Optional[Any], goal: bool, throw_in: bool,
                 player: Optional[str] = None, speed: Optional[float] = None, direction: Optional[int] = None) -> None:
        self.data = {
            "type": "touch",
            "player": player,
            "team_id": team_id,
            "time": game_time,
            "frame_no": frame_no,
            "position": ball_pos,
            "speed": speed,
            "direction": direction,
            "goal": goal,
            "throw_in": throw_in,
        }
    
    def get_json(self) -> Dict[str, Any]:
        """
        Returns the touch event data as a dictionary.
        """
        return self.data


class Error:
    """
    Represents an error event triggered when there is a discrepancy in player positions.
    """
    def __init__(self, non_involved_players_team0: Union[Dict[Any, Any], List[Any]],
                 non_involved_players_team1: Union[Dict[Any, Any], List[Any]],
                 game_time: int, frame_no: int) -> None:
        self.data = {
            "type": "error",
            "time": game_time,
            "frame_no": frame_no,
            "non_involved_players_team0": non_involved_players_team0,
            "non_involved_players_team1": non_involved_players_team1,
        }

    def get_json(self) -> Dict[str, Any]:
        """
        Returns the error event data as a dictionary.
        """
        return self.data
