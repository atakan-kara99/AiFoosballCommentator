import json
from typing import Any, Dict, List, Optional, Tuple, Union

from . import Ball
from . import Player
from .entity import Entity


class GameState(Entity):
    """
    Builds and represents the state of the game at a given frame.

    Each game state (or "digital sibling") is represented as a dictionary with the structure:
    
        {
            "id": int,                  # Unique identifier or frame timestamp (non-negative).
            "timestamp": Union[int, str],  # Timestamp in ms since 1970 (or as a formatted string).
            "game_state": {
                "ball": dict,         # Ball state as produced by Ball.get_json().
                "players": list       # List of player states as produced by Player.get_json().
            },
            "frame_no": int,            # Frame number.
            "frame_shape": Tuple[int, ...]  # Shape of the frame (e.g., (height, width, channels)).
        }
    """

    def __init__(
        self,
        ball_data: Ball,
        ps_data: List[Player],
        frame_time: Union[int, str],
        frame_no: int,
        shape: Tuple[int, ...],
        id: int = 0
    ) -> None:
        """
        Initialize a GameState instance with ball and player data.

        Args:
            ball_data (Ball): The ball data for the current frame.
            ps_data (List[Player]): A list of player data for the current frame.
            frame_time (Union[int, str]): The timestamp for the current frame (in ms since 1970 or formatted).
            frame_no (int): The frame number.
            shape (Tuple[int, ...]): The shape of the frame (e.g., (height, width, channels)).
            id (int, optional): The unique identifier for the current state. Must be non-negative. Defaults to 0.

        Raises:
            ValueError: If the provided id is negative.
        """
        self.set_id(id)
        self.timestamp: Union[int, str] = frame_time
        self.set_ball_data(ball_data)
        self.set_player_data(ps_data)
        self.frame_shape: Tuple[int, ...] = shape
        self.frame_no: int = frame_no

    def set_id(self, id: int = 0) -> None:
        """
        Set the identifier for the game state.

        Args:
            id (int, optional): The identifier value (must be >= 0). Defaults to 0.

        Raises:
            ValueError: If id is negative.
        """
        if id < 0:
            raise ValueError(f"Id value must be greater or equal to 0, got {id}")
        self.id: int = id

    def set_ball_data(self, ball: Optional[Ball] = None) -> None:
        """
        Set the ball data for the game state.

        Args:
            ball (Optional[Ball]): The Ball object representing the current ball state.
        """
        self.ball: Optional[Ball] = ball

    def set_player_data(self, ps_data: Optional[List[Player]] = None) -> None:
        """
        Set the list of player data for the game state.

        Args:
            ps_data (Optional[List[Player]]): A list of Player objects for the current frame.
        """
        self.players: Optional[List[Player]] = ps_data
        # Optionally, enforce a specific number of players (e.g., 22).
        # if ps_data is not None and len(ps_data) != 22:
        #     raise ValueError(f"Expected exactly 22 players, but got {len(ps_data)}.")

    def append_player(self, player: Optional[Player] = None) -> None:
        """
        Append a single player to the current list of players.

        Args:
            player (Optional[Player]): A Player object to add.
        """
        if player is None:
            return
        if self.players is None:
            self.players = [player]
        else:
            self.players.append(player)

    def next_state(self, ball_data: Ball, ps_data: List[Player]) -> "GameState":
        """
        Create a new game state for the next frame with updated ball and player data,
        automatically computing delta values and incrementing the state id.

        The new ball's delta is set using the current state's ball. For each player,
        the corresponding previous player is identified (by matching player_id and team_id)
        and the delta is computed.

        Args:
            ball_data (Ball): The new ball data.
            ps_data (List[Player]): The new list of player data.

        Returns:
            GameState: A new GameState instance representing the next state.
        """
        # Increment the state id.
        inc_id: int = self.id + 1

        # Create a new ball dictionary from the new ball data.
        new_ball_dict: Dict[str, Any] = ball_data.get_json()
        # Set the previous ball data as delta.
        new_ball_dict["delta"] = self.ball
        # Create a new Ball instance with delta values.
        ball_props: Ball = Ball(**new_ball_dict)

        # Process each new player by matching with the previous state.
        new_players: List[Player] = []
        for tmp_player in ps_data:
            for prev_player in self.players or []:
                # Identify the same player by player_id and team_id.
                if prev_player.player_id == tmp_player.player_id and prev_player.team_id == tmp_player.team_id:
                    # Create a new Player instance from the new data.
                    p: Player = Player(**tmp_player.get_json())
                    # Set delta using the corresponding previous player.
                    p.set_delta(prev_player)
                    new_players.append(p)
                    break  # Exit inner loop once match is found.

        # Return a new GameState with the updated data, preserving the timestamp, frame number, and shape.
        return GameState(
            ball_data=ball_props,
            ps_data=new_players,
            frame_time=self.timestamp,
            frame_no=self.frame_no,
            shape=self.frame_shape,
            id=inc_id
        )

    def __str__(self) -> str:
        """
        Return a prettified JSON string representation of the game state.

        Returns:
            str: The JSON representation of the game state.
        """
        return json.dumps(self.get_json(), indent=self.INDENT)

    def get_json(self) -> Dict[str, Any]:
        """
        Convert the game state into a JSON-serializable dictionary.

        Returns:
            Dict[str, Any]: The dictionary representing the game state.
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "frame_no": self.frame_no,
            "frame_shape": self.frame_shape,
            "game_state": {
                "ball": self.ball.get_json() if self.ball else None,
                "players": [p.get_json() for p in self.players] if self.players else []
            }
        }