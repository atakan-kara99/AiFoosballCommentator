import json
from typing import Any, Dict, List, Optional, Union

from .entity import Entity
from cv.global_constants import PLAYER_RODS
from cv.foosball_enums import player_enum_values, team_enum_values, PlayerID


class Player(Entity):
    """
    Represents a player in a single game state.

    Attributes:
        player_id (str): Unique identifier for the player (must be one of the allowed values in player_enum_values).
        team_id (Optional[int]): Identifier for the player's team (typically 0 or 1), or None for special players (e.g., WALL).
        x (float): Normalized x-coordinate (between 0 and 1).
        y (float): Normalized y-coordinate (between 0 and 1).
        foot_x (float): Normalized x-coordinate for the player's foot.
        foot_y (float): Normalized y-coordinate for the player's foot.
        delta (Optional[Dict[str, Union[float, int]]]): Absolute differences in coordinates relative to a previous state.
    """

    def __init__(
        self,
        player_id: Optional[str] = None,
        team_id: Optional[int] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        foot_x: Optional[float] = None,
        foot_y: Optional[float] = None,
        prev_player: Optional["Player"] = None,
    ) -> None:
        """
        Initialize a Player instance with the provided parameters.

        Args:
            player_id (Optional[str]): Identifier for the player; must be in player_enum_values.
            team_id (Optional[int]): Team identifier (e.g., 0 or 1) or None for special cases.
            x (Optional[float]): Normalized x-coordinate (0 to 1).
            y (Optional[float]): Normalized y-coordinate (0 to 1).
            foot_x (Optional[float]): Normalized x-coordinate for the foot.
            foot_y (Optional[float]): Normalized y-coordinate for the foot.
            prev_player (Optional[Player]): Previous state for delta calculation.
        """
        self.set_player_id(player_id)
        self.set_team_id(team_id)
        self.set_xy(x, y)
        self.set_foot_xy(foot_x, foot_y)
        self.set_delta(prev_player)

    def set_player_id(self, pid: Optional[str]) -> None:
        """
        Set the player's identifier.

        Args:
            pid (Optional[str]): The player ID; must be one of the allowed enum values.

        Raises:
            ValueError: If pid is None or not in player_enum_values.
        """
        if pid is None or pid not in player_enum_values:
            raise ValueError(f"[ PlayerError ] PlayerID must be one of {player_enum_values} but got {pid}")
        self.player_id = pid

    def set_team_id(self, tid: Optional[int]) -> None:
        """
        Set the player's team identifier.

        Args:
            tid (Optional[int]): The team ID; must be in team_enum_values or None if the player is special (e.g., WALL).

        Raises:
            ValueError: If tid is not valid.
        """
        if (tid is None and self.player_id == PlayerID.WALL.value) or (tid is not None and tid in team_enum_values):
            self.team_id = tid
        else:
            raise ValueError(f"[ PlayerError ] TeamID must be one of {team_enum_values} or None for WALL, but got {tid}")

    def set_xy(self, x: Optional[float], y: Optional[float]) -> None:
        """
        Set the player's x and y coordinates.

        Args:
            x (Optional[float]): Normalized x-coordinate.
            y (Optional[float]): Normalized y-coordinate.

        Raises:
            ValueError: If either x or y is None.
        """
        if x is None or y is None:
            raise ValueError("[ PlayerError ] Both x and y coordinates must be provided and not None.")
        self.x = x
        self.y = y

    def set_foot_xy(self, x: Optional[float], y: Optional[float]) -> None:
        """
        Set the player's foot coordinates.

        Args:
            x (Optional[float]): Normalized x-coordinate for the foot.
            y (Optional[float]): Normalized y-coordinate for the foot.

        Raises:
            ValueError: If either foot coordinate is None.
        """
        if x is None or y is None:
            raise ValueError("[ PlayerError ] Both foot x and foot y coordinates must be provided and not None.")
        self.foot_x = x
        self.foot_y = y

    def set_delta(self, prev_player: Optional["Player"]) -> None:
        """
        Calculate and set the delta values relative to a previous player state.

        Delta values are the absolute differences between the current and previous x and y coordinates.

        Args:
            prev_player (Optional[Player]): The previous state of the player. If None, delta is set to None.
        """
        if prev_player is None:
            self.delta = None
            return

        self.delta: Dict[str, Union[float, int]] = {
            "x": abs(self.x - prev_player.x) if self.x is not None and prev_player.x is not None else 0,
            "y": abs(self.y - prev_player.y) if self.y is not None and prev_player.y is not None else 0,
        }

    def __str__(self) -> str:
        """
        Return a prettified JSON string representation of the player's state.

        Returns:
            str: JSON-formatted string of the player's data.
        """
        return json.dumps(self.get_json(), indent=self.INDENT)

    def get_json(self) -> Dict[str, Any]:
        """
        Convert the player's state to a JSON-serializable dictionary.

        Returns:
            Dict[str, Any]: Dictionary representing the player's state.
        """
        data: Dict[str, Any] = {
            "player_id": self.player_id,
            "team_id": self.team_id,
            "x": self.x,
            "y": self.y,
            "foot_x": self.foot_x,
            "foot_y": self.foot_y,
        }
        if hasattr(self, "delta") and self.delta is not None:
            data["delta"] = self.delta
        return data

    @staticmethod
    def generate_player_list(
        data: List[List[List[float]]],
        foot_data: List[List[List[float]]]
    ) -> Optional[List["Player"]]:
        """
        Generate a list of Player objects for a game state from detection data.

        The detection data is expected to be a nested list corresponding to rods defined in PLAYER_RODS.
        Each element in PLAYER_RODS should provide a tuple (player_id, team_id), and the detection data
        should provide corresponding coordinates [x, y] from `data` and [foot_x, foot_y] from `foot_data`.

        Args:
            data (List[List[List[float]]]): Nested list containing player coordinates.
                Expected structure: data[rod_index][player_index] = [x, y].
            foot_data (List[List[List[float]]]): Nested list containing foot coordinates.
                Expected structure: foot_data[rod_index][player_index] = [foot_x, foot_y].

        Returns:
            Optional[List[Player]]: A list of Player objects if successful; otherwise, None.
        """
        player_list: List[Player] = []
        try:
            for r, rod in enumerate(PLAYER_RODS):
                for p, (player_id, team_id) in enumerate(rod):
                    coords = data[r][p]
                    foot_coords = foot_data[r][p]
                    # Assume a player is detected if the y-coordinate exists (is not None)
                    if coords[1] is not None:
                        player = Player(
                            player_id=player_id,
                            team_id=team_id,
                            x=coords[0],
                            y=coords[1],
                            foot_x=foot_coords[0],
                            foot_y=foot_coords[1]
                        )
                        player_list.append(player)
            return player_list
        except Exception as e:
            print(f"Unable to generate player list: {e} at rod: {r}, player: {p}")
            return None
