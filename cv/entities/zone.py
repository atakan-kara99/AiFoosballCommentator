import cv2
import json
import numpy as np
from typing import Any, Dict, Optional, Tuple

from .entity import Entity


class Zone(Entity):
    """
    Represents a rectangular zone within an image frame.

    A Zone is defined by its top-left and bottom-right coordinates and a display color.
    It provides methods to draw itself on an image frame and to check whether a given
    point falls within its boundaries.
    """

    def __init__(self, top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> None:
        """
        Initialize the Zone with specified corner coordinates.

        Args:
            top_left (Tuple[int, int]): The (x, y) coordinates of the zone's top-left corner.
            bottom_right (Tuple[int, int]): The (x, y) coordinates of the zone's bottom-right corner.
        """
        self.top_left: Tuple[int, int] = top_left
        self.bottom_right: Tuple[int, int] = bottom_right
        self.color: Tuple[int, int, int] = (179, 255, 0)  # Display color in BGR format.

    def draw(self, frame: np.ndarray, fill: bool = False) -> np.ndarray:
        """
        Draw the zone on the provided image frame.

        Args:
            frame (np.ndarray): The image frame on which to draw the zone.
            fill (bool, optional): If True, fill the rectangle with color; otherwise, draw only the border.
                                   Defaults to False.

        Returns:
            np.ndarray: The image frame with the zone drawn on it.
        """
        border: int = -1 if fill else 2
        cv2.rectangle(frame, self.top_left, self.bottom_right, self.color, border)
        return frame

    def contains(self, position: Optional[Tuple[int, int]]) -> bool:
        """
        Determine whether a given point lies within the zone.

        Args:
            position (Optional[Tuple[int, int]]): The (x, y) coordinates to check.

        Returns:
            bool: True if the position is within the zone; otherwise, False.
        """
        if position is None:
            return False
        x, y = position
        (x1, y1), (x2, y2) = self.top_left, self.bottom_right
        return x1 <= x <= x2 and y1 <= y <= y2

    def __str__(self) -> str:
        """
        Return a prettified JSON string representation of the zone.

        Returns:
            str: A JSON-formatted string describing the zone.
        """
        return json.dumps(self.get_json(), indent=self.INDENT)

    def get_json(self) -> Dict[str, Any]:
        """
        Generate a JSON-serializable dictionary representation of the zone.

        Returns:
            Dict[str, Any]: A dictionary containing the zone's attributes.
        """
        return {
            "top_left": self.top_left,
            "bottom_right": self.bottom_right,
        }
