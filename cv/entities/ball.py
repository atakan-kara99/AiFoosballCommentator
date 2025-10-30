import cv2
import json
import numpy as np
from typing import Any, Dict, Optional, Tuple

from .entity import Entity
from cv.global_constants import FRAME_TIME_SEC, REAL_FIELD_HEIGHT, REAL_FIELD_WIDTH, CF_PADDING_X, CF_PADDING_Y


class Ball(Entity):
    """
    Represents a detected ball in a video frame and computes movement metrics relative to a previous state.

    Attributes:
        x (int): x-coordinate of the ball's center (in pixels).
        y (int): y-coordinate of the ball's center (in pixels).
        r (int): Radius of the ball (in pixels).
        direction (Optional[int]): Movement direction in degrees (0 to 360). None if stationary.
        velocity (Optional[float]): Velocity of the ball in meters per second.
        delta (Optional[Dict[str, Optional[float]]]): Differences in 'x', 'y', 'velocity', and 'direction'
            compared to a previous state.
    """

    def __init__(self, x: int, y: int, r: int, frame: np.ndarray, prev_ball: Optional["Ball"]) -> None:
        """
        Initialize a Ball instance with its center position, radius, and compute movement metrics.

        Validates the parameters against the frame dimensions and, if a previous ball state is provided,
        computes the change in position, direction, and velocity.

        Args:
            x (int): x-coordinate of the ball's center (in pixels).
            y (int): y-coordinate of the ball's center (in pixels).
            r (int): Radius of the ball (in pixels).
            frame (np.ndarray): The current video frame, used to determine frame dimensions.
            prev_ball (Optional[Ball]): Previous ball state for computing movement metrics.

        Raises:
            ValueError: If x, y, or r are outside the valid range based on the frame dimensions.
        """
        # Extract frame dimensions (height, width) from the provided frame.
        frame_height, frame_width = frame.shape[:2]

        # Validate the ball's position and radius.
        if not (0 <= x <= frame_width):
            raise ValueError(f"[BallError] x must be between 0 and {frame_width}, but got {x}")
        if not (0 <= y <= frame_height):
            raise ValueError(f"[BallError] y must be between 0 and {frame_height}, but got {y}")
        if not (0 <= r <= min(frame_width, frame_height)):
            raise ValueError(
                f"[BallError] r must be between 0 and {min(frame_width, frame_height)}, but got {r}"
            )

        self.x: int = x
        self.y: int = y
        self.r: int = r

        # Initialize movement-related attributes.
        self.direction: Optional[int] = None
        self.velocity: Optional[float] = None
        self.delta: Optional[Dict[str, Optional[float]]] = None

        # If a previous ball state exists, calculate movement metrics.
        if prev_ball is not None:
            # Calculate differences in position.
            delta = self.calc_delta_pos(prev_ball)
            # Determine movement direction based on coordinate differences.
            self.direction = Ball.calc_direction(delta)
            # Compute the ball's velocity using real-world scaling.
            self.velocity = self.calc_velocity(prev_ball, frame)

            # Calculate changes in direction and velocity relative to the previous state.
            direction_delta: Optional[int] = (
                self.direction - prev_ball.direction if self.direction is not None and prev_ball.direction is not None else None
            )
            velocity_delta: Optional[float] = (
                np.round(self.velocity - prev_ball.velocity, 3) if self.velocity is not None and prev_ball.velocity is not None else None
            )

            self.delta = {
                "x": delta[0],
                "y": delta[1],
                "direction": direction_delta,
                "velocity": velocity_delta,
            }

    def calc_delta_pos(self, prev_ball: "Ball") -> Tuple[int, int]:
        """
        Compute the difference in the ball's center coordinates relative to a previous state.

        Args:
            prev_ball (Ball): The previous state of the ball.

        Returns:
            Tuple[int, int]: A tuple (delta_x, delta_y) representing the change in x and y positions.
        """
        delta_x: int = self.x - prev_ball.x
        delta_y: int = self.y - prev_ball.y
        return delta_x, delta_y

    @staticmethod
    def calc_direction(delta: Tuple[int, int]) -> Optional[int]:
        """
        Calculate the movement direction in degrees based on the change in position.

        The angle is measured clockwise from the positive x-axis.
        If there is no movement (i.e., both delta components are zero), returns None.

        Args:
            delta (Tuple[int, int]): A tuple (delta_x, delta_y) indicating the change in position.

        Returns:
            Optional[int]: Movement direction (0 to 360 degrees) or None if stationary.
        """
        delta_x, delta_y = delta
        if delta_x == 0 and delta_y == 0:
            return None

        # Calculate the angle in radians and convert it to degrees.
        angle_radians: float = np.arctan2(delta_y, delta_x)
        angle_degrees: int = int(np.degrees(angle_radians) % 360)

        if not (0 <= angle_degrees < 360):
            raise ValueError(f"[BallError] Calculated direction must be between 0 and 360, but got {angle_degrees}")
        return angle_degrees

    def calc_velocity(self, prev_ball: "Ball", frame: np.ndarray) -> float:
        """
        Calculate the ball's velocity (in meters per second) based on displacement between frames.

        Converts pixel displacement to a real-world distance using known field dimensions. 
        Assumes that REAL_FIELD_WIDTH and REAL_FIELD_HEIGHT are provided in millimeters,
        converting them to centimeters, and then to meters.

        Args:
            prev_ball (Ball): The previous state of the ball.
            frame (np.ndarray): The current video frame (used to compute scaling factors).

        Returns:
            float: The computed velocity in meters per second.

        Raises:
            ValueError: If the computed velocity is negative.
        """
        # Calculate conversion factors from pixels to centimeters.
        cm_per_pixel_x: float = (REAL_FIELD_WIDTH / 10) / (frame.shape[1] - 2 * CF_PADDING_X)
        cm_per_pixel_y: float = (REAL_FIELD_HEIGHT / 10) / (frame.shape[0] - 2 * CF_PADDING_Y)

        # Compute displacement in pixels.
        dx, dy = self.calc_delta_pos(prev_ball)
        # Convert pixel displacement to a real-world distance (in centimeters).
        real_distance_cm: float = np.sqrt((dx * cm_per_pixel_x) ** 2 + (dy * cm_per_pixel_y) ** 2)
        # Convert the distance from centimeters to meters.
        real_distance_m: float = real_distance_cm / 100

        # Calculate velocity using the frame time constant (in seconds).
        velocity: float = real_distance_m / FRAME_TIME_SEC

        if velocity < 0:
            raise ValueError(f"[BallError] Computed velocity must be non-negative, but got {velocity}")
        return float(np.round(velocity, 3))

    def get_json(self) -> Dict[str, Any]:
        """
        Generate a JSON-serializable dictionary of the ball's properties.

        Returns:
            Dict[str, Any]: A dictionary containing 'x', 'y', 'r', 'direction', 'velocity', and 'delta'.
        """
        return {
            "x": self.x,
            "y": self.y,
            "r": self.r,
            "direction": self.direction,
            "velocity": self.velocity,
            "delta": self.delta,
        }

    def __str__(self) -> str:
        """
        Return a JSON-formatted string representation of the ball's properties.

        Returns:
            str: A pretty-printed JSON string of the ball's information.
        """
        return json.dumps(self.get_json(), indent=self.INDENT)

    def draw(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """
        Draw the ball on the provided video frame.

        This method draws a circle around the ball's perimeter and overlays its radius as text above the circle.

        Args:
            frame (np.ndarray): The image frame on which to draw the ball.
            color (Tuple[int, int, int], optional): BGR color for the circle and text. Defaults to (0, 0, 255).

        Returns:
            np.ndarray: The image frame with the ball drawn.
        """
        # Draw the circle representing the ball.
        cv2.circle(frame, (self.x, self.y), self.r, color, thickness=2)
        # Prepare the text to display (ball's radius).
        text: str = f"{self.r}"
        # Calculate the position for the text (above the circle).
        text_position: Tuple[int, int] = (self.x - 15, self.y - self.r - 5)
        cv2.putText(
            frame,
            text,
            text_position,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            color=color,
            thickness=2
        )
        return frame
