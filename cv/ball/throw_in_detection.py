import numpy.typing as npt
from typing import Optional, Tuple

from cv.entities import Ball, Zone
from cv.debug_player import DecordVideoProcessor
from cv.global_constants import ROD_SPACING_MIN_K, GOAL_WAIT_FRAMES


class ThrowInDetection:
    """
    Detects throw-in events for a specified side (TOP or BOTTOM) of the field.

    This class defines a rectangular detection zone near the field's edge and monitors
    the ball's movement within that zone. When the ball exits the zone, its movement
    is analyzed to determine if a throw-in occurred.
    """

    def __init__(self, width: int, height: int, side: str, verbose: bool = False) -> None:
        """
        Initialize the throw-in detection zone and parameters.

        Args:
            width (int): The width of the video frame.
            height (int): The height of the video frame.
            side (str): The side for detection; must be either "TOP" or "BOTTOM".
            verbose (bool, optional): If True, enables verbose logging. Defaults to False.

        Raises:
            ValueError: If side is not "TOP" or "BOTTOM".
        """
        # Validate side parameter.
        if side not in {"TOP", "BOTTOM"}:
            raise ValueError("The 'side' parameter must be 'TOP' or 'BOTTOM'.")
        self.side: str = side

        # Calculate the detection zone dimensions.
        rod_spacing_min: int = int(ROD_SPACING_MIN_K * width)
        top_left_x: int = width // 2 - rod_spacing_min // 4
        zone_height: int = height // 12

        if side == "TOP":
            top_left: Tuple[int, int] = (top_left_x, 0)
            bottom_right: Tuple[int, int] = (top_left_x + rod_spacing_min // 2, zone_height)
        else:  # side == "BOTTOM"
            top_left = (top_left_x, height - zone_height)
            bottom_right = (top_left_x + rod_spacing_min // 2, height)

        self.zone: Optional[Zone] = Zone(top_left=top_left, bottom_right=bottom_right)

        # Initialize ball tracking attributes.
        self.first_ball: Optional[Ball] = None
        self.last_ball: Optional[Ball] = None

        # Frame-based tracking variables.
        self.ball_not_seen: int = 0
        self.wait_threshold: int = GOAL_WAIT_FRAMES
        self.ball_last_seen_in_zone: bool = False

        # Optional verbose logging.
        self.verbose: bool = verbose
        if self.verbose:
            print(f"[{self.side}] ThrowInDetection initialized.\n")

    def reset(self) -> None:
        """
        Reset the ball tracking state and frame counter.
        """
        self.first_ball = None
        self.last_ball = None
        self.ball_not_seen = 0
        self.ball_last_seen_in_zone = False

    def check_throw_in(self, ball: Optional[Ball]) -> Optional[Ball]:
        """
        Check if a throw-in event is detected based on the ball's movement.

        This method updates the tracking state while the ball is in the detection zone.
        When the ball exits the zone, the movement between the first and last observed positions
        is analyzed to determine if it constitutes a valid throw-in.

        Args:
            ball (Optional[Ball]): The ball object with attributes 'x', 'y', and 'velocity'.
                                   If None, it indicates the ball is not currently detected.

        Returns:
            Optional[Ball]: The initial ball state if a valid throw-in is detected, otherwise None.
        """
        if ball is None:
            # If the ball is not detected and was previously in the zone, update the counter.
            if self.ball_last_seen_in_zone:
                self.ball_not_seen += 1
                if self.ball_not_seen > self.wait_threshold:
                    self.reset()
            return None

        # If the ball is within the detection zone, update the tracking state.
        if self.zone.contains((ball.x, ball.y)):
            self.ball_last_seen_in_zone = True
            if self.first_ball is None:
                self.first_ball = ball
            self.last_ball = ball
            return None

        # When the ball leaves the zone, ensure there are two recorded positions.
        if self.first_ball is None or self.last_ball is None:
            return None
        # Calculate ball movement
        dx, dy = self.last_ball.calc_delta_pos(self.first_ball)
        degree = Ball.calc_direction((dx, dy))
        # Determine if the throw-in is valid
        if degree is None or abs(dy) <= 25:
            is_valid = False
        elif self.side == "TOP":
            is_valid = 71 < degree < 109
        elif self.side == "BOTTOM":
            is_valid = 251 < degree < 289

        # Optional verbose logging of detection details.
        if self.verbose:
            print(f"[{self.side}] First: ({self.first_ball.x}, {self.first_ball.y}), "
                  f"Last: ({self.last_ball.x}, {self.last_ball.y}), Movement: dx={dx}, dy={dy}, "
                  f"dir={degree}°")
            if is_valid:
                print(f"[{self.side}] THROW-IN DETECTED!")
            print()

        # Capture the initial ball state before resetting.
        first_ball = self.first_ball
        self.reset()
        return first_ball if is_valid else None


def run(frame: npt.NDArray) -> npt.NDArray:
    """
    Draw throw-in detection zones on the video frame for both the TOP and BOTTOM sides.

    This function instantiates detection zones for both sides and overlays them onto the frame.

    Args:
        frame (npt.NDArray): The input video frame in BGR format.

    Returns:
        npt.NDArray: The video frame annotated with the throw-in detection zones.
    """
    frame = throw_in_top.zone.draw(frame)
    frame = throw_in_bottom.zone.draw(frame)
    return frame


if __name__ == "__main__":
    # Define frame dimensions.
    height, width = 720, 1280

    # Initialize the throw-ins.
    throw_in_top = ThrowInDetection(width, height, side="TOP", verbose=True)
    throw_in_bottom = ThrowInDetection(width, height, side="BOTTOM", verbose=True)

    # Create and configure the video processor.
    video = DecordVideoProcessor('cv/resources/yellow/test_011.mp4')
    video.register_window("Throw In Detection", run)
    video.process_video_multi()
