import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple

from cv.entities import Ball, Zone
from cv.debug_player import DecordVideoProcessor
from cv.global_constants import GOAL_WIDTH_MIN, GOAL_WIDTH_MAX_K, GOAL_WIDTH_MIN_K, GOAL_WAIT_FRAMES


class GoalDetection:
    """
    Detects goal events within a defined zone on a specified side of the field.

    This class defines a rectangular detection zone for a goal and monitors whether the ball
    is present within this zone over a sequence of frames. A goal is detected if the ball
    disappears from the zone for a predetermined number of frames.
    """

    def __init__(self, width: int, height: int, side: str, verbose: bool = False) -> None:
        """
        Initialize a GoalDetection instance with frame dimensions, target side, and detection parameters.

        Args:
            width (int): Width of the video frame.
            height (int): Height of the video frame.
            side (str): The side for goal detection; must be either "LEFT" or "RIGHT".
            verbose (bool, optional): If True, enables verbose logging. Defaults to False.

        Raises:
            ValueError: If the provided side is not "LEFT" or "RIGHT".
        """
        self.width: int = width
        self.height: int = height

        if side not in {"LEFT", "RIGHT"}:
            raise ValueError("The 'side' parameter must be 'LEFT' or 'RIGHT'.")
        self.side: str = side

        # Initialize the goal detection zone with a neutral scaling factor (0).
        self.set_zone(0)

        # Frame-based variables for tracking ball presence.
        self.ball_not_seen: int = 0
        self.wait_threshold: int = GOAL_WAIT_FRAMES
        self.ball_last_seen_in_zone: bool = False

        # Optional verbose logging.
        self.verbose: bool = verbose
        if self.verbose:
            print(f"[{self.side}] Initialized GoalDetection.\n")

    def set_zone(self, factor: float) -> None:
        """
        Define the goal detection zone based on the given scaling factor and frame dimensions.

        The detection zone is a rectangle whose width is scaled based on the factor (often derived
        from ball velocity) and clamped between a minimum and maximum width. The zone is centered
        vertically on the frame.

        Args:
            factor (float): A scaling factor to adjust the width of the detection zone.
        """
        width, height = self.width, self.height
        # Calculate the minimum width of GOAL_WIDTH_MIN.
        goal_width_min: float = GOAL_WIDTH_MIN / 4
        # Scale the goal width using the factor.
        scaled_goal_width: float = goal_width_min / 2 * np.exp(factor)
        # Clamp it between goal_width_min and GOAL_WIDTH_MIN.
        goal_width: int = int(min(max(scaled_goal_width, goal_width_min), GOAL_WIDTH_MIN))

        # Compute the goal zone length based on a constant proportion of the frame height.
        goal_length: int = int((GOAL_WIDTH_MAX_K + GOAL_WIDTH_MIN_K) / 2 * height)
        center_y: int = height // 2
        center_goal: int = goal_length // 2

        if self.side == "LEFT":
            top_left: Tuple[int, int] = (0, center_y - center_goal)
            bottom_right: Tuple[int, int] = (goal_width, center_y + center_goal)
        else:  # self.side == "RIGHT"
            top_left = (width - goal_width, center_y - center_goal)
            bottom_right = (width, center_y + center_goal)

        self.zone: Optional[Zone] = Zone(top_left, bottom_right)

    def reset(self) -> None:
        """
        Reset the frame counter and tracking state for the detection zone.

        This method is called when the ball is detected outside the zone or after a goal is registered.
        """
        self.ball_not_seen = 0
        self.ball_last_seen_in_zone = False

    def check_goal(self, ball: Optional[Ball]) -> bool:
        """
        Evaluate whether a goal has been scored based on the ball's presence in the detection zone.

        If the ball is not detected (i.e., ball is None) but was previously seen in the zone,
        a counter is incremented. Once this counter exceeds a preset threshold, a goal is declared,
        and the internal state is reset.

        Args:
            ball (Optional[Ball]): The ball object with attributes 'x', 'y', and 'velocity'.
                                   If None, it indicates that the ball is not detected.

        Returns:
            bool: True if a goal is detected, False otherwise.
        """
        if ball is None:
            if self.ball_last_seen_in_zone:
                # Increment frame count when the ball is missing after being seen.
                self.ball_not_seen += 1
                if self.verbose:
                    print(f"[{self.side}] Ball not seen in zone for {self.ball_not_seen} frames.")
                # Declare a goal if the missing count exceeds the threshold.
                if self.ball_not_seen >= self.wait_threshold:
                    if self.verbose:
                        print(f"[{self.side}] GOAL DETECTED!\n")
                    self.reset()
                    return True
        else:
            # Adjust the detection zone dynamically based on the ball's velocity.
            if ball.velocity is not None:
                self.set_zone(ball.velocity)
            # If the ball is within the detection zone, reset the missing counter.
            if self.zone.contains((ball.x, ball.y)):
                self.ball_last_seen_in_zone = True
                self.ball_not_seen = 0
            else:
                self.reset()

        return False


def run(frame: npt.NDArray) -> npt.NDArray:
    """
    Annotate the video frame with goal detection zones for both the LEFT and RIGHT sides.

    This function creates goal detection instances for both sides of the field, draws their
    respective detection zones using a utility function, and returns the annotated frame.

    Args:
        frame (npt.NDArray): The current video frame.

    Returns:
        npt.NDArray: The video frame with the drawn goal detection zones.
    """
    frame = goal_left.zone.draw(frame)
    frame = goal_right.zone.draw(frame)
    return frame


if __name__ == "__main__":
    # Define frame dimensions.
    height, width = 720, 1280

    # Initialize the goals.
    goal_left = GoalDetection(width, height, side="LEFT", verbose=True)
    goal_right = GoalDetection(width, height, side="RIGHT", verbose=True)

    # Create and configure the video processor.
    video = DecordVideoProcessor('cv/resources/yellow/test_011.mp4')
    video.register_window("Goal Detection", run)
    video.process_video_multi()
