import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple, Deque, Any

from cv.utils import Utils
from cv.entities import Ball
from cv.debug_player import DecordVideoProcessor
from cv.global_constants import BALL_MIN_K, BALL_MAX_K
from .ball_approx import BallApprox
from .color_detection import ColorDetection
from .hough_transform import HoughTransform


class BallDetection:
    """
    Detects and tracks a ball in video frames using multiple detection algorithms.

    This class provides methods to detect the ball in a frame either via color detection
    or using the Hough Transform. It also maintains a history of recent detections, which
    can be used to visualize the ball's movement across frames.
    """

    DRAW_COLOR: Tuple[int, int, int] = (0, 0, 255)

    def __init__(
        self, 
        width: int, 
        height: int, 
        last_positions_max_len: int = 25, 
        verbose: bool = False
    ) -> None:
        """
        Initialize a BallDetection instance with the specified frame dimensions and parameters.

        The ball's minimum and maximum radii are computed using the geometric mean of the frame's
        width and height along with global scaling constants. A history queue is also initialized
        to record the ball's detected positions.

        Args:
            width (int): The width of the video frame.
            height (int): The height of the video frame.
            last_positions_max_len (int, optional): Maximum number of past positions to store.
                Defaults to 25.
            verbose (bool, optional): If True, prints initialization parameters. Defaults to False.
        """
        # Queue to store recent ball positions as tuples: (x, y, radius)
        self.last_positions: Deque[Tuple[int, int, int]] = deque(maxlen=last_positions_max_len)
        # Compute the geometric mean of frame dimensions for scaling.
        geometric_mean: float = Utils.geometric_mean(height, width)
        self.ball_min_radius: int = int(BALL_MIN_K * geometric_mean)
        self.ball_max_radius: int = int(BALL_MAX_K * geometric_mean) + 1
        self.ball_base_radius: int = (self.ball_min_radius + self.ball_max_radius) // 2
        # Initialize the ball approximation algorithm.
        self.ball_approx: BallApprox = BallApprox(self.ball_base_radius, self.ball_max_radius, verbose)
        if verbose:
            print("BallDetection initialized.")

    def draw_last_positions(self, frame: np.ndarray) -> np.ndarray:
        """
        Annotate the frame with the history of previously detected ball positions.

        A small circle is drawn at each historical position using the DRAW_COLOR.

        Args:
            frame (np.ndarray): The image frame to be annotated.

        Returns:
            np.ndarray: The image frame with the drawn history of ball positions.
        """
        for x, y, _ in self.last_positions:
            cv2.circle(frame, (x, y), radius=5, color=self.DRAW_COLOR, thickness=-1)
        return frame

    def detect_ball(
        self, 
        frame: np.ndarray, 
        rod_positions: Optional[Any] = None, 
        player_mask: Optional[Any] = None
    ) -> Optional[Tuple[int, int, int]]:
        """
        Detect the ball in the given frame using either the legacy or a new detection method.

        If rod_positions or player_mask are not provided, the method uses the old detection
        approach (first via ColorDetection, then HoughTransform if needed). Otherwise, it uses
        the new detection approach from BallApprox.

        Args:
            frame (np.ndarray): The image frame (BGR format) in which to detect the ball.
            rod_positions (Optional[Any], optional): Rod positions for new detection. Defaults to None.
            player_mask (Optional[Any], optional): Player mask for new detection. Defaults to None.

        Returns:
            Optional[Tuple[int, int, int]]: A tuple (x, y, radius) representing the ball's
            center and radius if a valid detection is found; otherwise, None.
        """
        if rod_positions is None or player_mask is None:
            # Legacy ball detection.
            ball_position: Optional[Tuple[int, int, int]] = ColorDetection.detect_ball(frame, self.ball_max_radius)
            if ball_position is None:
                ball_position = HoughTransform.detect_ball(frame, self.ball_min_radius, self.ball_max_radius)
            else:
                self.last_positions.append(ball_position)
            return ball_position
        else:
            # New ball detection method using BallApprox.
            ball_position = self.ball_approx.detect_ball(frame, rod_positions, player_mask)
            if ball_position is not None:
                self.last_positions.append(ball_position)
            return ball_position

@staticmethod
def run(frame: np.ndarray) -> np.ndarray:
    """
    Process a single video frame to detect and annotate the ball.

    This function uses a global BallDetection instance (ball_detection) to detect the ball,
    draw it on the frame, and annotate the frame with the history of detected positions.

    Args:
        frame (np.ndarray): The current video frame (BGR format).

    Returns:
        np.ndarray: The annotated video frame.
    """
    ball_position = ball_detection.detect_ball(frame)
    if ball_position is not None:
        x, y, r = ball_position
        ball = Ball(x, y, r)  
        frame = ball.draw(frame, color=BallDetection.DRAW_COLOR)
    frame = ball_detection.draw_last_positions(frame)
    return frame

if __name__ == "__main__":
    # Create a global BallDetection instance.
    ball_detection = BallDetection(width=1280, height=720)
    # Create and configure the video processor.
    video_processor = DecordVideoProcessor('cv/resources/yellow/test_011.mp4')
    video_processor.register_window("Ball Detection", run)
    video_processor.process_video_multi()
