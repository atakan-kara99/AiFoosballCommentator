import cv2
import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple

from cv.utils import Utils
from cv.entities import Ball
from cv.debug_player import DecordVideoProcessor


class HoughTransform:
    """
    Detects and annotates circular objects (e.g., balls) in video frames using the Hough Circle Transform.

    This class provides methods to:
      - Convert frames to grayscale with custom weights and noise reduction.
      - Detect circles within a specified radius range.
      - Annotate detected circles on the original frame.
    """
    
    @staticmethod
    def detect_ball(frame: npt.NDArray, min_radius: int, max_radius: int) -> Optional[Tuple[int, int, int]]:
        """
        Detect the most prominent circle in the frame using the Hough Circle Transform.

        The method converts the frame to a custom grayscale image (with Gaussian blur applied) and then
        applies the Hough Circle Transform. It returns the circle with the largest radius that lies within
        the specified range.

        Args:
            frame (npt.NDArray): The image frame in BGR format.
            min_radius (int): The minimum circle radius to detect.
            max_radius (int): The maximum circle radius to detect.

        Returns:
            Optional[Tuple[int, int, int]]: A tuple (x, y, r) representing the detected circle's center
            coordinates and radius, or None if no circle is detected.
        """
        gray_frame = HoughTransform.get_gray(frame)
        # Apply the Hough Circle Transform to detect circles.
        circles = cv2.HoughCircles(
            gray_frame,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=1280,  # Minimum distance between detected centers.
            param1=60,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is not None:
            # Round the coordinates and radii, then select the circle with the largest radius.
            circles_rounded = np.round(circles[0]).astype(int)
            return max(circles_rounded, key=lambda circle: circle[2])
        return None

    @staticmethod
    def get_gray(frame: npt.NDArray) -> npt.NDArray:
        """
        Convert the frame to a custom grayscale image and apply Gaussian blur for noise reduction.

        The grayscale conversion uses custom channel weights (via Utils.custom_rgb_to_greyscale) and
        applies a Gaussian blur to reduce noise before circle detection.

        Args:
            frame (npt.NDArray): The input image frame in BGR format.

        Returns:
            npt.NDArray: The processed grayscale image.
        """
        # Convert frame to grayscale using custom weights (here, full weight for the red channel).
        gray_frame = Utils.custom_rgb_to_greyscale(frame, 1, 0, 0)
        # Apply Gaussian blur to reduce noise.
        blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), sigmaX=2)
        return blurred_frame


@staticmethod
def run(frame: npt.NDArray) -> npt.NDArray:
    """
    Process a video frame by detecting a circle (ball) and annotating it using the Hough Circle Transform.

    Args:
        frame (npt.NDArray): The input video frame in BGR format.

    Returns:
        npt.NDArray: The annotated video frame with the detected circle (if any).
    """
    ball_position = HoughTransform.detect_ball(frame, 15, 20)
    if ball_position is not None:
        x, y, r = ball_position
        ball = Ball(x, y, r)
        ball.draw(frame)
    return frame

if __name__ == "__main__":
    # Create and configure the video processor.
    video = DecordVideoProcessor('cv/resources/yellow/test_011.mp4')
    video.register_window("Hough Circle Transform", run)
    video.register_window("Grayscale", HoughTransform.get_gray)
    video.process_video_multi()
