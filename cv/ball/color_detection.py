import cv2
import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Optional, Tuple, Union

from cv.entities import Ball
from cv.debug_player import DecordVideoProcessor


class ColorDetection:
    """
    Detect and track a ball of specific colors (e.g., yellow) within video frames.

    This class provides methods to generate color masks, detect ball contours,
    and annotate frames. Detection is based on predefined HSV color ranges and
    morphological operations. Although red ranges are defined, the current
    implementation focuses on the yellow color range.
    """

    # Predefined HSV color ranges for detecting red and yellow balls.
    # Red ranges are defined for potential future use.
    RED_RANGES: List[Tuple[List[int], List[int]]] = [
        ([0, 130, 25], [10, 255, 255]),   # Red range 1
        ([170, 130, 25], [180, 255, 255])   # Red range 2
    ]
    YELLOW_RANGE: Tuple[List[int], List[int]] = ([15, 125, 60], [33, 255, 255]) # Hue 15 is min, because of hands

    # Kernel size for morphological operations to clean up the mask.
    MORPH_KERNEL: Tuple[int, int] = (7, 7)

    @staticmethod
    def detect_ball(
        frame: np.ndarray,
        max_radius: int,
        approx: int = cv2.CHAIN_APPROX_SIMPLE,
        returnContour: bool = False
    ) -> Optional[Union[Tuple[int, int, int], np.ndarray]]:
        """
        Detect a ball in the provided frame using color segmentation and contour analysis.

        The method converts the frame to a binary mask (using a predefined yellow color range),
        applies morphological operations to reduce noise, and then extracts contours from the mask.
        It selects the contour whose minimum enclosing circle has the largest radius that does not exceed
        the specified max_radius. Depending on the `returnContour` flag, it returns either the contour
        or a tuple containing the ball's center coordinates and radius.

        Args:
            frame (np.ndarray): The input image frame in BGR format.
            max_radius (int): The maximum allowed radius for a valid ball detection.
            approx (int, optional): Contour approximation method. Defaults to cv2.CHAIN_APPROX_SIMPLE.
            returnContour (bool, optional): If True, return the detected contour; 
                                            otherwise, return a tuple (x, y, radius). Defaults to False.

        Returns:
            Optional[Union[Tuple[int, int, int], np.ndarray]]:
                - A tuple (x, y, radius) if a valid ball is detected and returnContour is False.
                - The contour (as a numpy array) if returnContour is True.
                - None if no valid ball detection is found.
        """
        # Generate a binary mask for the ball using color segmentation.
        mask: np.ndarray = ColorDetection.get_combined_mask(frame)
        
        # Find contours in the mask.
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, approx)
        
        # Initialize variables to track the largest valid contour's circle.
        largest_valid_radius: int = 0
        ball_position: Optional[Tuple[int, int, int]] = None
        selected_contour: Optional[np.ndarray] = None

        # Iterate through each contour to find the best candidate.
        for contour in contours:
            # Compute the minimum enclosing circle for the current contour.
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x_int, y_int, r_int = int(x), int(y), int(radius)
            
            # Update if the circle's radius is within the allowed range and is larger than any previous candidate.
            if r_int <= max_radius and r_int > largest_valid_radius:
                largest_valid_radius = r_int
                ball_position = (x_int, y_int, r_int)
                selected_contour = contour

        # Return the contour if requested; otherwise, return the ball's position tuple.
        return selected_contour if (largest_valid_radius > 0 and returnContour) else ball_position
    
    @staticmethod
    def get_combined_mask(frame: npt.NDArray) -> npt.NDArray:
        """
        Generate a binary mask for the target color range using HSV thresholding.

        The function converts the input frame from BGR to HSV color space and applies a
        threshold for the defined yellow range. A morphological opening is then performed
        to remove noise. (Additional color ranges such as red can be added if necessary.)

        Args:
            frame (npt.NDArray): The input image frame in BGR format.

        Returns:
            npt.NDArray: A binary mask highlighting regions within the target color range.
        """
        # Convert frame from BGR to HSV color space.
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Generate a mask for the yellow color range.
        mask = cv2.inRange(
            hsv_frame,
            np.array(ColorDetection.YELLOW_RANGE[0], dtype="uint8"),
            np.array(ColorDetection.YELLOW_RANGE[1], dtype="uint8")
        )
        # To add red ranges, you can iterate over RED_RANGES as shown below:
        # for lower, upper in ColorDetection.RED_RANGES:
        #     mask += cv2.inRange(
        #         hsv_frame,
        #         np.array(lower, dtype="uint8"),
        #         np.array(upper, dtype="uint8")
        #     )
        # Apply morphological opening to eliminate small noise in the mask.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ColorDetection.MORPH_KERNEL)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    

@staticmethod
def run(frame: npt.NDArray) -> npt.NDArray:
    """
    Process a video frame to detect and annotate the ball using color detection.

    The function detects the ball, draws an enclosing circle around it,
    and returns the annotated frame.

    Args:
        frame (npt.NDArray): The input video frame in BGR format.

    Returns:
        npt.NDArray: The processed frame with the detected ball highlighted.
    """
    ball_position = ColorDetection.detect_ball(frame, max_radius=20)
    if ball_position is not None:
        x, y, r = ball_position
        ball = Ball(x, y, r)
        ball.draw(frame)
    return frame

if __name__ == "__main__":
    # Create and configure the video processor.
    video_processor = DecordVideoProcessor('cv/resources/yellow/test_011.mp4')
    video_processor.register_window("Color Tracking", run)
    video_processor.register_window("Color Mask", ColorDetection.get_combined_mask)
    video_processor.process_video_multi()
