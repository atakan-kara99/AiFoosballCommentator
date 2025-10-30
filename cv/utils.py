import cv2
import numpy as np
from typing import List, Tuple

from cv.global_constants import CF_PADDING_X, CF_PADDING_Y


class Utils:
    """
    A collection of utility functions for image processing tasks.

    This class provides methods to compute geometric means, convert between color spaces,
    generate custom grayscale images, compute histograms, crop and warp frames, and convert
    pixel coordinates to relative percentages.
    """

    @staticmethod
    def geometric_mean(height: int, width: int) -> float:
        """
        Calculate the geometric mean of two values.

        Args:
            height (int): The first value.
            width (int): The second value.

        Returns:
            float: The geometric mean of height and width.
        """
        return np.sqrt(height * width)

    @staticmethod
    def xy_to_hsv(x: int, y: int, frame: np.ndarray) -> np.ndarray:
        """
        Convert the BGR (or grayscale) pixel value at (x, y) in a frame to HSV.

        If the frame is grayscale, the pixel value is replicated across BGR channels.

        Args:
            x (int): The x-coordinate of the pixel.
            y (int): The y-coordinate of the pixel.
            frame (np.ndarray): The input image frame (BGR or grayscale).

        Returns:
            np.ndarray: The HSV value of the pixel as a 1D array of 3 elements.
        """
        if len(frame.shape) == 2:  # Grayscale image.
            gray: int = frame[y, x]
            bgr: np.ndarray = np.array([gray, gray, gray])
        else:  # Color image.
            bgr = frame[y, x]
        return Utils.bgr_to_hsv(bgr)

    @staticmethod
    def bgr_to_hsv(bgr: np.ndarray) -> np.ndarray:
        """
        Convert a single BGR pixel value to HSV.

        Args:
            bgr (np.ndarray): A 1D array or list of 3 elements representing a BGR pixel.

        Returns:
            np.ndarray: A 1D array representing the HSV value of the input pixel.
        """
        # Create a 1x1 image from the BGR pixel.
        bgr_pixel: np.ndarray = np.uint8([[bgr]])
        hsv_pixel: np.ndarray = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)
        return hsv_pixel[0, 0]

    @staticmethod
    def custom_rgb_to_greyscale(
        frame: np.ndarray, red: float = 0.299, green: float = 0.587, blue: float = 0.114
    ) -> np.ndarray:
        """
        Convert a BGR image to grayscale using custom weights for the RGB channels.

        Note: In OpenCV, the image is in BGR format. The red channel is at index 2.

        Args:
            frame (np.ndarray): The input image frame in BGR format.
            red (float, optional): Weight for the red channel. Defaults to 0.299.
            green (float, optional): Weight for the green channel. Defaults to 0.587.
            blue (float, optional): Weight for the blue channel. Defaults to 0.114.

        Returns:
            np.ndarray: The resulting grayscale image as a uint8 array.
        """
        gray = red * frame[:, :, 2] + green * frame[:, :, 1] + blue * frame[:, :, 0]
        return gray.astype('uint8')

    @staticmethod
    def custom_hsv_to_greyscale(
        frame: np.ndarray, hue: float = 0.11, sat: float = 0.30, val: float = 0.59
    ) -> np.ndarray:
        """
        Convert a BGR image to grayscale based on weighted HSV channels.

        The input image is first converted to HSV, and then the channels are combined
        using the provided weights.

        Args:
            frame (np.ndarray): The input image frame in BGR format.
            hue (float, optional): Weight for the Hue channel. Defaults to 0.11.
            sat (float, optional): Weight for the Saturation channel. Defaults to 0.30.
            val (float, optional): Weight for the Value channel. Defaults to 0.59.

        Returns:
            np.ndarray: The resulting grayscale image as a uint8 array.
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = hue * hsv_frame[:, :, 0] + sat * hsv_frame[:, :, 1] + val * hsv_frame[:, :, 2]
        return gray.astype('uint8')

    @staticmethod
    def cumulative_histogram(frame: np.ndarray) -> np.ndarray:
        """
        Compute the normalized cumulative histogram of the Hue channel of a BGR image in HSV space.

        Args:
            frame (np.ndarray): The input image frame in BGR format.

        Returns:
            np.ndarray: A 1D array representing the cumulative histogram of the Hue channel.
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_channel = hsv_frame[:, :, 0]
        # Calculate histogram with 180 bins (one for each possible hue value).
        hist = cv2.calcHist([hue_channel], [0], None, [180], [0, 180])
        hist /= hist.sum()  # Normalize the histogram.
        return np.cumsum(hist)

    @staticmethod
    def crop_frame(frame: np.ndarray, corners: List[Tuple[int, int]]) -> np.ndarray:
        """
        Warp and crop an image frame based on the specified corner coordinates, applying padding.

        The transformation maps the provided corners to a rectangle defined by the global padding
        constants CF_PADDING_X and CF_PADDING_Y.

        Args:
            frame (np.ndarray): The input image frame.
            corners (List[Tuple[int, int]]): A list of four (x, y) coordinates representing the corners 
                                             of the region to be cropped. The order should be top-left, 
                                             top-right, bottom-left, and bottom-right.

        Returns:
            np.ndarray: The cropped and warped image.
        """
        height, width, _ = frame.shape
        dst_points = np.float32([
            [0 + CF_PADDING_X, 0 + CF_PADDING_Y], 
            [width - CF_PADDING_X, 0 + CF_PADDING_Y], 
            [width - CF_PADDING_X, height - CF_PADDING_Y], 
            [0 + CF_PADDING_X, height - CF_PADDING_Y]
            ])
        src_points = np.float32(corners)
        # Compute the perspective transformation matrix.
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        # Apply the perspective warp.
        return cv2.warpPerspective(frame, matrix, (width, height))

    @staticmethod
    def coord_to_perc(
        pos: Tuple[int, int], 
        shape: Tuple[int, int, int]
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to normalized percentages of the effective frame area.

        This method converts a given (x, y) pixel coordinate into a normalized coordinate 
        relative to the cropped frame dimensions. The effective dimensions are computed by 
        subtracting the horizontal (CF_PADDING_X) and vertical (CF_PADDING_Y) padding from 
        the frame's width and height respectively. The resulting percentages are clamped 
        to the [0, 1] range and rounded to three decimal places.

        Args:
            pos (Tuple[Union[float, int], Union[float, int]]): The (x, y) coordinate in pixel space.
            shape (Tuple[int, int, int]): The shape of the frame (height, width, channels).

        Returns:
            Tuple[float, float]: The normalized (x, y) coordinate, each value between 0 and 1,
                                rounded to three decimal places.
        """
        fh, fw, _ = shape  # Frame height, width, and channel count (unused)
        
        # Compute normalized coordinates adjusted for horizontal and vertical padding.
        x = float(pos[0]) / float(fw - 2 * CF_PADDING_X)
        y = float(pos[1]) / float(fh - 2 * CF_PADDING_Y)
        
        # Clamp the values to ensure they remain within the [0, 1] range.
        x = min(max(x, 0), 1)
        y = min(max(y, 0), 1)
        
        # Round the normalized coordinates to three decimal places.
        return (round(x, 3), round(y, 3))
