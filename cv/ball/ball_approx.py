import cv2
import numpy as np
import scipy.optimize as opt
from typing import List, Optional, Tuple, Union

from cv.debug_player import DecordVideoProcessor
from .color_detection import ColorDetection

# Type alias for clarity.
CircleCenter = Tuple[int, int]


class BallApprox:
    """
    Class to approximate the ball's position and shape within an image.

    This class implements a multi-step detection process:
      1. Rough ball detection using color segmentation.
      2. Extraction and upscaling of a region-of-interest (ROI) around the ball.
      3. Binary mask generation and extraction of the largest valid contour.
      4. Filtering and contour shape approximation.
      5. Circle fitting using least squares from multiple initial guesses.
      6. Candidate selection based on consistency, proximity to rods, and player mask cues.
    """

    def __init__(self, base_radius: int = 17, max_radius: int = 20, verbose: bool = False) -> None:
        """
        Initialize the ball approximator with detection parameters.

        Args:
            base_radius (int): Base radius used for initial ball estimates.
            max_radius (int): Maximum allowed radius for ball detection.
            verbose (bool): If True, annotate the ROI with detection information.
        """
        # Scale factors for region extraction and resolution enhancement.
        self.ZOOM_SCALE: int = 3
        self.UPSCALE_FACTOR: int = 2

        self.BASE_RADIUS: int = base_radius
        self.MAX_RADIUS: int = max_radius

        # Derived radii for region extraction (ROI) and circle fitting.
        self.ROI_RADIUS: int = base_radius * self.ZOOM_SCALE
        self.R_FIT: int = base_radius * self.UPSCALE_FACTOR

        # Parameters for contour filtering.
        self.DISTANCE_THRESHOLD: float = 0.5
        self.EPSILON_FACTOR: float = 0.01

        # Storage for annotated frames (high-resolution ROI and binary mask).
        self.frames: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.verbose: bool = verbose

    def fit_circle_least_squares(
        self, x: np.ndarray, y: np.ndarray, init: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Fit a circle (with a fixed radius) to contour points using the least squares method.

        This function optimizes the circle's center (h, k) so that the residuals between the
        distances of the contour points and the fixed radius (self.R_FIT) are minimized.

        Args:
            x (np.ndarray): 1D array of x-coordinates of the contour points.
            y (np.ndarray): 1D array of y-coordinates of the contour points.
            init (Tuple[float, float]): Initial guess for the circle center (h, k).

        Returns:
            Tuple[float, float]: Optimized circle center coordinates (h_fit, k_fit).
        """
        def circle_residuals(
            params: Tuple[float, float], x: np.ndarray, y: np.ndarray, r: float
        ) -> np.ndarray:
            """
            Compute residuals between the Euclidean distances of points to the center and the fixed radius.

            Args:
                params (Tuple[float, float]): Current estimate of the circle center (h, k).
                x (np.ndarray): x-coordinates of the contour points.
                y (np.ndarray): y-coordinates of the contour points.
                r (float): Fixed radius to fit.

            Returns:
                np.ndarray: Residual differences (distance - r) for each point.
            """
            h, k = params
            distances = np.sqrt((x - h) ** 2 + (y - k) ** 2)
            return distances - r

        h_init, k_init = init
        result = opt.least_squares(
            circle_residuals, x0=[h_init, k_init], args=(x, y, self.R_FIT)
        )
        h_fit, k_fit = result.x
        return h_fit, k_fit

    def fit_circles(
        self, contour_points: np.ndarray, initials: List[Tuple[int, int]]
    ) -> List[CircleCenter]:
        """
        Fit circles to a set of contour points using multiple initial guesses.

        For each initial guess, the circle's center is refined by least squares,
        then rounded to integer values.

        Args:
            contour_points (np.ndarray): Array of contour points with shape (N, 2).
            initials (List[Tuple[int, int]]): List of initial guesses for the circle center.

        Returns:
            List[CircleCenter]: List of fitted circle centers (h, k) as integer tuples.
        """
        contour_x = contour_points[:, 0]
        contour_y = contour_points[:, 1]
        circle_fits: List[CircleCenter] = []
        for initial in initials:
            h_fit, k_fit = self.fit_circle_least_squares(contour_x, contour_y, initial)
            circle_fits.append((int(h_fit), int(k_fit)))
        return circle_fits

    def get_roi(
        self, frame: np.ndarray, ball_position: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract a square region-of-interest (ROI) around the ball's detected position.

        The ROI is centered at the ball's position and extends by a fixed radius in all directions.

        Args:
            frame (np.ndarray): Input image frame.
            ball_position (Tuple[int, int, int]): Detected ball position (x, y, r).

        Returns:
            Tuple[np.ndarray, Tuple[int, int, int, int]]:
                - ROI image extracted from the frame.
                - ROI boundaries as a tuple (x1, y1, x2, y2).
        """
        x, y, _ = ball_position
        roi_rad = self.ROI_RADIUS
        # Ensure ROI boundaries remain within the frame dimensions.
        x1 = max(0, x - roi_rad)
        y1 = max(0, y - roi_rad)
        x2 = min(frame.shape[1], x + roi_rad)
        y2 = min(frame.shape[0], y + roi_rad)
        roi = frame[y1:y2, x1:x2].copy()
        return roi, (x1, y1, x2, y2)

    def upscale_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Upscale the ROI image using cubic interpolation to enhance its resolution.

        Args:
            roi (np.ndarray): Input region-of-interest image.

        Returns:
            np.ndarray: Upscaled ROI image.
        """
        roi_highres = cv2.resize(
            roi,
            (roi.shape[1] * self.UPSCALE_FACTOR, roi.shape[0] * self.UPSCALE_FACTOR),
            interpolation=cv2.INTER_CUBIC
        )
        return roi_highres

    def filter_and_approximate_contour(self, contour: np.ndarray) -> np.ndarray:
        """
        Filter out outlier points from a contour and approximate its shape.

        The process involves:
          - Converting the contour to integer precision.
          - Computing the center and distance of each point from that center.
          - Filtering points that deviate significantly from the mean distance.
          - Approximating the refined contour using the Ramer–Douglas–Peucker algorithm.

        Args:
            contour (np.ndarray): Original contour points.

        Returns:
            np.ndarray: Approximated contour points with shape (N, 2).
        """
        # Convert contour points to integer precision and reshape for processing.
        refined_contour = contour.astype(np.int32)
        pts = refined_contour.reshape(-1, 2)
        center = np.mean(pts, axis=0)
        distances = np.linalg.norm(pts - center, axis=1)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        # Filter points that deviate significantly from the mean distance.
        if std_distance == 0:
            filtered_pts = pts
        else:
            filtered_pts = pts[np.abs(distances - mean_distance) <= self.DISTANCE_THRESHOLD * std_distance]
        # Ensure at least 3 points for a valid contour.
        if filtered_pts.shape[0] < 3:
            filtered_pts = pts

        filtered_contour = filtered_pts.reshape(-1, 1, 2).astype(np.int32)
        arc_length = cv2.arcLength(filtered_contour, True)
        epsilon = self.EPSILON_FACTOR * arc_length
        approx = cv2.approxPolyDP(filtered_contour, epsilon, True)
        # Flatten the approximated contour to shape (N, 2).
        return np.array([pt[0] for pt in approx])

    def check_if_near_rod(
        self, circle_fits: List[CircleCenter], rod_positions: List[int]
    ) -> Optional[CircleCenter]:
        """
        Select the best candidate circle based on its horizontal proximity to rods.

        For each candidate circle, compute the minimum horizontal distance to any rod.
        If a candidate's distance is less than 2 times the base radius, it qualifies.
        The candidate with the smallest such distance is returned.

        Args:
            circle_fits (List[CircleCenter]): List of fitted circle centers (h, k).
            rod_positions (List[int]): List of rod x-coordinate positions.

        Returns:
            Optional[CircleCenter]: The best candidate circle near a rod or None if none qualify.
        """
        rod_candidates: List[Tuple[CircleCenter, float]] = []
        for h, k in circle_fits:
            # Compute minimum horizontal distance to any rod.
            rod_distance: float = min((abs(h - rod) for rod in rod_positions), default=float('inf'))
            if rod_distance < self.BASE_RADIUS * 2:
                rod_candidates.append(((h, k), rod_distance))
        if rod_candidates:
            best_candidate = min(rod_candidates, key=lambda item: item[1])[0]
            return best_candidate
        return None

    def check_if_near_player(
        self, circle_fits: List[CircleCenter], player_roi: np.ndarray
    ) -> Optional[CircleCenter]:
        """
        Select the best candidate circle based on cues from the player mask.

        First, the function checks whether any candidate falls directly within the white region
        of the player mask. If none do, a distance transform on the inverted mask is performed,
        and the candidate with the smallest distance value is selected.

        Args:
            circle_fits (List[CircleCenter]): List of fitted circle centers (h, k).
            player_roi (np.ndarray): High-resolution ROI of the player mask.

        Returns:
            Optional[CircleCenter]: The best candidate circle center if found; otherwise, None.
        """
        # Create a binary mask where 255 indicates player presence.
        binary_mask: np.ndarray = ((player_roi == 255).astype(np.uint8)) * 255
        inverted_mask = 255 - binary_mask
        # Compute the distance transform on the inverted mask.
        dist_transform = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)

        best_circle: Optional[CircleCenter] = None
        best_score: float = float('inf')
        for h, k in circle_fits:
            if k >= player_roi.shape[0] or h >= player_roi.shape[1]:
                continue
            score = dist_transform[k, h]
            if score < best_score and score < self.BASE_RADIUS * 3:
                best_score = score
                best_circle = (h, k)
        return best_circle

    @staticmethod
    def draw_text(
        roi: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
        padding: int,
    ) -> None:
        """
        Draw text on an image at a specified location with a given padding.

        Args:
            roi (np.ndarray): Image on which to draw the text.
            text (str): Text string to render.
            position (Tuple[int, int]): (x, y) coordinates for the text's center.
            color (Tuple[int, int, int]): Text color in BGR format.
            padding (int): Padding distance above the text position.
        """
        thickness = 1
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = position[0] - text_width // 2
        y = position[1] - padding - text_height // 2
        cv2.putText(roi, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    def detect_ball(
        self, frame: np.ndarray, rod_positions: List[int], player_mask: np.ndarray
    ) -> Optional[Tuple[int, int, int]]:
        """
        Detect and approximate the ball's position in an image frame.

        The detection pipeline includes:
          1. Rough ball detection using color segmentation.
          2. ROI extraction around the ball and upscaling.
          3. Binary mask generation and detection of the largest valid contour.
          4. Contour filtering and shape approximation.
          5. Circle fitting from several initial guesses.
          6. Candidate selection based on consistency, rod proximity, and player mask cues.
          7. (Optional) Annotating the ROI with the detected circle and contour points if verbose.

        Args:
            frame (np.ndarray): Input image frame.
            rod_positions (List[int]): List of rod x-coordinate positions.
            player_mask (np.ndarray): Binary mask representing detected player regions.

        Returns:
            Optional[Tuple[int, int, int]]:
                - (x, y, base_radius) if detection succeeds, where (x, y) is the ball's center
                  in the original frame and base_radius is the ball's radius.
                - None if detection fails at any stage.
        """
        # Step 1: Rough ball detection via color segmentation.
        ball_position = ColorDetection.detect_ball(frame, max_radius=self.MAX_RADIUS)
        if ball_position is None:
            return None

        # Step 2: Extract ROI around the detected ball and upscale it.
        roi, (x1, y1, x2, y2) = self.get_roi(frame, ball_position)
        roi_highres: np.ndarray = self.upscale_roi(roi)

        # Step 3: Generate a binary mask and detect the largest contour.
        mask: np.ndarray = ColorDetection.get_combined_mask(roi_highres)
        largest_contour = ColorDetection.detect_ball(
            roi_highres,
            max_radius=self.MAX_RADIUS * self.UPSCALE_FACTOR,
            approx=cv2.CHAIN_APPROX_NONE,
            returnContour=True
        )
        if largest_contour is None:
            return None

        # Step 4: Filter and approximate the detected contour.
        contour_points: np.ndarray = self.filter_and_approximate_contour(largest_contour)

        # Step 5: Fit circles using multiple initial guesses (corners of the ROI).
        roi_h, roi_w = roi_highres.shape[:2]
        initials: List[CircleCenter] = [
            (0, 0),                     # Top-left corner.
            (roi_w - 1, 0),             # Top-right corner.
            (0, roi_h - 1),             # Bottom-left corner.
            (roi_w - 1, roi_h - 1)       # Bottom-right corner.
        ]
        circle_fits = self.fit_circles(contour_points, initials)

        # Check for consistency among the fitted circle centers.
        base_h, base_k = circle_fits[0]
        centers_consistent: bool = all(
            np.isclose(h, base_h) and np.isclose(k, base_k)
            for h, k in circle_fits
        )

        # Initialize variables for candidate selection and annotation.
        color: Optional[Tuple[int, int, int]] = None
        text: str = ""
        h: Optional[int] = None
        k: Optional[int] = None

        # Remove all circles with invalid positions
        valid_circles: List[Tuple[float, float]] = []
        for candidate in circle_fits:
            # Unpack the candidate's center coordinates
            h_candidate, k_candidate = candidate
            # Check if the candidate circle's center is within the valid frame bounds
            if 0 <= h_candidate < roi_w and 0 <= k_candidate < roi_h:
                valid_circles.append(candidate)

        if centers_consistent:
            # Use the consensus candidate if all centers are consistent.
            text = "CONSENS"
            color = (255, 255, 0)  # Cyan indicates consensus.
            h, k = valid_circles[0]
        else:
            # Extract the player ROI and upscale it.
            player_roi: np.ndarray = player_mask[y1:y2, x1:x2].copy()
            player_roi = self.upscale_roi(player_roi)
            roi_h, roi_w = roi_highres.shape[:2]
            # Adjust rod positions relative to the ROI coordinate system.
            adjusted_rod_positions = [(pos - x1) * self.UPSCALE_FACTOR for pos in rod_positions]

            # Step 2: Find the candidate that falls within a white region in the player mask
            # and is nearest to one of the adjusted rod positions.
            candidate_rod_player: Optional[CircleCenter] = None
            best_distance = float('inf')
            for candidate in valid_circles:
                h_candidate, k_candidate = candidate
                if k_candidate >= roi_h or h_candidate >= roi_w:
                    continue
                if player_roi[k_candidate, h_candidate] != 255:
                    continue
                rod_distance = min(abs(h_candidate - rod) for rod in adjusted_rod_positions)
                if rod_distance < best_distance:
                    best_distance = rod_distance
                    candidate_rod_player = candidate
            if candidate_rod_player is not None:
                text = "I_PLAYER"
                color = (255, 0, 255)  # Magenta indicates candidate meeting rod and player criteria.
                h, k = candidate_rod_player
            else:
                # Step 3: Select the candidate nearest to any rod (within a threshold).
                best_circle = self.check_if_near_rod(valid_circles, adjusted_rod_positions)
                if best_circle is not None:
                    text = "N_ROD"
                    color = (0, 0, 255)  # Red indicates candidate near a rod.
                    h, k = best_circle
                else:
                    # Step 4: Select the candidate closest to a white pixel in the player mask.
                    nearest_player = self.check_if_near_player(valid_circles, player_roi)
                    if nearest_player is not None:
                        text = "N_PLAYER"
                        color = (0, 255, 0)  # Green indicates candidate near player.
                        h, k = nearest_player

        if self.verbose and h is not None and k is not None:
            # Annotate the high-resolution ROI with the fitted circle and text.
            cv2.circle(roi_highres, (h, k), self.R_FIT, color, 2)
            self.draw_text(roi_highres, text, (h, k), color, self.R_FIT)
            # Draw each approximated contour point.
            for point in contour_points:
                cv2.circle(roi_highres, tuple(point), 1, (255, 255, 255), -1)

        # Store the annotated ROI and mask.
        self.frames = (roi_highres, mask)

        # If no valid circle center was selected, return None.
        if h is None or k is None:
            return None

        # Convert the circle center coordinates back to the original frame scale.
        x_ball = int(x1 + h / self.UPSCALE_FACTOR)
        y_ball = int(y1 + k / self.UPSCALE_FACTOR)
        return x_ball, y_ball, self.BASE_RADIUS


def run(frame: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Process a video frame to detect and annotate the ball.

    The function retrieves rod positions and a player mask, then attempts ball detection.
    If detection is successful, it returns a tuple containing the annotated ROI and mask.
    Otherwise, it returns the original frame.

    Args:
        frame (np.ndarray): Current video frame.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            - (annotated ROI, mask) if ball detection succeeds.
            - Original frame if ball detection fails.
    """
    # Obtain rod positions and player mask.
    rod_positions: List[int] = rod.rod_detection(frame)
    player_mask: np.ndarray = player_unet.get_player_unet_mask(frame)
    result = ball_approx.detect_ball(frame, rod_positions, player_mask)
    if result is None:
        return frame
    return ball_approx.frames


if __name__ == "__main__":
    from cv.player_detection import rod, player_unet
    # Initialize ball approximator with verbose annotation enabled.
    ball_approx = BallApprox(base_radius=18, max_radius=21, verbose=True)
    video = DecordVideoProcessor('cv/resources/test_010_1Tor.mp4')
    video.register_window(("Contour Approx", "Mask"), run, width=720, height=720)
    video.process_video_multi()
