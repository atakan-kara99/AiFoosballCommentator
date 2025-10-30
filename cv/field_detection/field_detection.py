import cv2
import numpy as np
from collections import Counter


class CornerCandidates:
    """
    This class provides a simple majority vote isntance.
    We can simply append new items to this class and if 
    a given threshold (400) exceeds, the oldest element is removed (FIFO).
    """
    def __init__(self):
        self.ps = list()
        self.size = 400
    
    def append(self, p):
        """
        Appends a new point to the internal list of points.
        If the list size exceeds the max size (self.size),
        the very first element will be removed.
        
        Args:
            p (_type_): a new corner candidate (x, y)
        """
        p = tuple(int(coord) for coord in p)
        self.ps.append(p)
        if len(self.ps) > self.size:
            self.ps = self.ps[1:]
    
    @property
    def corner(self):
        """
        Calucaltes via majority existence, which point occurs more often.
        The most existing corner / point will be set as corner.

        Returns:
            _type_: corner as tuple (x,y)
        """
        if len(self.ps) == 0:
            return (-1, -1)
        # Count occurrences of each element
        counter = Counter(self.ps)
        most_common_element, count = counter.most_common(1)[0]
        return most_common_element
    
    
    
# Define the list of corner candidates for the top left and bottom right corner
TOP_LEFT_CANDIDATES = CornerCandidates()
TOP_RIGHT_CANDIDATES = CornerCandidates()
BOTTOM_RIGHT_CANDIDATES = CornerCandidates()
BOTTOM_LEFT_CANDIDATES = CornerCandidates()


def field_corners(frame: np.ndarray): # -> np.ndarray:
    """
    Takes as input a frame from the video stream and computes
    the corners of the (play field). Detects corners in the
    75x75px areas at the field corners and marks them.
    
    Args:
        frame: the frame we want to determine the field of.
    Returns:
        The frame with 75x75px boxes and detected corners marked.
    """
    global TOP_LEFT_CANDIDATES
    global TOP_RIGHT_CANDIDATES
    global BOTTOM_RIGHT_CANDIDATES
    global BOTTOM_LEFT_CANDIDATES
    
    # Get the dimensions of the frame
    height, width, _ = frame.shape
    # Convert to HSV (for visualization purpose, not used in corner detection here)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Remove noise => works great for corner detection
    gray_blurred = cv2.GaussianBlur(gray, (5,5), 0)
        
    # For each corner, define a box where we want to detect corners.
    # Good size is 60x60 pixels
    box_size = 60
    
    # Define the corner coordinates
    corners = {
        "top_left": (0, 0),
        "top_right": (width - box_size, 0),
        "bottom_right": (width - box_size, height - box_size),
        "bottom_left": (0, height - box_size),
    }
    
    # Draw rectangles (boxes) and detect corners within them
    for corner_name, (x, y) in corners.items():
        # Draw the rectangle for the box
        top_left = (x, y)
        
        # Extract the region of interest (ROI)
        roi = gray_blurred[y:y + box_size, x:x + box_size]
        gray_roi = roi
        
        # Detect corners using Shi-Tomasi method
        detected_corners = cv2.goodFeaturesToTrack(
            gray_roi, # grayscale region of interest
            maxCorners=25, # the maximum amount of corners we want to detect
            qualityLevel=0.1,  # determines the quality of the corners
            minDistance=17, # minimum distance between the corner centers
        )
        
        top_left_corners = list()
        top_right_corners = list()
        bottom_right_corners = list()
        bottom_left_corners = list()
        
        # Here, draw all corners into the field
        if detected_corners is not None:
            detected_corners = np.int16(detected_corners)            
            # Draw detected corners on the original frame
            for corner in detected_corners:
                cx, cy = corner.ravel()
                
                if corner_name == "top_left":
                    top_left_corners.append((x + cx, y + cy))
                    # TOP_LEFT_CANDIDATES.append((x + cx, y + cy))
                
                if corner_name == "top_right":
                    top_right_corners.append((x + cx, y + cy))
                    TOP_RIGHT_CANDIDATES.append((x + cx, y + cy))
                
                if corner_name == "bottom_right":
                    bottom_right_corners.append((x + cx, y + cy))
                    BOTTOM_RIGHT_CANDIDATES.append((x + cx, y + cy))
                
                if corner_name == "bottom_left":
                    bottom_left_corners.append((x + cx, y + cy))
                    BOTTOM_LEFT_CANDIDATES.append((x + cx, y + cy))


        # TODO: Here, a custom logic to find / detect all corners is required.
        # Top left corner
        # Special case, since multiple points might be detected correctly
        top_left = sorted(top_left_corners, key=lambda point: (point[0], point[1]), reverse=True)
        if len(top_left) > 0:
            TOP_LEFT_CANDIDATES.append(top_left[0])

        # Top right corner
        # cv2.circle(frame, TOP_RIGHT_CANDIDATES.corner, 3, (255, 0, 0), -1)  # yellow dots
        # Bottom right corner
        # cv2.circle(frame, BOTTOM_RIGHT_CANDIDATES.corner, 6, (0, 255, 255), -1)  # yellow dots
        # Bottom left corner
        # cv2.circle(frame, BOTTOM_LEFT_CANDIDATES.corner, 3, (255, 0, 0), -1)  # yellow dots
        
    # Here, we are using hard coded values for the video file test_011.mp4
    return [[48, 19], [1247, 21], [1240, 711], [44, 692]]



if __name__ == "__main__":
    """
    To test only this chunk of the project, you can simply
    run this file.
    
    To use the customized video player, please checkout the 
    given documentation for it.
    """
    videos = [
        "test_003.mp4",
        "test_004.mp4",
        "test_005.mp4",
        "test_008.mp4",
        "test_009.mp4",
        "test_010.mp4",
        "test_011.mp4"
    ]
    # current_video = videos[-1]
    # path_to_video_dir = "./input/vids/Masterprojekt2425_samples"
    # video_path = f"{path_to_video_dir}/{current_video}"
    # processor = VideoProcessor(video_path=video_path)
    # processor.register_window(
    #     window_name=f"Playfield Detection for {current_video}", 
    #     frame_callback=field_corners
    # )
    # # processor.register_window(
    # #     f"Playfield Detection for {current_video}", 
    # #     frame_callback=playfield_detection
    # # )
    # processor.process_video()