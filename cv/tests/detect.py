import sys
import os
import numpy as np
from decord import VideoReader, cpu
from typing import List, Tuple, Optional

from ..entities import Ball
from ..ball import BallDetection, ThrowInDetection, GoalDetection


class Detector:
    """
    Processes a video file to detect game events or ball positions.

    This detector reads frames from a video using Decord, applies detection algorithms,
    and logs either events (e.g., THROW-IN, GOAL) or ball positions detected in each frame.
    The results are saved to a file for further analysis. For ball positions, the file name
    is appended with '_ball'.
    """

    def __init__(self, video_path: str, mode: str = "events") -> None:
        """
        Initialize the Detector with the specified video file and detection mode.

        Args:
            video_path (str): Path to the video file.
            mode (str): Detection mode, either "events" for event detection or "ball" for ball position detection.

        Raises:
            FileNotFoundError: If the video file cannot be opened.
        """
        try:
            self.video_reader: VideoReader = VideoReader(video_path, ctx=cpu(0))
        except Exception as e:
            raise FileNotFoundError(f"Error: Unable to open video file: {video_path}. {e}")

        self.total_frames: int = len(self.video_reader)
        self.progress_bar_len: int = 66
        self.events: List[Tuple[int, str]] = []
        self.mode = mode.lower()

        # File-related attributes for saving events or ball positions.
        self.base_name: str = os.path.splitext(os.path.basename(video_path))[0]
        if self.mode == "ball":
            self.events_file: str = f"cv/tests/detected/{self.base_name}_ball.txt"
        else:
            self.events_file: str = f"cv/tests/detected/{self.base_name}.txt"
        if os.path.exists(self.events_file):
            user_input: str = input(
                f"The file {self.events_file} already exists. Do you want to overwrite it? (y/n): "
            ).strip().lower()
            if user_input != 'y':
                print("Operation aborted by the user.")
                sys.exit(0)

    def print_progress_bar(self, progress: int) -> None:
        """
        Print a progress bar to the terminal.

        Args:
            progress (int): Current progress as a percentage (0 to 100).
        """
        filled_length: int = int(self.progress_bar_len * progress // 100)
        bar: str = '█' * filled_length + '░' * (self.progress_bar_len - filled_length)
        sys.stdout.write(f'\rLoading.. {bar} {progress}% ')
        sys.stdout.flush()

    def process_video(self) -> None:
        """
        Process the video frame by frame, detect events or ball positions, and display a progress bar.

        For each frame, the detector converts it to a NumPy array, processes it for detections,
        updates the progress bar, and saves all detected information to a file after processing.
        """
        print(f"Start video processing for {self.base_name}.mp4 ({self.mode} detection)")
        last_printed_progress: int = -1

        for frame_counter, frame in enumerate(self.video_reader):
            # Convert frame to NumPy array and adjust color channel order if needed.
            frame_np: np.ndarray = frame.asnumpy()
            frame_np = frame_np[:, :, ::-1]  # Convert from RGB to BGR.
            self.process_frame(frame_np, frame_counter)

            progress: int = int((frame_counter / self.total_frames) * 100)
            if progress > last_printed_progress:
                self.print_progress_bar(progress)
                last_printed_progress = progress

        self.print_progress_bar(100)
        print("\nVideo processing complete.")
        self.save_events()

    def save_events(self) -> None:
        """
        Save the detected events or ball positions to a text file.

        Each entry is written as a line containing the frame number and detection result.
        """
        with open(self.events_file, "w") as f:
            for frame_number, detection in self.events:
                f.write(f"{frame_number}: {detection}\n")
        print(f"Results saved to {self.events_file}")

    def process_frame(self, frame: np.ndarray, frame_number: int) -> None:
        """
        Process a single video frame to detect game events or ball positions.

        For event detection mode:
            - Performs ball detection and checks for throw-in or goal events.
        For ball position mode:
            - Detects the ball and logs its position.

        Args:
            frame (np.ndarray): The current video frame in BGR format.
            frame_number (int): The sequential index of the current frame.
        """
        global prev_ball, ball_detection, throw_in_t, throw_in_b, goal_l, goal_r

        # PLAYER DETECTION
        rod_positions = rod.rod_detection(frame)
        player_mask = player_unet.get_player_unet_mask(frame)

        # BALL DETECTION
        ball_position: Optional[Tuple[int, int, int]] = ball_detection.detect_ball(frame, rod_positions, player_mask)
        
        if self.mode == "ball":
            # In ball detection mode, record the ball's position.
            if ball_position is not None:
                self.events.append((frame_number, f"{ball_position[0]}, {ball_position[1]}"))
        else:
            # Event detection mode.
            ball: Optional[Ball] = None if ball_position is None else Ball(*ball_position, frame, prev_ball)
            prev_ball = ball

            if throw_in_t.check_throw_in(ball):
                self.events.append((frame_number, "THROW-IN-TOP"))
            if throw_in_b.check_throw_in(ball):
                self.events.append((frame_number, "THROW-IN-BOTTOM"))
            if goal_l.check_goal(ball):
                self.events.append((frame_number, "GOAL-LEFT"))
            if goal_r.check_goal(ball):
                self.events.append((frame_number, "GOAL-RIGHT"))


if __name__ == "__main__":
    from cv.player_detection import rod, player_unet

    # Initialize global variables for detection.
    width, height = 1280, 720
    prev_ball = None
    ball_detection = BallDetection(width, height)
    throw_in_t = ThrowInDetection(width, height, side="TOP")
    throw_in_b = ThrowInDetection(width, height, side="BOTTOM")
    goal_l = GoalDetection(width, height, side="LEFT")
    goal_r = GoalDetection(width, height, side="RIGHT")

    # Hardcoded parameters
    video_path = 'cv/resources/test_011_2Tore.mp4'
    mode = 'ball'  # Change to 'events' for event detection mode

    # Create a Detector instance and start processing the video.
    Detector(video_path, mode=mode).process_video()
