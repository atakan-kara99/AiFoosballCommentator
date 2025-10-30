import numpy.typing as npt
from typing import Optional, Any

from cv.entities import Ball
from cv.debug_player import DecordVideoProcessor
from .ball_detection import BallDetection
from .goal_detection import GoalDetection
from .throw_in_detection import ThrowInDetection


def run(frame: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Process a video frame to detect players, the ball, throw-in events, and goal events.

    This function performs the following steps:
      - Detects player positions using rod detection and a player segmentation mask.
      - Detects the ball using the BallDetection instance (legacy or new method).
      - Draws the ball and its detection history on the frame.
      - Checks for throw-in events in designated zones and draws the zones.
      - Checks for goal events in designated zones and draws the zones.

    Args:
        frame (npt.NDArray[Any]): The current video frame in BGR format.

    Returns:
        npt.NDArray[Any]: The annotated video frame.
    """
    # PLAYER DETECTION
    # Obtain rod positions and player segmentation mask.
    rod_positions = rod.rod_detection(frame)
    player_mask = player_unet.get_player_unet_mask(frame)

    # BALL DETECTION
    global prev_ball  # Use global variable to track the previous ball state.
    ball_position = ball_detection.detect_ball(frame, rod_positions, player_mask)
    # Create a Ball instance if detection is successful.
    ball = None if ball_position is None else Ball(*ball_position, frame, prev_ball)
    prev_ball = ball  # Update the previous ball state.

    # Draw ball detection history on the frame.
    frame = ball_detection.draw_last_positions(frame)
    if ball is not None:
        ball.draw(frame)

    # THROW-IN DETECTION
    # Check for throw-in events in both the top and bottom zones.
    t_i_t = throw_in_t.check_throw_in(ball)
    t_i_b = throw_in_b.check_throw_in(ball)
    frame = throw_in_t.zone.draw(frame, fill=t_i_t)
    frame = throw_in_b.zone.draw(frame, fill=t_i_b)

    # GOAL DETECTION
    # Check for goal events on the left and right sides.
    g_l = goal_l.check_goal(ball)
    g_r = goal_r.check_goal(ball)
    frame = goal_l.zone.draw(frame, fill=g_l)
    frame = goal_r.zone.draw(frame, fill=g_r)

    return frame


if __name__ == "__main__":
    # Import player detection modules that provide rod detection and player segmentation.
    from cv.player_detection import rod, player_unet

    # Define frame dimensions.
    width, height = 1280, 720

    # Initialize global variables.
    prev_ball: Optional[Ball] = None
    ball_detection = BallDetection(width, height, verbose=True)

    # Initialize throw-in detection for TOP and BOTTOM zones.
    throw_in_t = ThrowInDetection(width, height, side="TOP", verbose=True)
    throw_in_b = ThrowInDetection(width, height, side="BOTTOM", verbose=True)

    # Initialize goal detection for LEFT and RIGHT sides.
    goal_l = GoalDetection(width, height, side="LEFT", verbose=True)
    goal_r = GoalDetection(width, height, side="RIGHT", verbose=True)

    # Create and configure the video processor.
    video = DecordVideoProcessor('cv/resources/yellow/test_009.mp4')
    video.register_window("Main", run)
    video.process_video_multi()
