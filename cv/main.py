"""
Soccer Game Video Analysis Script

This script processes a video feed of a soccer game to detect players, ball movements, and key game events such as 
goals and throw-ins. It uses computer vision techniques to extract and analyze game data.

Main Functionality:
- Detects players using deep learning-based segmentation (player detection).
- Identifies the ball and tracks its movement across frames.
- Recognizes key game events like throw-ins and goals.
- Logs detected events and optionally saves them for evaluation.
- Can operate in both live mode (real-time analysis) and offline mode (post-game video processing).

Usage:
- Live Mode: If a `publish_queue` is provided, detected events are sent in real-time.
- Offline Mode: Saves detected events as a JSON file for further evaluation.

Key Components:
- `analyse()`: The main function that processes video frames and extracts relevant game information.
- Player and ball detection models from `cv.player_detection` and `cv.ball` modules.
- Event detection mechanisms for goals and throw-ins.
- Touch detection and event logging using `TouchLogger`.

"""

from typing import Optional
import multiprocessing
import json
from datetime import datetime
import time
import cv2

from vislib import kpi
from cv.touchlog import TouchLogger
from cv.entities.gamestate import Player, GameState, Ball
from cv.player_detection.player_unet import player_detection
from cv.ball.goal_detection import GoalDetection
from cv.ball.throw_in_detection import ThrowInDetection
from cv.ball.ball_detection import BallDetection
from cv.field_detection import field_corners
from cv.utils import Utils

def analyse(
    publish_queue: Optional[multiprocessing.Queue] = None,
    logger=None,
    cap=None,
    eval: bool = False,
    video_name: str = None,
    verbose: bool = False,
    debug: bool = False,
    training_id="training_id_not_set"
):
    """
    Analyzes a video feed to detect players, ball movements, and game events.

    Parameters:
    - publish_queue (Optional[multiprocessing.Queue]): Queue for live mode publishing.
    - logger: Logging function.
    - cap: OpenCV VideoCapture object.
    - eval (bool): If True, saves detected touches for evaluation.
    - video_name (str): Name of the video being analyzed.
    - verbose (bool): If True, prints detailed logs.
    - debug (bool): If True, enables debug mode with additional outputs.
    - training_id (str): id of training current training (Markov)

    Returns:
    - None (Results are either logged or saved)
    """
    
    live = publish_queue is not None
    if live:
        debug = False  # Disable debug in live mode
        logger("Switched to live-mode.")

    # Initialize frame count and previous ball data
    curr_frame = 0
    prev_ball = None

    # Frame initialization flags
    is_init = False
    f_width, f_height = None, None

    # Timing variables
    start_time = int(time.time() * 1000)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no frame is read

        frame_time = int(time.time() * 1000) - start_time
        start_time2 = int(time.time() * 1000)

        # Field localization and masking
        try:
            corners = field_corners(frame)
            cropped_frame = Utils.crop_frame(frame, corners)
        except Exception as e:
            cropped_frame = frame
            if verbose:
                log_message = f"Error in field detection in frame {curr_frame}: {e}"
                logger(log_message) if live else print(log_message)

        if debug and not live:
            cv2.imshow("Cropped Frame", cropped_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Video closed!")
                break

        # Initialize frame-shape-dependent variables once
        if not is_init:
            f_height, f_width = frame.shape[:2]
            B = BallDetection(f_width, f_height)
            TI_T = ThrowInDetection(f_width, f_height, side="TOP", verbose=debug)
            TI_B = ThrowInDetection(f_width, f_height, side="BOTTOM", verbose=debug)
            G_L = GoalDetection(f_width, f_height, side="LEFT", verbose=debug)
            G_R = GoalDetection(f_width, f_height, side="RIGHT", verbose=debug)

            touch_count = 0
            touchlogger = TouchLogger()

            touches = {"logger": touchlogger.get_config_json()} if eval else {}
            touches["touches"] = []

            is_init = True

        # Player detection
        try:
            if frame is not None:
                player_data, player_foot_data, player_mask, rod_positions = player_detection(frame)
                player_list = Player.generate_player_list(player_data, player_foot_data)
            else:
                player_data = None
                player_list = []
        except Exception as e:
            player_list = []
            if verbose:
                log_message = f"Player detection failed in frame {curr_frame}: {e}"
                logger(log_message) if live else print(log_message)

        if debug and not live:
            print(f"Player Data: {player_data}")

        # Ball detection
        try:
            ball_pos = B.detect_ball(frame, rod_positions, player_mask)

            ball = Ball(*ball_pos, frame, prev_ball) if ball_pos else None
            prev_ball = ball

            if debug and ball:
                print(f"Ball Detected: {ball}")

            # Check ball events
            event = ""
            throw_in_ball = None

            if TI_T.check_throw_in(ball):
                event = "throw in top"
                throw_in_ball = ball
            elif TI_B.check_throw_in(ball):
                event = "throw in bottom"
                throw_in_ball = ball

            if G_L.check_goal(ball):
                event = "goal left"
            elif G_R.check_goal(ball):
                event = "goal right"

            if debug and event != "":
                print(f"Event Detected: {event}")

        except Exception as e:
            ball, event = None, ""
            if verbose:
                log_message = f"Ball detection failed in frame {curr_frame}: {e}"
                logger(log_message) if live else print(log_message)

        # Process game state and touches
        if curr_frame >= 3 and (ball or player_list):
            try:
                gamestate = GameState(
                    ball_data=ball,
                    ps_data=player_list,
                    id=curr_frame,
                    frame_time=frame_time,
                    frame_no=curr_frame,
                    shape=frame.shape,
                )

                if throw_in_ball:
                    gamestate.set_ball_data(throw_in_ball)

                touch = touchlogger.check_touch(gamestate, event, verbose, debug)

                if touch:
                    touch_count += 1
                    kpi("Touch Count", touch_count)
                    touch_data = touch.get_json()
                    if live:
                        logger(f"Sending touch to Markov: {touch_data}")
                        publish_queue.put(touch_data)
                    else:
                        touches["touches"].append(touch_data)
                        print(f"Touch Detected: {touch_data}")

                elif debug:
                    print("No touch detected.")

            except Exception as e:
                if verbose:
                    log_message = f"Touch processing failed in frame {curr_frame}: {e}"
                    logger(log_message) if live else print(log_message)

        curr_frame += 1  # Increment frame counter

        kpi("CV Avg Proc Time", f"{int((time.time() * 1000 - start_time) / curr_frame)} ms")

    # Save touch data if not in live mode
    if not live:
        print("Not live")
        now = datetime.now()
        formatted_date_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        if eval:
            path = f"cv/tests/detected/{video_name}/touches.json" # + "_" + formatted_date_time 
        else:
            path = f"training_resources/touches_{training_id}.json"
        with open(path, "w") as touches_file:
            json.dump(touches, touches_file, indent=4)
            print(f"Touches saved under {path}")


if __name__ == "__main__":
    video = "test_011_2Tore"
    camera = f"./cv/resources/{video}.mp4"

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
    else:
        print("Camera successfully accessed.")
        analyse(
            publish_queue=None, logger=None, cap=cap, eval=True, video_name=video, verbose=False, debug=True
        )
        cap.release()
        cv2.destroyAllWindows()