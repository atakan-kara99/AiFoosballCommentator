import sys, os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from multiprocessing import Queue

from queue import Empty
from vislib import kpi
import json
import csv
from resources.configs.constants import (
    PAUSE_BETWEEN_ERRORS, 
    PAUSE_BETWEEN_SAME_PLAYER_TOUCHES,
    DELAY_BEFORE_SENDING_LAST_TOUCH
)

def filter_pipeline(subscription_queue: Queue, publish_queue: Queue, log_func):
    """Processes live touch input and forwards a filtered version to the markov pipeline.
    
    - Errors: Limited to avoid excessive frequency.
    - Touches: Only the first touch from a player is sent, with subsequent touches filtered out
      unless PAUSE_BETWEEN_SAME_PLAYER_TOUCHES time has passed.
    """

    last_error_time = -PAUSE_BETWEEN_ERRORS
    last_touch_time = -PAUSE_BETWEEN_SAME_PLAYER_TOUCHES
    last_touch_player: str = None  

    filtered_errors = 0
    filtered_touches = 0

    kpi("Errors filtered", filtered_errors)
    kpi("Touches filtered", filtered_touches)

    while True:
        try:
            msg = subscription_queue.get(timeout=DELAY_BEFORE_SENDING_LAST_TOUCH)

            if msg == "FINISHED":
                log_func(f"Live touch processing: {msg}")
                publish_queue.put(msg)
                return

            assert isinstance(msg, dict), "Message must be a dictionary."

            if msg["type"] == "error":
                last_touch_player = None  # Reset last touch player on error
                if msg["time"] - last_error_time >= PAUSE_BETWEEN_ERRORS:
                    log_func(f"Live touch processing: {msg}")
                    publish_queue.put(msg)
                    last_error_time = msg["time"]
                else:
                    filtered_errors += 1
                    kpi("Errors filtered", filtered_errors + 1)

            elif msg["type"] == "touch":
                # Send immediately if goal or throw-in
                if msg["goal"] or msg["throw_in"]:
                    log_func(f"Live touch processing: {msg}")
                    publish_queue.put(msg)
                    last_touch_time = msg["time"]
                    last_touch_player = None
                    continue

                player = f'{msg["player"]}{msg["team_id"]}' if msg["player"] and msg["team_id"] else None

                # Only send if it's a different player or enough time has passed since last touch
                if player != last_touch_player or msg["time"] - last_touch_time >= PAUSE_BETWEEN_SAME_PLAYER_TOUCHES:
                    log_func(f"Live touch processing: {msg}")
                    publish_queue.put(msg)
                    last_touch_time = msg["time"]
                    last_touch_player = player
                else:
                    # Filter out subsequent touches from the same player within the time window
                    filtered_touches += 1
                    kpi("Touches filtered", filtered_touches)

        except Empty:
            # No action needed on timeout since we're only sending first touches
            pass


def filter_touches_from_file(input_filepath, output_filepath, log_func, file_format="json"):
    """
    Reads a file containing touch events, filters to keep only the first touch by each player,
    and writes filtered touches to an output file. A new first touch is counted if 
    PAUSE_BETWEEN_SAME_PLAYER_TOUCHES time has passed.
    
    :param input_filepath: Path to the input file containing touch events.
    :param output_filepath: Path to the output file where filtered touches will be saved.
    :param log_func: Logging function for debugging output.
    :param file_format: Format of the file ('json' or 'csv').
    """
    
    # Read input file
    if file_format == "json":
        with open(input_filepath, "r") as infile:
            data = json.load(infile)
    elif file_format == "csv":
        with open(input_filepath, "r") as infile:
            reader = csv.DictReader(infile)
            data = {"touches": list(reader)}
    else:
        raise ValueError("Unsupported file format. Use 'json' or 'csv'.")

    last_touch_player = None
    last_touch_time = -PAUSE_BETWEEN_SAME_PLAYER_TOUCHES
    filtered_touches = []
    
    for msg in data["touches"]:
        if msg["type"] == "error":
            continue  # Ignore error messages

        # Send immediately if goal or throw-in
        if msg["goal"] or msg["throw_in"]:
            filtered_touches.append(msg)
            last_touch_player = None
            last_touch_time = -PAUSE_BETWEEN_SAME_PLAYER_TOUCHES
            continue

        # Process touches
        if msg["type"] == "touch":
            player = f'{msg["player"]}{msg["team_id"]}' if msg["player"] and msg["team_id"] is not None else None
            current_time = float(msg["time"]) if isinstance(msg["time"], str) else msg["time"]

            # Add touch if it's a different player or enough time has passed
            if player != last_touch_player or current_time - last_touch_time >= PAUSE_BETWEEN_SAME_PLAYER_TOUCHES:
                filtered_touches.append(msg)
                last_touch_player = player
                last_touch_time = current_time

    # Save filtered touches to output file
    with open(output_filepath, "w") as outfile:
        if file_format == "json":
            json.dump({"touches": filtered_touches}, outfile, indent=4)
        elif file_format == "csv":
            fieldnames = filtered_touches[0].keys() if filtered_touches else []
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered_touches)

    log_func("Touch filtering complete.")

# import sys, os
# sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
# from multiprocessing import Queue

# from queue import Empty
# from vislib import kpi
# import json
# import csv
# from resources.configs.constants import (
#     PAUSE_BETWEEN_ERRORS, 
#     PAUSE_BETWEEN_SAME_PLAYER_TOUCHES,
#     DELAY_BEFORE_SENDING_LAST_TOUCH
# )

# def filter_pipeline(subscription_queue: Queue, publish_queue: Queue, log_func):
#     """Processes live touch input and forwards a filtered version to the markov pipeline.
    
#     - Errors: Limited to avoid excessive frequency.
#     - Touches: Sent immediately if from a different figure or after a long pause.
#       - The last touch in a fast sequence is **buffered** and sent if no new touch replaces it within a short delay.
#     """

#     last_error_time = -PAUSE_BETWEEN_ERRORS
#     last_touch_time = -PAUSE_BETWEEN_SAME_PLAYER_TOUCHES
#     last_touch_player: str = None  

#     delayed_touch = None  # Buffer for the last touch in a sequence
#     filtered_errors = 0
#     filtered_touches = 0

#     kpi("Errors filtered", filtered_errors)
#     kpi("Touches filtered", filtered_touches)

#     while True:
#         try:
#             msg = subscription_queue.get(timeout=DELAY_BEFORE_SENDING_LAST_TOUCH)

#             if msg == "FINISHED":
#                 # Ensure any delayed touch is sent before finishing
#                 if delayed_touch:
#                     log_func(f"Sending final delayed touch: {delayed_touch}")
#                     publish_queue.put(delayed_touch)
#                     delayed_touch = None
#                 log_func(f"Live touch processing: {msg}")
#                 publish_queue.put(msg)
#                 return

#             assert isinstance(msg, dict), "Message must be a dictionary."

#             if msg["type"] == "error":
#                 if msg["time"] - last_error_time >= PAUSE_BETWEEN_ERRORS:
#                     log_func(f"Live touch processing: {msg}")
#                     publish_queue.put(msg)
#                     last_error_time = msg["time"]
#                 else:
#                     filtered_errors += 1
#                     kpi("Errors filtered", filtered_errors + 1)

#             elif msg["type"] == "touch":
#                 # Send immediately if goal or throw-in
#                 if msg["goal"] or msg["throw_in"]:
#                     if delayed_touch:
#                         log_func(f"Sending delayed touch: {delayed_touch}")
#                         publish_queue.put(delayed_touch)
#                         delayed_touch = None
#                     log_func(f"Live touch processing: {msg}")
#                     publish_queue.put(msg)
#                     last_touch_time = msg["time"]
#                     last_touch_player = None
#                     continue

#                 player = f'{msg["player"]}{msg["team_id"]}' if msg["player"] and msg["team_id"] else None

#                 # If different player or enough time has passed, send immediately
#                 if player != last_touch_player or msg["time"] - last_touch_time >= PAUSE_BETWEEN_SAME_PLAYER_TOUCHES:
#                     # Send any delayed touch before replacing it
#                     if delayed_touch:
#                         log_func(f"Sending delayed touch: {delayed_touch}")
#                         publish_queue.put(delayed_touch)
#                         delayed_touch = None

#                     delayed_touch = msg
#                     last_touch_time = msg["time"]
#                     last_touch_player = player

#                 else:
#                     filtered_touches += 1
#                     kpi("Touches filtered", filtered_touches)
#                     # Buffer this touch as the last in a quick sequence
#                     delayed_touch = msg

#         except Empty:
#             # If no new message comes in within the delay threshold, send the buffered touch
#             if delayed_touch:
#                 log_func(f"Sending last buffered touch due to timeout: {delayed_touch}")
#                 publish_queue.put(delayed_touch)
#                 delayed_touch = None


# def filter_touches_from_file(input_filepath, output_filepath, log_func, file_format="json"):
#     """
#     Reads a file containing touch events, filters out consecutive touches by the same player,
#     and writes filtered touches to an output file.
    
#     :param input_filepath: Path to the input file containing touch events.
#     :param output_filepath: Path to the output file where filtered touches will be saved.
#     :param log_func: Logging function for debugging output.
#     :param file_format: Format of the file ('json' or 'csv').
#     """
    
#     # Read input file
#     if file_format == "json":
#         with open(input_filepath, "r") as infile:
#             data = json.load(infile)
#     elif file_format == "csv":
#         with open(input_filepath, "r") as infile:
#             reader = csv.DictReader(infile)
#             data = {"touches": list(reader)}
#     else:
#         raise ValueError("Unsupported file format. Use 'json' or 'csv'.")

#     last_touch_player = None
#     delayed_touch = None  # Buffer for tracking the last touch to be stored
#     filtered_touches = []

    
#     for msg in data["touches"]:
#         if msg["type"] == "error":
#             continue  # Ignore error messages

#         # Send immediately if goal or throw-in
#         if msg["goal"] or msg["throw_in"]:
#             if delayed_touch:
#                 filtered_touches.append(delayed_touch)
#                 delayed_touch = None
#             filtered_touches.append(msg)
#             last_touch_player = None
#             continue

#         # Process touches
#         if msg["type"] == "touch":
#             player = f'{msg["player"]}{msg["team_id"]}' if msg["player"] and msg["team_id"] is not None else None

#             if player != last_touch_player:
#                 # Store the last touch of the previous sequence before switching players
#                 if delayed_touch:
#                     filtered_touches.append(delayed_touch)
#                     delayed_touch = None

#                 # Add new touch and update tracking variables
#                 filtered_touches.append(msg)
#                 last_touch_player = player
#             else:
#                 # If it's the same player, update the delayed touch (keeping only the last one)
#                 delayed_touch = msg

#     # Ensure the last touch from the last sequence is included
#     if delayed_touch:
#         filtered_touches.append(delayed_touch)
#     # Save filtered touches to output file
#     with open(output_filepath, "w") as outfile:
#         if file_format == "json":
#             json.dump({"touches": filtered_touches}, outfile, indent=4)
#         elif file_format == "csv":
#             fieldnames = filtered_touches[0].keys() if filtered_touches else []
#             writer = csv.DictWriter(outfile, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerows(filtered_touches)

#     log_func("Touch filtering complete.")