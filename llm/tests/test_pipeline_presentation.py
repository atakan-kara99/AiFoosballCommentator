"""
Pipeline Integration Test Suite
============================

This module provides an end-to-end test environment for the complete
commentary pipeline, simulating real-time game events and commentary
generation.

Components Tested:
---------------
1. Event Processing:
   - Event cleaning and validation
   - Priority assignment
   - Interrupt detection

2. Buffer Management:
   - Event queueing
   - Priority-based retrieval
   - Overflow handling

3. Commentary Generation:
   - Prompt creation
   - LLM response generation
   - Natural speech output

Performance Metrics:
-----------------
- Frame-based timing
- Generation latency
- Speech pacing
"""

# Standard library imports
import sys
import os
import time
import json

# Add parent directory to Python path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Pipeline component imports
from modules import event_cleaner as cleaner
from modules import buffer as b
from modules import prompt_generator as prompter
from modules import llm_interface as llm
from modules import commentator


def print_header():
    """
    Display formatted header for test environment.
    Provides visual separation and context in console output.
    """
    print("--------------------------------------------")
    print("|                  TEST                    |")
    print("|      Pipeline for the Presentation       |")
    print("--------------------------------------------")
    print("")


# Initialize test environment
print_header()

# Initialize pipeline components
buffer = b.Buffer()                                    # Event buffer for priority queueing
llm = llm.LLMInterface(model="meta-llama/Llama-3.2-3B-Instruct")  # LLM interface
speaker = commentator.Commentator()                    # Speech output manager

# Load test event data
with open('tests/eventlog.json', 'r') as file:
    event_log_dict = json.load(file)

# Extract events and initialize counters
event_log = event_log_dict["events"]
event_counter = 0                                      # Tracks processed events
frame_counter = 0                                      # Tracks game time progression

# Prepare output structure for generated comments
json_output = {"comments": []}

# Main event processing loop
while event_counter <= len(event_log)-1:
    next_event = event_log[event_counter]
    frame_id = next_event["frame_no"]
    interupted = False

    # Event Producer Phase
    if frame_counter == frame_id:
        # Process and validate event
        event = cleaner.clean(next_event)             # Clean raw event data
        event = cleaner.add_prio_entry(event)         # Assign priority level

        # Handle high-priority interrupts
        if cleaner.checkInterupt(event):
            interupted = True

            # Generate immediate commentary
            start_time = time.time()
            prompt = prompter.generate_prompt("event", event)
            comment = llm.generate_comment(prompt)
            end_time = time.time()
            
            # Calculate timing and frame alignment
            time_needed = end_time - start_time
            frames_id_timestamp = round(time_needed * 57 + frame_counter)

            # Format and deliver interrupt commentary
            comment = "Oh wait... " + comment.replace('"', '').replace(" ", "", 1)
            json_comment = {
                "frame_no": frames_id_timestamp,
                "comment": comment.replace('"', '')
            }
            json_output["comments"].append(json_comment)

            # Display and speak commentary
            print(f"Frame_id {frames_id_timestamp} ({round(frames_id_timestamp/30)}s) ({time_needed:.2}s), {event['event']} {event['involved_players']}: \n")
            speaker.speak(comment.replace('"', ''))
            print("")

        else:
            # Queue non-interrupt event
            event["time"] = time.time()
            buffer.add(event)

        event_counter += 1

    # Consumer Phase - Regular Commentary
    if (not(interupted)) and (frame_counter % 114 == 0) and (frame_counter != 0):
        try:
            # Retrieve highest priority event
            event = buffer.next()

            # Generate commentary
            start_time = time.time()
            prompt = prompter.generate_prompt("event", event)
            comment = llm.generate_comment(prompt)
            end_time = time.time()
            
            # Calculate timing
            time_needed = end_time - start_time
            frames_id_timestamp = round(time_needed * 57 + frame_counter)

            # Format and deliver commentary
            json_comment = {
                "frame_no": frames_id_timestamp,
                "comment": comment.replace('"', '').replace(" ", "", 1)
            }
            json_output["comments"].append(json_comment)

            # Display and speak commentary
            print(f"Frame_id {frames_id_timestamp} ({round(frames_id_timestamp/30)}s) ({time_needed:.2}s), {event['event']} {event['involved_players']}: \n")
            speaker.speak(comment.replace('"', ''))
            print("")

        except TypeError:
            # Handle empty buffer with statistics
            if frame_counter % 57 == 0:
                json_comment = {
                    "frame_no": frame_counter + 57,
                    "comment": comment.replace('"', '').replace(" ", "", 1)
                }
                json_output["comments"].append(json_comment)
                print(f"Frame_id {frame_counter + 57} ({round((frame_counter + 30)/30)}s) (1.00s): ")
                speaker.speak("Nothing important happened. Here is a interesting statistic! \n")

    # Advance simulation time
    interupted = False
    frame_counter += 1

# Uncomment to save generated comments
#with open("output_comments.json", "w", encoding="utf-8") as json_file:
#    json.dump(json_output, json_file, indent=4, ensure_ascii=False)