"""
LLM Pipeline Module
==================

This module implements the Large Language Model (LLM) pipeline for real-time table football commentary.
It serves as the main script, combining various components to generate contextual and engaging
commentary based on game events and statistics.

Key Components:
-------------
1. Event Processing:
   - Receives and buffers game events
   - Handles interrupts for dynamic commentary
   - Processes statistics during quiet periods

2. Commentary Generation:
   - Uses LLaMA model for natural language generation
   - Maintains context awareness across events
   - Balances between event-driven and statistics-based commentary

3. Speech Output:
   - Manages text-to-speech timing
   - Handles concurrent speech requests
   - Prevents commentary overlap

Input/Output:
-----------
Input: Dictionary containing game events (goals, touches, possession changes)
Output: Natural language commentary delivered to the frontend

Dependencies:
-----------
- event_cleaner: Filters and prioritizes game events
- buffer: Manages event queue and timing
- prompt_generator: Creates context-aware prompts
- llm_interface: Handles LLM interaction
- commentator: Manages text-to-speech output
"""

# Standard library imports
import threading
import queue
import time
import json
import random
import os
from multiprocessing import Queue

# Local imports from the project's modules
from .config import GENERAL_SETTINGS, COMMENTATOR_CONFIG, LLM_INTERFACE_CONFIG
from .modules import (
    event_cleaner as cleaner,
    buffer,
    prompt_generator as prompter,
    llm_interface as llm,
    commentator as speaker
)

# Configuration constants from settings
APP_NAME   = GENERAL_SETTINGS["APP_NAME"]
VERSION    = GENERAL_SETTINGS["VERSION"]
DEBUG_MODE = GENERAL_SETTINGS["DEBUG_MODE"]

# Model access configuration
MODEL_URL       = LLM_INTERFACE_CONFIG["DEFAULT_MODEL_URL"]
STATISTICS_PATH = LLM_INTERFACE_CONFIG["DEFAULT_STATISTICS_PATH"]

# Timing parameters for statistics generation
STATISTICS_INTERVAL = COMMENTATOR_CONFIG["STATISTICS_INTERVAL"]        # Minimum time between consecutive statistics comments
STATISTICS_TIME     = COMMENTATOR_CONFIG["STATISTICS_TIME"]         # Initial delay after events before statistics

class LLM:
    """
    Class representing the Large Language Model (LLM) pipeline.

    Attributes:
    ----------
    log_func : function
        Logging function for output messages
    push_text_function : function
        Function to push text to the output
    kpi : function
        Function to track key performance indicators (KPIs)
    event_queue : Queue
        Queue to receive game events
    model : str
        LLM model to use for commentary generation
    statistics_path : str
        Path to the statistics file

    Methods:
    -------
    __init__()
        Initializes the LLM pipeline
    event_listener_thread()
        Thread to process game events
    commentator_thread()
        Thread to generate commentary
    kill()
        Stops the LLM pipeline
    join()
        Waits for the LLM pipeline to finish
    """

    def __init__(
        self,
        log_func,
        push_text_function,
        kpi,
        event_queue: Queue,
        model: str = MODEL_URL,
        statistics_path: str = STATISTICS_PATH
    ) -> None:
        """
        Initializes the LLM pipeline.

        Args:
        ----
        log_func (function): 
            Logging function for output messages
        push_text_function (function): 
            Function to push text to the output
        kpi (function): 
            Function to track key performance indicators (KPIs)
        event_queue (Queue): 
            Queue to receive game events
        model (str): 
            LLM model to use for commentary generation
        statistics_path (str): 
            Path to the statistics file
        """
        self.log_func = log_func
        self.push_text_function = push_text_function
        self.kpi = kpi

        self.event_counter = 0
        
        # Convert relative path to absolute path in Docker environment
        if not os.path.isabs(statistics_path):
            # In Docker, files are mounted under /app
            base_path = "/app"
            statistics_path = os.path.join(base_path, statistics_path)
            
        self.statistics_path = statistics_path
        self.log_func(f"Using statistics path: {self.statistics_path}")
        
        self.last_event_time = time.time() * 1000  # Convert to milliseconds
        self.last_statistics_time = time.time() * 1000

        # Initialize buffer
        self.buffer = buffer.Buffer()
        self.log_func("Buffer initialized.")
        
        # Initialize LLM
        self.llm = llm.LLMInterface(model = model, log_func = log_func)
        self.log_func("LLM initialized.")

        # Initialize speaker
        self.speaker = speaker.Commentator(push_func=push_text_function)

        # Initialize interupt queue
        self.interupt_queue = Queue(maxsize=100)  

        # Initialize running flag
        self.running = True
        
        # Flag to track if first event has been commented
        self.first_event_commented = False

        # Initialize lock
        self.lock = threading.Lock()

        # Create and start threads
        self.listener = threading.Thread(target=self.event_listener_thread, args=(event_queue,), name="Event Listener")
        self.listener.start()
        self.log_func("Event Listener initialized.")
        
        self.commentator = threading.Thread(target=self.commentator_thread, args=(), name="Commentator")
        self.commentator.start()
        self.log_func("Speaker initialized.")

    def event_listener_thread(
        self,
        event_queue: Queue
    ):
        """
        Thread to process game events from the event queue. This thread:
        1. Continuously monitors the event queue for new events
        2. Processes incoming events and assigns event numbers
        3. Determines if events should trigger interrupts
        4. Routes events to either interrupt queue or regular buffer

        Args:
        ----
        event_queue (Queue): 
            Queue to receive game events
        """
        while self.running:     
            
            # Recieve next event
            curr_event_log = event_queue.get()

            # Check for termination signal
            if curr_event_log == "FINISHED":
                self.log_func(f"Received FINISHED token.")
                self.kill()
                return
            
            # Track event count and log KPI
            self.event_counter += 1
            self.kpi('LLM received Events count', self.event_counter)

            # Add event number and log the event
            curr_event_log['event no'] = self.event_counter
            self.log_func(f"Received Event {curr_event_log['event no']}: {curr_event_log}")

            # Preprocess event: clean and add priority
            event = cleaner.clean(curr_event_log)
            event = cleaner.add_prio_entry(event)
            
            # Route event based on interrupt status
            if cleaner.checkInterupt(event):
                # High-priority events go to interrupt queue
                self.interupt_queue.put(event)
                self.log_func(f"Event {event['event no']}: Interrupt!")
            else:
                # Regular events go to buffer
                with self.lock:
                    self.buffer.add(event)
                    self.log_func(f"Event {event['event no']}: Added to buffer!")

    def commentator_thread(self):
        """
        Main commentary generation thread that:
        1. Processes events from both interrupt queue and regular buffer
        2. Generates statistics-based commentary during quiet periods
        3. Manages output and interruptions
        4. Ensures smooth transitions between different types of commentary

        The thread prioritizes:
        - Interrupt events over regular events
        - New interrupts over current speech
        - Statistics generation during quiet periods
        """
        speaker_thread = None
        current_interrupt = None  # Track the current interrupt being processed

        while self.running:
            # First check if we can process any events
            if speaker_thread is None or not speaker_thread.is_alive():
                try:
                    # Check for interrupts first
                    event = self.interupt_queue.get_nowait()
                    is_interrupt = True
                    self.buffer.clear()
                    current_interrupt = event  # Track this interrupt
                    self.log_func(f"Event {event['event no']}: Comment Interrupt!")
                except queue.Empty:
                    # No interrupts, try buffer
                    try:
                        # Get next event from buffer
                        with self.lock:
                            event = self.buffer.next()
                        if not event:
                            time.sleep(0.1)
                            continue
                        
                        # Reset interrupt tracking for new event
                        is_interrupt = False
                        current_interrupt = None
                        
                        # Mark that we've commented our first event
                        self.first_event_commented = True
                        
                    except Exception as e:
                        # No events in buffer - check if it's time for statistics commentary
                        current_time = time.time() * 1000
                        if self.first_event_commented and current_time - self.last_event_time >= STATISTICS_TIME and current_time - self.last_statistics_time >= STATISTICS_INTERVAL:
                            try:
                                # Skip statistics if shutting down
                                if not self.running:
                                    break

                                # Read and validate statistics file
                                with open(self.statistics_path, 'r') as f:
                                    stats = json.load(f)
                                    required_keys = ["score", "ball_posession", "touches_by_player", "goals_by_player"]
                                    if not all(key in stats for key in required_keys):
                                        self.log_func("Error reading statistics: Missing required keys")
                                        continue
                                        
                                    # Choose random statistic type and prepare event
                                    stat_type = random.choice(COMMENTATOR_CONFIG["STATISTICS_TYPES"])
                                    event = {
                                        "event no": self.event_counter,
                                        "event": "statistics",
                                        "type": stat_type
                                    }
                                    
                                    # Format statistics based on type
                                    if stat_type == "score":
                                        score1, score2 = stats['score'][0], stats['score'][1]
                                        diff = abs(score1 - score2)
                                        leading_team = "Team 1" if score1 > score2 else "Team 2"
                                        event.update({
                                            "score": f"{score1}-{score2}",
                                            "statistic": f"Score: {score1}-{score2}, {leading_team} leads by {diff}" if diff > 0 else f"All square at {score1}-{score2}"
                                        })
                                    elif stat_type == "ball_posession":
                                        team0_poss = stats['ball_posession']['team0'] * 100
                                        team1_poss = stats['ball_posession']['team1'] * 100
                                        dominant_team = "Team 1" if team0_poss > team1_poss else "Team 2"
                                        diff = abs(team0_poss - team1_poss)
                                        event.update({
                                            "team_id": "1" if team0_poss > team1_poss else "2",
                                            "statistic": f"Dominant {dominant_team} controls {max(team0_poss, team1_poss):.1f}% of possession, {diff:.1f}% advantage"
                                        })
                                    elif stat_type in ["touches_by_player", "goals_by_player"]:
                                        # Find player with most touches/goals
                                        player_stats = stats[stat_type]
                                        if player_stats:
                                            max_player = max(player_stats.items(), key=lambda x: x[1])
                                            player_id = max_player[0][:-1]  # Remove team ID from player code
                                            team_id = max_player[0][-1]     # Get team ID from player code
                                            stat_value = max_player[1]
                                            stat_name = "touches" if stat_type == "touches_by_player" else "goals"
                                            event.update({
                                                "player": player_id,
                                                "team_id": team_id,
                                                "value": stat_value,
                                                "statistic": f"Team {team_id} striker dominates with {stat_value} {stat_name}"
                                            })
                                    
                                    # Generate and deliver statistics comment
                                    prompt = prompter.generate_prompt("statistics", event)
                                    comment = self.llm.generate_comment(prompt)
                                    comment = comment.replace('"', '').replace(" ", "", 1)
                                    
                                    # Speak the comment through text-to-speech
                                    speaker_thread = threading.Thread(target=self.speaker.speak, args=(comment,))
                                    speaker_thread.start()
                                    self.log_func(f"Statistics comment: {comment}")
                                    
                                    # Update timing variables to prevent rapid statistics comments
                                    # last_event_time: controls initial delay after events
                                    # last_statistics_time: enforces minimum interval between statistics
                                    self.last_event_time = current_time
                                    self.last_statistics_time = time.time() * 1000
                                    
                            except FileNotFoundError:
                                # Statistics file not found
                                # This is expected behavior before the first statistics are generated
                                # and after the last statistics are deleted
                                time.sleep(0.1)
                                continue
                            except json.JSONDecodeError:
                                self.log_func("Error decoding statistics file")
                            except Exception as e:
                                self.log_func(f"Error reading statistics: {str(e)}")
                        time.sleep(0.1)
                        continue
                try:
                    # Generate comment
                    start_time = time.time()
                    prompt = prompter.generate_prompt("event", event)
                    comment = self.llm.generate_comment(prompt)
                    generation_time = (time.time() - start_time) 
                    self.kpi('LLM generation time', f'{generation_time:.2f} s')
                    self.log_func(f"Event {event['event no']}: Comment generation took {generation_time:.2f}s")

                    # Process comment
                    if is_interrupt:
                        comment = "Oh wait... " + comment.replace('"', '').replace(" ", "", 1)
                    else:
                        comment = comment.replace('"', '').replace(" ", "", 1)
                    
                    # Update last event time
                    self.last_event_time = time.time() * 1000

                    # Start new speech
                    speaker_thread = threading.Thread(target=self.speaker.speak, args=(comment,))
                    speaker_thread.start()
                    self.log_func(f"Event {event['event no']}: Start commenting: {comment}")
                except Exception as e:
                    print(f"Error generating comment: {e}")
                    time.sleep(0.1)

            else:
                # Check for interrupts while speaking
                try:
                    event = self.interupt_queue.get_nowait()
                    # Only process if this is a different interrupt
                    if event != current_interrupt:
                        # Interrupt current speech immediately for new events
                        self.speaker.interrupt(finish_sentence=False)
                        speaker_thread.join(timeout=0.1)

                        self.buffer.clear()
                        current_interrupt = event  # Track this new interrupt
                        self.log_func(f"Event {event['event no']}: Comment Interrupt!")

                        # Generate interrupt comment
                        start_time = time.time()
                        prompt = prompter.generate_prompt("event", event)
                        comment = self.llm.generate_comment(prompt)
                        generation_time = (time.time() - start_time)
                        self.kpi('LLM generation time', f'{generation_time:.2f} s')
                        self.log_func(f"Event {event['event no']}: Comment generation took {generation_time:.2f}s")
                        comment = "Oh wait... " + comment.replace('"', '').replace(" ", "", 1).replace("\n", "")

                        # Update last event time
                        self.last_event_time = time.time() * 1000

                        # Start new speech
                        speaker_thread = threading.Thread(target=self.speaker.speak, args=(comment,))
                        speaker_thread.start()
                        self.log_func(f"Event {event['event no']}: Start commenting: {comment}")
                except:
                    time.sleep(0.1)

        # Clean up
        if speaker_thread and speaker_thread.is_alive():
            self.speaker.interrupt(finish_sentence=True)  # Allow current sentence to finish
            speaker_thread.join(timeout=2.0)  # Give more time to finish the sentence

    def kill(self):
        """
        Stops the LLM pipeline by:
        1. Setting running flag to False to stop all threads
        2. Waiting for commentator thread to finish
        3. Waiting for listener thread to finish
        
        Note: Avoids deadlock by not joining the current thread
        """
        self.log_func("Killing LLM Pipeline...")
        self.running = False
        
        current_thread = threading.current_thread()
        
        if hasattr(self, 'commentator') and self.commentator != current_thread:
            self.commentator.join()
        if hasattr(self, 'listener') and self.listener != current_thread:
            self.listener.join()

    def join(self):
        """
        Waits for all pipeline threads to complete:
        1. Joins the commentator thread if it's not the current thread
        2. Joins the listener thread if it's not the current thread
        
        This ensures clean shutdown of all background processes
        """
        current_thread = threading.current_thread()
        
        if hasattr(self, 'commentator') and self.commentator != current_thread:
            self.commentator.join()
        if hasattr(self, 'listener') and self.listener != current_thread:
            self.listener.join()