"""
Threaded Pipeline Test Suite
=========================

This module implements a multi-threaded test environment for the commentary
pipeline, simulating real-time game event processing and commentary
generation with interrupt handling.

Architecture:
-----------
1. Producer-Consumer Pattern:
   - Event Listener (Producer): Processes game events
   - Commentator (Consumer): Generates and delivers commentary

2. Thread Communication:
   - Shared Buffer: Priority-based event queue
   - Interrupt Queue: High-priority events
   - Thread synchronization via locks

3. Pipeline Components:
   - Event cleaning and validation
   - Priority-based buffering
   - LLM-based commentary
   - Natural speech output

Performance Features:
------------------
- Real-time event processing
- Interrupt handling
- Thread-safe operations
- Resource cleanup
"""

# Standard library imports
import sys
import os
import threading
import queue
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


def event_listener_thread(event_log_dict, buffer: b.Buffer(), interupt_queue: queue.Queue):
    """
    Producer thread that processes game events.
    
    Reads events from the event log, processes them through the cleaner,
    and either queues them in the buffer or sends them as interrupts.
    
    Args:
        event_log_dict: Dictionary containing game events
        buffer: Shared buffer for normal priority events
        interupt_queue: Queue for high-priority interrupt events
    
    Threading:
        - Uses global 'running' flag for shutdown
        - Thread-safe buffer access via lock
        - Real-time frame-based processing
    """
    global running

    # Initialize event processing
    frame_counter = 0
    event_counter = 0
    seconds_per_frame = 1 / 30  # 30 FPS
    
    event_log = event_log_dict["events"]
    next_event = event_log[event_counter]
    
    try:
        while running:          
            if frame_counter == next_event["frame_no"]:
                # Process and validate event
                event = cleaner.clean(next_event)
                event = cleaner.add_prio_entry(event)
                
                # Route event based on priority
                if cleaner.checkInterupt(event):
                    interupt_queue.put(event)  # High-priority event
                else:
                    with lock:
                        buffer.add(event)      # Normal-priority event
                
                # Move to next event
                event_counter += 1
                try:
                    next_event = event_log[event_counter]
                except:
                    time.sleep(0.5)
                    running = False  # Signal completion
                    break

            # Maintain frame timing
            frame_counter += 1
            time.sleep(seconds_per_frame)
    
    except KeyboardInterrupt:
        running = False


def commentator_thread(buffer: b.Buffer(), llm: llm.LLMInterface, speaker: commentator.Commentator, interupt_queue: queue.Queue):
    """
    Consumer thread that generates and delivers commentary.
    
    Monitors both the interrupt queue and event buffer, generates
    appropriate commentary, and manages speech output with interrupts.
    
    Args:
        buffer: Shared buffer for normal priority events
        llm: Language model interface for commentary generation
        speaker: Speech output manager
        interupt_queue: Queue for high-priority interrupt events
    
    Threading:
        - Uses global 'running' flag for shutdown
        - Thread-safe buffer access via lock
        - Manages speech output thread
    """
    global running
    speaker_thread = None

    try:
        while running:
            # Check if we can process new events
            if speaker_thread is None or not speaker_thread.is_alive():
                try:
                    # Priority 1: Check interrupt queue
                    event = interupt_queue.get_nowait()
                    is_interrupt = True
                    buffer.clear()  # Clear pending events
                except queue.Empty:
                    # Priority 2: Check normal event buffer
                    try:
                        with lock:
                            event = buffer.next()
                        if not event:
                            time.sleep(0.1)
                            continue
                        is_interrupt = False
                    except Exception as e:
                        time.sleep(0.1)
                        continue

                try:
                    # Generate commentary
                    prompt = prompter.generate_prompt("event", event)
                    comment = llm.generate_comment(prompt)

                    # Format based on event type
                    if is_interrupt:
                        comment = "Oh wait... " + comment.replace('"', '').replace(" ", "", 1)
                    else:
                        comment = comment.replace('"', '').replace(" ", "", 1)

                    # Start speech output
                    speaker_thread = threading.Thread(target=speaker.speak, args=(comment,))
                    speaker_thread.start()
                except Exception as e:
                    print(f"Error generating comment: {e}")
                    time.sleep(0.1)

            else:
                # Handle interrupts during active speech
                try:
                    event = interupt_queue.get_nowait()
                    
                    # Stop current speech
                    speaker.interrupt()
                    speaker_thread.join(timeout=0.1)
                    buffer.clear()

                    # Generate interrupt commentary
                    prompt = prompter.generate_prompt("event", event)
                    comment = llm.generate_comment(prompt)
                    comment = "Oh wait... " + comment.replace('"', '').replace(" ", "", 1)

                    # Start new speech
                    speaker_thread = threading.Thread(target=speaker.speak, args=(comment,))
                    speaker_thread.start()
                except queue.Empty:
                    time.sleep(0.1)
                
    except KeyboardInterrupt:
        running = False
        # Cleanup speech thread
        if speaker_thread and speaker_thread.is_alive():
            speaker.interrupt()  
            speaker_thread.join(timeout=1.0) 


def print_header():
    """
    Display formatted header for test environment.
    Provides visual separation and context in console output.
    """
    print("--------------------------------------------")
    print("|                  TEST                    |")
    print("|                Pipeline                  |")
    print("--------------------------------------------")
    print("")


# Initialize test environment
print_header()

# Initialize pipeline components
model = "meta-llama/Llama-3.2-3B-Instruct"
buffer = b.Buffer()                            # Event buffer
llm = llm.LLMInterface(model=model)           # LLM interface
speaker = commentator.Commentator()            # Speech output

# Thread synchronization
interupt_queue = queue.Queue(maxsize=100)     # High-priority event queue
running = True                                # Global thread control
lock = threading.Lock()                       # Buffer access synchronization

# Load test event data
with open('llm/tests/resources/eventlog.json', 'r') as file:
    event_log_dict = json.load(file)

# Create and start pipeline threads
event_listener = threading.Thread(
    target=event_listener_thread,
    args=(event_log_dict, buffer, interupt_queue),
    name="Event Listener"
)
commentator = threading.Thread(
    target=commentator_thread,
    args=(buffer, llm, speaker, interupt_queue),
    name="Commentator"
)

event_listener.start()
commentator.start()

# Main thread: Monitor and handle shutdown
try:
    # Wait for pipeline completion
    event_listener.join()
    commentator.join()
except KeyboardInterrupt:
    print("\nShutting down pipeline...")
    running = False
    event_listener.join()
    commentator.join()
    print("Pipeline shutdown complete.")