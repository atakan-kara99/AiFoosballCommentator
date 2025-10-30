import cv2
import logging
import os
import sys
import queue

from datetime import datetime
from multiprocessing import Process, Queue, Event
from random import randint
from threading import Thread
from time import sleep
from typing import Optional
from vislib import kpi, pushText

# importing the cv pipeline
from cv.main import analyse
# importing the touch filtering
from markov.src.touch_filtering import filter_pipeline
# importing the markov pipeline
from markov.src.pipelines import LivePipeline
# importing the llm pipeline
from llm.llm import LLM

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#
# The class below provides a utility class to generate pretty terminal output.
# Feel free to customize it or to add your favorite color to it.
#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
class Colors:
    
    """
    A utility class for defining ANSI escape sequences for terminal text formatting and coloring.

    Usage:
        You can use these attributes to format text in the terminal, for example:
            print(f"{Colors.red}This is red text{Colors.reset}")
            print(f"{Colors.bold}This is bold text{Colors.reset}")
    """
    reset = '\033[0m'
    bold = '\033[01m'
    underline = '\033[04m'
    # only for the foreground
    black = '\033[30m'
    red = '\033[31m'
    green = '\033[32m'
    orange = '\033[33m'
    blue = '\033[34m'
    purple = '\033[35m'
    cyan = '\033[36m'
    lightgrey = '\033[37m'
    darkgrey = '\033[90m'
    lightred = '\033[91m'
    lightgreen = '\033[92m'
    yellow = '\033[93m'
    lightblue = '\033[94m'
    pink = '\033[95m'
    lightcyan = '\033[96m'
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#
# To get information about steps inside the entire pipeline, the InterProcessLogger provides 
# an interface to create log messages between multiple processes.
# Later on, each worker gets the logger instance and can simply put a logging message into the
# queue of that logger. The logger will retrieve the log and (optionally) print the message
# and also create a new entry in the given log file.
# 
# To increase performance or if you dont like realtime printing, simply use:
#   - `Get-Content <name_of_file>.log -Wait` (for windows)
#   - `tail -f log_file.log` (for macOs and linux)
# to monitor the logs in realtime.
#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
class InterProcessLogger:
    """
    Provide a inter process logger instance.
    It uses provides a queue to allow logging between multiple processes.
    Leverages the built-in logging functionality.
    
    If the value `live` is set to true, this instance also prints customized logs to the console.
    Otherwise, it `only` writes logs to the file defined below.
    
    If you need more information, checkout here:
        https://docs.python.org/3/library/logging.html
    
    LOG-Levels (https://docs.python.org/3/library/logging.html#logging-levels):
        - DEBUG
        - INFO
        - WARNING
        - ERROR
        - CRITICAL
    """
    def __init__(self, live: bool=True):
        log_filename = "cv_markov_llm.log"  # name of the logfile
        logging.basicConfig(
            filename=log_filename,
            level=logging.DEBUG,
            format="%(levelname)s - %(asctime)s - %(message)s",  # Custom format
            datefmt="%H:%M:%S - %d/%m/%Y "    # Date format for timestamps
        )
        self.live = live
        
    @staticmethod
    def _format_log(step_name: str, log_level: str, log_message: str) -> str:
        """Format log message for file output"""
        return f"[ {step_name.upper()} ] - {log_message}"
        
    def log(self, step_name: str="", step_color: str="", log_level: str="info", message: str="") -> None:
        """
        Appends a (new) log message to the logger queue.
        Provides interface for multiple processes.
        """
        try:
            if self.live:
                print(f"{step_color}[ {step_name.upper()} ] ~>{Colors.reset} ({log_level}) {message}")
            
            formatted_msg = self._format_log(step_name, log_level, message)
            if log_level == "debug":
                logging.debug(formatted_msg)
            elif log_level == "info":
                logging.info(formatted_msg)
            elif log_level == "warning":
                logging.warning(formatted_msg)
            elif log_level == "error":
                logging.error(formatted_msg)
            elif log_level == "critical":
                logging.critical(formatted_msg)
        except Exception as e:
            print(f"Logging error: {e}")

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#
# Every step in the entire pipeline represents a single worker.
# Each worker get at least the name (makes identification in log files easier) and optional values
# for the 
#   - subscription queue
#   - publish queue
# 
# The subscription queue is used to let the current worker wait for results/data from that queue.
#
# On the other hand, the publish queue lets the current worker put data / results into the queue.
# Other processes will consume that data.
#
# [   IMPORTANT   ]: Keep in mind to use the same structure inside a single queue!
#
# Some workers might not need both queues, since they might only be responsible for either producing data
# or receiving data.
#
# The logger will simply provide logging messages.
# Inside the customized worker, a custom log function can be created.
#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
def worker( 
    process_name: str, 
    subscription_queue: Optional[Queue] = None,
    publish_queue: Optional[Queue] = None,
    logger:InterProcessLogger = None,
) -> None:
    """
    The definition of a worker process for the entire pipeline.
    The `process_name` should be the logical step of the entire proccessing pipeline (cv, markov or llm).
    Later on, the name is used for logging and printing for live logs.
    
    The args `subscription_queue` and `publish_queue` are used to either receive data from another 
    queue or to send data via a queue (channel) to another process.
    
    Some processes might not need both of the channels:
        - i.e the cv process only need to produce data and send it to the markov process.
        - i.e the llm process only needs to receive data from markov to produce comments.
    

    Args:
        process_name (str): name of the process -> should be something from cv, markov or llm.
        subscription_queue (Optional[Queue], optional): the queue for receiving data from. Defaults to None.
        publish_queue (Optional[Queue], optional): the queue to send data to. Defaults to None.
        logger (InterProcessLogger): the logger instance to log data with / by multiple processes. Defaults to None.
    """
    pass
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#
# ==== S I M P L E    E X A M P L E ====
#
# Task: Provide a pipeline, which generates net_prices and calculates the taxes for that prices and 
# finally compute the vat prices (net_price + vat_price).
#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
def price_in_eur(price_in_cents: int) -> str:
    return f"{price_in_cents / 100:.2f}".replace('.', ',')


def generate_net_prices( 
    process_name: str, 
    subscription_queue: Optional[Queue] = None,
    publish_queue: Optional[Queue] = None,
    logger: InterProcessLogger = None,
) -> None:
    # Custom log function example
    def log(message: str) -> None:
        logger.log(step_name=process_name, step_color=Colors.lightred, log_level="info", message=message)
    
    for _ in range(20): # generate 50 random net prices (in cents)
        net_price = randint(100, 100000)
        log(f"Generated net_price (EUR): {price_in_eur(net_price)}")
        publish_queue.put({ 'net_price': net_price }) # use dict for the data
        sleep(1.5)
    
    publish_queue.put('exit')   # last exit message to shutdown pipeline
    return


def calc_taxes( 
    process_name: str, 
    subscription_queue: Optional[Queue] = None,
    publish_queue: Optional[Queue] = None,
    logger: InterProcessLogger = None,
) -> None:
    # Custom log function example
    def log(message: str) -> None:
        logger.log(step_name=process_name, step_color=Colors.yellow, log_level="info", message=message)
    
    while True:
        response = subscription_queue.get() # receive the dict
        if response == "exit":
            publish_queue.put('exit')   # last exit message to shutdown pipeline
            return
        net_price = response['net_price']
        tax = net_price * .19
        sleep(.5)
        log(f"Calculated for  (EUR) {price_in_eur(net_price)} a tax amount of (EUR) {price_in_eur(tax)}")
        publish_queue.put((net_price, tax))

def calc_vat_price( 
    process_name: str, 
    subscription_queue: Optional[Queue] = None,
    publish_queue: Optional[Queue] = None,
    logger: InterProcessLogger = None,
) -> None:
    # Custom log function example
    def log(message: str) -> None:
        logger.log(step_name=process_name, step_color=Colors.lightblue, log_level="info", message=message)
        
    while True:
        response = subscription_queue.get()
        if response == "exit":
            return
        net_price, tax = response
        vat_price = net_price + tax
        sleep(.5)
        log(f"Vat price (EUR): {price_in_eur(vat_price)}")
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#
# The main entry point simply implements the logic of the entire pipeline.
# The first process is simply responsible to create net_prices and propagate them via the
# `net_price_queue`.
# At the end of the pipeline, the vat_prices are simply calculated based on the previous steps.
# 
#
#
#   ================= LOGGER ==================
#    ^            ^            ^            ^
#    |            |            |            |
# step01 ----> step02 ----> step03 ----> step04
#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

def cv_worker(publish_queue: Queue, logger: InterProcessLogger):
    def log_func(msg: str):
        logger.log('cv', step_color=Colors.green, log_level='info', message=msg)

    ##### CV Pipeline #####
    video_dir = './cv/resources'

    # video path
    camera = "./cv/resources/test_011_2Tore.mp4"
    # camera = "./cv/resources/vid1.mp4"
    # camera = "./videos/v1.mp4"
    # capture camera

    mp4_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    log_func('Videos in resources:')
    log_func(mp4_files)

    cap = cv2.VideoCapture(camera) # device, ip url or video file path
    # cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Frame", 500, 350)
    log_func("Cv starts processing.")

    # check for video
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
    else:
        print("Camera successfully accessed.")
        analyse(publish_queue, log_func, cap, verbose=True) # Leave this at true! We want to see exceptions!
        cap.release()
        cv2.destroyAllWindows()
        
    publish_queue.put("FINISHED")
    return


def touch_filter_worker(subscription_queue: Queue, publish_queue: Queue, logger: InterProcessLogger):
    def log_func(msg: str):
        logger.log('touch_filtering', step_color=Colors.yellow, log_level='info', message=msg)
        
    log_func('Touch filtering is waiting for input.')
    # filter the queue
    filter_pipeline(subscription_queue, publish_queue, log_func)


def markov_worker(subscription_queue: Queue, publish_queue: Queue, logger: InterProcessLogger):
    def log_func(msg: str):
        logger.log('markov', step_color=Colors.lightcyan, log_level='info', message=msg)
    
    log_func('Markov is waiting for input')
    
    # Define necessary path variables for resources used by the Markov live pipeline
    # Convert relative paths to absolute paths under /app in Docker
    base_path = "/app"  # Docker container path
    path_to_autoencoder = os.path.join(base_path, "live_resources/autoencoder.pickle")
    path_to_model = os.path.join(base_path, "live_resources/model.pickle")
    path_to_stats = os.path.join(base_path, "live_resources/statistics.json")
    
    # Ensure live_resources directory exists
    os.makedirs(os.path.dirname(path_to_stats), exist_ok=True)
    
    # Markov pipeline
    markov_pipeline = LivePipeline(path_to_autoencoder, path_to_model, path_to_stats, sub_queue=subscription_queue, pub_queue=publish_queue, log_func=log_func, verbose=True)
    markov_pipeline.execute()
    
    # Once the transmission ends, notify subscribers
    publish_queue.put("FINISHED")
    return
        
def llm_worker(subscription_queue: Queue, llm_ready_event: Event, logger: InterProcessLogger):
    def log_func(msg: str):
        logger.log('llm', step_color=Colors.lightred, log_level='info', message=msg)

    ##### LLM Pipeline #####

    log_func('Initializing LLM Pipeline...')

    try:
        llm_pipeline = LLM(
            log_func=log_func, 
            push_text_function=pushText, 
            kpi=kpi, 
            event_queue=subscription_queue)  # Update the statistics path to match the Markov pipeline
        
        # Signal that LLM is initialized and ready
        log_func('LLM Pipeline initialized and waiting for input...')
        llm_ready_event.set()
        
        llm_pipeline.join()
    except Exception as e:
        log_func(f'Error initializing LLM Pipeline: {str(e)}')
        # Signal error in initialization
        llm_ready_event.set()
        raise e

    log_func('LLM Pipeline finished.')

if __name__ == '__main__':
    logger = InterProcessLogger(live=True)
    logger.log('pipeline', step_color=Colors.yellow, message='Initialized pipeline')
    pipeline_processes = list()
    
    # put all queues here
    cv_markov_queue = Queue()
    filtered_touch_queue = Queue()
    markov_llm_queue = Queue()
    
    # Create event for LLM initialization
    llm_ready_event = Event()
    
    # Start LLM worker first to ensure model is loaded
    llm_process = Process(target=llm_worker, args=(markov_llm_queue, llm_ready_event, logger))
    llm_process.start()
    logger.log('pipeline', step_color=Colors.yellow, message='Starting LLM worker first...')
    
    # Wait for LLM to be initialized
    llm_ready_event.wait()
    logger.log('pipeline', step_color=Colors.yellow, message='LLM worker initialized, starting other workers...')
    
    # append remaining processes
    pipeline_processes.append(llm_process)
    pipeline_processes.append(
        Process(target=cv_worker, args=(cv_markov_queue, logger))
    )
    pipeline_processes.append(
        Process(target=touch_filter_worker, args=(cv_markov_queue, filtered_touch_queue, logger))
    )
    pipeline_processes.append(
        Process(target=markov_worker, args=(filtered_touch_queue, markov_llm_queue, logger))
    )

    try:
        kpi('pipeline process count', len(pipeline_processes))
        
        # Start the remaining processes
        for i in range(1, len(pipeline_processes)):  # Start from 1 since LLM is already started
            pipeline_processes[i].start()
            
        # Join all processes
        for p in pipeline_processes:
            p.join()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        logger.log('init', step_color=Colors.yellow, message='Finished entire pipeline!')
        print('Finished')
