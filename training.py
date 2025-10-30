import cv2

from multiprocessing import Queue

import logging
from threading import Thread
from typing import Optional

# importing the cv pipeline
from cv.main import analyse
# importing the markov pipeline
from markov.src.pipelines import TrainingPipeline
# importing the touch filtering pipeline
from markov.src.touch_filtering import filter_touches_from_file
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
        self.log_queue = Queue()
        consume_thread = Thread(target=self._consume_queue_data, args=())
        consume_thread.daemon = True
        consume_thread.start()
        
    def _consume_queue_data(self) -> None:
        """
        Consumes the logs from other processes.
        Allows to collect all values and only print from one process.
        
        Only provide terminal outputs, if the live flag is given (set to true).
        """
        
        while True:
            # Obtain the log with information
            log_data = self.log_queue.get()
            
            (step_name, step_color, log_level, log_message) = log_data
            
            if self.live:
                print(self._create_stdout(step_name, step_color, log_level, log_message))
            if log_level == "debug":
                logging.debug(f"[ {step_name.upper()} ] - {log_message}")
            elif log_level == "info":
                logging.info(f"[ {step_name.upper()} ] - {log_message}")
            elif log_level == "warning":
                logging.warning(f"[ {step_name.upper()} ] - {log_message}")
            elif log_level == "error":
                logging.error(f"[ {step_name.upper()} ] - {log_message}")
            elif log_level == "critical":
                logging.critical(f"[ {step_name.upper()} ] - {log_message}")
            else:
                pass
    
    def _create_stdout(self, step_name: str, step_color: str, log_level: str, log_message: str) -> str:
        """
        Creates standard output for the logging message.

        Args:
            step_name (str): name of the current step (cv, markov, llm).
            step_color (str): color of the current step. Default is "" (no color).
            log_level (str): provide a level for the log message. Later useful to identify errors in the logfile.
            log_message (str): the subject of the current log -> message.

        Returns:
            str: a custom and pretty string we can return in the stdout.
        """
        return f"{Colors.bold}{step_color}[ {step_name.upper()} ] ~>{Colors.reset} ({log_level}) {log_message}"
    
    def log(self, step_name: str="", step_color: str="", log_level: str="info", message: str="") -> None:
        """
        Appends a (new) log message to the logger queue.
        Provides interface for multiple processes.

        Args:
            step_name (str, optional): name of the step (cv, markov, llm). Defaults to "".
            step_color (str, optional): color of the step [See Colors above]. Defaults to "".
            log_level (str, optional): level of the log [For more information, see docstring of `InterProcessLogger`]. Defaults to "info".
            message (str, optional): the subject / message we want to log. Defaults to "".
        """
        self.log_queue.put((step_name, step_color, log_level, message))
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

def cv_worker(logger: InterProcessLogger, training_id="training_id_not_set"):
    def log_func(msg: str):
        logger.log("cv", step_color=Colors.green, log_level="info", message=msg)

    ##### CV Pipeline #####

    # video path
    # camera = "./cv/resources/test_011_2Tore.mp4"
    camera = "./cv/resources/rec-20250203-134229.mp4"
    # capture camera
    cap = cv2.VideoCapture(camera) # device, ip url or video file path
    # cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Frame", 500, 350)
    log_func("Cv starts processing.")

    # check for video
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return
    else:
        print("Camera successfully accessed.")
        analyse(None, log_func, cap, training_id=training_id)
        cap.release()
        cv2.destroyAllWindows()
        
    return

def filter_worker(logger: InterProcessLogger, training_id="training_id_not_set"):
    def log_func(msg: str):
        logger.log("filter", step_color=Colors.lightblue, log_level="info", message=msg)
    
    log_func("Filter is waiting for input")
    
    # Filter pipeline
    filter_touches_from_file(f"training_resources/touches_{training_id}.json", f"training_resources/filtered_touches_{training_id}.json", log_func, file_format="json")
    return

def markov_worker(logger: InterProcessLogger, training_id="training_id_not_set"):
    def log_func(msg: str):
        logger.log("markov", step_color=Colors.lightcyan, log_level="info", message=msg)
    
    log_func("Markov is waiting for input")
    
    # Define necessary path variables for resources used by the Markov live pipeline
    path_to_autoencoder = "live_resources/autoencoder.pickle"
    path_to_model = "live_resources/model.pickle"
    path_to_training_data = f"training_resources/filtered_touches_{training_id}.json"
    # Markov pipeline
    markov_pipeline = TrainingPipeline(path_to_autoencoder, path_to_model, path_to_training_data, log_func=log_func, verbose=True)
    markov_pipeline.execute()
    return

if __name__ == "__main__":
    logger = InterProcessLogger(live=True)
    logger.log("pipeline", step_color=Colors.yellow, message="Initialized pipeline")

    # generate training id to append to file names
    training_id = "rec-20250203-134229"
    logger.log("pipeline", step_color=Colors.yellow, message=f"Training ID: {training_id}")

    cv_worker(logger, training_id)
    filter_worker(logger, training_id)
    markov_worker(logger, training_id)

    logger.log("pipeline", step_color=Colors.yellow, message="Pipeline finished")
    