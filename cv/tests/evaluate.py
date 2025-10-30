import os
from typing import List, Tuple, Dict, Union

from cv.global_constants import FPS as TOLERANCE

class Evaluator:
    """
    Evaluates detected events against annotated events for a video.

    This class loads annotated and detected events from files based on a given video.
    It then compares the events using a temporal tolerance and computes evaluation
    metrics (precision, recall, F1-score, and accuracy). Finally, it prints the results.
    """

    def __init__(self, video_path: str) -> None:
        """
        Initialize the Evaluator with the paths for annotated and detected event files.

        The evaluator derives the base name from the video file and constructs the paths
        to the annotated and detected event files. If the detected file already exists,
        the user is prompted to confirm overwriting.

        Args:
            video_path (str): The path to the video file.
        """
        self.base_name: str = os.path.splitext(os.path.basename(video_path))[0]
        self.annotated_file_path: str = f"cv/tests/annotated/{self.base_name}.txt"
        self.detected_file_path: str = f"cv/tests/detected/{self.base_name}.txt"
        self.annotated_data: List[Tuple[int, str]] = self.load_file(self.annotated_file_path)
        self.detected_data: List[Tuple[int, str]] = self.load_file(self.detected_file_path)
        self.events: List[Tuple[int, str]] = []

    @staticmethod
    def load_file(file_path: str) -> List[Tuple[int, str]]:
        """
        Load event data from a given file.

        Each line in the file should have the format "frame: event", where frame is an integer
        and event is a string.

        Args:
            file_path (str): The path to the file containing event data.

        Returns:
            List[Tuple[int, str]]: A list of tuples, each containing a frame number and event string.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be parsed correctly.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        data: List[Tuple[int, str]] = []
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    frame_str, event = line.strip().split(": ")
                    data.append((int(frame_str), event))
        except (ValueError, OSError) as e:
            raise ValueError(f"Error reading file {file_path}: {e}")
        return data

    def evaluate(self) -> None:
        """
        Evaluate the detected events against the annotated events and print the metrics.

        This method compares the annotated and detected events frame by frame using a tolerance
        window defined by TOLERANCE. It computes true positives, false positives, and false negatives,
        then calculates precision, recall, F1-score, and accuracy. Finally, it prints the metrics along
        with details of false positives and negatives.
        """
        # Copy detected data to allow removal of matched events.
        false_positives: List[Tuple[int, str]] = self.detected_data[:]
        true_positives: int = 0
        false_negatives: List[Tuple[int, str]] = []

        # Evaluate each annotated event.
        for a_frame, a_event in self.annotated_data:
            matched: bool = False
            for det in false_positives:
                d_frame, d_event = det
                # Check if the detected event matches the annotated event within the tolerance.
                if d_event == a_event and 0 < d_frame - a_frame < TOLERANCE:
                    matched = True
                    break

            if matched:
                true_positives += 1
                false_positives.remove(det)
            else:
                false_negatives.append((a_frame, a_event))

        # Compute evaluation metrics.
        precision: float = self.safe_division(true_positives, true_positives + len(false_positives))
        recall: float = self.safe_division(true_positives, true_positives + len(false_negatives))
        f1_score: float = self.safe_division(2 * precision * recall, precision + recall)
        accuracy: float = self.safe_division(true_positives, len(self.annotated_data))

        metrics: Dict[str, Union[int, float]] = {
            "True Positives": true_positives,
            "False Positives": len(false_positives),
            "False Negatives": len(false_negatives),
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_score,
            "Accuracy": accuracy,
        }

        self.print_metrics(metrics, false_positives, false_negatives)

    @staticmethod
    def safe_division(numerator: Union[int, float], denominator: Union[int, float]) -> float:
        """
        Safely divide two numbers, returning 0 if the denominator is zero.

        Args:
            numerator (Union[int, float]): The numerator.
            denominator (Union[int, float]): The denominator.

        Returns:
            float: The result of the division, or 0 if denominator is 0.
        """
        return numerator / denominator if denominator else 0

    @staticmethod
    def print_metrics(
        metrics: Dict[str, Union[int, float]],
        false_positives: List[Tuple[int, str]],
        false_negatives: List[Tuple[int, str]]
    ) -> None:
        """
        Print the evaluation metrics and details about false positives and negatives.

        Args:
            metrics (Dict[str, Union[int, float]]): A dictionary of evaluation metrics.
            false_positives (List[Tuple[int, str]]): List of false positive events.
            false_negatives (List[Tuple[int, str]]): List of false negative events.
        """
        print(f"\n{'Metric':<20}{'Value':>10}")
        print("-" * 30)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:<20}{value:>10.2f}")
            else:
                print(f"{key:<20}{value:>10}")
        print("-" * 30)
        print("\nFalse Positives:", false_positives)
        print("\nFalse Negatives:", false_negatives)


if __name__ == "__main__":
    # Example usage:
    Evaluator('cv/resources/test_011_2Tore.mp4').evaluate()
