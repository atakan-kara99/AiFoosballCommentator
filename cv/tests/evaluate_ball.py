import os
import math
import numpy as np
from typing import List, Tuple, Dict, Union

class BallEvaluator:
    """
    Evaluates detected ball center positions against annotated ball positions.

    This class loads annotated and detected ball position data from files based on a given video.
    It then compares the ball center positions using the Euclidean distance error and computes
    regression metrics, including:
      - Mean Error
      - RMSE (Root Mean Squared Error)
      - Standard Deviation
      - Maximum Error
      - Median Error
      - 1st Quartile (25th percentile)
      - 3rd Quartile (75th percentile)
    
    It creates a violin plot to visualize the error distribution and lists the worst 5 detections
    (frame number and error value).

    Expected file format for both annotated and detected files:
        Each line should be formatted as: "frame: x, y"
        where 'frame' is an integer and x, y are floating point numbers.
    
    The annotated file is expected at:
        "cv/tests/annotated/{base_name}.txt"
    
    The detected file is expected at:
        "cv/tests/detected/{base_name}_ball.txt"
    """

    def __init__(self, video_path: str) -> None:
        """
        Initialize the BallEvaluator with paths for annotated and detected ball position files.

        Args:
            video_path (str): The path to the video file.
        """
        self.base_name: str = os.path.splitext(os.path.basename(video_path))[0]
        self.annotated_file_path: str = f"cv/tests/annotated/{self.base_name}_ball.txt"
        self.detected_file_path: str = f"cv/tests/detected/{self.base_name}_ball.txt"
        self.annotated_data: List[Tuple[int, Tuple[float, float]]] = self.load_ball_file(self.annotated_file_path)
        self.detected_data: List[Tuple[int, Tuple[float, float]]] = self.load_ball_file(self.detected_file_path)

    @staticmethod
    def load_ball_file(file_path: str) -> List[Tuple[int, Tuple[float, float]]]:
        """
        Load ball position data from a file formatted as "frame: x, y".

        Args:
            file_path (str): The path to the file.

        Returns:
            List[Tuple[int, Tuple[float, float]]]: A list of tuples where each tuple contains a frame number
                                                   and a tuple (x, y) representing the ball center position.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        data: List[Tuple[int, Tuple[float, float]]] = []
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split(": ")
                    if len(parts) != 2:
                        raise ValueError(f"Line in file {file_path} is not in expected format: {line}")
                    frame_str, pos_str = parts
                    frame = int(frame_str)
                    try:
                        x_str, y_str = pos_str.split(", ")
                        x = float(x_str)
                        y = float(y_str)
                    except Exception as e:
                        raise ValueError(f"Error parsing line in file {file_path}: {line}. Error: {e}")
                    data.append((frame, (x, y)))
        except (ValueError, OSError) as e:
            raise ValueError(f"Error reading file {file_path}: {e}")
        return data

    def evaluate(self) -> None:
        """
        Evaluate the detected ball center positions against annotated positions, compute regression metrics,
        create a violin plot of the error distribution, and list the worst 5 detections.

        Only frames that appear in both annotated and detected data are evaluated.
        """
        # Build dictionaries for fast lookup by frame number.
        annotated_dict: Dict[int, Tuple[float, float]] = {frame: pos for frame, pos in self.annotated_data}
        detected_dict: Dict[int, Tuple[float, float]] = {frame: pos for frame, pos in self.detected_data}

        # Only consider frames that are present in both files.
        common_frames = set(annotated_dict.keys()).intersection(detected_dict.keys())

        # Compute Euclidean distance errors for common frames and store (frame, error) pairs.
        error_data: List[Tuple[int, float]] = []
        for frame in common_frames:
            a_pos = annotated_dict[frame]
            d_pos = detected_dict[frame]
            error = math.sqrt((d_pos[0] - a_pos[0]) ** 2 + (d_pos[1] - a_pos[1]) ** 2)
            error_data.append((frame, error))
        
        # Extract error values for metrics.
        errors = [e for _, e in error_data]

        # Compute regression metrics.
        if errors:
            mean_error = sum(errors) / len(errors)
            rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
            variance = sum((e - mean_error) ** 2 for e in errors) / len(errors)
            std_dev = math.sqrt(variance)
            max_error = max(errors)
            median_error = np.median(errors)
            q1 = np.percentile(errors, 25)
            q3 = np.percentile(errors, 75)
        else:
            mean_error = rmse = std_dev = max_error = median_error = q1 = q3 = 0.0

        metrics: Dict[str, Union[int, float]] = {
            "Mean Error": mean_error,
            "RMSE": rmse,
            "Standard Deviation": std_dev,
            "Max Error": max_error,
            "Median Error": median_error,
            "1st Quartile": q1,
            "3rd Quartile": q3,
            "Evaluated Frames": len(common_frames),
        }

        # Print evaluation metrics.
        print(f"\nBall Center Position Evaluation Metrics for {self.base_name}")
        print("-" * 50)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:<25}: {value:>10.2f}")
            else:
                print(f"{key:<25}: {value:>10}")
        print("-" * 50)

        # List worst 5 detections based on error.
        worst5 = sorted(error_data, key=lambda x: x[1], reverse=True)[:5]
        print("\nWorst 5 Detections (Frame: Error):")
        for frame, error in worst5:
            print(f"Frame {frame}: {error:.2f}")


if __name__ == "__main__":
    # Hardcoded parameters for ball evaluation.
    video_path = 'cv/resources/test_009_1Tor.mp4'
    evaluator = BallEvaluator(video_path)
    evaluator.evaluate()
