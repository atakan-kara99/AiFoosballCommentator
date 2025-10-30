"""
This script evaluates touch detection in video footage by comparing detected touches
against manually annotated ground truth data. 

Main Features:
- Runs touch detection with cv.analyse.
- Extracts touches from detected and annotated JSON files.
- Compares detections against annotations, calculating precision, recall, and F1 score.
- Processes multiple videos to aggregate evaluation metrics.
- Runs video analysis multiple times and computes average results.
- Saves evaluation results in JSON format.

Dependencies:
- OpenCV (cv2) for video processing.
- JSON for reading and writing results.
- Subprocess for retrieving the Git commit hash.
"""

import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import cv2
from cv.main import analyse  # Importing the core analysis function

NUM_RUNS_SINGLE = 3  # Number of runs for averaging results

NUM_RUNS_SINGLE = 1

def get_git_commit_hash(short: bool = False) -> str:
    """Returns the current Git commit hash (short or full)."""
    return subprocess.check_output(["git", "rev-parse", "--short" if short else "HEAD"]).decode().strip()


def load_json(file_path: str) -> Any:
    """Loads a JSON file and returns its contents."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_touches(touchlog: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Extracts touch events from the touch log."""
    return {t["frame_no"]: t for t in touchlog.get("touches", []) if t.get("type") == "touch"}


def evaluate_touches(detected_touches: Dict[int, Dict[str, Any]], annotated_touches: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compares detected touches with annotated ground truth touches.

    Returns precision, recall, F1 score, and detailed mismatches.
    """
    true_positives, false_positives, false_negatives = 0, 0, 0
    false_reason_counts = defaultdict(int)
    matched_frames = set()
    false_positive_touches, false_negative_touches = {}, {}

    for frame_no, detected_touch in detected_touches.items():
        for offset in [0, 1, 2, -2, -1]:  # Allow small frame offset for matching
            match_frame = str(frame_no + offset)
            if match_frame in annotated_touches:
                mismatches = [key for key in ["player", "team_id", "goal", "throw_in"] 
                              if str(detected_touch.get(key, "")) != str(annotated_touches[match_frame].get(key, " "))]
                if not mismatches:
                    true_positives += 1
                    matched_frames.add(match_frame)
                    break  # Stop checking once a match is found
                for reason in mismatches:
                    false_reason_counts[reason] += 1
        else:
            false_positives += 1
            false_positive_touches[frame_no] = detected_touch

    for frame_no in annotated_touches:
        if frame_no not in matched_frames:
            false_negatives += 1
            false_negative_touches[frame_no] = annotated_touches[frame_no]

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1_score = 2 * (precision * recall) / max(precision + recall, 1)

    return {
        "total_detections": len(detected_touches),
        "total_annotations": len(annotated_touches),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "false_reason_counts": dict(false_reason_counts),
        "false_positive_touches": false_positive_touches,
        "false_negative_touches": false_negative_touches,
    }


def evaluate_multiple_videos(video_files: List[str]) -> Dict[str, Any]:
    """
    Runs touch detection evaluation on multiple videos and aggregates the results.
    """
    aggregate_results = defaultdict(float)
    all_false_reasons = defaultdict(int)

    for video in video_files:
        try:
            detected_touches = extract_touches(load_json(f"cv/tests/detected/{video}/touches.json"))
            annotations_json = load_json(f"cv/tests/annotated/{video}.json")
            annotations = annotations_json.get("touches", {}) if annotations_json.get("video") == video else {}

            results = evaluate_touches(detected_touches, annotations)
            
            for key in ["total_detections", "total_annotations", "true_positives", "false_positives", "false_negatives"]:
                aggregate_results[key] += results[key]
            for reason, count in results["false_reason_counts"].items():
                all_false_reasons[reason] += count

            # Averaging precision, recall, and F1 score
            for key in ["precision", "recall", "f1_score"]:
                aggregate_results[key] += results[key]

        except Exception as e:
            print(f"Skipping {video} due to error: {e}")

    total_videos = max(len(video_files), 1)
    for key in ["precision", "recall", "f1_score"]:
        aggregate_results[key] /= total_videos  # Average across videos

    aggregate_results["false_reason_counts"] = dict(all_false_reasons)
    return dict(aggregate_results)


def process_video(video: str, save: bool) -> None:
    """
    Runs multiple evaluation iterations on a single video and saves the average results.
    """
    camera_path = f"cv/resources/{video}.mp4"
    annotations_path = f"cv/tests/annotated/{video}.json"

    annotations_json = load_json(annotations_path)
    annotations = annotations_json.get("touches", {}) if annotations_json.get("video") == video else {}

    if not annotations:
        print(f"Annotations are for the wrong video: {annotations_json.get('video')}")
        return

    detected_path = f"cv/tests/detected/{video}/touches.json"
    runs_results = []

    for _ in range(NUM_RUNS_SINGLE):
        cap = cv2.VideoCapture(camera_path)
        if not cap.isOpened():
            print(f"Error: Unable to access the camera for {video}.")
            return

        analyse(None, None, cap, eval=True, video_name=video, verbose=False, debug=False)
        cap.release()

        detected_touches = extract_touches(load_json(detected_path))
        runs_results.append(evaluate_touches(detected_touches, annotations))

    avg_results = {key: sum(run[key] for run in runs_results) / NUM_RUNS_SINGLE
                   for key in ["total_detections", "total_annotations", "true_positives", "false_positives", "false_negatives", "precision", "recall", "f1_score"]}

    avg_results["false_reason_counts"] = {reason: sum(run["false_reason_counts"].get(reason, 0) for run in runs_results) / NUM_RUNS_SINGLE
                                          for reason in ["player", "team_id", "goal", "throw_in"]}

    final_results = {
        "logger_config": load_json(detected_path).get("logger"),
        "repo_commit": get_git_commit_hash(),
        "annotation_quality": annotations_json.get("annotation_quality"),
        "avg_over_n_runs": NUM_RUNS_SINGLE,
        **avg_results
    }

    if save:
        output_path = f"cv/tests/detected/{video}/results/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as res_file:
            json.dump(final_results, res_file, indent=4)
        print(f"Results saved under {output_path}")
    else:
        print(json.dumps(final_results, indent=4))


if __name__ == "__main__":
    process_video("test_011_2Tore", save=True)
