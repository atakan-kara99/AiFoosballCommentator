from ..debug_player import OpenCVVideoProcessor
import cv2
import numpy as np
from scipy.signal import find_peaks
from cv.player_detection.aggregator import DataStreamAggregator
from ..player_detection import util

rod_distance_dsa = DataStreamAggregator()


def rod_detection(frame):
    mask = util.get_field_mask(frame)

    y_histogram = util.get_mask_histogram(cv2.bitwise_not(mask), 0)
    y_histogram = np.convolve(y_histogram, np.ones(11) / 11, mode="same")
    y_histogram = np.convolve(y_histogram, np.ones(11) / 11, mode="same")
    y_histogram = np.convolve(y_histogram, np.ones(11) / 11, mode="same")

    # rod_histogram_image = util.get_rod_histogram_image(
    #     frame.shape[1], frame.shape[0], y_histogram
    # )

    peaks1, _ = find_peaks(y_histogram, width=8, height=200, distance=100)
    peaks2, _ = find_peaks(
        y_histogram[::-1],
        width=8,
        height=300,
        distance=rod_distance_dsa.get_mean() - rod_distance_dsa.get_stdv(),
    )
    peaks2 = [frame.shape[1] - i for i in peaks2]
    rod_distance_dsa.update(np.mean(peaks1))
    rod_distance_dsa.update(np.mean(peaks2))

    if len(peaks1) == 8:
        rod_positions = peaks1
    elif len(peaks2) == 8:
        rod_positions = peaks2
    else:
        # default values that work for most videos
        rod_positions = [97, 244, 397, 556, 714, 872, 1031, 1182]

    return rod_positions


def update_rods_get_overlay(frame):
    rod_positions = rod_detection(frame)
    util.draw_rods(frame, rod_positions)
    return frame


if __name__ == "__main__":
    video_path = "cv/resources/rec-20250110-130045.mp4"
    processor = OpenCVVideoProcessor(video_path=video_path)
    processor.register_window(
        window_names=("rod_overlay"),
        frame_callback=update_rods_get_overlay,
    )
    processor.process_video_multi()
