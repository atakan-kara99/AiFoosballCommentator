from ..debug_player import OpenCVVideoProcessor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import imutils
from cv.player_detection.aggregator import DataStreamAggregator
from cv.player_detection.long_exp_generator import CleanLongExposureGenerator
import os
from ..player_detection import rod, util

########################################################################################################################################################################################################################################################################
# util
########################################################################################################################################################################################################################################################################
debug = False
frame_count = 1
rod_positions = [97, 244, 397, 556, 714, 872, 1031, 1182]  # x pixel positions of rods
players_per_rod = [3, 2, 3, 5, 5, 3, 2, 3]  # number of players per rod
player_positions = [
    [None for _ in range(players_per_rod[i])] for i in range(8)
]  # player positions per rod
player_contours_per_rod = [
    None for i in rod_positions
]  # holds detected player contours
player_distance_per_rod = [
    100,
    243,
    176,
    113,
    113,
    176,
    243,
    100,
]  # y pixel distances between players on the rods

rod_distance_dsa = DataStreamAggregator()


player_distance_aggregators = []
for i in range(8):
    dsa = DataStreamAggregator()
    player_distance_aggregators.append(dsa)
    dsa.update(player_distance_per_rod[i])
    dsa.update(player_distance_per_rod[i])

player_hist_height_aggregator = DataStreamAggregator()

long_exp = CleanLongExposureGenerator((720, 1280, 3))


def get_custom_mask(
    lower_color, upper_color, image, filter_size=8, highlight_color=(0, 0, 255)
):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_color, upper_color)
    kernel = np.ones((filter_size, filter_size), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    bright_red = np.full_like(image, highlight_color)
    mask_image = image.copy()
    mask_image[mask_cleaned > 0] = bright_red[mask_cleaned > 0]
    return mask_image, mask_cleaned


def apply_mask(image, mask, color, inv=False):
    if inv:
        mask = cv2.bitwise_not(mask)
    bright_red = np.full_like(image, color)
    image[mask > 0] = bright_red[mask > 0]
    return image


def dilate(img, k_size, k_type=None):
    if type(k_size) == int:
        k_size = k_size, k_size
    k = None
    if k_type == None:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size))
    else:
        k = cv2.getStructuringElement(k_type, (k_size))
    return cv2.dilate(img, k, iterations=1)


def erode(img, k_size, k_type=None):
    if type(k_size) == int:
        k_size = k_size, k_size
    k = None
    if k_type == None:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size))
    else:
        k = cv2.getStructuringElement(k_type, (k_size))
    return cv2.erode(img, k, iterations=1)


def bgr_to_saturation(frame):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsv_image[:, :, 2]


def get_blurr_diff(f1, f2, kernel_size=(9, 9)):

    f1_blured = cv2.GaussianBlur(f1, kernel_size, 0)
    f2_blured = cv2.GaussianBlur(f2, kernel_size, 0)
    f1_diff = cv2.absdiff(f1, f1_blured)
    f2_diff = cv2.absdiff(f2, f2_blured)
    return cv2.absdiff(f1_diff, f2_diff)


blurr_diff_img = None
blurr_diff_img_thres = None


def get_blurr_difference(frames):
    global blurr_diff_img, blurr_diff_img_thres
    f1, f2, f3 = frames
    player_movement_segmentation_threshold = 7
    kernel_size = (7, 7)

    f1 = cv2.GaussianBlur(f1, kernel_size, 0)
    f2 = cv2.GaussianBlur(f2, kernel_size, 0)
    f3 = cv2.GaussianBlur(f3, kernel_size, 0)

    f1 = bgr_to_saturation(f1)
    f2 = bgr_to_saturation(f2)
    f3 = bgr_to_saturation(f3)

    blurr_diff1 = get_blurr_diff(f1, f2)
    blurr_diff2 = get_blurr_diff(f2, f3)
    blurr_diff = cv2.bitwise_and(blurr_diff1, blurr_diff2)
    blurr_diff_img = blurr_diff.copy()
    blurr_diff = blurr_diff2
    _, blurr_diff = cv2.threshold(
        blurr_diff,
        player_movement_segmentation_threshold,
        255,
        cv2.THRESH_BINARY,
    )
    blurr_diff_img_thres = blurr_diff.copy()
    blurr_diff = util.dilate_and_erode(
        blurr_diff,
        dilate_iterations=1,
        erode_iterations=1,
        dilate_kernel_size=(23, 53),
        erode_kernel_size=(27, 43),
    )

    return blurr_diff


########################################################################################################################################################################################################################################################################
# semantic utils
########################################################################################################################################################################################################################################################################


def draw_player_positions(frame, radius=10, color=(255, 255, 0)):
    for i, rod in enumerate(player_positions):
        for player_y in rod:
            if player_y is not None:
                cv2.circle(frame, (rod_positions[i], player_y), radius, color, -1)


def is_player_position_noise(positions, rod_idx):
    global player_distance_aggregators
    player_distance = np.mean(np.diff(positions))
    player_distance_aggregators[rod_idx].update(player_distance)
    return player_distance_aggregators[rod_idx].is_noise(player_distance)


def update_long_exp(frame, i):
    sections = []
    if player_positions[i][0] is None:
        exit()
    boundaries = util.get_boundaries(rod_positions)

    x1 = boundaries[i]
    x2 = boundaries[i + 1]
    sections.append((x1, 0, x2, player_positions[i][0] - 40))

    for idx in range(len(player_positions[i]) - 1):
        sections.append(
            (
                x1,
                player_positions[i][idx] + 32,
                x2,
                player_positions[i][idx + 1] - 32,
            )
        )
    sections.append((x1, player_positions[i][-1] + 40, x2, 720))

    long_exp.add_sections(frame, sections)


########################################################################################################################################################################################################################################################################
# rod and player position detection
########################################################################################################################################################################################################################################################################
rod_histograms_stacked = None
player_mask = None
field_mask = None
field_mask_histogram = None


def rod_detection(frames):
    global rod_positions, frame_count, field_mask, field_mask_histogram
    frame_count += 1

    mask = util.get_field_mask(frames[-1])

    y_histogram = util.get_mask_histogram(cv2.bitwise_not(mask), 0)
    y_histogram = np.convolve(y_histogram, np.ones(11) / 11, mode="same")
    y_histogram = np.convolve(y_histogram, np.ones(11) / 11, mode="same")
    y_histogram = np.convolve(y_histogram, np.ones(11) / 11, mode="same")

    rod_histogram_image = util.get_rod_histogram_image(
        frames[-1].shape[1], frames[-1].shape[0], y_histogram
    )

    peaks1, properties = find_peaks(y_histogram, width=8, height=200, distance=100)
    peaks2, properties = find_peaks(
        y_histogram[::-1],
        width=8,
        height=300,
        distance=rod_distance_dsa.get_mean() - rod_distance_dsa.get_stdv(),
    )
    peaks2 = [frames[-1].shape[1] - i for i in peaks2]
    rod_distance_dsa.update(np.mean(peaks1))
    rod_distance_dsa.update(np.mean(peaks2))

    if len(peaks1) == 8:
        rod_positions = peaks1
    elif len(peaks2) == 8:
        rod_positions = peaks2

    rod_histogram_image = util.draw_rods(rod_histogram_image, rod_positions)
    field_mask = mask
    field_mask_histogram = rod_histogram_image


def update_player_positions(frames):
    global player_contours_per_rod, rod_histograms_stacked, player_mask, player_distance_aggregators, player_positions, player_hist_height_aggregator, long_exp
    frame1, frame2, f3 = frames

    last_frame_diff = get_blurr_difference((frame1, frame2, f3))
    rod_images = util.get_rod_images(last_frame_diff, rod_positions)
    rod_histogram_images = []
    for i in range(len(rod_images)):
        player_contours = util.get_filtered_contours(rod_images, i)
        rod_image = rod_images[i]
        rod_histogram = util.get_mask_histogram(rod_image, 1)

        ##################################################################
        # // rod histogram
        smoothed_histogram = util.gaussian_histogram_smoothing(rod_histogram)
        if debug:
            rod_histogram_image = util.draw_rod_histogram(
                rod_image.shape, smoothed_histogram / 4
            )
            rod_histogram_images.append(rod_histogram_image)
        # \\ rod histogram
        ##################################################################
        peaks, properties = find_peaks(
            smoothed_histogram,
            height=5,
            width=10,
            distance=player_distance_per_rod[i] - 15,
        )

        sorted_indices = np.argsort(properties["peak_heights"])[::-1]
        sorted_heights = properties["peak_heights"][
            sorted_indices[: players_per_rod[i]]
        ]

        for height in sorted_heights:
            player_hist_height_aggregator.update(height)

        min_height = (
            player_hist_height_aggregator.get_mean()
            - player_hist_height_aggregator.get_stdv()
            if player_hist_height_aggregator.get_n() > 42
            else 5
        )
        peaks, properties = find_peaks(
            smoothed_histogram,
            height=min_height,
            width=10,
            distance=player_distance_per_rod[i] - 15,
        )
        sorted_indices = np.argsort(properties["peak_heights"])[::-1]
        peaks = peaks[sorted_indices[: players_per_rod[i]]]
        peaks.sort()

        players_detected = False
        # if (i == 0 or i == 7) and len(peaks) == 3:
        #     players_detected = True
        #     if not is_player_position_noise(peaks, i):
        #         peaks.sort()
        #         player_positions[i] = np.array([(min(peaks) + max(peaks)) // 2])
        #         player_contours_per_rod[i] = player_contours

        if len(peaks) == players_per_rod[i]:  # and i not in [0, 7]:
            players_detected = True
            if not is_player_position_noise(peaks, i):
                player_positions[i] = peaks
                player_contours_per_rod[i] = player_contours

        if players_detected:
            update_long_exp(f3, i)

            # players were found -> update mean and stdv of distances between players
            player_distance_aggregators[i].update(np.mean(np.diff(peaks)))
    if debug:
        rod_histograms_stacked = util.stack_rod_images(rod_histogram_images)
        player_mask = util.stack_rod_images(rod_images)


def player_detection(frames):
    rod_detection(frames)
    # print("player_detection/update_rod_positions successful")
    update_player_positions(frames)
    # print("player_detection/update_player_positions successful")
    ret = []
    for i, pixel_pos in enumerate(rod_positions):
        ret.append([])
        if i in {0, 7}:
            ret[-1].append((pixel_pos, 1))
            continue

        for j in player_positions[i]:
            ret[-1].append((pixel_pos, j))
    # for goal_keeper_rod in (ret[0], ret[7]):

    assert len(ret) == 8
    assert len([player for rod in ret for player in rod]) == 22
    # debug_frames = get_debug_views(frames)
    # for i in range(len(debug_frames)):
    #     cv2.imshow(f"debug_frames[{i}]",frames[i])
    return ret


########################################################################################################################################################################################################################################################################
# data generation
########################################################################################################################################################################################################################################################################


def labeled_data_generator(frames):
    long_exp_display_frame = loaded_long_exp.copy()
    # TODO try blurring the current frame first
    long_exp_diff = cv2.cvtColor(
        cv2.absdiff(long_exp_display_frame, frames[-1]), cv2.COLOR_BGR2GRAY
    )
    _, mask = cv2.threshold(
        long_exp_diff,
        14,
        255,
        cv2.THRESH_BINARY,
    )
    mask_cleaned = erode(mask, (5, 3), cv2.MORPH_RECT)
    mask_cleaned = dilate(mask_cleaned, (5, 3), cv2.MORPH_ELLIPSE)
    mask_cleaned = dilate(mask_cleaned, 3, cv2.MORPH_ELLIPSE)
    mask_cleaned = erode(mask_cleaned, 3, cv2.MORPH_RECT)
    mask_cleaned = dilate(mask_cleaned, 5, cv2.MORPH_ELLIPSE)
    mask_cleaned = erode(mask_cleaned, 3, cv2.MORPH_RECT)

    contours = cv2.findContours(
        mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = imutils.grab_contours(contours)
    for j, c in enumerate(cnts):
        area = cv2.contourArea(c)
        if area < 200 or area > 10000:
            cv2.fillPoly(mask_cleaned, pts=[c], color=0)
    mask_cleaned = dilate(mask_cleaned, (10, 3), cv2.MORPH_ELLIPSE)
    mask_cleaned = erode(mask_cleaned, (8, 3), cv2.MORPH_RECT)

    applied = apply_mask(frames[-1].copy(), mask_cleaned, (0, 0, 255))

    return long_exp_display_frame, applied


########################################################################################################################################################################################################################################################################
# debug player
########################################################################################################################################################################################################################################################################


def get_debug_views(frames):
    global long_exp
    # for i, j in enumerate(player_detection_ret):
    #     print(i, j)
    # // debug info drawings
    ret = frames[-1].copy()
    util.draw_rods(ret, rod_positions)
    boundaries = util.get_boundaries(rod_positions)
    util.draw_boundaries(ret, boundaries)
    util.draw_player_contours(ret, player_contours_per_rod, boundaries)
    draw_player_positions(ret)
    draw_player_positions(rod_histograms_stacked, radius=3)

    return (
        # frames[2],
        ret,
        player_mask,
        rod_histograms_stacked,
        blurr_diff_img,
        blurr_diff_img_thres,
        field_mask,
        field_mask_histogram,
        long_exp.get_display_image(),
        # long_exp_diff,
        # mask,
        # mask_cleaned,
    )


def debug_player_views(frames):
    player_detection(frames)
    return get_debug_views(frames)


if __name__ == "__main__":
    from pathlib import Path

    debug = True
    # Initialize video path
    video_path = "cv/resources/rec-20250110-131121.mp4"
    video_file_type = ".mp4"
    output_folder = f"cv/player_detection/training_images/{os.path.basename(video_path).removesuffix(video_file_type)}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_folder), "masks").mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_folder), "images").mkdir(parents=True, exist_ok=True)
    long_exp_path = os.path.join(
        "cv",
        "player_detection",
        "training_images",
        "video_long_exp",
        os.path.basename(video_path).removesuffix(".mp4") + ".png",
    )
    loaded_long_exp = None  # cv2.imread(long_exp_path)

    # Alternatively, use multithreading for enhanced performance
    processor = OpenCVVideoProcessor(video_path=video_path, num_frames=3, overlay=False)
    # processor.register_window(
    #     window_names=("field_v2", "field_histogram_image_v2"),
    #     frame_callback=field_detection_v2,
    # )
    processor.register_window(
        window_names=(
            # "frame",
            "debug view",
            "player_mask",
            "rod_histograms_stacked",
            "blurr_diff_img",
            "blurr_diff_img_thres",
            "field_mask",
            "field_mask_histogram",
            "long_exp",
            # "long_exp_diff",
            # "mask",
            # "mask_cleaned",
        ),
        frame_callback=debug_player_views,
    )
    if loaded_long_exp is not None:
        processor.register_window(
            window_names=("long_exp", "applied"), frame_callback=labeled_data_generator
        )

    try:
        print("processing video")
        processor.process_video_multi()
    except:
        if loaded_long_exp is None and frame_count > 10_000:
            print(f"saving image at {long_exp_path}")
            cv2.imwrite(long_exp_path, long_exp.get_display_image())

    # save long exposure image
