import cv2
import numpy as np
import torch
from scipy.signal import find_peaks

from ..player_detection import rod, util
from cv.player_detection.aggregator import DataStreamAggregator
from ..player_detection.unet import ShallowUNet
import time

# Constants
IMG_SIZE = (224, 384)  # Image dimensions (width, height)
PLAYERS_PER_ROD = [3, 2, 3, 5, 5, 3, 2, 3]  # Number of players per rod
PLAYER_DISTANCES = [100, 243, 176, 113, 113, 176, 243, 100]  # Y pixel distances

# Player data initialization
player_positions = [[None] * PLAYERS_PER_ROD[i] for i in range(8)]
player_feet_positions = [[None] * PLAYERS_PER_ROD[i] for i in range(8)]
player_contours_per_rod = [[] for _ in range(8)]

# Aggregators and masks
player_hist_height_aggregator = DataStreamAggregator()
rod_histograms_stacked = None
player_mask = None
mask = None
image = None
debug = False

time_aggregators = {
    "UNET": DataStreamAggregator(),
    "ROD": DataStreamAggregator(),
    "TOTAL": DataStreamAggregator(),
}


def load_unet_model(path):
    """Loads the ShallowUNet model with pre-trained weights."""
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ShallowUNet(input_channels=3, output_channels=1)
    model.to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.share_memory()
    model.eval()
    return model


# Load the model
model = load_unet_model("cv/player_detection/unet_foosball.pth")


def get_player_unet_mask(frame):
    """Generates a binary mask for player detection using the UNet model."""
    global mask, image
    image = frame.copy()

    # Preprocess frame
    img = cv2.resize(frame, (IMG_SIZE[1], IMG_SIZE[0]))  # Resize to (width, height)
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    img = np.transpose(img, (2, 0, 1))  # Change shape from (H, W, C) → (C, H, W)
    img = torch.tensor(img.tolist(), dtype=torch.float32) - 1.0  # Convert to tensor

    # Predict mask
    input_tensor = img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)

    # Post-process mask
    output_mask = output.cpu().squeeze(0).squeeze(0).numpy()
    output_mask = (output_mask > 0.5).astype(np.uint8)  # Threshold to binary mask
    output_mask = cv2.resize(
        output_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    _, mask = cv2.threshold(output_mask, 0, 255, cv2.THRESH_BINARY)

    return mask


def update_player_positions(frame, rod_positions):
    """Updates player positions based on the processed mask."""
    global rod_histograms_stacked, player_mask
    start_time = time.time()
    mask = get_player_unet_mask(frame)
    time_aggregators["UNET"].update(time.time() - start_time)

    rod_images = util.get_rod_images(mask, rod_positions)
    rod_histogram_images = []

    for i, rod_image in enumerate(rod_images):
        player_contours = util.get_filtered_contours(rod_images, i)
        rod_histogram = util.get_mask_histogram(rod_image, 1)

        # Smooth histogram
        smoothed_histogram = util.gaussian_histogram_smoothing(
            rod_histogram, kernel_size=15, iterations=4
        )

        if debug:
            rod_histogram_images.append(
                util.draw_rod_histogram(rod_image.shape, smoothed_histogram / 4)
            )

        # Find peaks
        peaks, properties = find_peaks(smoothed_histogram, height=4, width=10)
        sorted_heights = sorted(properties["peak_heights"], reverse=True)[
            : PLAYERS_PER_ROD[i]
        ]

        for height in sorted_heights:
            player_hist_height_aggregator.update(height)

        min_height = (
            player_hist_height_aggregator.get_mean()
            - player_hist_height_aggregator.get_stdv() * 2
            if player_hist_height_aggregator.get_n() > 42
            else 5
        )

        peaks, properties = find_peaks(
            smoothed_histogram,
            height=min_height,
            width=10,
            distance=PLAYER_DISTANCES[i] - 15,
        )
        sorted_indices = np.argsort(properties["peak_heights"])[::-1]
        peaks = peaks[sorted_indices[: PLAYERS_PER_ROD[i]]]
        peaks.sort()

        # Update player positions and contours
        if len(peaks) == PLAYERS_PER_ROD[i]:
            player_positions[i] = peaks
            player_contours_per_rod[i] = []
            rod_pos = rod_image.shape[1] / 2

            for j, y in enumerate(player_positions[i]):
                for contour in player_contours:
                    x_contour, y_contour, w_contour, h_contour = cv2.boundingRect(
                        contour
                    )
                    if y_contour <= y <= y_contour + h_contour:
                        player_contours_per_rod[i].append(contour)
                        player_feet_positions[i][j] = (
                            x_contour
                            if rod_pos - x_contour > (x_contour + w_contour) - rod_pos
                            else x_contour + w_contour
                        )
                        break  # One match per y-coordinate

    if debug:
        rod_histograms_stacked = util.stack_rod_images(rod_histogram_images)
        player_mask = util.stack_rod_images(rod_images)


def player_detection(frame):
    """Detects player positions and returns their center locations, feet positions, player mask and rod positions."""
    global mask
    start_time = time.time()
    rod_positions = rod.rod_detection(frame)
    time_aggregators["ROD"].update(time.time() - start_time)
    update_player_positions(frame, rod_positions)

    # Compute player positions
    player_position_ret = [
        (
            [(int(rod_positions[i]), int(player_positions[i][1]))]
            if i in {0, 7}
            else [
                (int(rod_positions[i]), int(j)) if j != None else None
                for j in player_positions[i]
            ]
        )
        for i in range(8)
    ]

    # Compute feet positions
    boundaries = util.get_boundaries(rod_positions)
    feet_position_ret = [
        (
            [(rod_p[1] + boundaries[i], player_positions[i][1])]
            if i in {0, 7} and rod_p[1] is not None
            else [
                (x + boundaries[i], player_positions[i][j])
                for j, x in enumerate(rod_p)
                if x is not None
            ]
        )
        for i, rod_p in enumerate(player_feet_positions)
    ]

    # Assertions
    assert len(player_position_ret) == 8
    assert sum(len(rod) for rod in player_position_ret) == 22
    assert len(feet_position_ret) == 8
    time_aggregators["TOTAL"].update(time.time() - start_time)
    # print(
    #     f"Average times: UNET: {time_aggregators['UNET'].get_mean():.6f}, ROD: {time_aggregators['ROD'].get_mean():.6f}, Total :{time_aggregators['TOTAL'].get_mean():.6f}, Other: {time_aggregators['TOTAL'].get_mean()-time_aggregators['UNET'].get_mean() - time_aggregators['ROD'].get_mean():.6f}",
    # )
    return player_position_ret, feet_position_ret, mask, rod_positions


def get_debug_views(frame):
    """Generates debug visualizations for player detection."""
    global debug
    debug = True
    player_positionss, feet_positionss, _, _ = player_detection(frame)
    rod_positions = rod.rod_detection(frame)

    # Debug frame
    debug_frame = frame.copy()
    util.draw_rods(debug_frame, rod_positions)
    boundaries = util.get_boundaries(rod_positions)
    util.draw_boundaries(debug_frame, boundaries)
    util.draw_player_contours(
        debug_frame, player_contours_per_rod, boundaries, thickness=2
    )
    util.draw_dot_lists(
        debug_frame, player_positionss, dot_radius=10, dot_color=(255, 255, 0)
    )
    util.draw_dot_lists(
        debug_frame, feet_positionss, dot_radius=7, dot_color=(0, 255, 255)
    )
    util.draw_dot_lists(
        rod_histograms_stacked,
        player_positionss,
        dot_radius=7,
        dot_color=(0, 0, 255),
    )
    bgr_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    util.draw_dot_lists(
        bgr_mask,
        player_positionss,
        dot_radius=7,
        dot_color=(0, 0, 255),
    )

    return debug_frame, rod_histograms_stacked, bgr_mask


def get_current_mask_and_image():
    """Returns the current mask and image."""
    return mask, image
