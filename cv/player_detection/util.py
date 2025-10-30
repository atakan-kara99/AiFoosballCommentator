import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import imutils


def dilate_and_erode(
    image,
    dilate_iterations=3,
    erode_iterations=3,
    dilate_kernel_size=(7, 7),
    erode_kernel_size=(7, 7),
):
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel_size)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, erode_kernel_size)

    image = cv2.dilate(image, dilate_kernel, iterations=dilate_iterations)
    image = cv2.erode(image, erode_kernel, iterations=erode_iterations)
    return image


def get_mask_histogram(mask, axis, visualize=False):
    binary_mask = (mask > 0).astype(np.uint8)
    x_histogram = np.sum(binary_mask, axis=axis)
    if visualize:
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(x_histogram)), x_histogram, color="blue", width=1)
        plt.bar(
            range(len(x_histogram)),
            sorted(x_histogram),
            color=(1, 0, 0, 0.4),
            width=1,
        )
        plt.title("Pixel Count by X-Coordinate")
        plt.xlabel("X-Coordinate")
        plt.ylabel("Count of Pixels")
        plt.show()
    return x_histogram


def get_field_mask(frame):
    kernel_size = (5, 5)

    frame = cv2.GaussianBlur(frame, kernel_size, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel_size = (9, 9)
    blur = cv2.GaussianBlur(frame, kernel_size, 0)
    bd = cv2.absdiff(frame, blur)

    _, bd = cv2.threshold(
        bd,
        6,
        255,
        cv2.THRESH_BINARY,
    )

    bd = dilate_and_erode(
        bd,
        dilate_iterations=2,
        erode_iterations=1,
        dilate_kernel_size=(3, 3),
        erode_kernel_size=(5, 5),
    )
    bd = cv2.bitwise_not(bd)
    return bd


def draw_rods(frame, rod_positions, width=3, color=(0, 0, 222)):
    for rod in rod_positions:
        cv2.line(
            frame,
            (rod, 0),
            (rod, frame.shape[1]),
            color=color,
            thickness=width,
        )
    frame = cv2.flip(frame, 0)
    return frame


def draw_field_histogram(shape, histogram):
    histogram_image = np.zeros(
        (shape),
        dtype=np.uint8,
    )
    for col, count in enumerate(histogram):
        cv2.line(histogram_image, (col, 0), (col, int(count)), 255, 1)
    return histogram_image


def get_rod_histogram_image(width, height, histogram):
    rod_histogram_image = draw_field_histogram((height, width), histogram)
    rod_histogram_image = resize_img(rod_histogram_image, width, height)
    rod_histogram_image = cv2.cvtColor(rod_histogram_image, cv2.COLOR_GRAY2BGR)
    return rod_histogram_image


def resize_img(image, width, height):
    return cv2.resize(image, [width, height], interpolation=cv2.INTER_LINEAR)


def save_as_trainingdata(mask, image, output_folder, video_path, current_frame):
    img_path = os.path.join(
        output_folder,
        "images",
        f"{os.path.basename(video_path)}_frame_{current_frame}.png",
    )
    cv2.imwrite(img_path, image)

    mask_path = os.path.join(
        output_folder,
        "masks",
        f"{os.path.basename(video_path)}_frame_{current_frame}.png",
    )
    cv2.imwrite(mask_path, mask)
    print(
        f"saved image and mask of frame {current_frame}\n\timage: {img_path}\n\tmask: {mask_path}"
    )


def get_rod_images(normalized_image, rod_x_coords, visualize=False):
    rod_x_coords = np.sort(rod_x_coords)
    image_width = normalized_image.shape[1]
    boundaries = [
        (rod_x_coords[i] + rod_x_coords[i + 1]) // 2
        for i in range(len(rod_x_coords) - 1)
    ]
    boundaries = [0] + boundaries + [image_width]
    segments = []
    for i in range(len(boundaries) - 1):
        x_start = int(boundaries[i])
        x_end = int(boundaries[i + 1])
        segment = normalized_image[:, x_start:x_end]
        segments.append(segment)

    if visualize:
        for i, segment in enumerate(segments):
            cv2.imshow(f"Segment {i + 1}", segment)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
    return segments


def get_filtered_contours(rod_images, i):
    rod_image = rod_images[i]
    contours = cv2.findContours(rod_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(contours)
    contour_centers = []
    lower_bound, upper_bound = 250, 50_000
    for j, c in enumerate(cnts):
        M = cv2.moments(c)
        cx = int(M["m10"] / max(M["m00"], 1))
        cy = int(M["m01"] / max(M["m00"], 1))
        contour_centers.append((cx, cy))
        rod_pos = rod_image.shape[1] / 2
        if (
            rod_pos - 40 > cx
            or rod_pos + 40 < cx
            or not (lower_bound < cv2.contourArea(c) < upper_bound)
        ):
            cv2.fillPoly(rod_image, pts=[c], color=0)
            pass
    contours, _ = cv2.findContours(
        rod_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    rod_images[i] = rod_image
    # Filter contours by size
    return [c for c in contours if lower_bound < cv2.contourArea(c) < upper_bound]


def gaussian_histogram_smoothing(histogram, kernel_size=23, iterations=8):
    # kernel
    sigma = 2
    kernel = np.exp(-np.linspace(-3, 3, kernel_size) ** 2 / (2 * sigma**2))
    kernel = kernel / kernel.sum() * 1.2
    # application
    smoothed_histogram = histogram
    for _ in range(iterations):
        smoothed_histogram = np.convolve(smoothed_histogram, kernel, mode="same")
    return smoothed_histogram


def draw_rod_histogram(shape, histogram):
    rod_histogram_image = np.zeros(
        (shape),
        dtype=np.uint8,
    )
    for row, count in enumerate(histogram):
        cv2.line(rod_histogram_image, (0, row), (int(count), row), 255, 1)
    return rod_histogram_image


def get_boundaries(rod_positions):
    boundaries = [
        (rod_positions[i] + rod_positions[i + 1]) // 2
        for i in range(len(rod_positions) - 1)
    ]
    return [0] + boundaries + [1279]


def draw_boundaries(frame, boundaries, color=(127, 127, 0)):
    for boundary in boundaries:
        cv2.line(
            frame,
            (boundary, 0),
            (boundary, frame.shape[1]),
            color=color,
            thickness=1,
        )


def draw_player_contours(
    frame, player_contours_per_rod, boundaries, color=(222, 0, 222), thickness=3
):
    for i, player_contours in enumerate(player_contours_per_rod):
        if player_contours is not None:
            if player_contours == []:
                continue
            contours = [contour + (boundaries[i], 0) for contour in player_contours]
            cv2.drawContours(frame, contours, -1, color, thickness)


def stack_rod_images(rod_histogram_images):
    rod_histograms_stacked = rod_histogram_images[0]
    for i in range(1, 8):
        rod_histograms_stacked = np.hstack(
            (rod_histograms_stacked, rod_histogram_images[i])
        )
    return cv2.cvtColor(rod_histograms_stacked, cv2.COLOR_GRAY2BGR)


def draw_dot_lists(frame, dot_list, dot_radius=5, dot_color=(0, 0, 255)):
    for sublist in dot_list:
        for dot in sublist:
            cv2.circle(
                frame,
                dot,
                dot_radius,
                dot_color,
                -1,
            )
