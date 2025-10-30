import os
import cv2

# ---------------------------#
#        Configuration       #
# ---------------------------#
TEST_NAME = "test_011_2Tore"
VIDEO_PATH = f"cv/resources/{TEST_NAME}.mp4"
SAVE_PATH = f"cv/tests/annotated/{TEST_NAME}_ball.txt"

# Drawing and zooming constants
MAIN_RADIUS = 17          # Radius for drawing circle on main view
ZOOM_FACTOR = 6           # Factor by which the cropped region is zoomed
CROP_SIZE = 100           # Width and height of crop around the clicked point

# Global variable to store the ball's position
ball_pos = None


def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function to handle ball annotation events.

    This function updates the global ball_pos variable when the user clicks or drags the mouse.
    It also prints out the current frame number and ball position.

    Parameters:
        event (int): The type of mouse event.
        x (int): The x-coordinate of the mouse event.
        y (int): The y-coordinate of the mouse event.
        flags (int): Any relevant flags passed by OpenCV.
        param (dict): A dictionary containing additional parameters (e.g., frame number).
    """
    global ball_pos
    if event == cv2.EVENT_LBUTTONUP:
        ball_pos = (x, y)
        print(f"Frame {param['frame']}: annotated at {ball_pos}")
    # Update position continuously while dragging
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
        ball_pos = (x, y)
        print(f"Frame {param['frame']}: dragging at {ball_pos}")


def get_zoomed_view(frame, center, crop_size=CROP_SIZE, zoom_factor=ZOOM_FACTOR):
    """
    Extract a cropped region from the frame around a given center and zoom it.

    This function ensures that the crop region remains within the frame boundaries,
    and then resizes the cropped area using the specified zoom factor.

    Parameters:
        frame (numpy.ndarray): The current video frame.
        center (tuple): The (x, y) coordinates around which to crop.
        crop_size (int): The size (both width and height) of the crop region.
        zoom_factor (int): The factor by which to enlarge the cropped image.

    Returns:
        tuple: A tuple containing:
            - zoomed (numpy.ndarray): The zoomed (resized) cropped image.
            - center_zoom (tuple): The new center coordinates in the zoomed view.
    """
    x, y = center
    h, w = frame.shape[:2]
    half_crop = crop_size // 2

    # Calculate crop boundaries and ensure they remain within the frame
    x1 = max(0, x - half_crop)
    y1 = max(0, y - half_crop)
    x2 = min(w, x + half_crop)
    y2 = min(h, y + half_crop)

    crop = frame[y1:y2, x1:x2]
    zoomed = cv2.resize(crop, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    # Compute the new center in the zoomed view
    center_zoom = ((x - x1) * zoom_factor, (y - y1) * zoom_factor)
    return zoomed, center_zoom


def save_annotations(annotations, path):
    """
    Save the collected annotations to a file.

    Each annotation is saved on a new line.

    Parameters:
        annotations (list of str): The list of annotation strings.
        path (str): The file path to save the annotations.
    """
    with open(path, "w") as f:
        f.write("\n".join(annotations))
    print(f"Annotations saved to {path}")


def main():
    """
    Main function to run the video annotation tool.

    It checks for existing annotation files, loads the video,
    displays each frame for annotation, and handles user inputs.
    The user can annotate by clicking or dragging the mouse. Pressing 'n'
    proceeds to the next frame, and 'q' quits the tool saving all annotations.
    """
    # Check if the annotation file exists and confirm overwrite with the user
    if os.path.exists(SAVE_PATH):
        answer = input(f"File {SAVE_PATH} already exists. Overwrite? (y/n): ")
        if answer.lower() != 'y':
            print("Exiting without overwriting the file.")
            return

    global ball_pos
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Setup main window for annotation
    main_window = "Annotate Video (press 'n' for next, 'q' to quit)"
    cv2.namedWindow(main_window)

    annotations = []      # List to store annotations for each frame
    frame_number = 0      # Initialize frame counter

    # Parameters passed to mouse callback (e.g., current frame number)
    params = {"frame": frame_number}
    cv2.setMouseCallback(main_window, mouse_callback, params)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video stream

        ball_pos = None  # Reset ball position for the new frame
        params["frame"] = frame_number  # Update frame number in callback parameters

        # Process the current frame until user indicates to move on
        while True:
            display_frame = frame.copy()

            # If a ball position is set, draw circles on both main and zoomed views
            if ball_pos is not None:
                # Draw circle on main frame
                cv2.circle(display_frame, ball_pos, MAIN_RADIUS, (0, 0, 255), 2)
                # Generate and display zoomed view
                zoomed, center_zoom = get_zoomed_view(frame, ball_pos)
                cv2.circle(zoomed, center_zoom, MAIN_RADIUS * ZOOM_FACTOR, (0, 0, 255), 2)
                cv2.imshow("Zoomed View", zoomed)
            else:
                # Close zoomed view if no annotation is available
                try:
                    cv2.destroyWindow("Zoomed View")
                except cv2.error:
                    pass

            cv2.imshow(main_window, display_frame)
            key = cv2.waitKey(1) & 0xFF

            # 'n' for next frame, 'q' to quit and save annotations
            if key == ord('n'):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                save_annotations(annotations, SAVE_PATH)
                return

        # If a ball position was annotated, record it with the current frame number
        if ball_pos is not None:
            annotations.append(f"{frame_number}: {ball_pos[0]}, {ball_pos[1]}")
        frame_number += 1

    # Release video capture, close windows, and save the annotations file
    cap.release()
    cv2.destroyAllWindows()
    save_annotations(annotations, SAVE_PATH)


if __name__ == "__main__":
    main()
