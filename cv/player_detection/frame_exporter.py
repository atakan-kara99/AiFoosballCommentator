import cv2
import numpy as np
import random
import os


def extract_random_frames(video_path, num_frames=100, output_folder="output_frames"):
    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Generate random frame indices
    random_frame_indices = random.sample(range(total_frames), num_frames)

    # Make output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract and save frames
    for i, frame_idx in enumerate(random_frame_indices):
        # Set the video capture to the random frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read the frame
        ret, frame = cap.read()

        if ret:
            # Save the frame as an image
            output_path = os.path.join(
                output_folder, f"{os.path.basename(video_path)}_frame_{frame_idx}.png"
            )
            cv2.imwrite(output_path, frame)
            print(f"Extracted frame {frame_idx} to {output_path}")
        else:
            print(f"Error reading frame {frame_idx}")

    # Release the video capture object
    cap.release()


# Example usage:
video_path = "cv/resources/test_011.mp4"

extract_random_frames(
    video_path,
    num_frames=10,
    output_folder="cv/player_detection/training_images/unlabeled_images",
)
