from ..debug_player import OpenCVVideoProcessor
from pathlib import Path
import os
from ..player_detection.player_unet import *


def save_as_trainingdata():
    mask, image = get_current_mask_and_image()
    util.save_as_trainingdata(
        mask,
        image,
        output_folder=output_folder,
        video_path=video_path,
        current_frame=processor.current_frame,
    )


if __name__ == "__main__":
    device = None
    debug = True
    video_path = "cv/resources/rec-20250110-131121.mp4"
    # create directories for saving training data
    output_folder = f"cv/player_detection/training_images/{os.path.basename(video_path).removesuffix(".mp4")}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_folder), "masks").mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_folder), "images").mkdir(parents=True, exist_ok=True)

    processor = OpenCVVideoProcessor(video_path=video_path, num_frames=1, overlay=False)
    processor.register_window(
        window_names=("overlay", "rod_histogram_image", "mask"),
        frame_callback=get_debug_views,
    )
    processor.register_key("m", save_as_trainingdata)

    processor.process_video_multi()
