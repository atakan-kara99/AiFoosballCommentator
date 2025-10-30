from ..player_detection.player_unet import *


video_file = "cv/resources/test_011_2Tore.mp4"

cap = cv2.VideoCapture(video_file)


while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    player_detection(frame)
