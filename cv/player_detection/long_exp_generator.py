import numpy as np
import cv2


class CleanLongExposureGenerator:
    def __init__(self, shape):
        self.long_exposure_frame = np.zeros(shape, dtype=np.float32)
        self.frame_count_image = np.zeros(shape, dtype=np.float32)

    def add_sections(self, frame, sections):
        """
        frame: frame that includes the sections
        sections: list of tuples where the tuples are (x1, y1, x2, y2)
        """
        for section in sections:
            # TODO check if datatype conversion needs to be done (frame is typically given as uint8)
            x1, y1, x2, y2 = section
            # frame = frame.astype(np.float32)
            roi = frame[y1:y2, x1:x2].astype(np.float32)
            self.long_exposure_frame[y1:y2, x1:x2] = cv2.add(
                self.long_exposure_frame[y1:y2, x1:x2], roi
            )
            self.frame_count_image[y1:y2, x1:x2] = cv2.add(
                self.frame_count_image[y1:y2, x1:x2], np.ones_like(roi, np.float32)
            )

    def get_display_image(self):
        display_image = self.long_exposure_frame / self.frame_count_image
        return display_image.astype(np.uint8)
