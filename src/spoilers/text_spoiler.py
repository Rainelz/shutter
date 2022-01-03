import logging

import cv2
import numpy as np
import PIL

from dice_roller import roll_value
from spoilers.abstract_filter import AbstractFilter


class TextSpoiler(AbstractFilter):
    """Dilate text and replace with grey."""

    def __init__(self, grey=127, dilate_k=3, **kwargs):
        super().__init__(**kwargs)
        self.grey = grey
        self.dilate_k = dilate_k

    def run(self, image):
        grey = roll_value(self.grey)
        dilate_k = roll_value(self.dilate_k)
        logging.debug(f"Running TextSpoilere with grey: {grey} and kernel {dilate_k}")
        cv_im = np.array(image._img)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (dilate_k, dilate_k))
        dilated = cv2.morphologyEx(cv_im, cv2.MORPH_ERODE, kernel)
        dilated[dilated < 120] = grey
        pil_im = PIL.Image.fromarray(dilated)
        image._img = pil_im
        return image
