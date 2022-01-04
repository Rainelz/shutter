import logging

import numpy as np
import PIL

from dice_roller import roll_value
from spoilers.abstract_filter import AbstractFilter


class WhiteNoise(AbstractFilter):
    """Create noise mask and apply non white pixels."""

    def __init__(self, ratio=0.05, **kwargs):
        super(WhiteNoise, self).__init__(**kwargs)
        self.ratio = ratio

    def run(self, image):
        logging.debug("Running WhiteNoise")
        cv2_im = np.array(image._img)
        mask = np.where(cv2_im != 255)  # get black pixels
        mask = np.array(list(zip(*mask)))  # couples
        n_pixels = len(mask)
        ratio = roll_value(self.ratio)
        idx = np.random.choice(
            len(mask), int(n_pixels * ratio), replace=False
        )  # random choice pixels to set white
        mask = mask[idx]
        mask = tuple(zip(*mask))
        cv2_im[mask] = 255
        image.update(PIL.Image.fromarray(cv2_im))
        return image
