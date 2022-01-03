import logging

import PIL.ImageFilter

from dice_roller import roll_value
from spoilers.abstract_filter import AbstractFilter


class Blur(AbstractFilter):
    """Apply blur noise."""

    DEFAULT_R = 2

    def __init__(self, r=DEFAULT_R, **kwargs):
        super().__init__(**kwargs)
        self.r = r

    def run(self, image):
        r = roll_value(self.r)
        logging.debug(f"Running Blur with radius {r}")
        data = {"type": self.type(), "r": r}
        self.annotate(image, data)
        return image.filter(PIL.ImageFilter.GaussianBlur(r))
