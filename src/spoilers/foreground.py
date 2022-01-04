import logging

import PIL

from dice_roller import roll_value
from spoilers.abstract_filter import AbstractFilter
from spoilers.background import _white_noise


class Foreground(AbstractFilter):
    """Create noise grid and apply to foreground."""

    DEFAULT_GREY = [0, 200]
    DEF_GRID_RATIO = 2

    def __init__(self, grey=DEFAULT_GREY, grid_ratio=DEF_GRID_RATIO, **kwargs):
        super(Foreground, self).__init__(**kwargs)
        self.grey = grey
        self.grid_ratio = grid_ratio

    def run(self, image):
        logging.debug(f"Running Foreground with grey {self.grey}")
        w, h = image.size
        grid_ratio = roll_value(self.grid_ratio)
        noise = _white_noise(w, h, self.grey, grid_ratio=grid_ratio)
        data = {"type": self.type(), "grey": self.grey}
        self.annotate(image, data)
        return PIL.ImageChops.lighter(image, noise)
