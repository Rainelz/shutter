import logging

import PIL

from dice_roller import get_value_generator
from dice_roller import roll_value
from spoilers.abstract_filter import AbstractFilter


def _white_noise(width, height, gray_p, grid_ratio=2):
    """Create downscaled noise grid."""
    w = width // grid_ratio
    h = height // grid_ratio
    # w = width
    # h = height
    pil_map = PIL.Image.new("L", (w, h), 255)
    values = get_value_generator(gray_p)
    random_grid = list(map(lambda x: int(next(values)), [0] * w * h))
    pil_map.putdata(random_grid)

    return pil_map.resize((width, height), PIL.Image.LINEAR)


class Background(AbstractFilter):
    """Create noise grid and apply to background."""

    DEFAULT_GREY = [220, 255]
    DEF_GRID_RATIO = 2

    def __init__(self, grey=DEFAULT_GREY, grid_ratio=DEF_GRID_RATIO, **kwargs):
        super(Background, self).__init__(**kwargs)
        self.grey = grey
        self.grid_ratio = grid_ratio

    def run(self, image):

        w, h = image.size
        grid_ratio = roll_value(self.grid_ratio)
        logging.debug(
            f"Running Background with grey {self.grey}, grid_ratio={grid_ratio}"
        )

        noise = _white_noise(w, h, self.grey, grid_ratio)
        data = {"type": self.type(), "grey": self.grey}
        self.annotate(image, data)
        return PIL.ImageChops.darker(image, noise)
