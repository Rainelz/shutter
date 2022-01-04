import logging

import PIL

from dice_roller import roll_value
from spoilers.abstract_filter import AbstractFilter


class Pad(AbstractFilter):
    """Draw component border (outside)"""

    DEFAULT_N = 2

    def __init__(self, n=DEFAULT_N, **kwargs):
        super(Pad, self).__init__(**kwargs)
        self.n = n

    def run(self, image):
        n = roll_value(self.n)
        logging.debug(f"Running Pad with n:{n}")

        w, h = image.size
        data = {"type": self.type(), "n": n}
        self.annotate(image, data)
        draw = PIL.ImageDraw.Draw(image)
        draw.rectangle((0, 0, w, h), width=n)

        # bg = PIL.Image.new(image.mode, (w + 2 * n, h + 2 * n), 0)
        # bg.paste(image, (n, n))
        return image
