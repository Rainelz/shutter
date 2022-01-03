import logging
import math

import PIL

from dice_roller import roll_value
from spoilers.abstract_filter import AbstractFilter


class Rotate(AbstractFilter):
    """Rotate image by angle."""

    DEFAULT_ANGLE = 0

    def __init__(self, angle=DEFAULT_ANGLE, **kwargs):
        super().__init__(**kwargs)
        self.angle = angle

    def center_box(self, outer, inner, c, s):
        W, H = outer
        w, h = inner
        w1 = int(c * w + s * h) + 1
        h1 = int(s * w + c * h) + 1
        return ((W - w1) // 2, (H - h1) // 2, (W - w1) // 2 + w1, (H - h1) // 2 + h1)

    def run(self, image):
        angle = round(roll_value(self.angle), 1)
        alpha = math.pi * angle / 180
        c = abs(math.cos(alpha))
        s = abs(math.sin(alpha))
        logging.debug(f"Running Rotate with angle {angle}")
        # TODO check this
        # rotated = Pad(100).run(image).convert('RGBA').rotate(self.angle, expand = 1)
        rotated = image.convert("RGBA").rotate(
            angle,
            resample=PIL.Image.BICUBIC,
            expand=True,
            fillcolor="white",
        )
        box = self.center_box(rotated.size, image.size, c, s)
        data = {"type": self.type(), "angle": angle}
        self.annotate(image, data)
        return rotated.crop(box)
