import numpy as np
import PIL

from dice_roller import roll_value
from spoilers.abstract_filter import AbstractFilter


class SaltPepper(AbstractFilter):
    DEF_RATIO = 0.5
    DEF_AMOUNT = 0.05

    def __init__(self, ratio=DEF_RATIO, amount=DEF_AMOUNT, **kwargs):
        super(SaltPepper, self).__init__(**kwargs)
        self.ratio = ratio
        self.amount = amount

    def run(self, image):
        ratio = roll_value(self.ratio)
        amount = roll_value(self.amount)
        w, h = image.size
        w = w // 4
        h = h // 4
        s_vs_p = ratio

        out = np.copy(np.array(image))
        # Salt mode
        num_salt = np.ceil(amount * w * h * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.size]
        coords = tuple((coords[1], coords[0]))
        # coords = [coord[1], coord[0] for coord in coords]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * w * h * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.size]
        coords = tuple((coords[1], coords[0]))

        out[coords] = 0
        intermediate = PIL.ImageChops.lighter(
            image, PIL.Image.fromarray(out).resize(image.size, PIL.Image.CUBIC)
        )
        return PIL.ImageChops.darker(
            intermediate, PIL.Image.fromarray(out).resize(image.size, PIL.Image.CUBIC)
        )
