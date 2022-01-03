import numpy.random as random
import PIL

from spoilers.abstract_filter import AbstractFilter


class VerticalLine(AbstractFilter):
    """Draw a vertical line on the component."""

    def run(self, image):
        w, h = image.size
        try:
            a = random.randint(0, h // 8)
            b = h - random.randint(0, h // 8)
            x = random.randint(0, w)
            draw = PIL.ImageDraw.Draw(image)
            draw.line((x, a, x, b), fill=30, width=4)
            data = {"type": self.type(), "pos": [x, a, x, b], "fill": 30, "width": 4}
            self.annotate(image, data)
        except Exception as e:
            print(e)
            return image
        return image
