import PIL

from spoilers.abstract_filter import AbstractFilter


class Crop(AbstractFilter):
    """Center Crop component."""

    DEFAULT_BORDER = 0

    def __init__(self, border=DEFAULT_BORDER, **kwargs):
        super().__init__(**kwargs)

        self.border = border

    def run(self, image):
        w, h = image.size
        blank = PIL.Image.new("L", (w, h), 255)
        diff = PIL.ImageChops.difference(image, blank)
        if diff.getbbox() is None:
            return image

        box = list(diff.getbbox())  # x,y, x2, y2
        box[0] -= self.border + 10
        box[1] -= 10 + self.border * 3
        box[2] += self.border + 10
        box[3] += 10 + self.border * 3
        data = {"type": self.type, "box": box}
        self.annotate(image, data)
        return image.crop(tuple(box))
