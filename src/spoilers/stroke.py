import numpy.random as random
import PIL

from spoilers.abstract_filter import AbstractFilter


class Stroke(AbstractFilter):
    """Draw random stroke."""

    def __init__(self, num_signs=None, num_strokes=None, step=None, **kwargs):
        super().__init__(**kwargs)
        self.num_signs = num_signs or random.randint(1, 6)
        self.num_strokes = num_strokes or random.randint(3, 14)
        self.step = step or random.randint(10, 50)

    def move(self, position, size):
        x, y = position
        return (x + random.randint(-size, size), y + random.randint(-size, size))

    def draw_sign(self, image):
        w, h = image.size
        draw = PIL.ImageDraw.Draw(image)
        position = random.randint(w // 3, 2 * w // 3), random.randint(
            h // 3, 2 * h // 3
        )
        for _ in range(self.num_strokes):
            new_position = self.move(position, self.step)
            draw.line((position, new_position), width=3)
            position = new_position

    def run(self, image):
        if any(dim < 200 for dim in image.size):
            return image
        for _ in range(self.num_signs):
            self.draw_sign(image._img)
        return image
