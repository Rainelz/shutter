import PIL

from spoilers.abstract_filter import AbstractFilter


class Gradient(AbstractFilter):
    """Apply a gradient foregound noise."""

    def __init__(self, gradient_magnitude=1.0, direction=0, color=0, **kwargs):
        super(Gradient, self).__init__(**kwargs)
        self.gradient_mg = gradient_magnitude
        self.direction = direction
        self.color = color

    INITIAL_VAL = 0.9

    def run(self, image):
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        width, height = image.size
        if self.direction:
            gradient_size = (width, 1)
            side = width
        else:
            gradient_size = (1, height)
            side = height

        gradient = PIL.Image.new("L", gradient_size, color=0xFF)

        for x in range(side):
            a = int(
                (self.INITIAL_VAL * 255.0) * (1.0 - self.gradient_mg * float(x) / side)
            )
            # gradient.putpixel((x, 0), 255-x)
            # gradient.putpixel((x, 0), int(255 * (1 - self.gradient_mg * float(x) / side)))
            if a < 0:
                a = 0
            if self.direction:
                gradient.putpixel((x, 0), a)
            else:
                gradient.putpixel((0, x), a)

        alpha = gradient.resize(image.size)
        black_im = PIL.Image.new(
            "RGBA", (width, height), color=(self.color, self.color, self.color)
        )  # i.e. gray
        black_im.putalpha(alpha)
        gradient_im = PIL.Image.alpha_composite(image, black_im)
        return gradient_im
