from pathlib import Path

import numpy.random as random
import PIL

from spoilers.abstract_filter import AbstractFilter


class Overlay(AbstractFilter):
    """Random paste an image."""

    def __init__(self, path, size, probabilities=[], **kwargs):
        super(Overlay, self).__init__(**kwargs)
        self.path = Path(path)
        assert self.path.exists()
        self.probabilities = probabilities
        self.w, self.h = size["height"], size["width"]

    @staticmethod
    def pad_overlay_at(overlay, size):
        w, h = size
        bg = PIL.Image.new("L", size, 255)
        w1 = random.randint(0, w - overlay.width)
        h1 = random.randint(0, h - overlay.height)
        bg.paste(overlay, (w1, h1))
        return bg, (w1, h1)

    def run(self, image):
        if self.path.is_dir():
            paths = list(self.path.glob("*.png"))

            if self.probabilities:
                map = [(name, value) for name, value in self.probabilities.items()]
                files, probs = list(zip(*map))
                paths = sorted(
                    paths, key=lambda x: list(files).index(str(x.stem))
                )  # order Paths like zip result to keep coupling with probs
            else:
                probs = None
            file_path = random.choice(paths, p=probs)
        else:
            file_path = self.path

        overlay = PIL.Image.open(file_path).convert("L").resize((self.w, self.h))

        overlay, pos = Overlay.pad_overlay_at(overlay, image.size)
        data = {
            "type": self.type(),
            "fname": file_path.name,
            "box": [*pos, self.w, self.h],
        }
        self.annotate(image, data)
        return PIL.ImageChops.darker(image, overlay)

    @staticmethod
    def open(path, size):
        return Overlay(PIL.Image.open(path), size)
