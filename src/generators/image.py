from pathlib import Path

import numpy.random as random
import PIL.Image

from dice_roller import roll_value
from generators.component import Component
from generators.generator import Generator


class Image(Generator):
    #    STAMPS = list(Path('resources/heading_stamps/').glob('*.png'))

    def generate(self, container_size=None, last_w=0, last_h=0):

        files_node = self.node.get("files", None)
        if files_node:
            f_path = Path(files_node["path"])

            if not f_path.exists():
                raise ValueError(f"Path {f_path} does not exist")

            paths = list(f_path.glob("*.png"))
            probabilities = files_node.get("probabilities", None)
            if probabilities:
                paths = [path for path in paths if path.stem in probabilities.keys()]
                map = [(name, value) for name, value in probabilities.items()]
                files, probs = list(zip(*map))
                paths = sorted(
                    paths, key=lambda x: list(files).index(str(x.stem))
                )  # order Paths like zip result to keep coupling with probs
            else:
                probs = None
            file_path = random.choice(paths, p=probs)
        else:
            file_path = self.node["file"]

        original = PIL.Image.open(file_path)

        size = self.get_size(container_size, last_w, last_h)
        img = Component(str(self), size, self.node, background_color=self.background)
        w_border = roll_value(self.node.get("w_border", 0))  # %
        w_border = int(w_border * size[0])
        h_border = roll_value(self.node.get("h_border", 0))
        h_border = int(h_border * size[1])

        cropped = (size[0] - w_border), (size[1] - h_border)
        im_size = original.size

        ratio = min(cropped[0] / float(im_size[0]), cropped[1] / float(im_size[1]))
        new_size = int(im_size[0] * ratio), int(im_size[1] * ratio)

        resized = original.resize(new_size, PIL.Image.ANTIALIAS)

        rand_left = roll_value([w_border, cropped[0] - resized.size[0]])
        rand_top = roll_value([h_border, cropped[1] - resized.size[1]])
        position = rand_left, rand_top
        img.annotate({"image": str(file_path), "box": [*position, *new_size]})
        img.paste(resized, position, resized.convert("RGBA"))  # use alpha mask to paste
        return img
