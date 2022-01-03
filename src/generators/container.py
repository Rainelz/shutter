import logging

from dice_roller import roll_value
from generators.component import Component
from generators.generator import Generator
from generators.generator import get_position_range


class Container(Generator):
    def generate(self, container_size=None, last_w=0, last_h=0):
        """Runs sub-elements generation and computes positions based on the
        config parameters."""

        size = self.get_size(container_size, last_w, last_h)

        logging.debug(f"Generating container with size {size}")

        img = Component(str(self), size, self.node)
        # available_x, available_y = width, height = size
        # total_units = 100
        # unit = (height // total_units)
        probs = [gen.p for gen in self.generators]
        if sum(probs) != 1:
            probs = None  # undefined p, using uniform

        chosen = roll_value(list(zip(self.generators, probs)))

        im = chosen.generate(size)
        # node = chosen.node
        x, y = get_position_range(im, container_size)
        img.add(im, (x, y))
        img.render()
        return img
