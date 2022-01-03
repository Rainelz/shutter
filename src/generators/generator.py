import logging
from math import floor

import numpy.random as random

from dice_roller import get_size_generator
from dice_roller import roll
from dice_roller import roll_value
from generators.component import Component
from generators.factory import GeneratorFactory
from interfaces import AbstractGenerator

MIN = 0
MAX = 1


# TODO refactor this
def get_position_range(component, container_size, last_x=0, last_y=0):
    parent_w, parent_h = container_size
    width, height = component.size
    position = component.node.get("position", dict())
    x = position.get("x", 0)
    y = position.get("y", 0)

    if isinstance(x, list):
        if all(isinstance(val, str) for val in x):
            x = random.choice(x)
    if isinstance(x, str):
        if x in ["head", "left"]:
            x = 0
        elif x == "center":
            x = (parent_w - width) // 2
        elif x in ["tail", "right"]:
            x = parent_w - width
        elif x == "concatenate":
            x = last_x
        else:
            raise ValueError(f"Unsupported position value: {x}")
        x /= parent_w  # result in % relative to parent
    if isinstance(y, list):
        if all(isinstance(val, str) for val in y):
            y = random.choice(y)
    if isinstance(y, str):
        if y in ["head", "top"]:
            y = 0
        elif y == "center":
            y = (parent_h - height) // 2
        elif y in ["tail", "bottom"]:
            y = parent_h - height
        elif y == "concatenate":
            y = last_y
        else:
            raise ValueError(f"Unsupported position value: {y}")
        y /= parent_h

    if isinstance(x, (float, int)):
        x = [x, x]

    if isinstance(y, (float, int)):
        y = [y, y]

    baseline_x = floor(parent_w * x[MIN])
    baseline_y = floor(parent_h * y[MIN])
    max_x = floor(x[MAX] * parent_w)
    max_y = floor(y[MAX] * parent_h)
    try:
        x = roll_value([baseline_x, max(baseline_x, max_x)])
        y = roll_value([baseline_y, max(baseline_y, max_y)])
    except ValueError:
        logging.warning("Illegal configuration position")

    return x, y


class Generator(AbstractGenerator):
    """Class containing criteria and probabilities to generate a composable
    image."""

    def __init__(self, opt):
        super(Generator, self).__init__()
        node = opt[self.__class__.__name__]
        self.node = node
        self.sizes = get_size_generator(node)
        self.generators = GeneratorFactory.get_generators(node)
        self.background = self.node.get("background_color", (255, 255, 255))
        if isinstance(self.background, str):
            vals = self.background.replace("(", "").replace(")", "").split(",")
            self.background = tuple([int(val) for val in vals])
        self.p = node.get("p", 1)
        self.components = []

    def __str__(self):
        return str(self.__class__.__name__)

    def get_size(self, container_size, last_w, last_h):

        width, height = next(self.sizes)

        if width == "fill":
            width = (container_size[0] - last_w) / container_size[0]
        if height == "fill":
            height = (container_size[1] - last_h) / container_size[1]

        if container_size is not None and width <= 1:
            width *= container_size[0]
        if container_size is not None and height <= 1:
            height *= container_size[1]

        size = int(width), int(height)
        return size

    def generate(self, container_size=None, last_w=0, last_h=0):
        """Runs sub-elements generation and computes positions based on the
        config parameters."""
        size = self.get_size(container_size, last_w, last_h)
        logging.info(f"Generating image with size {size}")

        img = Component(str(self), size, self.node)  # create component with rolled size
        # available_x, available_y = width, height = size
        # total_units = 100
        # unit = (height // total_units)
        last_x2 = last_y2 = 0

        for gen in self.generators:  # iterate over sub_generators

            if roll() > gen.p:
                continue  # skip gen
            try:
                component = gen.generate(size, last_x2, last_y2)  # generate component
            except Exception:
                logging.exception(f"Problems generating {gen} {gen.node}")
                break
            # node = gen.node
            x, y = get_position_range(component, size, last_x2, last_y2)
            x, y = img.check_position_for(x, y, component)

            last_x2 = max(x + component.size[0], last_x2)
            last_y2 = max(y + component.size[1], last_y2)

            # available_x -= x - baseline_x
            # available_y -= y - baseline_y

            img.add(component, (x, y))  # paste generated component
        img.render()
        return img
