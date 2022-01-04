import logging
from collections import defaultdict
from functools import wraps

import PIL
from PIL import ImageChops

from interfaces import BaseComponent


class Component(BaseComponent):
    """Basic component structuring the resulting image.

    Can be used (almost) as it is a PIL.Image. Implements the
    Composite/Component pattern.
    """

    def __init__(self, type, size, node, background_color=(255, 255, 255, 255)):
        super(Component, self).__init__()
        if len(background_color) > 1:
            color_space = "RGB"
        else:
            color_space = "L"

        self.type = type
        # color = node.get('background_color', None)
        # color = color or background_color
        self._img = PIL.Image.new(color_space, size, background_color)
        self.elements = []
        self.data = defaultdict(dict)
        self.node = node

    def __getattr__(self, item):
        """Maps unresolved function calls to pillow Image calls."""
        attr = object.__getattribute__(self._img, item)
        if hasattr(attr, "__call__"):

            @wraps(attr)
            def _wrapped(*args, **kwargs):
                return attr(*args, **kwargs)

            return _wrapped
        else:
            return attr

    def __str__(self):
        return self.type

    def empty(self):
        return not ImageChops.invert(self).getbbox()

    def update(self, im):
        """Updates internal image."""
        if im is not None and isinstance(im, PIL.Image.Image):
            self._img = im

    def render(self):
        """Render sub-elements pasting each component at given position."""
        for el, pos in self.elements:
            self.paste(el.copy(), pos)

    def add(self, *items):
        self.elements.append(tuple(items))

    def check_position_for(self, x, y, component):
        if self.elements:
            last_component, last_pos = self.elements[-1]
            if x == -3:  # TODO check this ??
                x = last_pos[0] + last_component.width
            if y == -3:
                y = last_pos[1] + last_component.height
        if x + component.width > self.size[0]:
            logging.warning(
                f"Forcing position of {component} on {self} -  ({x + component.width} > {self.size[0]} (width)"
            )
            x = self.size[0] - component.width
        if y + component.height > self.size[1]:
            logging.warning(
                f"Forcing position of {component} on {self} -  ({y + component.height} > {self.size[1]} (height)"
            )
            y = self.size[1] - component.height

        return x, y

    def annotate(self, data):
        self.data["data"].update(data)
