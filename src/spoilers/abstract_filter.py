from abc import ABC
from abc import abstractmethod

from dice_roller import roll
from generators.component import Component


class AbstractFilter(ABC):
    """Implements basic filter behaviour."""

    def __init__(self, p=1, exclude=[], **_):
        assert 0 <= p <= 1
        self.exclude = exclude
        self.p = p

    def type(self):
        return str(self.__class__.__name__)

    def annotate(self, component: Component, data):
        component.data["spoilers"].update(data)

    def roll_and_run(self, image: Component):
        """Rolls and eventually applies the filter."""
        if roll() <= self.p:
            img = self.run(image)
            for filter in self.exclude:
                image_spoilers = image.node.get("spoilers", dict())
                image_spoilers.get(filter, {"p": 0}).update(
                    p=0
                )  # clear filter probability
            return img

    @abstractmethod
    def run(self, image: Component):
        pass
