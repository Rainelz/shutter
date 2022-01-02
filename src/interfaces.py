from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class BaseComponent(ABC):
    """Abstract class defining the component interface for visitors."""

    elements = NotImplemented
    node = NotImplemented
    data = NotImplemented

    @abstractmethod
    def update(self, val):
        pass

    def accept(self, visitor: Visitor, **kwargs):

        return visitor.visit(self, **kwargs)


class Visitor(ABC):
    @abstractmethod
    def visit(self, component: BaseComponent, **kwargs):
        pass
