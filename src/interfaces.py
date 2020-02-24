from __future__ import annotations
from abc import ABC, abstractmethod


class BaseComponent(ABC):
    """
    Abstract class defining the component interface for visitors
    """
    elements = NotImplemented
    node = NotImplemented

    @abstractmethod
    def update(self, val):
        pass

    def accept(self, visitor: Visitor):

        return visitor.visit(self)


class Visitor(ABC):

    @abstractmethod
    def visit(self, component: BaseComponent):
        pass


class Exporter(Visitor):
    def visit(self, component: BaseComponent, *kwargs):
        if len(component.elements) == 0:
            return