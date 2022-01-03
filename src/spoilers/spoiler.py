from generators.component import Component
from interfaces import Visitor
from spoilers.factory import SpoilerFactory


class Spoiler(Visitor):
    def __init__(self):
        """Loads implemented spoilers using reflection on the module."""
        super(Spoiler, self).__init__()

    def visit(self, component: Component, **kwargs):
        """Define spoiler base behavior, visit leaves, check node name and call
        its constructor."""
        for el, _ in component.elements:
            self.visit(el)
            component.render()
        for spoiler_name, kwargs in component.node.get("spoilers", dict()).items():

            kwargs = kwargs or dict()  # handle no args, default values

            cls = SpoilerFactory.get_spoiler(spoiler_name)
            spoiler = cls(**kwargs)

            component._img = component.convert("L")

            component.update(spoiler.roll_and_run(component))  # do nothing if not run
