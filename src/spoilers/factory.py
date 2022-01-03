import importlib
import inspect
import pkgutil

import spoilers

DEFAULT_FONT = "Arial"


def filter_class(cls):
    match cls.__module__.split("."):
        case [module_name, *_] if module_name == spoilers.__name__:
            return not inspect.isabstract(cls)
        case _:
            return False


def _load_classes():
    modules = []
    for importer, modname, ispkg in pkgutil.walk_packages(
        path=spoilers.__path__, prefix=spoilers.__name__ + "."
    ):
        modules += [importlib.import_module(modname)]

    # create dict class name - constructor
    local_modules = [inspect.getmembers(module, inspect.isclass) for module in modules]
    local_classes = {
        name: cls
        for module in local_modules
        for name, cls in module
        if name not in ["Component", "Spoiler"] and filter_class(cls)
    }
    return local_classes


class SpoilerFactory:
    module_classes = None

    @classmethod
    def get_spoiler(cls, spoilername):
        """Given a YML node, iterate over specified elements and instantiate
        Generators.

        Returns a list of generator objects
        """
        if not cls.module_classes:
            cls.module_classes = _load_classes()
        if spoilername in cls.module_classes.keys():
            return cls.module_classes[spoilername]
        else:
            raise ValueError(f"Cannot find implementation for spoiler: {spoilername}")
