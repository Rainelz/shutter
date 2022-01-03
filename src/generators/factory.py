import importlib
import inspect
import pkgutil

import generators

DEFAULT_FONT = "Arial"


def filter_class(cls):
    match cls.__module__.split("."):
        case [module_name, *_] if module_name == generators.__name__:
            return not inspect.isabstract(cls)
        case _:
            return False


def _load_classes():
    modules = []
    for importer, modname, ispkg in pkgutil.iter_modules(
        path=generators.__path__, prefix=generators.__name__ + "."
    ):
        modules += [importlib.import_module(modname)]

    # create dict class name - constructor
    local_modules = [inspect.getmembers(module, inspect.isclass) for module in modules]
    local_classes = {
        name: cls
        for module in local_modules
        for name, cls in module
        if name not in ["Component", "BaseComponent"] and filter_class(cls)
    }
    return local_classes


class GeneratorFactory:
    module_classes = None

    @classmethod
    def get_generators(cls, node):
        """Given a YML node, iterate over specified elements and instantiate
        Generators.

        Returns a list of generator objects
        """
        if not cls.module_classes:
            cls.module_classes = _load_classes()
        elements = node.get("elements", None)  # iterate over yaml nodes
        if elements is None:
            return []

        objects = []
        for el in elements:
            class_name = list(el.keys())[0]
            if class_name in cls.module_classes.keys():
                imported_class = cls.module_classes[class_name]
                objects.append(imported_class(el))  # create generator object
            else:
                raise AttributeError("error instantiating element", el)

        return objects
