from __future__ import annotations
from pathlib import Path
import json
import inspect
import sys, os
from interfaces import Visitor, BaseComponent
import PIL.Image
import logging


class LocalExporter(Visitor):
    def __init__(self, path, **kwargs):
        self.depth = 0
        self.path = Path(path)

    def visit(self, component: BaseComponent, file_name=None, **kwargs):

        self.depth += 1
        data = {'type': component.type, 'size': component.size, 'elements': []}

        for el, position in component.elements:
            sub_data = self.visit(el)
            sub_data.update(position=position)
            data['elements'].append(sub_data)
            
        self.depth -= 1

        data.update(component.data)
        if self.depth == 0:

            assert file_name
            with open(self.path/f'{file_name}.json', 'w') as f:
                json.dump(data, f)

        return data


class GlobalExporter(Visitor):
    def __init__(self, path, **kwargs):
        self.depth = 0
        self.path = Path(path)

    def visit(self, component: BaseComponent, global_x=0, global_y=0, file_name=None, **kwargs):

        self.depth += 1
        data = {'type': component.type, 'size': component.size, 'elements': []}

        for el, position in component.elements:
            x, y = position

            sub_data = self.visit(el, global_x+x, global_y+y)
            sub_data.update(position=(global_x+x, global_y+y))
            data['elements'].append(sub_data)
        box = component.data.get('data', dict()).get('box')
        if box:
            box[0] = box[0]+global_x
            box[1] = box[1]+global_y
        self.depth -= 1
        data.update(component.data)
        if self.depth == 0:
            assert file_name
            with open(self.path / f'{file_name}.json', 'w') as f:
                json.dump(data, f)

        return data


def get_concat_h(im1, im2):
    dst = PIL.Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


class CoupleExporter(Visitor):
    def __init__(self, exp_path, **kwargs):

        self.path = Path(exp_path) / 'couple'
        os.makedirs(str(self.path), exist_ok=True)
        self.orig_path = self.path.parent.parent / 'original'

    def visit(self, component: BaseComponent, file_name=None, **kwargs):
        f_name = f"{file_name}.png"
        orig = PIL.Image.open(str(self.orig_path / f_name))
        result = get_concat_h(orig, component)
        result.save(str(self.path/f_name))


def from_options(opt, export_dir):
    exporters = opt.get('Exporters', None)  # iterate over yaml nodes
    if exporters is None:
        return []
    # create dict class name - constructor
    local_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    local_classes = {name: cls for name, cls in local_classes
                     if not inspect.isabstract(cls)}
    objects = []
    for el in exporters:
        class_name = list(el.keys())[0]
        if class_name in local_classes.keys():
            args = el[class_name] or dict()
            cls = local_classes[class_name]
            objects.append(cls(export_dir, **args))  # create generator object
        else:
            raise AttributeError("error instantiating element", el)

    return objects