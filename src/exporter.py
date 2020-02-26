from __future__ import annotations
from interfaces import Visitor, BaseComponent
from pathlib import Path
import json
import logging

class LocalExporter(Visitor):
    def __init__(self, path):
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
