from functools import wraps
import sys
import inspect
import logging
from pathlib import Path
from math import ceil
from collections import defaultdict
import PIL.Image
from PIL import ImageDraw
from itertools import product

import numpy as np
import numpy.random as random
import textwrap

from interfaces import BaseComponent
from dice_roller import roll, roll_value, fn_map, SAMPLES, get_value_generator
#from tablegen import TableGen
#from tablegen import Table


    #
    # def should_visit_leaves(self):
    #     return False


class Component(BaseComponent):
    """
    Basic component structuring the resulting image. Can be used (almost) as it is a PIL.Image.
    Implements the Composite/Component pattern.

    """

    def __init__(self, type, size, node, background_color=(255,255,255,255)):
        if len(background_color) > 1:
            color_space = 'RGB'
        else:
            color_space = 'L'

        self.type = type
        color = node.get('background_color', None)
        color = color or background_color
        self._img = PIL.Image.new(color_space, size, color)
        self.elements = []
        self.data = defaultdict(list)
        self.node = node

    def __getattr__(self, item):
        """Maps unresolved function calls to pillow Image calls"""
        attr = object.__getattribute__(self._img, item)
        if hasattr(attr, '__call__'):
            @wraps(attr)
            def _wrapped(*args, **kwargs):
                return attr(*args, **kwargs)
            return _wrapped
        else:
            return attr

    def __str__(self):
        return str(self.__class__.__name__)

    def update(self, im):
        "Updates internal image"
        if im is not None and isinstance(im, PIL.Image.Image):
            self._img = im

    def render(self):
        """Render sub-elements pasting each component at given position"""
        for el, pos in self.elements:
            self.paste(el.copy(), pos)

    def add(self, *items):
        self.elements.append(tuple(items))

    def check_position_for(self, x,y, component):
        if self.elements:
            last_component, last_pos = self.elements[-1]
            if x == -3:
                x = last_pos[0]+last_component.width
            if y == -3 :
                y = last_pos[1]+last_component.height
        if x + component.width > self.size[0]:
            logging.debug("Forcing position on x")
            x = self.size[0] - component.width
        if y + component.height > self.size[1]:
            logging.debug("Forcing position on y")
            y = self.size[1] - component.height

        return x, y

    def annotate(self, data):
        self.data['data'].append(data)

def get_generators(node):
    elements = node.get('elements', None) # iterate over yaml nodes
    if elements is None:
        return []
    # create dict class name - constructor
    local_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    local_classes = {name: cls for name, cls in local_classes
                     if name not in ['Component','BaseComponent'] and not inspect.isabstract(cls)}
    objects = []
    for el in elements:
        class_name = list(el.keys())[0]
        if class_name in local_classes.keys():
            cls = local_classes[class_name]
            objects.append(cls(el)) # create generator object
        else:
            raise AttributeError("error instantiating element", el)

    return objects

# TODO refactor this
def get_position_range(component, container_size, last_x=0, last_y=0):
    parent_w, parent_h = container_size
    width, height = component.size
    position = component.node.get('position', dict())
    x = position.get('x', 0)
    y = position.get('y', 0)

    if isinstance(x, list):
        if all(isinstance(val, str) for val in x):
            x = random.choice(x)
    if isinstance(x, str):
        if x == 'head':
            x = 0
        elif x == 'center':
            x = (parent_w - width) // 2
        elif x == 'tail':
            x = parent_w - width
        elif x == 'concatenate':
            x = last_x
        else:
            raise ValueError(f'Unsupported position value: {x}')
        x /= parent_w  # result in % relative to parent
    if isinstance(y, list):
        if all(isinstance(val, str) for val in y):
            y = random.choice(y)
    if isinstance(y, str):
        if y == 'head':
            y = 0
        elif y == 'center':
            y = (parent_h - height) // 2
        elif y == 'tail':
            y = parent_h-height
        elif y == 'concatenate':
            y = last_y
        else:
            raise ValueError(f'Unsupported position value: {y}')
        y /= parent_h

    if isinstance(x, (float, int)):
        x = [x, x]

    if isinstance(y, (float, int)):
        y = [y, y]
    baseline_x = ceil(parent_w * x[MIN])
    baseline_y = ceil(parent_h * y[MIN])
    max_x = ceil(x[MAX] * parent_w)
    max_y = ceil(y[MAX] * parent_h)
    try:
        x = random.randint(baseline_x, max(baseline_x, max_x) + 1)
        y = random.randint(baseline_y, max(baseline_y, max_y) + 1)
    except ValueError as e:
        logging.info("Illegal configuration position")

    return x, y

# TODO move this
def get_sizes(node):
    size = node.get('size', dict())

    width = size.get('width', 1)
    height = size.get('height', 1)

    while True:
        ws = get_value_generator(width)
        hs = get_value_generator(height)

        for couple in zip(ws, hs):
            yield couple


MIN = 0
MAX = 1
DEF_F_NAME = 'Arial'
DEFAULT_NOISE_P = 0.5
class Generator:
    """
    Class containing criteria and probabilities to generate a composable image
    """
    def __init__(self, opt):
        node = opt[self.__class__.__name__]
        self.node = node
        self.sizes = get_sizes(node)
        self.generators = get_generators(node)
        #assert all(gen.p == 1 for gen in self.generators) or sum(gen.p for gen in self.generators) == 1
        self.p = node.get('p', 1)
        self.components = []

    def __str__(self):
        return str(self.__class__.__name__)

    def get_size(self, container_size, last_w, last_h):
        width, height = next(self.sizes)
        if width == 'fill':
            width = (container_size[0] - last_w) / container_size[0]
        if height == 'fill':
            height = (container_size[1] - last_h) / container_size[1]
        if container_size is not None and width <=1:
            width *= container_size[0]
        if container_size is not None and height <= 1:
            height *= container_size[1]

        size = int(width), int(height)
        return size

    def get_spoilers(self):
        noises = []
        for noise in self.node.get('spoilers', list()):
            if isinstance(noise, str):
                p = DEFAULT_NOISE_P
            else:
                noisenode = list(noise.items())[0][1]
                noisename = list(noise.items())[0][0]

                p = noisenode.get('p', DEFAULT_NOISE_P)
                noise = noisename
            if roll() <= p:
                noises.append(noise)
        return noises

    def generate(self, container_size=None, last_w=0, last_h=0):
        """Runs sub-elements generation and computes positions based on the config parameters"""
        size = self.get_size(container_size, last_w, last_h)
        logging.info(f"Generating image with size {size}")
#        spoilers = self.get_spoilers()
        img = Component(str(self), size, self.node)
        available_x, available_y = width, height = size
        # total_units = 100
        # unit = (height // total_units)
        last_x2 = last_y2 = 0

# TODO add concatenate position here
        for gen in self.generators:

            if roll() > gen.p:
                continue
            component = gen.generate(size, last_x2, last_y2)
            node = gen.node
            x, y = get_position_range(component, size, last_x2, last_y2)
            x, y = img.check_position_for(x,y,component)
            last_x2, last_y2 = max(x + component.size[0], last_x2),\
                               max(y+component.size[1], last_y2)

            # available_x -= x - baseline_x
            # available_y -= y - baseline_y

            img.add(component, (x,y))
        img.render()
        return img


class Container(Generator):
    def generate(self, container_size=None, last_w=0, last_h=0):
        """Runs sub-elements generation and computes positions based on the config parameters"""

        size = self.get_size(container_size, last_w, last_h)

        logging.debug(f"Generating container with size {size}")
        img = Component(str(self), size, self.node)
        available_x, available_y = width, height = size
        # total_units = 100
        # unit = (height // total_units)
        probs = [gen.p for gen in self.generators]
        if sum(probs) != 1:
            probs = None # undefined p, using uniform
        chosen = random.choice(self.generators, p=probs)
        im = chosen.generate(size)
        node = chosen.node
        x, y = get_position_range(im, container_size)
        img.add(im, (x, y))
        img.render()
        return img

# TODO find a way to refactor these
class TextGroup(Generator):
    font_sizes = [30, 38, 44, 52]
    #font_sizes = [50]
    style_map = {'bold': ' Bold', 'italic': ' Italic'}

    def __init__(self, opt):
        super().__init__(opt)
        self.data_path = self.node.get('source_path', None)
        self.n_lines = self.node.get('n_lines', -1)
        self.font = self.node.get('font', dict())
        self.f_name = self.font.get('name', DEF_F_NAME)
        self.font_size = self.font.get('size', 24)
        self.fill = self.font.get('fill', 0)
        self.bold = self.font.get('bold', 0)

    def text_gen(self):
        import os
        if self.data_path:
            file_size = os.path.getsize(self.data_path)
            offset = roll_value([1, file_size])
            with open(self.data_path, 'r') as file:
                file.seek(offset)
                file.readline()

                while True:
                    line = file.readline()
                    if not line:
                        file.seek(0, 0)
                        continue
                    yield line
        else:
            while True:
                text = self.node.get('text', "Placeholder TEXT")
                for line in text.split('\n'):
                    yield line

    def generate(self, container_size=None, last_w=0, last_h=0):


        size = self.get_size(container_size, last_w, last_h)
        #spoilers = self.get_spoilers()
        img = Component(str(self), size, self.node)

        n_lines = roll_value(self.n_lines)

        w_border = roll_value(self.node.get('w_border', 0))  # %
        w_border = int(w_border * size[0])
        h_border = roll_value(self.node.get('h_border', 0))
        h_border = int(h_border * size[1])
        cropped = (size[0] - w_border * 2), (size[1] - h_border * 2)

        font_name = roll_value(self.f_name)
        font_data = {'name': font_name}

        for style, value in TextGroup.style_map.items():
            if roll() < self.font.get(style, 0):
                font_data.update({style: True})
                font_name += value

        font_name = font_name.replace(' ', '_')
        f_size = roll_value(self.font_size)
        try:
            font = PIL.ImageFont.truetype(font_name, f_size)
        except OSError:
            logging.exception(f"Cannot open font {font_name} with size {f_size}")
            exit(1)
        width, l_height = font.getsize('Ag')
        while l_height + h_border > cropped[1]:
            f_size -= 1
            font = PIL.ImageFont.truetype(font_name, f_size)
            width, l_height = font.getsize('A g')

        font_data.update({'size': f_size})
        draw = ImageDraw.Draw(img)
        y = h_border
        width = int(cropped[0]//width*2)

        text_gen = self.text_gen()
        fill = roll_value(self.fill)
        font_data.update({'fill': fill})
        x = w_border

        while y + l_height <= cropped[1] and n_lines != 0:

            for line in textwrap.wrap(next(text_gen), width=width):
                l_height = font.getsize(line)[1]
                if y + l_height > cropped[1] or n_lines==0:
                    break
                img.annotate({'text': line, 'font': font_data, 'box': [x, y, width, l_height]})
                draw.text((x, y), line, font=font, fill=fill)
                n_lines -= 1
                y += l_height
        return img


class Text(Generator):
    alignments = ['left', 'center', 'right']
    style_map = {'bold': ' Bold', 'italic': ' Italic'}

    def __init__(self, opt):
        super().__init__(opt)
        self.data_path = self.node.get('source_path', None)
        self.n_lines = self.node.get('n_lines', -1)
        self.font = self.node.get('font', dict())
        self.f_name = self.font.get('name', DEF_F_NAME)

        self.font_size = self.font.get('size', 'fill')
        self.fill = self.font.get('fill', 0)
        self.bold = self.font.get('bold', 0)
        self.align = self.node.get('align', 'center')
        self.v_align = self.node.get('v_align', 'center')
    def text_gen(self):
        import os
        if self.data_path:
            file_size = os.path.getsize(self.data_path)
            offset = random.randint(1, file_size)
            with open(self.data_path, 'r') as file:
                file.seek(offset)
                file.readline()

                while True:
                    line = file.readline()
                    if not line:
                        file.seek(0, 0)
                        continue
                    yield line
        else:
            while True:
                text = self.node.get('text', "Placeholder TEXT")
                for line in text.split('\n'):
                    yield line

    def get_font(self, text, size):
        width, height = size
        font_name = roll_value(self.f_name)

        font_data = {'name': font_name}
        for style, value in Text.style_map.items():
            if roll() < self.font.get(style, 0):
                font_data.update({style: True})
                font_name += value
        font_name = font_name.replace(' ', '_')
        f_size = roll_value(self.font_size)
        if isinstance(f_size, int):  # check if it fits
            try:
                font = PIL.ImageFont.truetype(font_name, f_size)
            except OSError:
                logging.exception(f"Cannot open font {font_name} with size {f_size}")
                exit(1)
            l_width, l_height = font.getsize(text)
            c_width = l_width / len(text)
            max_chars = width // c_width

            lines = textwrap.wrap(text, width=int(max_chars))
            if l_height > height or l_width > width or f_size < 1:  # doesn't fit, go for filling. N.B. single line!
                f_size = 'fill'  #

        if f_size == 'fill':
            font_data.update({'filled':True})
            f_size = int(height * 0.8)

            while True:
                try:
                    font = PIL.ImageFont.truetype(font_name, f_size)
                except OSError:
                    logging.exception(f"Cannot open font {font_name} with size {f_size}")
                    exit(1)
                c_width, l_height = font.getsize('Ag')
                c_width = c_width / 2
                max_chars = width // c_width
                if max_chars > 0:
                    #print(f"f_size {f_size}, max_ch {max_chars}")
                    lines = textwrap.wrap(text, width=int(max_chars))
                    if l_height < height and len(lines) == 1 or f_size < 1:
                        break

                f_size -= 1
        try:
            font = PIL.ImageFont.truetype(font_name, f_size)
        except OSError:
            logging.exception(f"Cannot open font {font_name} with size {f_size}")
            exit(1)
        font_data.update({'size': f_size})


        return font, font_data

    def generate(self, container_size=None, last_w=0, last_h=0):

        size = self.get_size(container_size, last_w, last_h)

        # spoilers = self.get_spoilers()
        img = Component(str(self), size, self.node)
        
        #n_lines = roll_value(self.n_lines)

        # TODO param these
        w_border = roll_value(self.node.get('w_border', 0))  # %
        w_border = int(w_border * size[0])
        h_border = roll_value(self.node.get('h_border', 0))
        h_border = int(h_border * size[1])
        width, height = cropped = (size[0] - w_border * 2), (size[1] - h_border * 2)
        
        draw = ImageDraw.Draw(img)
        y = h_border

        text = next(self.text_gen())

        font, font_data = self.get_font(text, cropped)

        fill = roll_value(self.fill)
        font_data.update({'fill': fill})

        _, l_height = font.getsize('Ag')

        align = roll_value(self.align)
        v_align = roll_value(self.v_align)
        x = w_border
        l_width, _ = draw.textsize(text, font)
        if v_align == 'bottom':
            y = height-l_height
        elif v_align == 'center':
            y = (height-l_height) // 2

        if align == 'right':
            x = width-l_width
        elif align == 'center':
            x = (width-l_width) // 2

        draw.text((x,y), text, font=font, fill=fill, align=align)
        img.annotate({'text': text, 'font': font_data, 'box': [x, y, l_width, l_height]})

        return img


class Image(Generator):
#    STAMPS = list(Path('resources/heading_stamps/').glob('*.png'))

    def generate(self, container_size=None, last_w=0, last_h=0):

        files_node = self.node.get('files', None)
        if files_node:
            f_path = Path(files_node['path'])

            if not f_path.exists():
                raise ValueError(f"Path {f_path} does not exist")

            paths = list(f_path.glob('*.png'))
            probabilities = files_node.get('probabilities', None)
            if probabilities:
                paths = [path for path in paths if path.stem in probabilities.keys()]
                map = [(name, value) for name, value in probabilities.items()]
                files, probs = list(zip(*map))
                paths = sorted(paths, key=lambda x: list(files).index(str(x.stem))) # order Paths like zip result to keep coupling with probs
            else:
                probs = None
            file_path = random.choice(paths, p=probs)
        else:
            file_path = self.node['file']

        original = PIL.Image.open(file_path)

        size = self.get_size(container_size, last_w, last_h)
        spoilers = self.get_spoilers()
        img = Component(str(self), size, self.node, background_color=(255,255,255,255))
        w_border = random.randint(5,15) #
        h_border = random.randint(5,15)

        cropped = (size[0] - int(size[0] * w_border/100)), (size[1] - int(size[1]*h_border/100))
        im_size = original.size

        ratio = min(cropped[0]/float(im_size[0]), cropped[1]/float(im_size[1]))
        new_size = int(im_size[0]*ratio), int(im_size[1]*ratio)

        resized = original.resize(new_size, PIL.Image.ANTIALIAS)
        #self.stamp.show()
        rand_left = random.randint(0, w_border + cropped[0]-resized.size[0])
        rand_top = random.randint(0, h_border + cropped[1]-resized.size[1])
        position = rand_left, rand_top
        img.annotate({'image': str(file_path), 'box': [*position, *new_size]})
        img.paste(resized, position)
        return img


class TableCell(Generator):

    def __init__(self, opt):
        super().__init__(opt)
        self.values_file = self.node.get('values_file', None)

        self.w_border = self.node.get('w_border', 0)
        self.h_border = self.node.get('h_border', 0)
        self.frame = self.node.get('frame', 2)
        self.font = self.node.get('font', dict())
        self.f_name = self.font.get('name', DEF_F_NAME)
        self.font_size = self.font.get('size', 'fill')
        self.fill = self.font.get('fill', 0)
        self.bold = self.font.get('bold', 0)
        self.align = self.node.get('align', 'center')
        self.v_align = self.node.get('v_align', 'top')

    def text_gen(self):
        import os
        if self.values_file:
            file_size = os.path.getsize(self.values_file)
            offset = random.randint(1, file_size)
            with open(self.values_file, 'r') as file:
                file.seek(offset)
                file.readline()

                while True:
                    line = file.readline()
                    if not line:
                        file.seek(0, 0)
                        continue
                    yield line
        else:
            while True:
                text = self.node.get('value', "123456")
                for line in text.split('\n'):
                    yield line

    def add_frame(self, img, b_color=0):
        b_size = roll_value(self.frame)
        border_w = PIL.Image.new("L", (img.width, b_size), b_color)
        border_h = PIL.Image.new("L", (b_size, img.height), b_color)
        img.paste(border_w, (0, 0))
        img.paste(border_w, (0, img.height - b_size))
        img.paste(border_h, (0, 0))
        img.paste(border_h, (img.width - b_size, 0))
        return b_size, b_size, b_size, b_size
        #return img

    def write_value(self, cell, frame):

        size = cell.size
        # -- white border
        w_border = roll_value(self.w_border)  # in %
        l_border = int(w_border * size[0]) + frame[0]
        r_border = int(w_border * size[0]) + frame[2]
        h_border = roll_value(self.h_border)
        t_border = int(h_border * size[1]) + frame[1]
        b_border = int(h_border * size[1]) + frame[3]
        width, height = (size[0] - l_border - r_border), (size[1] - t_border - b_border)

        # Creating text generator with calculated size, default alignment and my font info
        value_node = {'Text': {'size': {'width': width, 'height': height},
                               'source_path': self.values_file, 'n_lines': 1,
                               'font': self.font}}

        value_gen = Text(value_node)
        value = value_gen.generate(container_size=size)
        value.annotate({'value':True, 'key':False})
        cell.add(value, (l_border, t_border))
        cell.render()
        return cell

    def generate(self, container_size=None, last_w=0, last_h=0):
        size = self.get_size(container_size, last_w, last_h)

        cell = Component(str(self), size, self.node)
        frame = self.add_frame(cell)
        cell.annotate({'frame': frame})
        cell = self.write_value(cell, frame)

        return cell


class Table(Generator):

    def __init__(self, opt):
        super().__init__(opt)
        self.cols = self.node.get('cols', 1)
        self.rows = self.node.get('rows', 1)
        self.cell_size = self.node.get('cell_size', dict())
        self.cell_w = self.cell_size.get('width', 1)
        self.cell_h = self.cell_size.get('height', 1)
        self.fix_rows = self.node.get('fix_rows', 0.5)
        self.plain_table = self.node.get('plain_table', True)

        self.font = self.node.get('font', dict())
        self.f_name = self.font.get('name', DEF_F_NAME)
        self.font_size = self.font.get('size', 'fill')
        self.fill = self.font.get('fill', 0)
        self.bold = self.font.get('bold', 0)
        self.align = self.node.get('align', 'center')
        self.v_align = self.node.get('v_align', 'top')
        self.values_file = self.node.get('values_file', None)
        self.keys_file = self.node.get('keys_file', None)

    def generate(self, container_size=None, last_w=0, last_h=0):
        size = self.get_size(container_size, last_w, last_h)

        table = Component(str(self), size, self.node, background_color=(255,255,255))
        n_rows = roll_value(self.rows)
        n_cols = roll_value(self.cols)

        schema = self.make_table_schema(table, n_rows, n_cols)
        cells = self.gen_cells(schema)
        self.make_table(table, cells)

        return table

    def gen_cells(self, schema):

        for coord in schema:
            _, (cell_w, cell_h) = schema[coord]
            cell_node = {'TableCell': {'size': {'width': cell_w, 'height': cell_h},
                                       'font': self.font,
                                       'values_file': self.values_file
                                       }
                         }
            cell_gen = TableCell(cell_node)
            cell_im = cell_gen.generate()

            schema[coord] = cell_im, (cell_w, cell_h)
        return schema

    def make_table_fixed(self, table, axis):

        def roll_cells(table, n_rows, n_cols, axis):
            # N.B. axis is the axis to use for varying values e.g. 0 -> varying widths
            # trying to keep the values coupled to enable swap based on axis
            dims = table.size
            grid = n_cols, n_rows
            opposite_axis = abs(axis-1)

            measure = dims[axis]  # width
            mu = measure / grid[axis]  # width / n_cols

            fixed_side = dims[opposite_axis] // grid[opposite_axis]  # height // n_rows

            rem = measure  # width
            min = mu * 0.5
            max_ = mu * 1.5
            sigma = mu * 0.3
            sizes = []
            n_cells = grid[axis]  # per row/col

            #  create 1 row/col sizes, calculating varying width/height, permute if axis == 1
            for i in range(n_cells-1):

                var_side = int(roll_value({'distribution': 'normal', 'mu': mu, 'sigma': sigma, 'min': min, 'max': max_}))
                rem -= var_side

                mu = rem / (n_cells - (i + 1))
                max_ = int(mu * 1.5)
                min = int(mu * 0.5)

                size = var_side, fixed_side
                #if axis == 1:  # transpose if rolling heights
                size = size[axis], size[opposite_axis]
                sizes.append(size)

            # === last cell
            size = rem, fixed_side  # fill with last cell
            #if axis == 1:
            size = size[axis], size[opposite_axis]
            sizes.append(size)
            # ===

            values = list(zip(*sizes))[axis]  # get rolled values on axis

            #  iterate on other dim and calc positions for each cell
            table_cells = []
            for i in range(grid[opposite_axis]):
                var_pos = [sum(values[:j]) for j in range(n_cells)]  # cumulative positions
                fix_pos = [fixed_side*i for _ in range(n_cells)]  # fixed side positions
                pos = var_pos, fix_pos
                # couple positions (var & fixed), eventually swap based on axis
                pos = list(zip(pos[axis], pos[opposite_axis]))

                group = []
                for pos, size in zip(pos,sizes): # for each (row/col) create entry (position, size)
                    cell = (pos, size)
                    group.append(cell)

                table_cells.append(group)

            # this block permutes the matrix to have values ordered by row
            # if axis == 1:
            #     table_cells = np.array(table_cells).reshape((n_cols, n_rows, 2, 1))
            #     table_cells = np.swapaxes(table_cells, 0, 1).tolist()

            return table_cells

        pos_mapping = dict()

        n_cols = roll_value(self.cols)
        n_rows = roll_value(self.rows)
        cells = roll_cells(table, n_cols, n_rows, axis)
        for row in cells:
            for pos, size in row:
                pos_mapping[pos] = (None, size)

        return pos_mapping

    def make_table_schema(self, table, n_rows, n_cols):
        pos_mapping = dict()
        if self.plain_table:
            width, height = table.size
            cell_h = height // n_rows
            cell_w = width // n_cols
            couples = product([x for x in range(n_cols)], [y for y in range(n_rows)])
            for couple in couples:
                coord = (couple[0] * cell_w, couple[1] * cell_h)
                pos_mapping[coord] = (None, (cell_w, cell_h))
            return pos_mapping
        else:
            axis = roll_value([0, 1])

            return self.make_table_fixed(table, axis=axis)

    @staticmethod
    def make_table(table, cells):
        for cell_pos in cells:
            table.add(cells[cell_pos][0], cell_pos)
        table.render()




