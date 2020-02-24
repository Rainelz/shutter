from functools import wraps
import sys
import inspect
import logging
from pathlib import Path
from math import ceil

import PIL.Image
from PIL import ImageDraw

import numpy as np
import numpy.random as random
import textwrap

from interfaces import BaseComponent
from dice_roller import roll, roll_value, fn_map, SAMPLES, get_value_generator
from tablegen import Tablegen


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
        return self.__class__.__name__

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

def get_components(node):
    elements = node.get('elements', None) # iterate over yaml nodes
    if elements is None:
        return []
    # create dict class name - constructor
    local_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    local_classes = {name: cls for name, cls in local_classes
                     if name not in ['BaseComponent', 'ABC'] and not inspect.isabstract(cls)}
    objects = []
    for el in elements:
        class_name = list(el.keys())[0]
        if class_name in local_classes.keys():
            cls = local_classes[class_name]
            objects.append(cls(el)) # create generator object
        else:
            raise AttributeError("error instantiating element", el)

    return objects


def get_position_range(parent_size, node):
    width, height = parent_size
    position = node.get('position', dict())
    x = position.get('x', 0)
    y = position.get('y', 0)
    distribution = position.get('distribution', dict()).get('type', 'uniform')

    if isinstance(x, (float, int)):
        x = [x, x]

    if isinstance(y, (float, int)):
        y = [y, y]

    baseline_x = ceil(width * x[MIN])
    baseline_y = ceil(height * y[MIN])
    max_x = ceil(x[MAX] * width)
    max_y = ceil(y[MAX] * height)
    try:
        x = random.randint(baseline_x, max(baseline_x, max_x) + 1)
        y = random.randint(baseline_y, max(baseline_y, max_y) + 1)
    except ValueError as e:
        logging.info("Illegal configuration (position")

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

DEFAULT_NOISE_P = 0.5
class Generator:
    """
    Class containing criteria and probabilities to generate a composable image
    """
    def __init__(self, opt):
        node = opt[self.__class__.__name__]
        self.node = node
        self.sizes = get_sizes(node)
        self.generators = get_components(node)
        #assert all(gen.p == 1 for gen in self.generators) or sum(gen.p for gen in self.generators) == 1
        self.p = node.get('probability', 1)
        self.components = []

    def __str__(self):
        return self.__class__.__name__

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

    def generate(self, container_size=None):
        """Runs sub-elements generation and computes positions based on the config parameters"""
        size = next(self.sizes)

        if container_size is not None:
            size = [int(dim * size[i]) for i, dim in enumerate(container_size)]
        size = int(size[0]), int(size[1])
        logging.info(f"Generating image with size {size}")
#        spoilers = self.get_spoilers()
        img = Component(str(self), size, self.node)
        available_x, available_y = width, height = size
        # total_units = 100
        # unit = (height // total_units)

        for gen in self.generators:

            if roll() > gen.p:
                continue
            component = gen.generate(size)
            node = gen.node
            x, y = get_position_range(size, node)
            x, y = img.check_position_for(x,y,component)

            # available_x -= x - baseline_x
            # available_y -= y - baseline_y

            img.add(component, (x,y))
        img.render()
        return img


class Container(Generator):
    def generate(self, container_size=None):
        """Runs sub-elements generation and computes positions based on the config parameters"""
        size = next(self.sizes)

        if container_size is not None:
            size = [int(dim * size[i]) for i, dim in enumerate(container_size)]
        size = int(size[0]), int(size[1])
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
        x, y = get_position_range(size, node)
        img.add(im, (x, y))
        img.render()
        return img


class TextGroup(Generator):
    font_sizes = [18,20, 22, 24]
    font_sizes = [50]

    def __init__(self, opt):
        super().__init__(opt)
        self.data_path = self.node.get('source_path', None)
        self.n_lines = self.node.get('n_lines', -1)

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

    def generate(self, container_size=None, dataloader=None, fonts=('Courier',) ):

        factors = next(self.sizes)
        size = [int(dim*factors[i]) for i, dim in enumerate(container_size)]
        #spoilers = self.get_spoilers()
        img = Component(str(self), size, self.node)
        height = random.choice(TextGroup.font_sizes)
        font_name = random.choice(fonts)
        font = PIL.ImageFont.truetype(font_name, height)
        n_lines = roll_value(self.n_lines)

        w_border = random.randint(5, 15)  # %
        h_border = random.randint(5, 15)

        cropped = (size[0] - int(size[0] * w_border / 100)), (size[1] - int(size[1] * h_border / 100))

        draw = ImageDraw.Draw(img)
        offset = h_border
        width, l_height = font.getsize('Ag')
        width = int(cropped[0]//width*2)

        text_gen = self.text_gen()
        while offset + l_height < cropped[1] and n_lines != 0:

            for line in textwrap.wrap(next(text_gen), width=width):
                l_height=font.getsize(line)[1]
                if offset + l_height > cropped[1] or n_lines==0:
                    break
                draw.text(((size[0]-cropped[0])//2, offset), line, font=font, fill=0)
                n_lines -= 1
                offset += l_height
        return img


class Text(Generator):
    alignments = ['left', 'center', 'right']

    def __init__(self, size, font=PIL.ImageFont.truetype('Arial', 16), txt='Testo Prova   ', cfg=None):
        super().__init__(size, dict())

        w_border = random.randint(5,15) #  %
        h_border = random.randint(5,15)

        cropped = (size[0] - int(size[0] * w_border/100)), (size[1] - int(size[1]*h_border/100))

        draw = ImageDraw.Draw(self._img)

        draw.text(((size[0]-cropped[0])/2, (size[1]-cropped[1])//2),txt, font=font, fill=0, align=random.choice(Text.alignments))


class Image(Generator):
#    STAMPS = list(Path('resources/heading_stamps/').glob('*.png'))

    def generate(self, container_size=None):
        files_node = self.node.get('files', None)
        if files_node:
            f_path = Path(files_node['path'])

            if not f_path.exists():
                raise ValueError(f"Path {f_path} does not exist")

            paths = list(f_path.glob('*.png'))
            probabilities = files_node.get('probabilities', None)
            if probabilities:
                map = [(name, value) for name, value in probabilities.items()]
                files, probs = list(zip(*map))
                paths = sorted(paths, key=lambda x: list(files).index(str(x.stem))) # order Paths like zip result to keep coupling with probs
            else:
                probs = None
            file_path = random.choice(paths, p=probs)
        else:
            file_path = self.node['file']

        original = PIL.Image.open(file_path)

        factors = next(self.sizes)
        size = [int(dim * factors[i]) for i, dim in enumerate(container_size)]
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

        img.paste(resized, position)
        return img

#########--------------------

# -- To be fixed
class Table(Generator):
    def generate(self, container_size=None):

        factors = next(self.sizes)
        size = [ceil(dim * factors[i]) for i, dim in enumerate(container_size)]
        compose_type = self.node.get('compose_type')
        img = Component(str(self), size, self.node, background_color=(255,))

        ## PASSARE A TABLE GEN IL TYPE DELLA TABELLA? IN BASE A QUELLO DECIDERE e passare il sottonodo
        t = Tablegen(size[0],size[1],compose_type,self.node)
        t.compose(img, (0,0,*size))
        img.render()
        return img

# -- To be completed
class Tablecells(Generator):
    def generate(self, container_size=None):
        return None





