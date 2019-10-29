from __future__ import annotations
from functools import wraps
import PIL.Image
from PIL import ImageDraw
import numpy as np
import numpy.random as random
from scipy.stats import truncnorm
import inspect
import logging
from pathlib import Path
import textwrap
from abc import ABC, abstractmethod


from tablegen import Tablegen


class BaseComponent(ABC):
    """
    Abstract class defining the component interface for visitors
    """
    elements = NotImplemented

    @abstractmethod
    def update(self, val):
        pass

    def accept(self, visitor: Visitor):

        return visitor.visit(self)


class Visitor(ABC):

    @abstractmethod
    def visit(self, component: BaseComponent):
        pass

    @abstractmethod
    def should_visit_leaves(self):
        pass


class Exporter(Visitor):
    def visit(self, component: BaseComponent, *kwargs):
        if len(component.elements) == 0:
            return

    def should_visit_leaves(self):
        return False


class Component(BaseComponent):
    """
    Basic component structuring the resulting image. Can be used (almost) as it is a PIL.Image.
    Implements the Composite/Component pattern.

    """

    def __init__(self, size, spoilers=[], background_color=(255,255,255)):
        if len(background_color) > 1:
            color_space = 'RGBA'
        else:
            color_space = 'L'
        self._img = PIL.Image.new(color_space, size, background_color)
        self.elements = []
        self.spoilers = spoilers

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
            #el.render()
            self.paste(el.copy(), pos)

    def add(self, *items):
        self.elements.append(tuple(items))


def get_components(node):
    elements = node.get('elements', None)
    if elements is None:
        return []
    import sys
    local_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    local_classes = {name: cls for name, cls in local_classes
                     if name not in ['BaseComponent', 'ABC'] and not inspect.isabstract(cls)}
    objects = []
    for el in elements:
        class_name = list(el.keys())[0]
        if class_name in local_classes.keys():
            cls = local_classes[class_name]
            objects.append(cls(el))

        else:
            raise AttributeError("error instantiating element", el)

    return objects


def get_position_range(parent_size, node):
    width, height = parent_size
    position = node.get('position', dict())
    x = position.get('x', [0, 1])
    y = position.get('y', [0, 1])
    distribution = position.get('distribution', dict()).get('type', 'uniform')
    if not isinstance(x, list):
        x = [x, x]

    if not isinstance(y, list):
        y = [y, y]

    baseline_x = int(width * x[MIN])
    baseline_y = int(height * y[MIN])
    max_x = int(x[MAX] * width)
    max_y = int(y[MAX] * height)
    try:
        x = random.randint(baseline_x, min(baseline_x, max_x) + 1)
        y = random.randint(baseline_y, min(baseline_y, max_y) + 1)
    except ValueError as e:
        logging.info("Illegal configuration (position")

    return x, y


def get_sizes(node):
    size = node.get('size', dict())

    width = size.get('width', 1)
    height = size.get('height', 1)

    while True:
        if isinstance(width, list):
            assert(len(width) == 2)

            ws = random.uniform(*width, SAMPLES)
        elif isinstance(width, dict):
            distribution = width.get('distribution', lambda *x: None)
            pdf = fn_map[distribution]
            args = width['mu'], width['sigma'], width['min'], width['max']
            ws = pdf(*args, SAMPLES)
            # import matplotlib.pyplot as plt
            # plt.hist(ws, bins='auto')  # arguments are passed to np.histogram
            # plt.title("Histogram with 'auto' bins")
            # plt.show()

        else:
            ws = np.full(SAMPLES, fill_value=width)

        if isinstance(height, list):
            assert (len(height) == 2)
            hs = random.uniform(*height, SAMPLES)
        elif isinstance(height, dict):
            distribution = height.get('distribution', 'uniform')
            pdf = fn_map[distribution]
            args = height['mu'], height['sigma'], height['min'], height['max']
            hs = pdf(*args, SAMPLES)
        else:
            hs = np.full(SAMPLES, fill_value=height)
        for couple in zip(ws, hs):
            yield couple


def calc_probabilities():
    """Generate samples in range [0,1]"""
    while True:
        chances = random.uniform(0,1, SAMPLES)
        for val in chances:
            yield val


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
        self.p = node.get('probability',1)
        self.dice = calc_probabilities()
        self.components = []

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
            roll = next(self.dice)
            if roll <= p:
                noises.append(noise)
        return noises

    def __str__(self):
        return self.__class__.__name__

    def generate(self, container_size=None):
        """Runs sub-elements generation and computes positions based on the config parameters"""
        size = next(self.sizes)

        if container_size is not None:
            size = [int(dim * size[i]) for i, dim in enumerate(container_size)]
        size = int(size[0]), int(size[1])
        logging.info(f"Generating image with size {size}")
        spoilers = self.get_spoilers()
        img = Component(size, spoilers)
        available_x, available_y = width, height = size
        # total_units = 100
        # unit = (height // total_units)

        for gen in self.generators:
            if next(self.dice) > gen.p:
                continue
            im = gen.generate(size)
            node = gen.node
            x, y = get_position_range(size, node)

            if x+im.width > size[0] or y + im.height > size[1]:
                logging.warning("Placing Component outside image range")

            # available_x -= x - baseline_x
            # available_y -= y - baseline_y

            img.add(im, (x,y))
        img.render()
        return img


class Container(Generator):
    def generate(self, container_size=None):
        """Runs sub-elements generation and computes positions based on the config parameters"""
        size = next(self.sizes)

        if container_size is not None:
            size = [int(dim * size[i]) for i, dim in enumerate(container_size)]
        size = int(size[0]), int(size[1])
        logging.info(f"Generating image with size {size}")
        img = Component(size)
        available_x, available_y = width, height = size
        # total_units = 100
        # unit = (height // total_units)
        probs = [gen.p for gen in self.generators]
        chosen = random.choice(self.generators, p=probs)
        im = chosen.generate(size)
        node = chosen.node
        x, y = get_position_range(size, node)
        img.add(im, (x, y))
        img.render()
        return img


class TextGroup(Generator):
    font_sizes = [18,20, 22, 24]

    def __init__(self, opt):
        super().__init__(opt)
        self.data_path = self.node.get('source_path', None)

    def text_gen(self):
        import os
        if self.data_path:
            file_size = os.path.getsize(self.data_path)
            offset = random.randint(0, file_size)
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

    def generate(self, container_size=None, dataloader=None, fonts=('Arial',) ):

        factors = next(self.sizes)
        size = [int(dim*factors[i]) for i, dim in enumerate(container_size)]
        spoilers = self.get_spoilers()
        img = Component(size, spoilers)
        height = random.choice(TextGroup.font_sizes)
        font_name = random.choice(fonts)
        font = PIL.ImageFont.truetype(font_name, height)
        ascent, descent = font.getmetrics()

        # line_height = ascent + descent
        # n_lines = size[1] // line_height
        # start = random.randint(0, max(1,len(lorem_ipsum.text) - n_lines))
        # text = lorem_ipsum.text[start: start + n_lines] #dataloader.get_lines(n_lines)
        # text = '\n'.join(text)

        w_border = random.randint(5, 15)  # %
        h_border = random.randint(5, 15)

        cropped = (size[0] - int(size[0] * w_border / 100)), (size[1] - int(size[1] * h_border / 100))

        draw = ImageDraw.Draw(img)
        offset = h_border
        width, l_height = font.getsize('A a')
        width = int(cropped[0]//width*3.5)

        text_gen = self.text_gen()
        while offset + l_height < cropped[1]:

            for line in textwrap.wrap(next(text_gen), width=width):
                l_height=font.getsize(line)[1]
                if offset + l_height > cropped[1]:
                    break
                draw.text(((size[0]-cropped[0])//2, offset), line, font=font, fill=0)
                offset += l_height
        return img


class Text(Component):
    alignments = ['left', 'center', 'right']

    def __init__(self, size, font=PIL.ImageFont.truetype('Arial', 16), txt='Testo Prova   ', cfg=None):
        super().__init__(size)

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

            probabilities = files_node['probabilities']
            map = [(name, value) for entry in probabilities for name, value in entry.items()]
            files, probs = list(zip(*map))
            paths = list(f_path.glob('*.png'))
            paths = sorted(paths, key=lambda x: list(files).index(str(x.stem))) # order Paths like zip result to keep coupling with probs
            file_path = random.choice(paths, p=probs)
        else:
            file_path = self.node['file']

        original = PIL.Image.open(file_path)

        factors = next(self.sizes)
        size = [int(dim * factors[i]) for i, dim in enumerate(container_size)]
        spoilers = self.get_spoilers()
        img = Component(size, spoilers)
        w_border = random.randint(5,15) #  %
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


class Table(Generator):
    def generate(self, container_size=None):
        w_border = random.randint(5, 15)  # %
        h_border = random.randint(5, 15)
        factors = next(self.sizes)
        size = [int(dim * factors[i]) for i, dim in enumerate(container_size)]
        spoilers = self.get_spoilers()
        img = Component(size)
        cropped = (size[0] - int(size[0] * w_border / 100)), (size[1] - int(size[1] * h_border / 100))

        t = Tablegen(*cropped)
        t.compose(img, ( (size[0]-cropped[0] ) // 2 , (size[1]-cropped[1] ) //2, *cropped))
        img.render()
        return img


class Footer(Generator):

    def generate(self, container_size=None):
        factors = next(self.sizes)
        size = [int(dim * factors[i]) for i, dim in enumerate(container_size)]
        spoilers = self.get_spoilers()
        img = Component(size, spoilers)
        w_border = random.randint(5, 15)  # %
        h_border = random.randint(5, 15)


        width, height = size
        total_units = 10
        unit = (width // total_units)

        long_el = 4 #  units
        short_el = 0.5

        l = Text((unit * long_el, height))
        c = Text((unit * long_el, height), txt = 'ABCDEFGHIJKLMNOPQRSTUVZ')
        r = Text((int(unit * short_el), height), txt='3/6')

        c_pos = (l.width + random.randint(0, width - (l.width + c.width + r.width)), 0)
        r_pos = (img.width - r.width, 0)

        img.add(l, (0, 0))
        img.add(c, c_pos)
        img.add(r, r_pos)
        img.render()
        return img




def truncated_normal(mean=0, sd=1, low=0, upp=10, samples=1):
    a, b = (low - mean) / sd, (upp - mean) / sd
    return truncnorm(a, b, loc=mean, scale=sd).rvs(samples)


fn_map = {'uniform': random.uniform,
          'normal': truncated_normal}
        #visitor.visit(self)
SAMPLES = 1000


