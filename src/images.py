from __future__ import annotations
from functools import wraps
import PIL.Image
from PIL.Image import isImageType
from PIL import ImageDraw
import numpy as np
import numpy.random as random
from pathlib import Path
import lorem_ipsum
import textwrap
import logging
# from tablegen import Tablegen
from abc import ABC, abstractmethod
from tablegen import Tablegen
import inspect
import logging
from typing import List


class BaseComponent(ABC):

    @abstractmethod
    def accept(self, visitor: Visitor):
        pass

    @abstractmethod
    def update(self, val):
        pass

    def accept(self, visitor: Visitor):
        # if visitor.should_visit_leaves():
        #     for el, _ in self.elements:
        #         el._img = el.convert('L')
        #         el.accept(visitor)
        # el.render()
        # self.render()
        return visitor.visit(self)
        # self.render()


class Visitor(ABC):

    @abstractmethod
    def visit(self, component: BaseComponent):
        pass

    @abstractmethod
    def should_visit_leaves(self):
        pass


class Exporter(Visitor):
    def visit(self, component: BaseComponent, *kwargs):
        if len(self.element) == 0:
            return

class Component(BaseComponent):

    def __init__(self, size, background_color=(255,255,255)):
        if len(background_color) > 1:
            color_space = 'RGBA'
        else:
            color_space = 'L'
        self._img = PIL.Image.new(color_space, size, background_color)
        self.elements = []

    def __getattr__(self, item):
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
        if im is not None and isinstance(im, PIL.Image.Image):
            self._img = im


    def render(self):
        for el, pos in self.elements:
            #el.render()
            self.paste(el.copy(), pos)

    def add(self, *items):
        self.elements.append(tuple(items))


class Generator:
    def __init__(self, opt):
        node = opt[self.__class__.__name__]
        self.node = node
        self.sizes = get_sizes(node)
        self.generators = get_components(node, opt)

    def __str__(self):
        return self.__class__.__name__

    def generate(self, container_size=None):

        size = next(self.sizes)

        if container_size is not None:
            size = [int(dim * size[i]) for i, dim in enumerate(container_size)]
        size = int(size[0]), int(size[1])
        img = Component(size)
        available_x, available_y = width, height = size
        # total_units = 100
        # unit = (height // total_units)

        for i, gen in enumerate(self.generators):
            component = gen.generate(size)
            available_x -= component.width
            available_y -= component.height
            self.generators[i] = (gen, component)

        # assert available_x >= 0
        # assert available_y >= 0

        available_x = 0
        available_y = 0
        for gen, im in self.generators:

            node = gen.node
            x_minmax, y_minmax = get_position_range(node)
            baseline_x = int(width*x_minmax[MIN])
            baseline_y = int(height*y_minmax[MIN])
            max_x = int(x_minmax[MAX]*width+1)
            max_y = int(y_minmax[MAX]*height+1)
            try:
                x = np.random.randint(baseline_x, min(baseline_x+available_x, max_x)+1)
                y = np.random.randint(baseline_y, min(baseline_y+available_y, max_y)+1)
            except ValueError as e:
                print("Illegal configuration (position")
                continue

            if x+im.width > size[0] or y + im.height > size[1]:
                logging.warning("Placing Component outside image range")

            # available_x -= x - baseline_x
            # available_y -= y - baseline_y

            img.add(im, (x,y))

        img.render()
        return img

class TextGroup(Generator):
    font_sizes = [18,20, 22, 24]

    def generate(self, container_size=None, dataloader=None, fonts=('Arial',) ):

        factors = next(self.sizes)
        size = [int(dim*factors[i]) for i, dim in enumerate(container_size)]

        img = Component(size)
        height = random.choice(TextGroup.font_sizes)
        font_name = random.choice(fonts)
        font = PIL.ImageFont.truetype(font_name, height)
        ascent, descent = font.getmetrics()

        line_height = ascent + descent
        n_lines = size[1] // line_height
        start = random.randint(0, max(1,len(lorem_ipsum.text) - n_lines))
        text = lorem_ipsum.text[start: start + n_lines] #dataloader.get_lines(n_lines)
        text = '\n'.join(text)

        w_border = random.randint(5, 15)  # %
        h_border = random.randint(5, 15)

        cropped = (size[0] - int(size[0] * w_border / 100)), (size[1] - int(size[1] * h_border / 100))

        draw = ImageDraw.Draw(img)
        offset = h_border
        width = font.getsize('A a')[0]
        width = int(cropped[0]//width*3.5)
        for line in textwrap.wrap(text, width=width):
            if offset + font.getsize(line)[1] > cropped[1]:
                break
            draw.text(((size[0]-cropped[0])//2, offset), line, font=font, fill=0)
            offset += font.getsize(line)[1]
        return img
        #body = Text(size, font, text)
        #self.paste(body, (0,0))
        #(width, baseline), (offset_x, offset_y) = font.font.getsize("Placeholder with different letters - gjkxJ")



class Text(Component):
    alignments = ['left', 'center', 'right']

    def __init__(self, size, font=PIL.ImageFont.truetype('Arial', 16), txt='Testo Prova   ', cfg=None):
        super().__init__(size)

        w_border = random.randint(5,15) #  %
        h_border = random.randint(5,15)

        cropped = (size[0] - int(size[0] * w_border/100)), (size[1] - int(size[1]*h_border/100))

        draw = ImageDraw.Draw(self._img)

        draw.text(((size[0]-cropped[0])/2, (size[1]-cropped[1])//2),txt, font=font, fill=0, align=random.choice(Text.alignments))



class HeadingStamp(Generator):
    STAMPS = list(Path('resources/heading_stamps/').glob('*.png'))

    def generate(self, container_size):
        stamp = PIL.Image.open(random.choice(HeadingStamp.STAMPS))

        factors = next(self.sizes)
        size = [int(dim * factors[i]) for i, dim in enumerate(container_size)]

        img = Component(size)
        height = random.choice(TextGroup.font_sizes)
        w_border = random.randint(5,15) #  %
        h_border = random.randint(5,15)

        cropped = (size[0] - int(size[0] * w_border/100)), (size[1] - int(size[1]*h_border/100))
        stamp_size = stamp.size

        ratio = min(cropped[0]/float(stamp_size[0]), cropped[1]/float(stamp_size[1]))

        new_size = int(stamp_size[0]*ratio), int(stamp_size[1]*ratio)

        self.stamp = stamp.resize(new_size, PIL.Image.ANTIALIAS)
        #self.stamp.show()

        rand_left = random.randint(0, w_border + cropped[0]-self.stamp.size[0])
        rand_top = random.randint(0, h_border + cropped[1]-self.stamp.size[1])
        position = rand_left, rand_top

        img.paste(self.stamp, position)
        return img


class Header(Generator):
    """
    |-----------|-------|-----------|
    |     L     |   C   |     R     |
    |___________|_______|___________|

    """
    def __init__(self, opt):
        self.sizes = get_sizes(opt)

        self.elements = get_components(opt, opt)

    def generate(self, size):
        super().__init__(size)
        width, height = size
        unit = (width // 10)
        l = TextGroup((unit * 4, height))
        c = Text((unit * 2, height))
        r = HeadingStamp((unit*4, height))
        self.add(l, (0,0))
        self.add(c, (unit*4, 0))
        self.add(r, (unit*6, 0))
        self.render()
        #self.save('test_header.png')


    @staticmethod
    def random(cfg):
        p = cfg.get('probability', 50)
        if random.randint(0, 100) > p :
            pass
        else:
            pass

class Table(Component):
    def generate(self, size):
        super().__init__(size)

        w_border = random.randint(5, 15)  # %
        h_border = random.randint(5, 15)

        cropped = (size[0] - int(size[0] * w_border / 100)), (size[1] - int(size[1] * h_border / 100))
        print(size)
        print(cropped)
        print('---------')
        t = Tablegen(*cropped)
        t.compose(self._img, ( (size[0]-cropped[0] ) // 2 , (size[1]-cropped[1] ) //2, *cropped))


class Body(Component):

    def __init__(self, size):
        super().__init__(size)
        width, height = size
        unit = (width // 5)
        l = TextGroup((size[0], size[1]//2))
        #t = Table
        # c = Text((unit, height))
        # r = HeadingStamp((unit*2, height))
        self.add(l, (0, 0))
        self.render()
        #self.paste(r, (unit*3, 0))
        #self.save('test_body.png')



class Footer(Component):

    def __init__(self, size):
        super().__init__(size)

        width, height = size
        total_units = 10
        unit = (width // total_units)

        long_el = 4 #  units
        short_el = 0.5

        l = Text((unit * long_el, height))
        c = Text((unit * long_el, height), txt = 'ABCDEFGHIJKLMNOPQRSTUVZ')
        r = Text((int(unit * short_el), height), txt='3/6')

        c_pos = (l.width + random.randint(0, width - (l.width + c.width + r.width)), 0)
        r_pos = (self.width - r.width, 0)

        self.add(l, (0, 0))
        self.add(c, c_pos)
        self.add(r, r_pos)
        self.render()
    # def render(self):
    #     for el, pos in self.elements:
    #         self.paste(el.copy(), pos)
    #     self.save('test_footer.png')


fn_map = {'uniform': np.random.uniform,
          'normal': np.random.normal}
        #visitor.visit(self)
SAMPLES = 500

def get_sizes(node):
    size = node['size']
    width = size.get('width', 1)
    height = size.get('height', 1)
    distribution = size.get('distribution', 'uniform')
    pdf = fn_map[distribution]
    if isinstance(width, list):
        assert(len(width) == 2)
        ws = pdf(*width, SAMPLES)
    else:
        ws = np.full(SAMPLES, fill_value=width)

    if isinstance(height, list):
        assert (len(height) == 2)
        hs = pdf(*height, SAMPLES)
    else:
        hs = np.full(SAMPLES, fill_value=height)
    return zip(ws, hs)


def get_components(node, opt):
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


def get_position_range(node):
    position = node.get('position', dict())
    x = position.get('x', [0, 1])
    y = position.get('y', [0, 1])
    distribution = position.get('distribution', dict()).get('type', 'uniform')
    if not isinstance(x, list):
        x = [x,x]

    if not isinstance(y, list):
        y = [y,y]

    return x, y


MIN = 0
MAX = 1

class Image(Generator):

    def generate(self, container_size=None):
        super().generate(container_size)

    # def generate(self, container_size=None):
    #
    #     size = next(self.sizes)
    #     size = int(size[0]), int(size[1])
    #     img = Component(size)
    #     available_x, available_y = width, height = size
    #     # total_units = 100
    #     # unit = (height // total_units)
    #
    #     for i, gen in enumerate(self.generators):
    #         component = gen.generate(size)
    #         available_x -= component.width
    #         available_y -= component.height
    #         self.generators[i] = (gen, component)
    #
    #     assert available_x >= 0
    #     assert available_y >= 0
    #
    #     available_x += 1
    #     available_y += 1
    #     for gen, im in self.generators:
    #
    #         node = gen.node
    #         x_minmax, y_minmax = get_position_range(node)
    #         baseline_x = int(width*x_minmax[MIN])
    #         baseline_y = int(height*y_minmax[MIN])
    #         max_x = int(x_minmax[MAX]*width+1)
    #         max_y = int(y_minmax[MAX]*height+1)
    #         try:
    #             x = np.random.randint(baseline_x, min(baseline_x+available_x, max_x))
    #             y = np.random.randint(baseline_y, min(baseline_y+available_y, max_y))
    #         except ValueError as e:
    #             print("Illegal configuration (position")
    #             continue
    #
    #         if x+im.width > size[0] or y + im.height > size[1]:
    #             logging.warning("Placing Component outside image range")
    #
    #         available_x -= x - baseline_x
    #         available_y -= y - baseline_y
    #
    #         img.add(im, (x,y))
    #
    #     img.render()
    #     return img



def image(lines, font, line_height=1.2, underline=False):
    sizes = []
    h_ = 0
    w_ = 0
    for line in lines:
        w, h = font.getsize(line)
        sizes.append((w, int(h_), int(h_ + h)))
        h_ += line_height * h
        w_  = max(w_, w)
    #img = PIL.Image.new("L", (2 * (w_ + border) , int(2 * (h_+border))), 255)
    img = PIL.Image.new("L", (2 * (w_ ) , int(2 * (h_))), 255)

    draw = PIL.ImageDraw.Draw(img)
    ws, hs = int((w_ )/ 2), int((h_) / 2)
    for i, line in enumerate(lines):
        if line.strip() != '':
            w, ht, hb = sizes[i]
            draw.text((ws, hs + ht), line, font=font)
            if underline:
                draw.line((ws - 1, hs + hb - 1, ws + w, hs + hb - 1))
    return img


def load_fonts(path, size=30):
    with open(path) as f:
        font_names = list(f.readlines())
        return [PIL.ImageFont.truetype(f.strip(), size) for f in font_names if f.strip() != '']


def random_image(lines, fonts, options, num_lines=50):
    start = random.randint(0, len(lines) - num_lines)
    text_lines = lines[start:start+num_lines]
    text = ''.join(text_lines).strip()
    #font = random.choice(fonts)
    font = PIL.ImageFont.truetype('Arial', 18)
    im = image(text_lines, font, line_height=random.uniform(1.05, 1.3), underline=random.choice([True, False], p=[0.1, 0.9]))
    #im = PIL.ImageOps.expand(im, border=20, fill='black') #add border

    noisy = spoil(im, options)
    return text, noisy.convert('RGB'), im.convert('RGB')