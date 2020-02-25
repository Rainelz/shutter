from __future__ import annotations

from pathlib import Path
import math
import inspect
import logging

import numpy as np
import numpy.random as random
import PIL, PIL.ImageFont, PIL.Image, PIL.ImageDraw, PIL.ImageChops, PIL.ImageOps, PIL.ImageFilter

from dice_roller import roll, roll_value, get_value_generator
from interfaces import Visitor
from generators import Component


class Spoiler(Visitor):
    def __init__(self):
        """
        Loads implemented spoilers using reflection on the module
        """
        import sys
        local_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
        local_classes = {name: cls for name, cls in local_classes
                         if name not in ['Spoiler', 'Component']  and not inspect.isabstract(cls)}
        self.filter_classes = local_classes

    def visit(self, component: Component, **kwargs):
        """ Define spoiler base behavior, visit leaves, check node name and call its constructor"""
        for el, _ in component.elements:
            self.visit(el)
            component.render()
        for spoiler_name, kwargs in component.node.get('spoilers', dict()).items():

            kwargs = kwargs or dict() # handle no args, default values

            cls = self.filter_classes[spoiler_name]
            spoiler = cls(**kwargs)

            component._img = component.convert('L')

            component.update(spoiler.roll_and_run(component)) # do nothing if not run


class Filter:
    """Implements basic filter behaviour"""

    def __init__(self, p=1, **_):
        assert 0 <= p <= 1
        self.p = p

    def type(self):
        return str(self.__class__.__name__)

    def annotate(self, component: Component, data):
        component.data['spoilers'].append(data)

    def roll_and_run(self, image: Component):
        """Rolls and eventually applies the filter"""
        if roll() <= self.p:
            return self.run(image)

    def run(self, image: Component):
        pass




class Crop(Filter):
    """Center Crop component"""
    DEFAULT_BORDER = 0

    def __init__(self, border=DEFAULT_BORDER, **kwargs):
        super().__init__(**kwargs)

        self.border = border

    def run(self, image):
        w, h = image.size
        blank = PIL.Image.new("L", (w, h), 255)
        diff = PIL.ImageChops.difference(image, blank)
        if diff.getbbox() is None:
            return image


        box = list(diff.getbbox()) #x,y, x2, y2
        box[0] -= self.border + 10
        box[1] -= (10 + self.border * 3)
        box[2] += self.border + 10
        box[3] += (10 + self.border * 3)
        data = {'type': self.type, 'box':box}
        self.annotate(image, data)
        return image.crop(tuple(box))


class Pad(Filter):
    """Draw component border (outside)"""
    DEFAULT_N = 2

    def __init__(self, n=DEFAULT_N, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def run(self, image):
        logging.debug(f"Running Pad with n:{self.n}")
        n = roll_value(self.n)
        w, h = image.size
        data = {'type': self.type(), 'n': n}
        self.annotate(image, data)
        bg = PIL.Image.new(image.mode, (w + 2 * n, h + 2 * n), 0)
        bg.paste(image, (n, n))
        return bg


class Rotate(Filter):
    """Rotate image by angle"""
    DEFAULT_ANGLE = 0

    def __init__(self, angle=DEFAULT_ANGLE, **kwargs):
        super().__init__(**kwargs)
        self.angle = angle


    def center_box(self, outer, inner, c, s):
        W, H = outer
        w, h = inner
        w1 = int(c * w + s * h) + 1
        h1 = int(s * w + c * h) + 1
        return ((W - w1) // 2, (H - h1) // 2, (W - w1) // 2 + w1, (H - h1) // 2 + h1)

    def run(self, image):
        angle = round(roll_value(self.angle), 1)
        alpha = math.pi * angle / 180
        c = abs(math.cos(alpha))
        s = abs(math.sin(alpha))
        logging.debug(f"Running Rotate with angle {angle}")
        #TODO check this
        #rotated = Pad(100).run(image).convert('RGBA').rotate(self.angle, expand = 1)
        rotated = image.convert('RGBA').rotate(angle, resample=PIL.Image.BICUBIC, expand=True, fillcolor='white',)
        box = self.center_box(rotated.size, image.size, c, s)
        data = {'type': self.type(), 'angle': angle}
        self.annotate(image, data)
        return rotated.crop(box)



def _white_noise(width, height, gray_p):
    """Create downscaled noise grid """
    w = width // 8
    h = height // 8
    # w = width
    # h = height
    pil_map = PIL.Image.new("L", (w, h), 255)
    values = get_value_generator(gray_p)
    random_grid = map(lambda x: next(values), [0] * w * h)
    pil_map.putdata(list(random_grid))
    return pil_map.resize((width, height), PIL.Image.LINEAR)


class Background(Filter):
    """Create noise grid and apply to background"""
    DEFAULT_GREY=[220,255]

    def __init__(self, grey=DEFAULT_GREY, **kwargs):
        super().__init__(**kwargs)
        self.grey = grey

    def run(self, image):
        logging.debug(f"Running Background with grey {self.grey}")

        w, h = image.size
        noise = _white_noise(w, h, self.grey)
        data = {'type': self.type(), 'grey': self.grey}
        self.annotate(image, data)
        return PIL.ImageChops.darker(image, noise)




class Foreground(Filter):
    """Create noise grid and apply to foreground"""
    DEFAULT_GREY=[0,200]

    def __init__(self, grey=DEFAULT_GREY, **kwargs):
        super().__init__(**kwargs)
        self.grey = grey

    def run(self, image):
        logging.debug(f"Running Foreground with grey {self.grey}")

        w, h = image.size
        noise = _white_noise(w, h, self.grey)
        data = {'type': self.type(), 'grey': self.grey}
        self.annotate(image, data)
        return PIL.ImageChops.lighter(image, noise)

class Blur(Filter):
    """Apply blur noise"""
    DEFAULT_R = 2

    def __init__(self, r=DEFAULT_R, **kwargs):
        super().__init__(**kwargs)
        self.r = r

    def run(self, image):
        r = roll_value(self.r)
        logging.debug(f"Running Blur with radius {r}")
        data = {'type': self.type(), 'r': r}
        self.annotate(image, data)
        return image.filter(PIL.ImageFilter.GaussianBlur(r))

class SaltPepper(Filter):
    DEF_DENSITY=50

    def __init__(self, density=DEF_DENSITY, **kwargs):
        super().__init__(*kwargs)
        self.density = density

    def run(self, image):
        density = roll_value(self.density)
        w,h = image.size
        w = w//4
        h = h//4
        s_vs_p = 1
        amount = 0.4
        out = np.copy(np.array(image))
        # Salt mode
        num_salt = np.ceil(amount * w*h * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.size]
        coords = tuple((coords[1], coords[0]))
        #coords = [coord[1], coord[0] for coord in coords]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * w*h * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.size]
        coords = tuple((coords[1], coords[0]))

        out[coords] = 0
        return PIL.ImageChops.darker(image, PIL.Image.fromarray(out).resize(image.size, PIL.Image.LINEAR))


class Stroke(Filter):
    """Draw random stroke"""
    def __init__(self, num_signs=None, num_strokes=None, step=None, **kwargs):
        super().__init__(*kwargs)
        self.num_signs = num_signs or random.randint(1, 6)
        self.num_strokes = num_strokes or random.randint(3, 14)
        self.step = step or random.randint(10, 50)

    def move(self, position, size):
        x, y = position
        return (x + random.randint(-size, size), y + random.randint(-size, size))

    def draw_sign(self, image):
        w, h = image.size
        draw = PIL.ImageDraw.Draw(image)
        position = random.randint(w // 3, 2 * w // 3), random.randint(h // 3, 2 * h // 3)
        for _ in range(self.num_strokes):
            new_position = self.move(position, self.step)
            draw.line((position, new_position), width=3)
            position = new_position

    def run(self, image):
        if any(dim < 200 for dim in image.size):
            return image
        for _ in range(self.num_signs):
            self.draw_sign(image._img)
        return image

class Dilate(Filter):
    """Dilate black blobs in component"""
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        #self.morphs = [cv2.MORPH_RECT,cv2.MORPH_CROSS,cv2.MORPH_ELLIPSE]
    
    def get_kernel(self):
        #morph = random.choice(self.morphs)
        ksize = random.choice([1,3])
        #return cv2.getStructuringElement(morph, (ksize, ksize))
        return ksize

    def run(self,image):
        kernel = self.get_kernel()
        #image = cv2.erode(np.array(image),kernel,iterations=2)
        return image.filter(PIL.ImageFilter.MinFilter(kernel))

class Erode(Filter):
    """Erode black blobs in component"""
    DEFAULT_K = 3

    def __init__(self,k=DEFAULT_K,**kwargs):
        super().__init__(**kwargs)
        self.k = k
        #self.morphs = [cv2.MORPH_RECT,cv2.MORPH_CROSS,cv2.MORPH_ELLIPSE]
    
    def get_kernel(self):
        #morph = random.choice(self.morphs)
        ksize = random.choice([1,3])
        #return cv2.getStructuringElement(morph, (ksize, ksize))
        return ksize

    def run(self,image):
        kernel = self.get_kernel()
        k = roll_value(self.k)
        data = {'type': self.type(), 'k': k}
        self.annotate(image, data)
        #image = cv2.erode(np.array(image),kernel,iterations=2)
        return image.filter(PIL.ImageFilter.MaxFilter(self.k))


class Overlay(Filter):
    """Random paste an image"""
    def __init__(self, path, size, probabilities=[], **kwargs):
        super().__init__(**kwargs)
        self.path = Path(path)
        assert self.path.exists()
        self.probabilities = probabilities
        self.w, self.h = size['height'], size['width']

    @staticmethod
    def pad_overlay_at(overlay, size):
        w, h = size
        bg = PIL.Image.new("L", size, 255)
        w1 = random.randint(0, w - overlay.width)
        h1 = random.randint(0, h - overlay.height)
        bg.paste(overlay, (w1, h1))
        return bg, (w1, h1)

    def run(self, image):
        if self.path.is_dir():
            paths = list(self.path.glob('*.png'))

            if self.probabilities:
                map = [(name, value) for name, value in self.probabilities.items()]
                files, probs = list(zip(*map))
                paths = sorted(paths, key=lambda x: list(files).index(
                    str(x.stem)))  # order Paths like zip result to keep coupling with probs
            else:
                probs = None
            file_path = random.choice(paths, p=probs)
        else:
            file_path = self.path

        overlay = PIL.Image.open(file_path).convert('L').resize((self.w, self.h))

        overlay, pos = Overlay.pad_overlay_at(overlay, image.size)
        data = {'type': self.type(), 'fname': file_path.name, 'box':[*pos, self.w, self.h] }
        self.annotate(image, data)
        return PIL.ImageChops.darker(image, overlay)

    @staticmethod
    def open(path, size):
        return Overlay(PIL.Image.open(path), size)


class VerticalLine(Filter):
    """Draw a vertical line on the component"""
    def run(self, image):
        w, h = image.size
        try:
            a = random.randint(0, h // 8)
            b = h - random.randint(0, h // 8)
            x = random.randint(0, w)
            draw = PIL.ImageDraw.Draw(image)
            draw.line((x, a, x, b), fill=30, width=4)
            data = {'type': self.type(), 'pos': [x,a,x,b], 'fill':30, 'width' : 4 }
            self.annotate(image, data)
        except Exception as e:
            print(e)
            return image
        return image

class Gradient(Filter):
    """Apply a gradient foregound noise"""
    def __init__(self, gradient_magnitude=1., direction=0, color=0, **kwargs):
        super().__init__(**kwargs)
        self.gradient_mg = gradient_magnitude
        self.direction = direction
        self.color = color

    INITIAL_VAL = 0.9

    def run(self, image):
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        width, height = image.size
        if self.direction:
            gradient_size = (width, 1)
            side = width
        else:
            gradient_size = (1, height)
            side = height

        gradient = PIL.Image.new('L', gradient_size, color=0xFF)

        for x in range(side):
            a = int((self.INITIAL_VAL * 255.) * (1. - self.gradient_mg * float(x) / side))
            # gradient.putpixel((x, 0), 255-x)
            # gradient.putpixel((x, 0), int(255 * (1 - self.gradient_mg * float(x) / side)))
            if a < 0:
                a = 0
            if self.direction:
                gradient.putpixel((x, 0), a)
            else:
                gradient.putpixel((0, x), a)

        alpha = gradient.resize(image.size)
        black_im = PIL.Image.new('RGBA', (width, height), color=(self.color, self.color, self.color))  # i.e. gray
        black_im.putalpha(alpha)
        gradient_im = PIL.Image.alpha_composite(image, black_im)
        return gradient_im
    #
    # @staticmethod
    # def random():
    #     # if random.randint(0, 10) < 7:
    #     #     return Filter()
    #     # else:
    #     grad = random.randint(1, 10)
    #     dir = random.randint(0, 1)
    #     color = random.randint(100, 200)
    #     return Gradient(grad/10, dir, color)

# def filters_from_cfg(cfg):
#     filters = [
#         Pad.random(),
#         Foreground.random(),
#         Blur.random(),
#         Stroke.random(),
#         VerticalLine.random(),
#         Overlay.random('resources/stamps', (100, 100)),
#         Gradient.random(),
#         Rotate.random(),
#         Background.random()
#     ]
#     return filters
#
# def spoil(im, options):
#     im = Crop.random().run(im)
#     filters = [
#         VerticalLine.random(),
#         Overlay.random('resources/stamps', (100, 100)),
#         Rotate.random() if not options.no_skew else Filter(),
#         Pad.random(),
#         Background.random(),
#         Foreground.random(),
#         Stroke.random(),
#         Blur.random(),
#         #Gradient.random()
#     ]
#
#     noisy = im
#     for f in filters:
#         noisy = f.run(noisy)
#
#     return noisy