from __future__ import annotations

from pathlib import Path
import math
import inspect
import logging
from io import BytesIO

import numpy as np
import numpy.random as random
import PIL, PIL.ImageFont, PIL.Image, PIL.ImageDraw, PIL.ImageChops, PIL.ImageOps, PIL.ImageFilter

import cv2

from dice_roller import roll, roll_value, get_value_generator
from interfaces import Visitor
from generators import Component, TableCell


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

    def __init__(self, p=1, exclude=[], **_):
        assert 0 <= p <= 1
        self.exclude = exclude
        self.p = p

    def type(self):
        return str(self.__class__.__name__)

    def annotate(self, component: Component, data):
        component.data['spoilers'].update(data)

    def roll_and_run(self, image: Component):
        """Rolls and eventually applies the filter"""
        if roll() <= self.p:
            img = self.run(image)
            for filter in self.exclude:
                image_spoilers = image.node.get('spoilers', dict())
                image_spoilers.get(filter, {'p': 0}).update(p=0)  # clear filter probability
            return img

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
        n = roll_value(self.n)
        logging.debug(f"Running Pad with n:{n}")

        w, h = image.size
        data = {'type': self.type(), 'n': n}
        self.annotate(image, data)
        draw = PIL.ImageDraw.Draw(image)
        draw.rectangle((0,0,w,h), width=n)


        # bg = PIL.Image.new(image.mode, (w + 2 * n, h + 2 * n), 0)
        # bg.paste(image, (n, n))
        return image


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



def _white_noise(width, height, gray_p, grid_ratio=2):
    """Create downscaled noise grid """
    w = width // grid_ratio
    h = height // grid_ratio
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
    DEF_GRID_RATIO=2

    def __init__(self, grey=DEFAULT_GREY, grid_ratio=DEF_GRID_RATIO,**kwargs):
        super().__init__(**kwargs)
        self.grey = grey
        self.grid_ratio=grid_ratio

    def run(self, image):
        logging.debug(f"Running Background with grey {self.grey}")

        w, h = image.size
        grid_ratio = roll_value(self.grid_ratio)
        noise = _white_noise(w, h, self.grey, grid_ratio)
        data = {'type': self.type(), 'grey': self.grey}
        self.annotate(image, data)
        return PIL.ImageChops.darker(image, noise)




class Foreground(Filter):
    """Create noise grid and apply to foreground"""
    DEFAULT_GREY=[0,200]
    DEF_GRID_RATIO=2

    def __init__(self, grey=DEFAULT_GREY, grid_ratio=DEF_GRID_RATIO, **kwargs):
        super().__init__(**kwargs)
        self.grey = grey
        self.grid_ratio = grid_ratio

    def run(self, image):
        logging.debug(f"Running Foreground with grey {self.grey}")
        w, h = image.size
        grid_ratio = roll_value(self.grid_ratio)
        noise = _white_noise(w, h, self.grey, grid_ratio=grid_ratio)
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
    DEF_RATIO=0.5
    DEF_AMOUNT=0.05

    def __init__(self, ratio=DEF_RATIO, amount=DEF_AMOUNT, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.amount = amount

    def run(self, image):
        ratio = roll_value(self.ratio)
        amount = roll_value(self.amount)
        w,h = image.size
        w = w//4
        h = h//4
        s_vs_p = ratio

        out = np.copy(np.array(image))
        # Salt mode
        num_salt = np.ceil(amount * w*h * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.size]
        coords = tuple((coords[1], coords[0]))
        #coords = [coord[1], coord[0] for coord in coords]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * w*h * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.size]
        coords = tuple((coords[1], coords[0]))

        out[coords] = 0
        intermediate = PIL.ImageChops.lighter(image, PIL.Image.fromarray(out).resize(image.size, PIL.Image.CUBIC) )
        return PIL.ImageChops.darker(intermediate, PIL.Image.fromarray(out).resize(image.size, PIL.Image.CUBIC))


class Stroke(Filter):
    """Draw random stroke"""
    def __init__(self, num_signs=None, num_strokes=None, step=None, **kwargs):
        super().__init__(**kwargs)
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

class JPEGCompression(Filter):
    """Apply a gradient foregound noise"""

    def __init__(self, quality=50, subsampling=-1, **kwargs):
        super().__init__(**kwargs)
        self.quality = quality
        self.subsampling = subsampling

    def run(self, image):
        quality = roll_value(self.quality)
        subsampling = roll_value(self.subsampling)
        logging.debug(f"Running JPEGCompression with quality: {quality} and subsampling {subsampling}")
        compressed = BytesIO()
        image.save(compressed, "JPEG", quality=quality, subsampling=subsampling)
        compressed.seek(0)
        compr = PIL.Image.open(compressed)
        image._img = compr
        return image

class TextSpoiler(Filter):
    """Dilate text and replace with grey"""

    def __init__(self, grey=127, dilate_k=3, **kwargs):
        super().__init__(**kwargs)
        self.grey = grey
        self.dilate_k = dilate_k

    def run(self, image):
        grey = roll_value(self.grey)
        dilate_k = roll_value(self.dilate_k)
        logging.debug(f"Running TextSpoilere with grey: {grey} and kernel {dilate_k}")
        cv_im = np.array(image._img)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (dilate_k, dilate_k))
        dilated = cv2.morphologyEx(cv_im, cv2.MORPH_ERODE, kernel)
        dilated[dilated < 120] = grey
        pil_im = PIL.Image.fromarray(dilated)
        image._img = pil_im
        return image

class InvertCellBackground(Filter):
    """Invert background of a cell of the table"""

    def run(self, image):
        logging.debug(f"Running InvertCellBackground spoiler")
        for text, pos in image.elements:  # key / value
            if text.data['data'].get('key', False) or image.node.get('is_key', False):  # invert only keys

                cell_inv = PIL.ImageOps.invert(text)
                image.paste(cell_inv, pos)
                image.node['invert'] = True


        return image

class CellBackground(Filter):
    """Create noise grid and apply to background of a cell of the table"""

    def __init__(self, grey=[220, 255], grid_ratio=2, **kwargs):
        super().__init__(**kwargs)
        self.grey = grey
        self.grid_ratio = grid_ratio

    def run(self, image):
        if image.type == 'TableCell' and image.node.get('is_key', None) is None: #or image.node.get('invert', None) is not None:
            elements = [(image, None)]
        elif image.type == 'Table' and image.node.get('invert', None) is None:
            elements = [cell for cell in image.elements if cell[0].node.get('is_key', None) is not None]
            image.node['modified_bg'] = True
        else:
            return image
        logging.debug(f"Running Cell Background with grey {self.grey}")
        grid_ratio = roll_value(self.grid_ratio)
        for el in elements:
            w, h = el[0].size
            noise = _white_noise(w, h, self.grey, grid_ratio)
            cell_with_noise = PIL.ImageChops.darker(el[0], noise)
            if image.type == 'TableCell':
                image._img = cell_with_noise
            else:
                image.paste(cell_with_noise, el[1])
        return image

class Whiteholes(Filter):
    """Create noise grid and apply to background of a cell of the table"""

    def __init__(self, scale=0.15, n=5, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.n = n

    def run(self, image):
        logging.debug(f"Running Erode Cell text")
        blank_im = PIL.Image.new('L', (image.size[0], image.size[1]), 0)
        scale = roll_value(self.scale)
        size = (int(image.size[0] * scale), int(image.size[1] * scale))
        mask = PIL.Image.new('L', size, 0)
        draw = PIL.ImageDraw.Draw(mask)
        draw.ellipse((0, 0, mask.size[0], mask.size[1]), fill=255)
        mask = mask.rotate(roll_value([-90, 90]), expand=True, fillcolor=0)
        for i in range(roll_value(self.n)):
            y_ = random.randint(0, image.size[1] - mask.size[1])
            x_ = random.randint(0, image.size[0] - mask.size[0])
            blank_im.paste(mask, (x_, y_))
        image.paste(PIL.ImageChops.lighter(image, blank_im))
        return image

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