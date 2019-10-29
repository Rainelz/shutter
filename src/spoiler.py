from __future__ import annotations
import numpy.random as random
import PIL, PIL.ImageFont, PIL.Image, PIL.ImageDraw, PIL.ImageChops, PIL.ImageOps, PIL.ImageFilter
import math
from abc import ABC, abstractmethod
from typing import List
from images import Component, Visitor, Image


class Filter(Visitor):
    """Implements basic filter behaviour"""

    def should_visit_leaves(self):
        return False

    def run(self, image: Component):
        pass

    def visit(self, component: Component):
        for el, _ in component.elements:
            self.visit(el)
        component.render()

        if self.__class__.__name__ in component.spoilers:
            component._img = component.convert('L')
            component.update(self.run(component))
        # if self.__class__.__name__ in component.spoilers:
        # component._img = component.convert('L')
        #
        # if self.should_visit_leaves():
        #     for el, _ in component.elements:
        #         self.visit(el)
        #     component.render()
        #
        # component.update(self.run(component))


class Crop(Filter):
    def __init__(self, border=0):
        self.border = border
        pass

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
        return image.crop(tuple(box))

    @staticmethod
    def from_cfg(cfg):
        return Crop(random.randint(0, 50))
        #return Crop()


class Pad(Filter):
    def __init__(self, n):
        self.n = n

    def run(self, image):
        w, h = image.size
        bg = PIL.Image.new(image.mode, (w + 2 * self.n, h + 2 * self.n), 0)
        bg.paste(image, (self.n, self.n))
        return bg

    @staticmethod
    def random():
        return Pad(random.randint(2, 5))

    def should_visit_leaves(self):
        return True


class Rotate(Filter):
    def __init__(self, angle):
        self.angle = angle
        alpha = math.pi * angle / 180
        self.c = abs(math.cos(alpha))
        self.s = abs(math.sin(alpha))

    def center_box(self, outer, inner):
        W, H = outer
        w, h = inner
        w1 = int(self.c * w + self.s * h) + 1
        h1 = int(self.s * w + self.c * h) + 1
        return ((W - w1) // 2, (H - h1) // 2, (W - w1) // 2 + w1, (H - h1) // 2 + h1)

    def run(self, image):
        if isinstance(image, Image):
            #rotated = Pad(100).run(image).convert('RGBA').rotate(self.angle, expand = 1)
            rotated = image.convert('RGBA').rotate(self.angle, expand=1, fillcolor='white')
            box = self.center_box(rotated.size, image.size)
            return rotated.crop(box)
        else:
            return image

    def should_visit_leaves(self):
        return True

    @staticmethod
    def random():
        # if random.randint(0, 10) < 7:
        #     return Filter()
        # else:
        return Rotate(random.randint(-10, 10   ))


def _white_noise(width, height, m=0, M=255):
    pil_map = PIL.Image.new("L", (width, height), 255)
    random_grid = map(lambda x: random.randint(m, M), [0] * width * height)
    pil_map.putdata(list(random_grid))
    return pil_map


class Background(Filter):
    def __init__(self, grey):
        self.grey = grey

    def run(self, image):
        w, h = image.size
        noise = _white_noise(w, h, self.grey, 255)
        return PIL.ImageChops.darker(image, noise)

    def should_visit_leaves(self):
        return True

    @staticmethod
    def random():
        # if random.randint(0, 10) < 7:
        #     return Filter()
        # else:
        return Background(random.randint(220, 245))


class Foreground(Filter):
    def __init__(self, grey):
        self.grey = grey

    def run(self, image):
        w, h = image.size
        noise = _white_noise(w, h, 0, self.grey)
        return PIL.ImageChops.lighter(image, noise)

    def should_visit_leaves(self):
        return True

    @staticmethod
    def random():
        # if random.randint(0, 10) < 6:
        #     return Filter()
        # else:
        return Foreground(random.randint(150, 255))


class Blur(Filter):
    def __init__(self, r):
        self.r = r

    def run(self, image):
        return image.filter(PIL.ImageFilter.GaussianBlur(self.r))

    @staticmethod
    def random():
        # if random.randint(0, 10) < 8:
        #     return Filter()
        # else:
        return Blur(1)

    def should_visit_leaves(self):
        return True

class Stroke(Filter):
    def __init__(self, num_signs, num_strokes, step):
        self.num_signs = num_signs
        self.num_strokes = num_strokes
        self.step = step

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

    def should_visit_leaves(self):
        return True

    @staticmethod
    def random():
        # if random.randint(0, 10) < 7:
        #     return Filter()
        # else:
        return Stroke(random.randint(1, 6), random.randint(3, 14), random.randint(10, 50))


class Overlay(Filter):
    def __init__(self, overlay, size):
        self.overlay = overlay.convert('L').resize(size)
        self.w, self.h = size

    def pad_overlay_at(self, size):
        w, h = size
        bg = PIL.Image.new("L", size, 255)
        w1 = random.randint(0, w - self.w)
        h1 = random.randint(0, h - self.h)
        bg.paste(self.overlay, (w1, h1))
        return bg

    def run(self, image):
        overlay = self.pad_overlay_at(image.size)
        return PIL.ImageChops.darker(image, overlay)

    @staticmethod
    def open(path, size):
        return Overlay(PIL.Image.open(path), size)

    @staticmethod
    def random(dir, size):
        import os
        # if random.randint(0, 10) < 7:
        #     return Filter()
        # else:
        file = random.choice(os.listdir(dir))
        full_path = os.path.join(dir, file)
        return Overlay.open(full_path, size)


class VerticalLine(Filter):
    def run(self, image):
        w, h = image.size
        try:
            a = random.randint(0, h // 8)
            b = h - random.randint(0, h // 8)
            x = random.randint(0, w)
            draw = PIL.ImageDraw.Draw(image)
            draw.line((x, a, x, b), fill=30, width=4)
        except Exception as e:
            print(e)
            return image
        return image

    @staticmethod
    def random():
        import os
        # if random.randint(0, 10) < 9:
        #     return Filter()
        # else:
        return VerticalLine()


class Gradient(Filter):
    def __init__(self, gradient_magnitude=1., direction=0, color=0):
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

    @staticmethod
    def random():
        # if random.randint(0, 10) < 7:
        #     return Filter()
        # else:
        grad = random.randint(1, 10)
        dir = random.randint(0, 1)
        color = random.randint(100, 200)
        return Gradient(grad/10, dir, color)

def filters_from_cfg(cfg):
    filters = [

        Rotate.random(),
        Pad.random(),
        Background.random(),
        Foreground.random(),
        Blur.random(),
        Stroke.random(),
        VerticalLine.random(),
        Overlay.random('resources/stamps', (100, 100)),
        Gradient.random()
    ]
    return filters

def spoil(im, options):
    im = Crop.random().run(im)
    filters = [
        VerticalLine.random(),
        Overlay.random('resources/stamps', (100, 100)),
        Rotate.random() if not options.no_skew else Filter(),
        Pad.random(),
        Background.random(),
        Foreground.random(),
        Stroke.random(),
        Blur.random(),
        #Gradient.random()
    ]

    noisy = im
    for f in filters:
        noisy = f.run(noisy)
        
    return noisy