import math
import os

import PIL
import numpy.random as random


from spoiler import spoil



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