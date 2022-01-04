import logging
import textwrap

from PIL import ImageDraw
from PIL import ImageFont

from .component import Component
from dice_roller import roll
from dice_roller import roll_value
from generators import DEFAULT_FONT
from generators.generator import Generator
from utils import text_gen


class Text(Generator):
    alignments = ["left", "center", "right"]
    style_map = {"bold": " Bold", "italic": " Italic"}

    def __init__(self, opt):
        super(Text, self).__init__(opt)
        self.data_path = self.node.get("source_path", None)
        self.n_lines = self.node.get("n_lines", -1)
        self.font = self.node.get("font", dict())
        self.f_name = self.font.get("name", DEFAULT_FONT)
        self.uppercase = self.node.get("uppercase", 0)
        self.text = self.node.get("text", "")
        self.font_size = self.font.get("size", "fill")
        self.font_min = self.font.get("min_size", 8)
        self.fill = self.font.get("fill", 0)
        self.bold = self.font.get("bold", 0)
        self.align = self.node.get("align", "center")
        self.v_align = self.node.get("v_align", "center")
        # self.background = self.node.get('background', (255,))

    def get_font(self, text, size):
        width, height = size
        font_name = roll_value(self.f_name)

        font_data = {"name": font_name}
        for style, value in Text.style_map.items():
            if roll() < self.font.get(style, 0):
                font_data.update({style: True})
                font_name += value
        font_name = font_name.replace(" ", "_")
        f_size = roll_value(self.font_size)
        if isinstance(f_size, int):  # check if it fits
            try:
                font = ImageFont.truetype(font_name, f_size)
            except OSError:
                logging.exception(f"Cannot open font {font_name} with size {f_size}")
                exit(1)
            l_width, l_height = font.getsize(text)
            c_width = l_width / len(text)
            max_chars = width // c_width

            # lines = textwrap.wrap(text, width=int(max_chars))
            if (
                l_height > height or l_width > width
            ):  # doesn't fit, go for filling. N.B. single line!
                f_size = "fill"  #
                # logging.debug(f"Can't fit with font size {f_size}, filling...")

        if f_size == "fill":
            font_data.update({"filled": True})
            f_size = int(height * 0.8)

            while True:
                try:
                    font = ImageFont.truetype(font_name, f_size)
                except OSError:
                    logging.exception(
                        f"Cannot open font {font_name} with size {f_size}"
                    )
                    exit(1)
                c_width, l_height = font.getsize(text)
                c_width = c_width / len(text)
                max_chars = width // c_width
                if max_chars > 0:
                    # print(f"f_size {f_size}, max_ch {max_chars}")
                    lines = textwrap.wrap(text, width=int(max_chars))
                    if l_height < height and len(lines) == 1:
                        break

                f_size -= 1
        if f_size < self.font_min:
            return None, None
        try:
            font = ImageFont.truetype(font_name, f_size)
        except OSError:
            logging.exception(f"Cannot open font {font_name} with size {f_size}")
            exit(1)
        # logging.debug(f"Using font size {f_size}")
        font_data.update({"size": f_size})

        return font, font_data

    def generate(self, container_size=None, last_w=0, last_h=0):

        size = self.get_size(container_size, last_w, last_h)

        # spoilers = self.get_spoilers()
        img = Component(str(self), size, self.node, background_color=self.background)

        # n_lines = roll_value(self.n_lines)

        w_border = roll_value(self.node.get("w_border", 0))  # %
        w_border = int(w_border * size[0])
        h_border = roll_value(self.node.get("h_border", 0))
        h_border = int(h_border * size[1])
        width, height = cropped = (size[0] - w_border * 2), (size[1] - h_border * 2)

        draw = ImageDraw.Draw(img)
        y = h_border

        text = next(text_gen(self.data_path, self.text))
        if roll() <= self.uppercase:
            text = text.upper()

        font, font_data = self.get_font(text, cropped)
        if font:
            fill = roll_value(self.fill)
            font_data.update({"fill": fill})

            _, l_height = font.getsize("Ag")

            align = roll_value(self.align)
            v_align = roll_value(self.v_align)
            x = w_border
            l_width, _ = draw.textsize(text, font)
            if v_align == "bottom":
                y = height - l_height
            elif v_align == "center":
                y = (height - l_height) // 2

            if align == "right":
                x = width - l_width
            elif align == "center":
                x = (width - l_width) // 2

            draw.text((x, y), text, font=font, fill=fill, align=align)
            img.annotate(
                {"text": text, "font": font_data, "box": [x, y, l_width, l_height]}
            )

        return img


class TextGroup(Generator):
    font_sizes = [30, 38, 44, 52]
    style_map = {"bold": " Bold", "italic": " Italic"}

    def __init__(self, opt):
        super(TextGroup, self).__init__(opt)
        self.data_path = self.node.get("source_path", None)
        self.n_lines = self.node.get("n_lines", -1)
        self.font = self.node.get("font", dict())
        self.f_name = self.font.get("name", DEFAULT_FONT)
        self.font_size = self.font.get("size", 24)
        self.fill = self.font.get("fill", 0)
        self.bold = self.font.get("bold", 0)

    def generate(self, container_size=None, last_w=0, last_h=0):

        size = self.get_size(container_size, last_w, last_h)

        img = Component(str(self), size, self.node)

        n_lines = roll_value(self.n_lines)

        w_border = roll_value(self.node.get("w_border", 0))  # %
        w_border = int(w_border * size[0])
        h_border = roll_value(self.node.get("h_border", 0))
        h_border = int(h_border * size[1])
        cropped = (size[0] - w_border * 2), (size[1] - h_border * 2)

        font_name = roll_value(self.f_name)
        font_data = {"name": font_name}

        for style, value in TextGroup.style_map.items():
            if roll() < self.font.get(style, 0):
                font_data.update({style: True})
                font_name += value

        font_name = font_name.replace(" ", "_")
        f_size = roll_value(self.font_size)
        try:
            font = ImageFont.truetype(font_name, f_size)
        except OSError:
            logging.exception(f"Cannot open font {font_name} with size {f_size}")
            exit(1)
        width, l_height = font.getsize("Ag")
        while l_height + h_border > cropped[1]:
            f_size -= 1
            font = ImageFont.truetype(font_name, f_size)
            width, l_height = font.getsize("A g")

        font_data.update({"size": f_size})
        draw = ImageDraw.Draw(img)
        y = h_border
        width = int(cropped[0] // width * 2)
        texts = text_gen(self.data_path)
        fill = roll_value(self.fill)
        font_data.update({"fill": fill})
        x = w_border
        x0, y0 = x, y
        x1 = 0
        text_data = []
        while y + l_height <= cropped[1] and n_lines != 0:

            for line in textwrap.wrap(next(texts), width=width):
                l_width, l_height = font.getsize(line)
                x1 = max(x1, l_width)
                if y + l_height > cropped[1] or n_lines == 0:
                    break
                draw.text((x, y), line, font=font, fill=fill)
                n_lines -= 1
                y += l_height
                text_data.append(
                    {"text": line, "font": font_data, "box": [x, y, width, l_height]}
                )
        img.annotate({"box": [x0, y0, x1, y], "text_data": text_data})

        return img
