from PIL import Image, ImageDraw, ImageFont
from dice_roller import roll, fn_map, SAMPLES, roll_value
from itertools import product

class TableGen():

    def __init__(self, width, height, node):
        self.height = height
        self.width = width
        self.node = node

        # --- general parameters which should be common for all configurations
        self.min_font = int(node.get('font_min_size', 14))  # 13
        self.base_font = max(self.min_font, self.height // 80)
        self.pad_font_min_size = int(roll_value(node.get('font_delta_size', 6)))
        # ---

        keys_file = node.get('keys_path', None)
        assert keys_file
        with open(keys_file, 'r') as f:
            self.textheads = f.read()
            self.textheads = self.textheads.split('\n')

        self.im = Image.new("L", (self.width, self.height), 255)

    @staticmethod
    def make_table_schema(n_rows, n_cols, width, height):
        pos_mapping = dict()
        cell_h = height // n_rows
        cell_w = width // n_cols
        couples = product([x for x in range(n_cols)], [y for y in range(n_rows)])
        for couple in couples:
            coord = (couple[0] * cell_w, couple[1] * cell_h)
            pos_mapping[coord] = None
        return pos_mapping, cell_h, cell_w

    def add_cells_to_table(self, schema, cell_h, cell_w):
        from generators import Component

        for cell in schema:
            #Color
            if roll() <= self.node.get('cell_p_iscolored'):
                color = (int(roll_value(self.node.get('cell_color'))),)
            else:
                color = None

            #text
            text = 'lautaro ' * 10
            if roll() <= float(self.node.get('text_p_isbold')):
                fontname = self.node.get('font') + ' Bold'
            else:
                fontname =  self.node.get('font')

            #border
            if self.node.get('zero_width') == True:
                border_width = 0
            else:
                border_width = int(roll_value(self.node.get('border_width')))

            img = Component(str(self), (self.width, self.height), self.node, background_color=(255,255,255))
            cell_istance = TableCell(cell_w, cell_h, fontname, self.base_font, self.pad_font_min_size,
                             color = color, text = text, border_width = border_width)

            img = cell_istance.generate(img)
            schema[cell] = img
        return schema

    @staticmethod
    def make_table(img, cells):
        for cell_pos in cells:
            img.add(cells[cell_pos], cell_pos)
        return img

    def compose(self, img):
        n_rows = int(roll_value(self.node.get('rows')))
        n_cols = int(roll_value(self.node.get('cols')))
        schema, cell_h, cell_w = self.make_table_schema(n_rows, n_cols, self.width, self.height)
        cells = self.add_cells_to_table(schema, cell_h, cell_w)
        table = self.make_table(img, cells)
        return table

class TableCell():

    def __init__(self, width, height, fontname, base_font, pad_font_min_size, color = None,
                 background_color=(255,255,255,255), text = None, border_width = 0):

        self.width = width
        self.height = height
        self.color = color or background_color
        self.text = text
        self.border_width = border_width
        self.fontname = fontname
        self.base_font = base_font
        self.pad_font_min_size = pad_font_min_size

    @staticmethod
    def add_border(img, b_size, b_color = 0):
        border_w = Image.new("L", (img.width, b_size), b_color)
        border_h = Image.new("L", (b_size, img.height), b_color)
        img.paste(border_w, (0, 0))
        img.paste(border_w, (0, img.height - b_size))
        img.paste(border_h, (0, 0))
        img.paste(border_h, (img.width - b_size, 0))
        return img

    def add_plain_text(self, img):
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(self.fontname, self.base_font + self.pad_font_min_size)

        text_out = []
        t = ''
        for word in self.text.split(' '):
            if font.getsize(t)[0] <= self.width * 0.7:
                t = t + word + ' '
            else:
                text_out.append(t.strip())
                t = ''

        text_out.append(t)
        MAX_W, MAX_H = img.size
        h_text = 0
        for phrase in text_out:
            w = ((MAX_W - self.border_width - font.getsize(phrase)[0])/MAX_W) / 2
            h = ((MAX_H - self.border_width - font.getsize(phrase)[1] * len(text_out))/MAX_H) / 2
            draw.text((MAX_W * w ,(h * MAX_H) + h_text), phrase, 0, font=font, align="center")
            h_text = h_text + font.getsize(phrase)[1]
        return img

    def generate(self, img):
        if len(self.color) > 1:
            color_space = 'RGB'
        else:
            color_space = 'L'

        size = (self.width, self.height)
        img._img = Image.new(color_space, size, self.color)
        if self.border_width > 0:
            img._img = self.add_border(img._img, self.border_width)
        if self.text is not None:
            img._img = self.add_plain_text(img._img)
        return img