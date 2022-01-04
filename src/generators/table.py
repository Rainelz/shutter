from copy import deepcopy

import numpy as np
from PIL import Image

from . import DEFAULT_FONT
from dice_roller import roll
from dice_roller import roll_value
from generators.component import Component
from generators.generator import Generator
from generators.text import Text
from utils import roll_axis_split
from utils import roll_table_sizes


class TableCell(Generator):
    # key_value_map = {'top': []}
    def __init__(self, opt):
        super(TableCell, self).__init__(opt)
        self.headers_file = self.node.get("headers_file", None)
        self.w_border = self.node.get("w_border", 0)
        self.h_border = self.node.get("h_border", 0)
        self.frame = self.node.get("frame", 2)
        self.cell_borders = self.node.get(
            "cell_borders", ["top", "bottom", "left", "right"]
        )

        self.key = self.node.get("key", dict())
        self.key_p = self.key.get("p", 0)
        self.key_font = self.key.get("font", dict())
        self.key_f_name = self.key_font.get("name", DEFAULT_FONT)
        self.key_upper = self.key.get("uppercase", 0.5)
        self.key_font_size = self.key_font.get("size", "fill")
        self.key_fill = self.key_font.get("fill", 0)
        self.key_bold = self.key_font.get("bold", 0)
        self.key_align = self.key.get("align", "center")  # head center tail
        self.key_v_align = self.key.get("v_align", "top")  # top center bottom
        self.keys_file = self.key.get("file", None)

        self.is_title = self.node.get("is_title", False)

        self.value = self.node.get("value", dict())
        self.value_font = self.value.get("font", dict())
        self.value_f_name = self.value_font.get("name", DEFAULT_FONT)
        self.value_upper = self.value.get("uppercase", 0)

        self.value_font_size = self.value_font.get("size", "fill")
        self.value_fill = self.value_font.get("fill", 0)
        self.value_bold = self.value_font.get("bold", 0)
        self.value_align = self.value.get("align", "center")
        self.value_v_align = self.node.get("v_align", "top")
        self.values_file = self.value.get("file", None)

    def add_frame(self, img, b_size, b_color=0):
        border_w = Image.new("L", (img.width, b_size), b_color)
        border_h = Image.new("L", (b_size, img.height), b_color)
        t_border_size = b_border_size = l_border_size = r_border_size = 0

        if "top" in self.cell_borders:
            img.paste(border_w, (0, 0))

            t_border_size = b_size

        if "bottom" in self.cell_borders:
            img.paste(border_w, (0, img.height - b_size))
            b_border_size = b_size

        if "left" in self.cell_borders:
            img.paste(border_h, (0, 0))

            l_border_size = b_size

        if "right" in self.cell_borders:
            img.paste(border_h, (img.width - b_size, 0))
            r_border_size = b_size
        # if len(self.cell_borders) > 1:  # full borders, put borders on key
        #     if len(img.elements) > 0:
        #         subcell = img.elements[0][0]
        #         if subcell.data['data']['key'] and subcell.data['data']['axis'] == 1:
        #             sub_height = subcell.height
        #             img.paste(border_w, (0, sub_height))
        #         if subcell.data['data']['key'] and subcell.data['data']['axis'] == 0:
        #             sub_width = subcell.width
        #             img.paste(border_h, (sub_width, 0))
        return l_border_size, t_border_size, r_border_size, b_border_size

    def populate(self, cell, frame):

        size = cell.size
        # -- white border
        w_border = roll_value(self.w_border)  # in %
        l_border = int(w_border * size[0]) + frame[0]
        r_border = int(w_border * size[0]) + frame[2]
        h_border = roll_value(self.h_border)
        t_border = int(h_border * size[1]) + frame[1]
        b_border = int(h_border * size[1]) + frame[3]
        size = width, height = (size[0] - l_border - r_border), (
            size[1] - t_border - b_border
        )

        if roll() <= self.key_p:
            axis_split = roll_axis_split(width, height)  # 1 for horizontal split
            opposite_ax = abs(axis_split - 1)
            split_size = (
                int(size[axis_split] * roll_value([0.3, 0.7])),
                size[opposite_ax],
            )  # calc random split on side
            width_key, height_key = (
                split_size[axis_split],
                split_size[opposite_ax],
            )  # permute if axis == 1

            key_node = {
                "Text": {
                    "size": {"width": width_key, "height": height_key},
                    "source_path": self.keys_file,
                    "n_lines": 1,
                    "uppercase": self.key_upper,
                    "font": self.key_font,
                }
            }

            key_gen = Text(key_node)
            key = key_gen.generate(container_size=size)
            if key.empty():
                pass
                # return cell
            key.annotate({"value": False, "key": True, "axis": axis_split})
            cell.add(key, (l_border, t_border))
            width = size[0] - (
                width_key * opposite_ax
            )  # keep side intact or decrement based on axis
            height = size[1] - (height_key * axis_split)
            l_border = (l_border * axis_split) + (
                size[0] - width + l_border
            ) * opposite_ax  # calc offset where to place cell
            t_border = (t_border * opposite_ax) + (
                size[1] - height + t_border
            ) * axis_split
        # Creating text generator with calculated size, default alignment and my font info
        value_node = {
            "Text": {
                "size": {"width": width, "height": height},
                "source_path": self.values_file,
                "n_lines": 1,
                "background_color": self.background,
                "uppercase": self.value_upper,
                "font": self.value_font,
            }
        }

        value_gen = Text(value_node)
        value = value_gen.generate(container_size=size)
        value.annotate({"value": True, "key": False})
        cell.add(value, (l_border, t_border))
        cell.render()
        return cell

    def generate(self, container_size=None, last_w=0, last_h=0):
        size = self.get_size(container_size, last_w, last_h)

        cell = Component(str(self), size, self.node, background_color=self.background)
        frame_size = roll_value(self.frame)
        frame = self.add_frame(cell, frame_size)
        cell = self.populate(cell, frame)
        cell.annotate({"frame": frame, "title": self.is_title})
        return cell


class Table(Generator):
    def __init__(self, opt):
        super(Table, self).__init__(opt)
        self.cols = self.node.get("cols", 1)
        self.rows = self.node.get("rows", 1)
        # self.cell_size = self.node.get('cell_size', dict())
        # self.cell_w = self.cell_size.get('width', 1)
        # self.cell_h = self.cell_size.get('height', 1)
        self.fix_rows = self.node.get("fix_rows", 0.5)
        self.fix_cols = self.node.get("fix_cols", 0.5)

        # self.plain_table = self.node.get('plain_table', True)
        self.cell_w_border = self.node.get("cell_w_border", 0)
        self.cell_h_border = self.node.get("cell_h_border", 0)
        self.row_frame = self.node.get("row_frame", 1)
        self.col_frame = self.node.get("col_frame", 1)

        self.cell_spoilers = self.node.get("cells_spoilers", dict())
        # if 'Cell' in self.node.get('spoilers', dict()):
        #     self.node['spoilers'] = self.node.get('spoilers').get('Cell')
        #     self.node['spoilers'].pop('Cell', None)

        self.font = self.node.get("font", dict())
        self.f_name = self.font.get("name", DEFAULT_FONT)
        self.font_size = self.font.get("size", "fill")
        self.fill = self.font.get("fill", 0)
        self.bold = self.font.get("bold", 0)
        self.align = self.node.get("align", "center")
        self.v_align = self.node.get("v_align", "top")

        self.values_file = self.node.get("values_file", None)
        self.title_file = self.node.get("headers_file", None)
        self.title = self.node.get("title", 0.5)
        self.keys_file = self.node.get("keys_file", None)

        self.fix_keys_col = self.node.get("fix_keys_col", 0.5)
        self.fix_keys_row = self.node.get("fix_keys_row", 0.5)

    def generate(self, container_size=None, last_w=0, last_h=0):
        size = self.get_size(container_size, last_w, last_h)

        table = Component(str(self), size, self.node)

        schema = self.make_schema(table)
        schema = self.put_borders(schema)
        # schema = self.fix_fonts(schema) #TODO
        for row in schema:
            for couple in row:
                node, position = couple
                cell_gen = TableCell(node)
                cell_im = cell_gen.generate()
                table.add(cell_im, position)
        table.render()

        return table

    # def fix_fonts(self, schema):
    #     np_schema = np.array(
    #         [row + [row[0]] * (len(schema[-1]) - len(row)) for row in schema]
    #     )
    #     first_row = np_schema[
    #         0, :, 0
    #     ]  # create list from first row, every column, first couple element
    #
    #     with open(self.keys_file, "r") as file:
    #         lines = file.readlines()
    #         lens = [len(line) for line in lines]
    #         median = np.argmax(lens)
    #         # gen = Text({'Text':{'font':roll_value(self.font)}})
    #         # _, font_data = gen.get_font(lens[median], )
    #     return schema

    def put_borders(self, schema):
        """Take a table schema and fill borders based on params N.B.

        this method works inplace thus modifies the original schema
        """

        def add_bs(bs, nodes):
            for node in nodes:
                if node is None:
                    continue
                borders = node["TableCell"]["cell_borders"]
                for b in bs:
                    if b not in borders:
                        borders.append(b)

        # replicate first cell if row has less cells (title)
        np_schema = np.array(
            [row + [row[0]] * (len(schema[-1]) - len(row)) for row in schema],
            dtype=object,
        )
        first_row = np_schema[
            0, :, 0
        ]  # create list from first row, every column, first couple element
        first_col = np_schema[:, 0, 0]
        last_row = np_schema[-1, :, 0]
        last_col = np_schema[:, -1, 0]

        # externals
        first_row_borders = ["top"]
        if first_row[0]["TableCell"].get("is_title", False) or first_row[0][
            "TableCell"
        ].get(
            "is_key", False
        ):  # if is title and then no row frame the title must be enclosed
            first_row_borders.append("bottom")

        add_bs(first_row_borders, first_row)
        add_bs(["left"], first_col)
        add_bs(["bottom"], last_row)
        add_bs(["right"], last_col)

        internal_borders = []
        if roll() <= self.row_frame:
            internal_borders.append("bottom")

        if roll() <= self.col_frame:
            internal_borders.append("right")

        add_bs(
            internal_borders, np_schema[:, :, 0].flatten()
        )  # add every cell right border

        return schema

    def make_schema(self, table):
        """Create a matrix row x cols with (Cell_node, position)"""
        n_cols = roll_value(self.cols)
        n_rows = roll_value(self.rows)
        if roll() <= self.fix_cols:  # fixed dims
            cell_w = table.width // n_cols
            widths = [cell_w] * n_cols
        else:
            widths = roll_table_sizes(table, n_cols, axis=0)
        if roll() <= self.fix_rows:
            cell_h = table.height // n_rows
            heights = [cell_h] * n_rows
        else:
            heights = roll_table_sizes(table, n_rows, axis=1)

        cell_node = {
            "size": {"width": 0, "height": 0},
            "value": {"file": self.values_file, "font": self.font},
            "key": {"p": 0.5, "file": self.keys_file, "font": self.font},
            "cell_borders": [],  # to be filled later on
            "spoilers": self.cell_spoilers,
        }
        pos_mapping = list()
        row_idx = 0
        if roll() <= self.title:  # create title, one row single cell
            title_node = deepcopy(cell_node)
            h = heights[0]
            del title_node["key"]
            title_node["value"].update(file=self.title_file, font={"size": "fill"})
            title_node.update(size={"width": table.width, "height": h}, is_title=True)
            pos_mapping.append(
                [({"TableCell": deepcopy(title_node)}, (0, 0))]
            )  # first row
            row_idx = 1

        if roll() <= self.fix_keys_col:  # create keys row

            h = heights[row_idx]
            y = sum(heights[:row_idx])

            key_node = deepcopy(cell_node)
            del key_node["key"]

            row = list()
            for j in range(len(widths)):
                x = sum(widths[:j])
                w = widths[j]
                position = x, y
                key_node.update(size={"width": w, "height": h}, is_key=True)
                key_node["value"].update(file=self.keys_file, uppercase=0.8)
                row.append(({"TableCell": deepcopy(key_node)}, position))
            row_idx += 1
            pos_mapping.append(row)

        # if roll() <= self.fix_keys_row:
        #     w = widths[col_idx]
        #     x = sum(widths[:col_idx]) # 0
        #     key_node = deepcopy(cell_node)
        #     del key_node['key']
        #     col = list()
        #     for i in range(len(heights)):

        while row_idx < len(heights):
            row = []

            for j in range(len(widths)):
                x = sum(widths[:j])
                y = sum(heights[:row_idx])
                w = widths[j]
                h = heights[row_idx]
                position = x, y
                cell_node.update(size={"width": w, "height": h}, is_val=True)
                row.append(({"TableCell": deepcopy(cell_node)}, position))

            pos_mapping.append(row)
            row_idx += 1

        return pos_mapping
