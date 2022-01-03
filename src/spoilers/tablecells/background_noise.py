import logging

import PIL

from dice_roller import roll_value
from spoilers.abstract_filter import AbstractFilter
from spoilers.background import _white_noise


class CellBackground(AbstractFilter):
    """Create noise grid and apply to background of a cell of the table."""

    def __init__(self, grey=(220, 255), grid_ratio=2, **kwargs):
        super().__init__(**kwargs)
        self.grey = grey
        self.grid_ratio = grid_ratio

    def run(self, image):
        if (
            image.type == "TableCell" and image.node.get("is_key", None) is None
        ):  # or image.node.get('invert', None) is not None:
            elements = [(image, None)]
        elif image.type == "Table" and image.node.get("invert", None) is None:
            elements = [
                cell
                for cell in image.elements
                if cell[0].node.get("is_key", None) is not None
            ]
            image.node["modified_bg"] = True
        else:
            return image
        logging.debug(f"Running Cell Background with grey {self.grey}")
        grid_ratio = roll_value(self.grid_ratio)
        for el in elements:
            w, h = el[0].size
            noise = _white_noise(w, h, self.grey, grid_ratio)
            cell_with_noise = PIL.ImageChops.darker(el[0], noise)
            if image.type == "TableCell":
                image._img = cell_with_noise
            else:
                image.paste(cell_with_noise, el[1])
        return image
