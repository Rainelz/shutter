import logging

import PIL

from spoilers.abstract_filter import AbstractFilter


class InvertCellBackground(AbstractFilter):
    """Invert background of a cell of the table."""

    def run(self, image):
        logging.debug("Running InvertCellBackground spoiler")
        for text, pos in image.elements:  # key / value
            if text.data["data"].get("key", False) or image.node.get(
                "is_key", False
            ):  # invert only keys

                cell_inv = PIL.ImageOps.invert(text)
                image.paste(cell_inv, pos)
                image.node["invert"] = True

        return image
