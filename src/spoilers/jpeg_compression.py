import logging
from io import BytesIO

import PIL

from dice_roller import roll_value
from spoilers.abstract_filter import AbstractFilter


class JPEGCompression(AbstractFilter):
    """Apply a gradient foregound noise."""

    def __init__(self, quality=50, subsampling=-1, **kwargs):
        super().__init__(**kwargs)
        self.quality = quality
        self.subsampling = subsampling

    def run(self, image):
        quality = roll_value(self.quality)
        subsampling = roll_value(self.subsampling)
        logging.debug(
            f"Running JPEGCompression with quality: {quality} and subsampling {subsampling}"
        )
        compressed = BytesIO()
        image.save(compressed, "JPEG", quality=quality, subsampling=subsampling)
        compressed.seek(0)
        compr = PIL.Image.open(compressed)
        image._img = compr
        return image
