import os
from dice_roller import roll_value, roll


def text_gen(file=None, value=None):
    if file:
        file_size = os.path.getsize(file)
        offset = roll_value([1, file_size])
        with open(file, 'r') as file:
            file.seek(offset)
            file.readline()

            while True:
                line = file.readline()
                if not line:
                    file.seek(0, 0)
                    continue
                yield line
    else:
        while True:
            text = value or "PLACEHOLDER TEXT"
            for line in text.split('\n'):
                yield line


# =========== Table utils
HORIZ_BIAS = 0.3
def roll_axis_split(width, height):
    """
    Calculating w-h / w+h which belongs to ]-1, 1[ , then normalize in ]0,1[ as p --> w / (w+h)/2

    """
    # augment height to make horizontal split more likely to be rolled (like most table cells)
    height = height + (height * HORIZ_BIAS)
    side_ratio = (width + height) / 2
    p_horizontal = (height / side_ratio)
    if roll() <= p_horizontal:
        return 1
    return 0
