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
def roll_axis_split(width, height):  # for key value positioning
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

def roll_table_sizes(table, splits, axis):
    # N.B. axis is the axis to use for varying values e.g. 0 -> varying widths
    # trying to keep the values coupled to enable swap based on axis
    dims = table.size
    measure = dims[axis]  # width
    mu = measure / splits  # width / n_cols

    #fixed_side = dims[opposite_axis] // grid[opposite_axis]  # height // n_rows

    rem = measure  # width
    min = mu * 0.5
    max_ = mu * 1.5
    sigma = mu * 0.3
    sizes = []

    #  create 1 row/col sizes, calculating varying width/height, permute if axis == 1
    for i in range(splits - 1):
        var_side = int(roll_value({'distribution': 'normal', 'mu': mu, 'sigma': sigma, 'min': min, 'max': max_}))
        rem -= var_side

        mu = rem / (splits - (i + 1))
        max_ = int(mu * 1.5)
        min = int(mu * 0.5)

        # size = var_side, fixed_side
        # # if axis == 1:  # transpose if rolling heights
        # size = size[axis], size[opposite_axis]
        sizes.append(var_side)

    # === last cell
    sizes.append(rem)  # fill with last cell
    return sizes

