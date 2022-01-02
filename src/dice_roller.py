import numpy.random as random
from scipy.stats import truncnorm

SAMPLES = 200


def truncated_normal(mean=0, sd=1, low=0, upp=10, samples=SAMPLES):
    a, b = (low - mean) / sd, (upp - mean) / sd
    return truncnorm(a, b, loc=mean, scale=sd).rvs(samples)


# fn_map = {"uniform": random.uniform, "normal": truncated_normal}


def sample_values():
    """Generate samples in range [0,1]"""
    while True:
        chances = random.uniform(0, 1, SAMPLES)
        for val in chances:
            yield val


generator = sample_values()


def all_strings(values):
    return all(isinstance(val, str) for val in values)


def all_couples(values):
    return all(isinstance(val, (list, tuple)) and len(val) == 2 for val in values)


def get_value_generator(node):
    """Returns a value generator from the provided distribution.

    If node is a scalar, return the scalar.
    if node is a list of strings, samples from the list.
    if node is a list of numbers uniformly samples in the interval.
    if nodes is a list of lists of len 2 defining values and probabilities of being picked, sample from that dist.
    if node is a dict defining a distribution, samples from the distribution (if supported).
    """
    match node:
        case float() | int() | str():  # scalar
            while True:
                yield node

        case [*values] if all_strings(values):  # list of strings
            while True:
                yield random.choice(node)

        case [int(low), int(high)]:  # list of two ints
            while True:
                yield random.randint(low, high + 1)

        case [float(low), float(high)]:  # list of two floats
            while True:
                yield random.uniform(low, high)

        case [0, float(high)]:  # list of [0, float] (QoL)
            while True:
                yield random.uniform(0, high)

        case [[_, _], *_] | [
            (_, _),
            *_,
        ] if all_couples(  # list of tuples (value, probability)
            node
        ):  # list of tuples or list of lists (len 2)
            values, probs = list(zip(*node))
            if sum(probs) != 1:
                raise ValueError(
                    f"Probabilities associated with values don't sum to 1. Node: {node}"
                )
            while True:
                yield random.choice(values, p=probs)

        case {
            "distribution": "normal",
            "mu": mu,
            "sigma": sigma,
            "min": min,
            "max": max,
        }:

            while True:
                vals = truncated_normal(mu, sigma, min, max)
                for val in vals:
                    yield val

        case {"distribution": "uniform", "min": min, "max": max}:
            random.uniform(min, max)

        case _:
            raise ValueError(f"Unrecognized Value distribution for list {node}")


def get_size_generator(node):
    """Generate width, height values."""
    size = node.get("size", dict())

    width = size.get("width", 1)
    height = size.get("height", 1)

    while True:
        ws = get_value_generator(width)
        hs = get_value_generator(height)

        for couple in zip(ws, hs):
            yield couple


def roll():
    """Pops a number in [0,1]"""
    return next(generator)


def roll_value(node):
    """Pops a number in a given distribution."""
    return next(get_value_generator(node))
