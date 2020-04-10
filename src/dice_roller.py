import numpy.random as random
from scipy.stats import truncnorm

SAMPLES = 200

def truncated_normal(mean=0, sd=1, low=0, upp=10, samples=SAMPLES):
    a, b = (low - mean) / sd, (upp - mean) / sd
    return truncnorm(a, b, loc=mean, scale=sd).rvs(samples)


fn_map = {'uniform': random.uniform,
          'normal': truncated_normal}



def sample_values():
    """Generate samples in range [0,1]"""
    while True:
        chances = random.uniform(0, 1, SAMPLES)
        for val in chances:
            yield val


generator = sample_values()

def get_value_generator(node):
    """Returns a value generator from the provided distribution.
    If node is a scalar, return the scalar
    if node is a list of strings, samples from the list
    if node is a list of numbers uniformly samples in the interval
    if nodes is a list of lists of len 2 defining values and probabilities of being picked, sample from that dist
    if node is a dict defining a distribution, samples from the distribution (if supported)
    """
    if isinstance(node, (int, float, str)):
        while True:
            yield node
    if isinstance(node, list):
        if all(isinstance(val, (list, tuple)) and len(val) == 2 for val in node):  # list of lists [value, p]
            values, probs = list(zip(*node))
            while True:
                yield random.choice(values, p=probs)

        elif all(isinstance(val, str) for val in node):  # uniform
            while True:
                yield random.choice(node)
        elif all(isinstance(val, int) for val in node) and len(node) == 2:
            while True:
                yield random.randint(node[0], node[1]+1)
        elif len(node) == 2:
            while True:
                yield random.uniform(*node)
        else:
            raise ValueError(f"Unrecognized Value distribution for list {node}")
    elif isinstance(node, dict):

        distribution = node.get('distribution', 'normal')
        pdf = fn_map[distribution]
        args = node['mu'], node['sigma'], node['min'], node['max']
        while True:
            vals = pdf(*args)
            for val in vals:
                yield val

def roll():
    "Pops a number in [0,1]"
    return next(generator)


def roll_value(node):
    "Pops a number in a given distribution"
    return next(get_value_generator(node))

