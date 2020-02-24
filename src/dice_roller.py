import numpy.random as random
from scipy.stats import truncnorm

SAMPLES = 1000

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
    if isinstance(node, (int, float)):
        while True:
            yield node
    if isinstance(node, list): ## uniform
        if all(isinstance(val, int) for val in node):
            while True:
                yield  random.randint(node[0], node[1]+1)
        else:
            while True:
                yield random.uniform(*node)
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
    return next(get_value_generator(node))