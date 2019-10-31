import numpy.random as random
from scipy.stats import truncnorm


def truncated_normal(mean=0, sd=1, low=0, upp=10, samples=1):
    a, b = (low - mean) / sd, (upp - mean) / sd
    return truncnorm(a, b, loc=mean, scale=sd).rvs(samples)


fn_map = {'uniform': random.uniform,
          'normal': truncated_normal}

SAMPLES = 1000


def sample_values():
    """Generate samples in range [0,1]"""
    while True:
        chances = random.uniform(0, 1, SAMPLES)
        for val in chances:
            yield val


generator = sample_values()


def roll():
    "Pops a number in [0,1]"
    return next(generator)


def roll_value(node):
    if isinstance(node, (int, float)):
        return node
    if isinstance(node, list): ## uniform
        if all(isinstance(val, int) for val in node):
            return random.randint(node[0], node[1]+1)
        else:
            return random.uniform(*node)
    elif isinstance(node, dict):

        distribution = node.get('distribution', 'normal')
        pdf = fn_map[distribution]
        args = node['mu'], node['sigma'], node['min'], node['max']
        return pdf(*args)[0]