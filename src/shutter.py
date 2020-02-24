import numpy.random as random
import argparse
import os, logging, sys
from multiprocessing import Process
import yaml

from images import Generator
from spoiler import Spoiler



def config_logger():
    logger = logging.getLogger()
    logging.getLogger('PIL').setLevel(logging.ERROR)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s - [%(process)s] %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(logging.DEBUG)
    logger.addHandler(stdout_handler)

def parse_options():
    parser = argparse.ArgumentParser(description='Generated text and images.')
    parser.add_argument('--config', required=True, help='path to YAML config file')
    parser.add_argument('--workers', required=False, type = int, help='Number of workers', default=1)

    parser.add_argument('--fonts', required=False, help='path of the list of fonts')
    parser.add_argument('--dir', required=True, help='path of the output directory')
    parser.add_argument('--size', type=int, default=100, help='how many examples to generate')
    parser.add_argument('--no-filters', type=bool, default=False, help='Don\'t apply filters')
    parser.add_argument('--no-skew', default=False, help='Don\'t skew images',  action="store_true")
    parser.add_argument('--dpi', type=int, default=70, help='Noisy images dpi')


    return parser.parse_args()


def gen_image(image_generator, visitors, options, i):
    logging.info(f"{i + 1}/{options.size}")

    img_file = '%s/spoiled/%04d.png' % (options.dir, i)
    truth_file = '%s/original/%04d_GT.png' % (options.dir, i)

    image = image_generator.generate()
    image.save(truth_file, dpi=(300, 300))

    logging.info("Generated Image. Applying spoilers")
    for visitor in visitors:
        image.accept(visitor)

    image.save(img_file, dpi=(options.dpi, options.dpi))

def gen_image_pool(generator, visitors, pool_list, opt, seed):
    random.seed(seed)
    for item in pool_list:
        gen_image(generator, visitors, opt, item)

def main():
    import numpy as np
    options = parse_options()
    config_logger()
    with open(options.config, 'r') as stream:
        try:
            opt = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.exception(exc)
            exit(1)

    os.makedirs(f"{options.dir}/original", exist_ok=True)
    os.makedirs(f"{options.dir}/spoiled", exist_ok=True)

    seed = opt.get('seed', None)
    if not seed:
        seed = random.randint(0, 2**32-1)
        opt['seed'] = seed
    logging.info(f"Starting generation with seed {seed} ")
    n_workers= options.workers
    filler = [None for _ in range(n_workers - (options.size % n_workers))]
    vals = np.concatenate([np.arange(options.size), filler]).reshape(-1, n_workers).transpose()
    vals = [[val for val in values if val is not None] for values in vals]
    processes = []

    random.seed(seed)

    for i in range(n_workers):
        seed = random.randint(0,2**32-1)
        visitors = [Spoiler(), ]
        p = Process(target=gen_image_pool, args=(Generator(opt), visitors, vals[i], options, seed))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    with open(f"{options.dir}/config.yml", 'w') as f:
        yaml.dump(opt, f)


if __name__ == '__main__':
    main()