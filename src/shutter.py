import numpy.random as random
import argparse
import os, logging, sys
from multiprocessing import Process
import yaml

from generators import Generator
from spoiler import Spoiler
from exporter import from_options



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
    parser.add_argument('--out-name', required=False, type = str, help='Base name for output files', default='img')
    parser.add_argument('--dir', required=True, help='path of the output directory')
    parser.add_argument('--size', type=int, default=100, help='how many examples to generate')
    parser.add_argument('--dpi', type=int, default=70, help='Noisy images dpi')


    return parser.parse_args()


def gen_image(image_generator, visitors, options, i):
    logging.info(f"{i}/{options.size:04}")

    img_file = '%s/spoiled/%s.png' % (options.dir, i)
    truth_file = '%s/original/%s.png' % (options.dir, i)

    image = image_generator.generate()
    image.save(truth_file, dpi=(300, 300))

    logging.info("Generated Image. Applying spoilers")
    for visitor in visitors:
        image.accept(visitor, file_name=str(i))

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
    gt_dir = f"{options.dir}/original"
    data_dir = f"{options.dir}/spoiled"
    export_dir = f"{options.dir}/export"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    seed = opt.get('seed', None)
    if not seed:
        seed = random.randint(0, 2**32-1)
        opt['seed'] = seed
        opt['n_workers'] = options.workers
    logging.info(f"Starting generation with seed {seed} ")
    n_workers = opt.get('n_workers', None)
    n_workers = n_workers or options.workers  # ensure determinism

    filler = [None for _ in range(n_workers - (options.size % n_workers))]
    vals = np.concatenate([np.arange(options.size), filler]).reshape(-1, n_workers).transpose()
    vals = [[f'{options.out_name}_{val:04}' for val in values if val is not None] for values in vals]
    processes = []

    random.seed(seed)

    for i in range(n_workers):
        seed = random.randint(0,2**32-1)
        generator = Generator(opt)
        spoiler = Spoiler()
        exporters = from_options(opt, export_dir)
        visitors = [spoiler] + exporters
        p = Process(target=gen_image_pool, args=(generator, visitors, vals[i], options, seed))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    with open(f"{options.dir}/config.yml", 'w') as f:
        yaml.dump(opt, f)


if __name__ == '__main__':
    main()