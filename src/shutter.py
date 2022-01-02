import argparse
import logging
import math
import multiprocessing
import os
import sys
from multiprocessing import Process

import numpy as np
import numpy.random as random
import yaml

from exporter import from_options
from generators import Generator
from progress_bar import ProgressBar
from spoiler import Spoiler


def config_logger(out_dir):
    logger = logging.getLogger()
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s - [%(process)s] %(message)s"
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(logging.ERROR)
    file_handler = logging.FileHandler(f"{out_dir}/shutter.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)


def create_dirs(path: str):
    gt_dir = f"{path}/original"
    data_dir = f"{path}/spoiled"
    export_dir = f"{path}/export"

    os.makedirs(path, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)
    return export_dir


def open_yaml(filepath: str):
    with open(filepath, "r") as stream:
        try:
            opt = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            logging.exception(exc)
            exit(1)
    return opt


def gen_image(image_generator, visitors, options, i):

    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()
    logging.info(f"{i}/{options.size:04}")

    img_file = "%s/spoiled/%s.png" % (options.dir, i)
    truth_file = "%s/original/%s.png" % (options.dir, i)

    image = image_generator.generate()
    image.save(truth_file, dpi=(300, 300))

    logging.info("Generated Image. Applying spoilers")
    for visitor in visitors:
        image.accept(visitor, file_name=str(i))

    image.save(img_file, dpi=(options.dpi, options.dpi))
    # pr.disable()
    # pr.print_stats()


def gen_image_pool(generator, visitors, pool_list, opt, seed, update_pbar):
    logging.debug(f"Initializing thread with seed : {seed}")
    random.seed(seed)
    for item in pool_list:
        gen_image(generator, visitors, opt, item)
        update_pbar(item)


def split_jobs_for_workers(n_images, n_workers, out_name="shutter"):
    filler = [None for _ in range(n_workers - (n_images % n_workers))]
    jobs_indexes = (
        np.concatenate([np.arange(n_images), filler]).reshape(-1, n_workers).transpose()
    )
    num_digits = "0" + str(int(math.log10(n_images)) + 1)
    jobs = [
        [
            f"{out_name}_{index:{num_digits}}"
            for index in job_indexes
            if index is not None
        ]
        for job_indexes in jobs_indexes
    ]
    return jobs


def main(options):

    export_dir = create_dirs(options.dir)

    config_logger(options.dir)
    opt = open_yaml(options.config)

    seed = opt.get("seed", None)
    if not seed:
        seed = random.randint(0, 2 ** 32 - 1)
        opt["seed"] = seed
        opt["n_workers"] = options.workers

    with open(f"{options.dir}/config.yml", "w") as f:
        yaml.dump(opt, f)

    logging.info(f"Starting generation with seed {seed} ")

    n_workers = opt.get("n_workers", None)
    n_workers = n_workers or options.workers  # ensure determinism
    outputs = split_jobs_for_workers(options.size, n_workers, options.out_name)

    processes = []

    random.seed(seed)
    pbar = ProgressBar(options.size)

    def update_pbar(item):
        pbar.update(f"Generated {item}")

    for i in range(n_workers):
        seed = random.randint(0, 2 ** 32 - 1)
        generator = Generator(opt)
        spoiler = Spoiler()
        exporters = from_options(opt, export_dir)
        visitors = [spoiler] + exporters
        p = Process(
            target=gen_image_pool,
            args=(generator, visitors, outputs[i], options, seed, update_pbar),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")  # from py 3.8 spawn is default

    parser = argparse.ArgumentParser(description="Generated text and images.")
    parser.add_argument("--config", required=True, help="path to YAML config file")
    parser.add_argument(
        "--workers", required=False, type=int, help="Number of workers", default=1
    )
    parser.add_argument(
        "--out-name",
        required=False,
        type=str,
        help="Base name for output files",
        default="img",
    )
    parser.add_argument("--dir", required=True, help="path of the output directory")
    parser.add_argument(
        "--size", type=int, default=100, help="how many examples to generate"
    )
    parser.add_argument("--dpi", type=int, default=70, help="Noisy images dpi")

    options = parser.parse_args()
    main(options)
