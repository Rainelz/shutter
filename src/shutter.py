import numpy.random as random
import argparse
import os, logging, sys

import yaml

from images import Generator
from spoiler import Spoiler



def config_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s - [%(threadName)s] %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(logging.DEBUG)
    logger.addHandler(stdout_handler)

def parse_options():
    parser = argparse.ArgumentParser(description='Generated text and images.')
    parser.add_argument('--config', required=True, help='path to YAML config file')
    parser.add_argument('--fonts', required=False, help='path of the list of fonts')
    #parser.add_argument('--text', required=True, help='path of the text file')
    parser.add_argument('--dir', required=True, help='path of the output directory')
    parser.add_argument('--size', type=int, default=100, help='how many examples to generate')
    parser.add_argument('--no-filters', type=bool, default=False, help='Don\'t apply filters')
    #parser.add_argument('--gen-truth', default=False, help='Generate ground truth images', action="store_true")
    parser.add_argument('--no-skew', default=False, help='Don\'t skew images',  action="store_true")
    parser.add_argument('--dpi', type=int, default=70, help='Noisy images dpi')


    return parser.parse_args()


def main():
    options = parse_options()
    config_logger()
    with open(options.config, 'r') as stream:
        try:
            opt = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.exception(exc)
            exit(1)

    #fonts = load_fonts(options.fonts, size=15)
    fonts=[]
    #with open(options.text) as f:
        #lines = f.readlines()

    #if options.gen_truth:
    os.makedirs(f"{options.dir}/GT", exist_ok=True)
    seed = opt.get('seed', None)
    if not seed:
        seed = random.randint(0, 2**32-1)
        opt['seed'] = seed
    logging.info(f"Starting generation with seed {seed} ")

    random.seed(seed)
    image_generator = Generator(opt)
    #filters = filters_from_cfg(None)
    spoiler = Spoiler()
    for i in range(options.size):
        logging.info(f"{i+1}/{options.size}")
        #im =
       # text, noisy, im = random_image(lines, fonts, options)
        text_file = '%s/%04d.txt' % (options.dir, i)
        img_file = '%s/%04d.png' % (options.dir, i)
        truth_file = '%s/GT/%04d_GT.png' % (options.dir, i)

        image = image_generator.generate()
        #if options.gen_truth:
        image.save(truth_file, dpi=(300,300))

        logging.info("Generated Image. Applying spoilers")
        image.accept(spoiler)
        # for filter in filters:
        #     image.accept(filter)
        # el.accept(Background(random.randint(220, 245)))
        # el.accept(Foreground(random.randint(200,255)))
        # image.render()
        image.save(img_file, dpi=(options.dpi, options.dpi) )


    with open(f"{options.dir}/config.yml", 'w') as f:
        yaml.dump(opt, f)

        # if not options.gen_truth:
        #     text_visitor = None
        #     text = image.accept(text_visitor)
        #     with open(text_file, 'w') as f:
        #         f.write(text)


        #im.save(img_file)


if __name__ == '__main__':
    main()