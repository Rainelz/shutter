import random
import argparse
import os
from images import Generator #random_image, load_fonts
from spoiler import filters_from_cfg
import yaml

def parse_options():
    parser = argparse.ArgumentParser(description='Generated text and images.')
    parser.add_argument('--config', required=True, help='path to YAML config file')
    parser.add_argument('--fonts', required=False, help='path of the list of fonts')
    parser.add_argument('--text', required=True, help='path of the text file')
    parser.add_argument('--dir', required=True, help='path of the output directory')
    parser.add_argument('--size', type=int, default=100, help='how many examples to generate')
    parser.add_argument('--no-filters', type=bool, default=False, help='Don\'t apply filters')
    parser.add_argument('--gen-truth', default=False, help='Generate ground truth images', action="store_true")
    parser.add_argument('--no-skew', default=False, help='Don\'t skew images',  action="store_true")
    parser.add_argument('--dpi', type=int, default=70, help='Noisy images dpi')


    return parser.parse_args()


def main():
    options = parse_options()

    with open(options.config, 'r') as stream:
        try:
            opt = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    #fonts = load_fonts(options.fonts, size=15)
    fonts=[]
    #with open(options.text) as f:
        #lines = f.readlines()

    if options.gen_truth:
        os.makedirs(f"{options.dir}/GT", exist_ok=True)

    for i in range(options.size):
        print(i, end='\r')
        #im =
       # text, noisy, im = random_image(lines, fonts, options)
        text_file = '%s/%04d.txt' % (options.dir, i)
        img_file = '%s/%04d.png' % (options.dir, i)
        truth_file = '%s/GT/%04d_GT.png' % (options.dir, i)

        image_generator = Generator(opt)
        image = image_generator.generate()
        if options.gen_truth:
            image.save(truth_file, dpi=(300,300))

        filters = filters_from_cfg(None)

        for filter in filters:
            image.accept(filter)
        # el.accept(Background(random.randint(220, 245)))
        # el.accept(Foreground(random.randint(200,255)))
        # image.render()
        image.save(img_file, dpi=(options.dpi, options.dpi) )


        if not options.gen_truth:
            text_visitor = None
            text = image.accept(text_visitor)
            with open(text_file, 'w') as f:
                f.write(text)


        #im.save(img_file)


if __name__ == '__main__':
    main()