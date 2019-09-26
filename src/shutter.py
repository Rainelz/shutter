import random
import argparse
import os
from images import random_image, load_fonts


def parse_options():
    parser = argparse.ArgumentParser(description='Generated text and images.')
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
    #fonts = load_fonts(options.fonts, size=15)
    fonts=[]
    with open(options.text) as f:
        lines = f.readlines()

    for i in range(options.size):
        text, noisy, im = random_image(lines, fonts, options)
        text_file = '%s/%04d.txt' % (options.dir, i)
        img_file = '%s/%04d.png' % (options.dir, i)
        truth_file = '%s/GT/%04d_GT.png' % (options.dir, i)
        if options.gen_truth:
            os.makedirs(f"{options.dir}/GT", exist_ok=True)
        if not options.gen_truth:

            with open(text_file, 'w') as f:
                f.write(text)

        noisy.save(img_file, dpi=(options.dpi, options.dpi))
        if options.gen_truth:
            im.save(truth_file, dpi=(300,300))
        #im.save(img_file)
        print(i, end='\r')

if __name__ == '__main__':
    main()