# Shutter

A Python Tool for the generation of images with different layouts and noise combinations.
Shutter is basically a metamodel leveraging the [Composite Pattern](https://en.wikipedia.org/wiki/Composite_pattern) for generating images + [Visitor Pattern](https://en.wikipedia.org/wiki/Visitor_pattern) to apply different noises or to navigate the structure.

The tool loads a YAML configuration file where the generation model is defined, i.e. each component is defined along with the optional noises to apply on it as well as the probability distributions of various parameters.

It exports by default the ground truth images, the spoiled images, and a yaml containing the parameters used for the generation (including the seed used to initialize the random state to enable determinism between run).
In addition to that, the tool runs the exporters defined into the yaml file. see [Exporters](#exporters)

## Requirements

- `pyenv` - `brew install pyenv` or visit `https://github.com/pyenv/pyenv-installer`
- `pipenv` - `pip install pipenv`

## Installation

- `pyenv install 3.8.5` if not yet installed
- `pipenv sync` should install py 3.8 environment with required packages

# Running

run `pipenv run python src/shutter.py -h` to know supported args

run `pipenv run python src/shutter.py --config $CONFIG_FILE --size $K --dir $PATH_TO_OUT_DIR [--workers N] [--out FILE_PREFIX]` to start the image generation

# Docker image

The docker image uses dumb-init to enable proper terminations of the workers. For more see https://github.com/Yelp/dumb-init

Simply build the docker image (which includes a font package), and run by mounting your local folders inside the docker container

```
docker build -t shutter .
```

Get font list

```
docker run shutter fc-list
```

e.g. running with `complex.yml` as model

```
export OUT_PATH=/local/path/to/dir
export IMG_DATAPATH=/opt/app/data

docker run -v $OUT_PATH:$IMG_DATAPATH shutter dumb-init pipenv run python src/shutter.py --config examples/complex.yml --dir $IMG_DATAPATH --size 3 --workers 8 --out complex_out --dpi 70
```

Example configurations available in [examples](examples)

Output examples using [complex.yml](examples/complex.yml) available in [examples/complex_out](examples/complex_out)

# Available Components

Each component has a default generation probability of 1. Overriding the default behaviour results in optional components.
See [Container](#container) for mutual exclusion.

## Generator

The basic component generator to play with in shutter. Defines its size (eventually a position relative to a parent) and a list of sub-components. Must instatiate a `Component` object (which is an augmented Pillow image) and return it.

### Basic parameters

```
size:
  height: H
  widht: W
position:
  x: X
  y: Y
elements:
  - Generators...
```

Each value node can be filled with an absolute value (H = 2000) or a value < 1 denoting its relation to the parent Generator.

#### Spatial values

```
position:
  x: 0.5
  y : concatenate
size:
  height: fill
```

will place the object sized `(1*parent_width, parent_free_height)` top left corner at `(0.5 * parent_width, parent_height - object_height)`

### Define Randomness

Wheter you'd like to define a random value for a particular field, you can assign a probability distribution function to it and let shutter do the rest. e.g.

```
Generator:
  size:
    width:
      distribution: normal
      mu: 1000
      sigma: 100
      min: 800
      max: 1200
    height: [1000, 1800]
  elements: ...
  spoilers: ...
```

Defines a Component generator which will produce images with:

**width** sampled from a normal distribution with μ = 1000, σ=100 truncated in the interval [800, 1200]

**height** sampled from a uniform distribution in the interval [1000, 1800]

## TextGroup

A block of text. Loads lazily random text from a `source_path` or fixed `text` taken from the yaml.`{h/w}_border` defines an optional % region of the image to be left blank

TODO: Rename to padding

## Text

A single line of text.

```
Text:
  h_border:[0,0.02]
  font:
    bold: 0.5
    name: font_names.txt
    italic: ~
    size: fill
```

## Image

An image that can be sampled from a set or be constant. if `files` is defined, `path` points to a folder where to find the files and `probabilities` defines a list of entries like `<file, likelihood to be selected>` see example in [config.yml](examples/paper.yml#L39)

## Container

A logical container useful to define mutually exclusive Components. The configuration syntax is almost equal to Generator less than it assumes that for each defined sub-component there is an associated probability of being picked for generation.

## Table

A Table with configurable layout.
`cols`: number of columns
`rows`: number of rows
`fix_keys_col` : probability of cells to have a top key text
`fix_rows` : probability of having fixed rows height
`fix_cols` : probability of having fixed cols width
`row_frame` : probability of cells having horizontal borders
`col_frame` : probability of cells having vertical borders
`title`: probability of table header row
`headers_file` : file of lines to sample from for `title`
`keys_file` : file of lines to sample from for cells `keys`
`values_file` : file of lines to sample from for cells `values`
`keys_uppercase`: probability of keys to be uppercase

`cells_spoiler` : list of spoilers to apply randomly to single cells
`spoilers`: list of spoiler to apply to the table

# Spoilers

Each of the defined Component can define a list of accepted spoilers and the associated likelihood.

## Available Spoilers

### Crop

Crops an external white border (if available) of the component.

### Pad

Adds an external black border to the component.

### Rotate

Rotates the component by a random angle and crops the resulting image.

### Background

Adds random noise to the background of the component.

### Foreground

Adds random noise to the foreground of the component.

### SaltPepper

Adds salt and pepper noise to the image

### Blur

Blurs the component by radius `r`

### Stroke

Adds a randomly moved line to the component.

### Overlay

Adds randomly one of the overlays available in `path`

### VerticalLine

Adds a vertical line

### Gradient

Adds a intensity gradient to the component

### WhiteNoise

`ratio`

### Background

`grey`

### JPEGCompression

`quality` / `subsampling`

# Exporters

Exporters are run on each model instance and visit the generated tree structure to output annotations (or some final postprocessing) on the generated image.
Currently the exporters dumps generation and spoilers informations such as type of component, size, applied spoilers, rolled values (when available) and data contained.

## GlobalExporter

Global Exporter outputs coordinates with respect to the image origin (top-left)

## LocalExporter

Local exporter outputs coordinates with respect to each component origin (doesn't account nested position offsets)

# Create custom Component

Components are resolved by reflection when they are named in the `config` file. It is sufficient to add a class extending `Generator`, a constructor accepting a dict with the params and a `generate`.

A generator takes its yaml node as argument from which it can take initialization parameters and store relevant informations for the actual generation step.

In the generate function, you basically pop random values and return a drawn component which will contain useful informations for future spoilers.

To pop a random value for a parameter (node with param name and distribution values): just call the `roll_value` function and pass the node as argument.
See some generators [implementations](src/generators.py)

# Create custom Spoiler

Spoilers must extend the `Filter` class, they received unpacked args from the yaml node, and must override a `run(self, image):` method returning the spoiled image

# Create custom Exporter
