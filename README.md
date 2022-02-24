# acl2022-code

Code for ACL2022 privacy paper

## Installation

All packages are specified in `requirements.txt`, this project uses `virtualenv` to ensure reproducibility.

```bash
$ sudo apt-get install virtualenv
$ virtualenv venv --python=python3
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Running main experiments

Needs at least 16 GB RAM, takes a while; generates `results.json`

```bash
$ python experiment.py
```

## Running unit tests

```bash
$ python -m unittest tests.py
```

## Plotting sampling functions

Run jupyter

```bash
$ python -m jupyter notebook --notebook-dir=.
```

And open and run `plots.ipynb` in Jupyter notebook using a web browser