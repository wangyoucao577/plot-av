# Development

## Prerequisites
- Latest [Python](https://www.python.org/downloads/)
- [pyav](https://pyav.org/docs/stable/)
- [matplotlib](https://matplotlib.org/)

## Example installation via conda

```bash
# base python env
conda create -n pyav python=3.11
conda activate pyav

# install pyav
conda install av -c conda-forge

# install matplot 
python -m pip install -U pip
python -m pip install -U matplotlib
```

## Usage 

```bash
$ python -m plot_av.plot_av -h
```
