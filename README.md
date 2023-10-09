# plot-av
Plot Audio/Video streams.

![](docs/images/plot-av.png)

## Installation 

### Prerequisites
- Latest [Python](https://www.python.org/downloads/)
- [pyav](https://pyav.org/docs/stable/)
- [matplotlib](https://matplotlib.org/)

### Example installation via conda

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

- Basic 

```bash
python plot-av.py -i test.mp4
```

- Partial subplots

```bash
python plot-av.py -i test.mp4 --plots dts,pts
```
