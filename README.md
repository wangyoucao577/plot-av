# plot-av
Plot details of Audio/Video streams of media files to help you gain better insights of them.      


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

```bash
# basic use
$ python plot-av.py -i test.mp4


# help 
$ python plot-av.py -h
usage: plot-av.py [-h] -i INPUT [-vn | -an | -map STREAMS_SELECTION] [--dpi DPI] [--plots PLOTS] [--interval INTERVAL] [--log LOGLEVEL]

plot audio/video streams.

options:
  -h, --help            show this help message and exit
  -i INPUT              input file url (default: None)
  -vn                   disable video stream (default: False)
  -an                   disable audio stream (default: False)
  -map STREAMS_SELECTION
                        manually select streams, pattern 'input_index:stream_type:stream_index', e.g. '0:v:0', '0:a:0' (default: None)
  --dpi DPI             resolution of the figure. If not provided, defaults to 100 by matplotlib. (default: None)
  --plots PLOTS         subplots to show, seperate by ','. options: dts,pts,size,bitrate,fps,avsync,dts_delta,duration (default:
                        dts,pts,size,bitrate,fps,avsync,dts_delta,duration)
  --interval INTERVAL   calculation interval in seconds for statistics metrics, such as bitrate, fps, etc. (default: 1.0)
  --log LOGLEVEL        log level (default: None)
```

### More examples

- Draw your interested subplots only

```bash
python plot-av.py -i test.mp4 --plots dts,pts
```

- Draw video or audio only 

```bash
# disable video
python plot-av.py -i test.mp4 -vn

# disable audio
python plot-av.py -i test.mp4 -an
```

- Manually select streams from mutiple inputs

```bash
python plot-av.py -i test1.mp4 -i test2.mp4 -map 0:v:0 -map 1:a:0 
```

