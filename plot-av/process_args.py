import argparse
import logging


def unsigned_limited_float_type(arg):
    """Type function for argparse - a float within some predefined bounds"""
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f <= 0.0:
        raise argparse.ArgumentTypeError("Argument must be non-negative")
    return f


def stream_selection_type(arg):
    l = arg.split(":")
    if len(l) != 3:
        raise argparse.ArgumentTypeError("must be something like '0:v:0' or '0:a:0'")

    if l[1] != "v" and l[1] != "a":
        raise argparse.ArgumentTypeError("stream type only accepts 'v' or 'a'")
    stream_type = l[1]

    try:
        input_index = int(l[0])
    except ValueError:
        raise argparse.ArgumentTypeError("input index must be a integer number")
    if input_index < 0:
        raise argparse.ArgumentTypeError("input index must be non-negative")

    try:
        stream_index = int(l[2])
    except ValueError:
        raise argparse.ArgumentTypeError("stream index must be a integer number")
    if stream_index < 0:
        raise argparse.ArgumentTypeError("stream index must be non-negative")

    return (input_index, stream_type, stream_index)


def process_args(available_subplots):
    PLOT_SPLIT_DELIMETER = ","
    available_subplots_options_str = PLOT_SPLIT_DELIMETER.join(available_subplots)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="plot audio/video streams.",
    )
    parser.add_argument(
        "-i", required=True, action="append", help="input file url", dest="input"
    )
    select_streams_group = parser.add_mutually_exclusive_group()
    select_streams_group.add_argument(
        "-vn", required=False, action="store_true", help="disable video stream"
    )
    select_streams_group.add_argument(
        "-an", required=False, action="store_true", help="disable audio stream"
    )
    select_streams_group.add_argument(
        "-map",
        required=False,
        type=stream_selection_type,
        action="append",
        help="manually select streams, pattern 'input_index:stream_type:stream_index', e.g. '0:v:0', '0:a:0'",
        dest="streams_selection",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        help="resolution of the figure. If not provided, defaults to 100 by matplotlib.",
    )
    parser.add_argument(
        "--plots",
        default=available_subplots_options_str,
        help=f"subplots to show, seperate by ','. options: {available_subplots_options_str}",
    )
    parser.add_argument(
        "--interval",
        type=unsigned_limited_float_type,
        default=1.0,
        help="calculation interval in seconds for statistics metrics, such as bitrate, fps, etc.",
    )
    parser.add_argument("--log", dest="loglevel", help="log level")
    args = parser.parse_args()
    # print(args)

    # config log level
    if args.loglevel:
        # assuming loglevel is bound to the string value obtained from the
        # command line argument. Convert to upper case to allow the user to
        # specify --log=DEBUG or --log=debug
        numeric_level = getattr(logging, args.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError("Invalid log level: %s" % args.loglevel)
        logging.basicConfig(level=numeric_level)

    # validate plots
    subplots = []
    for p in args.plots.split(PLOT_SPLIT_DELIMETER):
        if not p in available_subplots:
            logging.warning(f"warning: remove invalid plot '{p}'")
            continue
        if p in subplots:
            logging.warning(f"warning: remove duplicate plot '{p}'")
            continue
        subplots.append(p)

    args.plots = subplots  # make sure plots are all valid

    return args
