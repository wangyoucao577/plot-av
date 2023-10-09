import argparse


def unsigned_limited_float_type(arg):
    """Type function for argparse - a float within some predefined bounds"""
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f <= 0.0:
        raise argparse.ArgumentTypeError("Argument must be > 0.0")
    return f


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
    video_selection_group = parser.add_mutually_exclusive_group()
    video_selection_group.add_argument(
        "-vs",
        "--video_stream",
        nargs=2,
        type=int,
        metavar=("INPUT_INDEX", "STREAM_INDEX"),
        help="manually select video stream, INDEX starts from 0",
    )
    video_selection_group.add_argument(
        "-vn", required=False, action="store_true", help="disable video stream"
    )
    audio_selection_group = parser.add_mutually_exclusive_group()
    audio_selection_group.add_argument(
        "-as",
        "--audio_stream",
        nargs=2,
        type=int,
        metavar=("INPUT_INDEX", "STREAM_INDEX"),
        help="manually select audio stream, INDEX starts from 0",
    )
    audio_selection_group.add_argument(
        "-an", required=False, action="store_true", help="disable audio stream"
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
    args = parser.parse_args()
    # print(args)

    # validate plots
    subplots = []
    for p in args.plots.split(PLOT_SPLIT_DELIMETER):
        if not p in available_subplots:
            print(f"warning: remove invalid plot '{p}'")
            continue
        if p in subplots:
            print(f"warning: remove duplicate plot '{p}'")
            continue
        subplots.append(p)

    args.plots = subplots  # make sure plots are all valid

    return args
