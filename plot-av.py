import av
import argparse
import os

from stream_info import StreamInfo
from av_plotter import AVPlotter


def process_args():
    PLOT_SPLIT_DELIMETER = ","
    available_subplots = AVPlotter().available_subplots()
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
        type=float,
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


def select_stream(input_index, disable=None, select=None):
    if disable:
        return None

    if select is None:
        return 0  # auto selected first one

    if select[0] != input_index:  # input index not match
        return None

    return select[1]


def produce_streams(
    inputs,
    disable_video,
    video_stream_selection,
    disable_audio,
    audio_stream_selection,
    interval,
):
    v_stream_info = a_stream_info = None

    for input_index, input in enumerate(inputs):
        with av.open(input) as container:
            # select streams
            selected_video_index = selected_audio_index = None
            if v_stream_info is None:
                selected_video_index = select_stream(
                    input_index=input_index,
                    disable=disable_video,
                    select=video_stream_selection,
                )
                if selected_video_index is not None:
                    selected_video_stream = container.streams.get(
                        video=selected_video_index
                    )
                    v_stream_info = StreamInfo(selected_video_stream[0])
            if a_stream_info is None:
                selected_audio_index = select_stream(
                    input_index=input_index,
                    disable=disable_audio,
                    select=audio_stream_selection,
                )
                if selected_audio_index is not None:
                    selected_audio_stream = container.streams.get(
                        audio=selected_audio_index
                    )
                    a_stream_info = StreamInfo(selected_audio_stream[0])

            if selected_video_index is None and selected_audio_index is None:
                continue

            # retrieve info from packets
            for packet in container.demux(
                video=(
                    selected_video_index if selected_video_index is not None else []
                ),
                audio=(
                    selected_audio_index if selected_audio_index is not None else []
                ),
            ):
                # print(packet)
                if packet.size == 0:
                    continue  # empty packet for flushing, ignore it

                if (
                    v_stream_info is not None
                    and packet.stream_index == v_stream_info.stream_index
                ):
                    v_stream_info.capture_by_packet(packet)
                elif (
                    a_stream_info is not None
                    and packet.stream_index == a_stream_info.stream_index
                ):
                    a_stream_info.capture_by_packet(packet)

    # finalize data
    if v_stream_info is not None:
        v_stream_info.set_interval(interval)
        v_stream_info.finalize()
    if a_stream_info is not None:
        a_stream_info.set_interval(interval)
        a_stream_info.finalize()

    return (v_stream_info, a_stream_info)


def main():
    args = process_args()

    # retrieve a/v stream info
    (v_stream_info, a_stream_info) = produce_streams(
        args.input,
        args.vn,
        args.video_stream,
        args.an,
        args.audio_stream,
        args.interval,
    )

    # plot
    input_names = ",".join([os.path.basename(i) for i in args.input])
    window_title = os.path.basename(__file__) + " - " + input_names
    av_plotter = AVPlotter(
        window_title,
        v_stream_info,
        a_stream_info,
    )
    av_plotter.plot(args.plots, args.dpi)


if __name__ == "__main__":
    main()
