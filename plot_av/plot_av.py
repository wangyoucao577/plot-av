from plot_av.stream_info import StreamInfo
from plot_av.av_plotter import AVPlotter
from plot_av.process_args import process_args

import av
import os
import logging


def auto_select_stream(streams, input_index, disable=None):
    if disable:
        return None

    # auto selected first one
    return None if not streams else streams[0]


def manually_select_streams(v_streams, a_streams, input_index, streams_selection):
    selected_v_streams = []
    selected_a_streams = []

    for ss in streams_selection:
        if ss[0] != input_index:
            continue

        if ss[1] == "v":
            if len(v_streams) <= ss[2]:
                logging.warning(f"try to select stream {ss} but not found")
                continue
            selected_v_streams.append(v_streams[ss[2]])
        elif ss[1] == "a":
            if len(a_streams) <= ss[2]:
                logging.warning(f"try to select stream {ss} but not found")
                continue
            selected_a_streams.append(a_streams[ss[2]])
        else:
            assert False
    return (selected_v_streams, selected_a_streams)


def produce_streams_info(
    inputs,
    disable_video,
    disable_audio,
    streams_selection,
):
    manually_selection = True if streams_selection is not None else False

    v_streams_info = []
    a_streams_info = []

    for input_index, input in enumerate(inputs):
        with av.open(input) as container:
            # select streams
            selected_streams = []
            selected_v_streams_info = []
            selected_a_streams_info = []
            if manually_selection:
                (selected_v_streams, selected_a_streams) = manually_select_streams(
                    container.streams.video,
                    container.streams.audio,
                    input_index,
                    streams_selection,
                )
                for v in selected_v_streams:
                    selected_streams.append(v)
                    selected_v_streams_info.append(StreamInfo(v))
                for a in selected_a_streams:
                    selected_streams.append(a)
                    selected_a_streams_info.append(StreamInfo(a))
            else:
                if not v_streams_info:
                    selected_v_stream = auto_select_stream(
                        container.streams.video,
                        input_index,
                        disable=disable_video,
                    )
                    if selected_v_stream is not None:
                        selected_v_streams_info.append(StreamInfo(selected_v_stream))
                        selected_streams.append(selected_v_stream)
                if not a_streams_info:
                    selected_a_stream = auto_select_stream(
                        container.streams.audio,
                        input_index,
                        disable=disable_audio,
                    )
                    if selected_a_stream is not None:
                        selected_a_streams_info.append(StreamInfo(selected_a_stream))
                        selected_streams.append(selected_a_stream)

            if not selected_streams:  # no need demux the input
                continue

            # retrieve info from packets
            for packet in container.demux(selected_streams):
                # print(packet)
                if packet.size == 0:
                    continue  # empty packet for flushing, ignore it

                for v in selected_v_streams_info:
                    if packet.stream_index == v.stream_index:
                        v.capture_by_packet(packet)
                for a in selected_a_streams_info:
                    if packet.stream_index == a.stream_index:
                        a.capture_by_packet(packet)

            v_streams_info += selected_v_streams_info
            a_streams_info += selected_a_streams_info

    # finalize data
    for v in v_streams_info:
        v.finalize()
    for a in a_streams_info:
        a.finalize()

    return (v_streams_info, a_streams_info)


def main():
    args = process_args(available_subplots=AVPlotter().available_subplots())

    # retrieve a/v stream info
    (v_streams_info, a_streams_info) = produce_streams_info(
        args.input,
        args.vn,
        args.an,
        args.streams_selection,
    )

    # plot
    input_names = ",".join([os.path.basename(i) for i in args.input])
    window_title = os.path.basename(__file__) + " - " + input_names
    av_plotter = AVPlotter(
        window_title,
        # workaround since avplotter doesn't support multiple streams yet, may be supported soon.
        None if not v_streams_info else v_streams_info[0],
        None if not a_streams_info else a_streams_info[0],
    )
    av_plotter.set_interval(args.interval)
    av_plotter.plot(args.plots, args.dpi)


if __name__ == "__main__":
    main()
