import av
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np


class StreamInfo:
    def __init__(self, stream):
        # self.stream = stream
        self.stream_index = stream.index
        self.stream_type = stream.type
        self.time_base = stream.time_base

        # raw data from packets: [[dts, pts, duration, size], [dts, pts, duration, size], ...]
        self.raw_data_list = []

        # numpy array
        self.npdata = None

        # for statistics
        self.interval = 1.0

    def set_interval(self, interval):
        self.interval = interval

    def capture_by_packet(self, packet):
        self.raw_data_list.append(
            [packet.dts, packet.pts, packet.duration, packet.size]
        )

    def finalize(self):
        # construct data array
        # [[dts, pts, duration, size], [dts, pts, duration, size], ...]
        self.npdata = np.array(
            self.raw_data_list,
            dtype=np.float64,
        )
        self.raw_data_list = []  # clean up

        # [[dts, dts, ...], [pts, pts, ...], [duration, duration, ...], [size, size, ...]]
        self.npdata = self.npdata.transpose()

    def dts_in_seconds(self):
        return self.npdata[0] * self.time_base

    def pts_in_seconds(self):
        return self.npdata[1] * self.time_base

    def duration_in_seconds(self):
        return self.npdata[2] * self.time_base

    def size_in_KB(self):
        return self.npdata[3] / 1024.0

    def calc_dts_delta_in_seconds(self):
        dts_delta = self.npdata[0]

        # calculate dts_delta = dts - prev_dts
        dts_delta = np.roll(dts_delta, -1)
        dts_delta[-1] = np.nan  # ignore last value
        dts_delta = dts_delta - self.npdata[0]
        return dts_delta * self.time_base

    def calc_bitrate_in_kbps(self):
        interval = self.interval / self.time_base

        data_array = []

        start_ts = self.npdata[0][0]  # first dts
        size = 0

        for d in self.npdata.transpose():  # [[dts,pts,duration,size], ...]
            if d[0] > start_ts + interval:
                data_array.append([start_ts, size])
                start_ts += interval
                size = 0
            size += d[3]

        if size > 0:  # last interval
            data_array.append([start_ts, size])

        bitrate = np.array(
            data_array,
            dtype=np.float64,
        ).transpose()
        bitrate[0] = bitrate[0] * self.time_base  # seconds
        bitrate[1] = bitrate[1] * 8 / 1024 / self.interval  # kbps

        return bitrate

    def calc_fps(self):
        interval = self.interval / self.time_base  # 1 second

        data_array = []

        start_ts = self.npdata[0][0]  # first dts
        size = 0

        for d in self.npdata.transpose():  # [[dts,pts,duration,size], ...]
            if d[0] > start_ts + interval:
                data_array.append([start_ts, size])
                start_ts += interval
                size = 0
            size += 1

        if size > 0:  # last interval
            data_array.append([start_ts, size])

        fps = np.array(
            data_array,
            dtype=np.float64,
        ).transpose()
        fps[0] = fps[0] * self.time_base  # seconds
        fps[1] = fps[1] / self.interval  # fps

        return fps


def calc_avsync_in_seconds(base_ts, another_ts):
    diff_ts = np.empty(len(base_ts))
    diff_ts.fill(np.nan)

    # merge
    base_data = np.array(
        (base_ts, np.arange(len(base_ts)), np.zeros(len(base_ts)))
    )  # use type '0' as base
    another_data = np.array(
        (another_ts, np.arange(len(another_ts)), np.ones(len(another_ts)))
    )  # use type '1' as another
    full_data = np.concatenate((base_data.T, another_data.T))

    # sort by ts
    full_rec_array = np.core.records.fromarrays(
        full_data.T, names="ts,original_index,type", formats="float64,int64,int8"
    )
    full_rec_array.sort(order="ts")

    # calc diff for each base_ts
    for i in range(full_rec_array.shape[0]):
        if full_rec_array[i][2] != 0:  # find next base ts
            continue

        curr_base_ts = full_rec_array[i][0]
        original_index = full_rec_array[i][1]

        prev_another_ts = None
        next_another_ts = None

        # find prev & next another_ts
        for j in reversed(range(i)):
            if full_rec_array[j][2] == 1:
                prev_another_ts = full_rec_array[j][0]
                break
        for j in range(i + 1, full_rec_array.shape[0]):
            if full_rec_array[j][2] == 1:
                next_another_ts = full_rec_array[j][0]
                break

        candidates_array = np.array(
            [prev_another_ts, next_another_ts], dtype=np.float64
        )
        candidates_diff = np.abs(candidates_array - curr_base_ts)
        candidates_min_index = np.nanargmin(candidates_diff)
        diff_ts[original_index] = candidates_array[candidates_min_index] - curr_base_ts

    return (base_ts, diff_ts)


class AVPlotter:
    # line fmt
    VIDEO_LINE_COLOR = "y"
    VIDEO_LINE_FMT = VIDEO_LINE_COLOR + "^"
    AUDEO_LINE_COLOR = "b"
    AUDIO_LINE_FMT = AUDEO_LINE_COLOR + "+"
    AVSYNC_LINE_COLOR = "g"

    def __init__(self, window_title="", v_stream=None, a_stream=None):
        self.v_stream = v_stream
        self.a_stream = a_stream
        self.window_title = window_title

        self.subplots = {
            "dts": self.plot_dts,
            "pts": self.plot_pts,
            "size": self.plot_size,
            "bitrate": self.plot_bitrate,
            "fps": self.plot_fps,
            "avsync": self.plot_avsync,
            "dts_delta": self.plot_dts_delta,
            "duration": self.plot_duration,
        }

    def available_subplots(self):
        return self.subplots.keys()

    def decide_layout(self, subplots):
        # decide cols/rows
        if len(subplots) == 1 or len(subplots) == 2:  # 1x1 or 2x1
            ncols = len(subplots)
            nrows = 1
            subplots_2d = [subplots]
            return (ncols, nrows, subplots_2d)

        if len(subplots) == 3 or len(subplots) == 4:  # 2x2
            ncols = nrows = 2
            subplots_2d = [subplots[:ncols], subplots[ncols:]]
        elif len(subplots) == 5 or len(subplots) == 6:  # 3x2
            ncols = 3
            nrows = 2
            subplots_2d = [
                subplots[:ncols],
                subplots[ncols:],
            ]
        else:  # 3x3
            ncols = nrows = 3
            subplots_2d = [
                subplots[:ncols],
                subplots[ncols : ncols * 2],
                subplots[ncols * 2 :],
            ]

        for i in range(nrows):
            delta = ncols - len(subplots_2d[i])
            for j in range(delta):
                subplots_2d[i].append(None)  # make sure ncols == nrows

        return (ncols, nrows, subplots_2d)

    def plot(self, subplots, dpi=None):
        # layout
        (ncols, nrows, subplots_2d) = self.decide_layout(subplots)
        # print(subplots_2d)

        # create axis
        # fig, axs = plt.subplots(ncols=ncols, nrows=nrows, layout="constrained")
        fig, axs = plt.subplot_mosaic(subplots_2d, layout="constrained")
        fig.canvas.manager.set_window_title(self.window_title)

        # plot
        for p in subplots:
            plot_func = self.subplots[p]
            plot_func(axs[p])

        # fig.align_labels()
        if dpi is not None:
            fig.dpi = dpi
        plt.show()

    def plot_dts(self, ax):
        ax.set_title(f"dts")
        ax.set_xlabel("packet no.", loc="right")
        ax.set_ylabel("dts (s)")
        if self.v_stream:
            ax.plot(
                self.v_stream.dts_in_seconds(),
                self.VIDEO_LINE_FMT,
                label="video",
            )
        if self.a_stream:
            ax.plot(
                self.a_stream.dts_in_seconds(),
                self.AUDIO_LINE_FMT,
                label="audio",
            )
        ax.legend()
        # ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02, 0.0, 0.102), ncols=2)

    def plot_pts(self, ax):
        ax.set_title(f"pts")
        ax.set_xlabel("packet no.", loc="right")
        ax.set_ylabel("pts (s)")
        # ax.yaxis.tick_right()
        # ax.yaxis.set_label_position("right")
        if self.v_stream:
            ax.plot(
                self.v_stream.pts_in_seconds(),
                self.VIDEO_LINE_FMT,
                label="video",
            )
        if self.a_stream:
            ax.plot(
                self.a_stream.pts_in_seconds(),
                self.AUDIO_LINE_FMT,
                label="audio",
            )
        ax.legend()

    def plot_size(self, ax):
        ax.set_title(f"size")
        ax.set_xlabel("packet no.", loc="right")
        ax.set_ylabel("size (KB)")
        if self.v_stream:
            ax.plot(
                self.v_stream.size_in_KB(),
                self.VIDEO_LINE_FMT,
                label="video",
            )
        if self.a_stream:
            ax.plot(
                self.a_stream.size_in_KB(),
                self.AUDIO_LINE_FMT,
                label="audio",
            )
        ax.set_ylim(0)
        ax.legend()

    def plot_bitrate(self, ax):
        ax.set_title(f"bitrate")
        ax.set_xlabel("decode time (s)", loc="right")
        ax.set_ylabel("bitrate (kbps)")
        if self.v_stream:
            v_bitrate_array = self.v_stream.calc_bitrate_in_kbps()
            ax.plot(
                v_bitrate_array[0],
                v_bitrate_array[1],
                self.VIDEO_LINE_COLOR,
                label="video",
            )
        if self.a_stream:
            a_bitrate_array = self.a_stream.calc_bitrate_in_kbps()
            ax.plot(
                a_bitrate_array[0],
                a_bitrate_array[1],
                self.AUDEO_LINE_COLOR,
                label="audio",
            )
        ax.set_ylim(0)
        ax.legend()

    def plot_fps(self, ax):
        ax.set_title(f"fps")
        ax.set_xlabel("decode time (s)", loc="right")
        ax.set_ylabel("fps")
        if self.v_stream:
            v_fps_array = self.v_stream.calc_fps()
            ax.plot(
                v_fps_array[0],
                v_fps_array[1],
                self.VIDEO_LINE_COLOR,
                label="video",
            )
        if self.a_stream:
            a_fps_array = self.a_stream.calc_fps()
            ax.plot(
                a_fps_array[0],
                a_fps_array[1],
                self.AUDEO_LINE_COLOR,
                label="audio",
            )
        ax.set_ylim(0)
        ax.legend()

    def plot_avsync(self, ax):
        ax.set_title(f"av sync")
        ax.set_xlabel("presentation time (s)", loc="right")
        ax.set_ylabel("diff (s)")
        if self.v_stream and self.a_stream:
            (base_ts, sync_ts) = calc_avsync_in_seconds(
                base_ts=np.sort(self.a_stream.pts_in_seconds()),
                another_ts=np.sort(self.v_stream.pts_in_seconds()),
            )
            ax.plot(
                base_ts,
                sync_ts,
                self.AVSYNC_LINE_COLOR,
            )

    def plot_dts_delta(self, ax):
        ax.set_title(f"dts delta (dts-prev_dts)")
        ax.set_xlabel("dts (s)", loc="right")
        ax.set_ylabel("delta (s)")
        if self.v_stream:
            ax.plot(
                self.v_stream.dts_in_seconds(),
                self.v_stream.calc_dts_delta_in_seconds(),
                self.VIDEO_LINE_FMT,
                label="video",
            )
        if self.a_stream:
            ax.plot(
                self.a_stream.dts_in_seconds(),
                self.a_stream.calc_dts_delta_in_seconds(),
                self.AUDIO_LINE_FMT,
                label="audio",
            )
        ax.legend()

    def plot_duration(self, ax):
        ax.set_title(f"duration")
        ax.set_xlabel("dts (s)", loc="right")
        ax.set_ylabel("duration (s)")
        # ax.yaxis.tick_right()
        # ax.yaxis.set_label_position("right")
        if self.v_stream:
            ax.plot(
                self.v_stream.dts_in_seconds(),
                self.v_stream.duration_in_seconds(),
                self.VIDEO_LINE_FMT,
                label="video",
            )
        if self.a_stream:
            ax.plot(
                self.a_stream.dts_in_seconds(),
                self.a_stream.duration_in_seconds(),
                self.AUDIO_LINE_FMT,
                label="audio",
            )
        ax.legend()


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
