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
        interval = 1 / self.time_base  # 1 second

        data_array = []

        start_ts = self.npdata[0][0]  # first dts
        size = 0

        for d in self.npdata.transpose():  # [[dts,pts,duration,size], ...]
            if d[0] > start_ts + interval:
                data_array.append([start_ts, size])
                start_ts += interval
                size = 0
            size += d[3]

        bitrate = np.array(
            data_array,
            dtype=np.float64,
        ).transpose()
        bitrate[0] = bitrate[0] * self.time_base  # seconds
        bitrate[1] = bitrate[1] * 8 / 1024  # kbps

        return bitrate

    def calc_fps(self):
        interval = 1 / self.time_base  # 1 second

        data_array = []

        start_ts = self.npdata[0][0]  # first dts
        size = 0

        for d in self.npdata.transpose():  # [[dts,pts,duration,size], ...]
            if d[0] > start_ts + interval:
                data_array.append([start_ts, size])
                start_ts += interval
                size = 0
            size += 1

        fps = np.array(
            data_array,
            dtype=np.float64,
        ).transpose()
        fps[0] = fps[0] * self.time_base  # seconds

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

    def plot(self, subplots):
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

    parser = argparse.ArgumentParser(description="plot timestamps.")
    parser.add_argument("-i", required=True, help="input file url", dest="input")
    parser.add_argument(
        "--plots",
        required=False,
        default=available_subplots_options_str,
        help=f"subplots to show, seperate by ','. options: {available_subplots_options_str}",
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


def main():
    args = process_args()

    # retrieve info
    streams_info = []
    with av.open(args.input) as container:
        for stream in container.streams:
            streams_info.append(StreamInfo(stream))
        for packet in container.demux():
            # print(packet)
            if packet.size == 0:
                continue  # empty packet for flushing, ignore it
            si = streams_info[packet.stream_index]
            si.capture_by_packet(packet)

    # finalize data
    for s in streams_info:
        s.finalize()

    # select v/a stream
    v_stream = None
    a_stream = None
    for s in streams_info:
        if v_stream is None and s.stream_type == "video":
            v_stream = s
        if a_stream is None and s.stream_type == "audio":
            a_stream = s

    # plot
    window_title = os.path.basename(__file__) + " - " + os.path.basename(args.input)
    av_plotter = AVPlotter(
        window_title,
        v_stream,
        a_stream,
    )
    av_plotter.plot(args.plots)


if __name__ == "__main__":
    main()
