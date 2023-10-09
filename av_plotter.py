import matplotlib.pyplot as plt
import numpy as np


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

        # for statistics
        self.interval = 1.0

        # subplots
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

    def set_interval(self, interval):
        self.interval = interval

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
            v_bitrate_array = self.v_stream.calc_bitrate_in_kbps(self.interval)
            ax.plot(
                v_bitrate_array[0],
                v_bitrate_array[1],
                self.VIDEO_LINE_COLOR,
                label="video",
            )
        if self.a_stream:
            a_bitrate_array = self.a_stream.calc_bitrate_in_kbps(self.interval)
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
            v_fps_array = self.v_stream.calc_fps(self.interval)
            ax.plot(
                v_fps_array[0],
                v_fps_array[1],
                self.VIDEO_LINE_COLOR,
                label="video",
            )
        if self.a_stream:
            a_fps_array = self.a_stream.calc_fps(self.interval)
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
