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

    def calc_bitrate_in_kbps(self, interval_in_seconds=1.0):
        interval = interval_in_seconds / self.time_base

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
        bitrate[1] = bitrate[1] * 8 / 1024 / interval_in_seconds  # kbps

        return bitrate

    def calc_fps(self, interval_in_seconds=1.0):
        interval = interval_in_seconds / self.time_base  # 1 second

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
        fps[1] = fps[1] / interval_in_seconds  # fps

        return fps
