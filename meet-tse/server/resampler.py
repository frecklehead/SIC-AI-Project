"""Exact, stateful, arbitrary-ratio streaming resampler (windowed-sinc, numpy only).

Feed it chunks of any size; it returns the matching resampled samples and keeps
the filter history between calls, so chunk boundaries do not create artefacts.
Algorithmic delay is `half` input samples (<1 ms for 48k -> 16k).
"""
import numpy as np


class StreamResampler:
    def __init__(self, sr_in: int, sr_out: int):
        self.sr_in, self.sr_out = int(sr_in), int(sr_out)
        self.same = self.sr_in == self.sr_out
        self.ratio = self.sr_in / self.sr_out          # input samples per output sample
        self.cut = min(1.0, self.sr_out / self.sr_in) * 0.94
        self.half = int(np.ceil(10.0 / self.cut))
        self.buf = np.zeros(self.half, np.float32)      # pre-roll of zeros
        self.base = -self.half                          # absolute index of buf[0]
        self.j = 0                                      # next output index
        self._offs = np.arange(-self.half + 1, self.half + 1)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)
        if self.same:
            return x
        self.buf = np.concatenate([self.buf, x])
        end = self.base + self.buf.size
        jmax = int(np.floor((end - 1 - self.half) / self.ratio))
        if jmax < self.j:
            return np.zeros(0, np.float32)
        t = np.arange(self.j, jmax + 1, dtype=np.float64) * self.ratio
        i0 = np.floor(t).astype(np.int64)
        idx = i0[:, None] + self._offs[None, :]
        d = t[:, None] - idx
        w = self.cut * np.sinc(self.cut * d) * (0.5 + 0.5 * np.cos(np.pi * d / self.half))
        y = (self.buf[idx - self.base] * w).sum(axis=1)
        self.j = jmax + 1
        keep = int(np.floor(self.j * self.ratio)) - self.half + 1
        drop = keep - self.base
        if drop > 0:
            self.buf = self.buf[drop:]
            self.base += drop
        return y.astype(np.float32)
