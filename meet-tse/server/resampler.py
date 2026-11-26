"""Exact, stateful, arbitrary-ratio streaming resampler (windowed-sinc, numpy only).

Feed it chunks of any size; it returns the matching resampled samples and keeps
the filter history between calls, so chunk boundaries do not create artefacts.
Algorithmic delay is `half` input samples (<1 ms for 48k -> 16k).
"""
import numpy as np

_EMPTY_F32 = np.empty(0, np.float32)


class Polyphase3Resampler:
    """Preallocated, stateful 3:1 windowed-sinc resampler.

    It uses the same support, cutoff, and raised-cosine window as
    ``StreamResampler``. Windows are evaluated in vectorized phase groups.
    Output is a borrowed view valid until the next call; filter and output work
    buffers are reused so the streaming caller can consume samples in place.
    """
    _CAPACITY = 1 << 16

    def __init__(self, sr_in, sr_out):
        self.sr_in, self.sr_out = int(sr_in), int(sr_out)
        self.same = self.sr_in == self.sr_out
        self.ratio = self.sr_in / self.sr_out
        if self.ratio not in (3.0, 1.0 / 3.0):
            raise ValueError("Polyphase3Resampler requires an exact 3:1 rate ratio")
        self.down = self.ratio == 3.0
        self.cut = min(1.0, self.sr_out / self.sr_in) * 0.94
        self.half = int(np.ceil(10.0 / self.cut))
        self.offsets = np.arange(-self.half + 1, self.half + 1, dtype=np.float64)
        self.weights = np.empty((3, self.offsets.size), np.float32)
        phases = (0.0, 0.0, 0.0) if self.down else (0.0, 1 / 3, 2 / 3)
        for phase, frac in enumerate(phases):
            d = frac - self.offsets
            self.weights[phase] = (self.cut * np.sinc(self.cut * d) *
                                   (0.5 + 0.5 * np.cos(np.pi * d / self.half))).astype(np.float32)
        self.buf = np.zeros(self._CAPACITY, np.float32)
        self.size = self.half
        self.base = -self.half
        self.j = 0
        self._output_capacity = int(np.ceil(self._CAPACITY / self.ratio)) + 4
        self.out = np.empty(self._output_capacity, np.float32)

    def process(self, x):
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("resampler input must be one-dimensional mono audio")
        if x.dtype != np.float32:
            raise TypeError("Polyphase3Resampler input dtype must be float32")
        if self.same:
            return x
        if x.size > self._CAPACITY - self.size:
            parts = []
            offset = 0
            while offset < x.size:
                block_n = min(x.size - offset, self._CAPACITY - self.size)
                block_out = self._process_block(x[offset:offset + block_n])
                if block_out.size:
                    # _process_block returns a borrowed work-buffer view.
                    parts.append(block_out.copy())
                offset += block_n
            return np.concatenate(parts) if parts else _EMPTY_F32
        return self._process_block(x)

    def _process_block(self, x):
        np.copyto(self.buf[self.size:self.size + x.size], x)
        self.size += x.size
        end = self.base + self.size
        jmax = int(np.floor((end - 1 - self.half) / self.ratio))
        if jmax < self.j:
            return _EMPTY_F32
        count = jmax - self.j + 1
        if self.down:
            start = 3 * self.j + int(self.offsets[0]) - self.base
            view = np.lib.stride_tricks.as_strided(
                self.buf[start:], shape=(count, self.offsets.size),
                strides=(3 * self.buf.itemsize, self.buf.itemsize), writeable=False)
            np.einsum("ij,j->i", view, self.weights[0], out=self.out[:count],
                      optimize=False, dtype=np.float32)
        else:
            first = self.j
            last = jmax
            for phase in range(3):
                j0 = first + ((phase - first) % 3)
                if j0 > last:
                    continue
                n = (last - j0) // 3 + 1
                center = j0 // 3
                start = center + int(self.offsets[0]) - self.base
                view = np.lib.stride_tricks.as_strided(
                    self.buf[start:], shape=(n, self.offsets.size),
                    strides=(self.buf.itemsize, self.buf.itemsize), writeable=False)
                np.einsum("ij,j->i", view, self.weights[phase],
                          out=self.out[j0 - first:j0 - first + 3 * n:3],
                          optimize=False, dtype=np.float32)
        self.j = jmax + 1
        result = self.out[:count]
        keep = int(np.floor(self.j * self.ratio)) - self.half + 1
        drop = keep - self.base
        if drop > 0:
            remain = self.size - drop
            np.copyto(self.buf[:remain], self.buf[drop:self.size])
            self.size = remain
            self.base += drop
        return result


def make_stream_resampler(sr_in, sr_out):
    ratio = int(sr_in) / int(sr_out)
    if ratio in (3.0, 1.0 / 3.0):
        return Polyphase3Resampler(sr_in, sr_out)
    return StreamResampler(sr_in, sr_out)


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
