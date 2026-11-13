"""Sliding-window streaming wrapper for ANY (even non-causal) TSE model.

Every `hop` seconds the model is run on the most recent `window` seconds.
Only the slice that ends `lookahead` seconds before the newest sample is
emitted (non-causal models are weakest at the right edge, so we give them
some future context), and consecutive slices are cross-faded.

    algorithmic latency ~= hop + lookahead + xfade        (+ model compute time)
    compute per second  ~= (1/hop) * model_time(window)    -> need RTF < 1

If your model is truly causal/streaming, set lookahead=0 and a short window.
"""
import numpy as np


class SlidingWindowTSE:
    def __init__(self, fn, sr, window_s=2.0, hop_s=0.08, lookahead_s=0.16, xfade_ms=8.0):
        self.fn = fn                              # fn(np.float32[W]) -> np.float32[W]
        self.hop = int(round(hop_s * sr))
        self.L = int(round(lookahead_s * sr))
        self.xf = int(round(xfade_ms * 1e-3 * sr))
        self.W = max(int(round(window_s * sr)), self.hop + self.L + self.xf)
        self.a = self.W - self.L - self.xf - self.hop   # start of emitted slice inside window
        assert self.a >= 0
        self.hist = np.zeros(self.W, np.float32)
        self.pending = np.zeros(0, np.float32)
        self.tail = None
        self.ramp = np.linspace(0.0, 1.0, self.xf, endpoint=False, dtype=np.float32)
        self.last_step_ms = 0.0

    @property
    def algorithmic_latency_ms(self):
        return 1000.0 * (self.hop + self.L + self.xf) / 16000.0  # indicative only

    def push(self, x: np.ndarray) -> np.ndarray:
        import time
        self.pending = np.concatenate([self.pending, x.astype(np.float32, copy=False)])
        outs = []
        while self.pending.size >= self.hop:
            chunk, self.pending = self.pending[: self.hop], self.pending[self.hop:]
            self.hist = np.concatenate([self.hist[self.hop:], chunk])
            t0 = time.perf_counter()
            est = self.fn(self.hist)
            self.last_step_ms = (time.perf_counter() - t0) * 1e3
            seg = np.array(est[self.a: self.a + self.hop + self.xf], dtype=np.float32)
            if self.tail is not None and self.xf:
                seg[: self.xf] = self.tail * (1.0 - self.ramp) + seg[: self.xf] * self.ramp
            outs.append(seg[: self.hop].copy())
            self.tail = seg[self.hop:].copy()
        return np.concatenate(outs) if outs else np.zeros(0, np.float32)
