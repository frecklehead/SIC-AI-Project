"""Causal frame-synchronous streaming engine for TSEBSRNN."""
import time

import numpy as np
import torch
import torch.nn.functional as F


class StatefulTSE:
    """Process fixed frame chunks while retaining causal LSTM state."""

    def __init__(self, runner, chunk_frames=2, max_mask=0.0, fixed_scale=None):
        assert runner.causal and runner.cond_t is not None
        self.m, self.dev = runner.model, runner.device
        self.batched_ops = getattr(runner, "batched_ops", None)
        self.onnx_session = getattr(runner, "onnx_session", None)
        self.n, self.h, self.cf = self.m.n_fft, self.m.hop, int(chunk_frames)
        if self.cf < 1:
            raise ValueError("chunk_frames must be positive")
        self.win = self.m.window.to(self.dev)
        self.env = (self.win[:self.h] ** 2 + self.win[self.h:] ** 2).repeat(self.cf)
        with torch.inference_mode():
            self.spk = self.m.spk_enc(F.normalize(runner.cond_t.float(), dim=-1))
        self.buf = torch.zeros(self.n - self.h, device=self.dev)
        self.tail = torch.zeros(self.h, device=self.dev)
        self.states = [None] * len(self.m.layers)
        self.onnx_states = []
        if self.onnx_session is not None:
            for layer in self.m.layers:
                shape = (1, len(self.m.bands), layer.time.rnn.hidden_size)
                self.onnx_states.extend((np.zeros(shape, np.float32), np.zeros(shape, np.float32)))
        self.max_mask, self.fixed_scale = float(max_mask), fixed_scale
        self.pow = None
        self.alpha = 1.0 - float(np.exp(-self.cf * self.h / 16000.0))
        self.pending = np.zeros(0, np.float32)
        self.last_step_ms = 0.0

    def _layer(self, ly, x, st):
        B, K, T, N = x.shape
        g, b = ly.film(self.spk).chunk(2, -1)
        x = x * (1 + g[:, None, None]) + b[:, None, None]
        xr = x.reshape(B * K, T, N)
        y, st = ly.time.rnn(ly.time.norm(xr), st)
        xr = xr + ly.time.proj(y)
        x = xr.reshape(B, K, T, N).permute(0, 2, 1, 3).reshape(B * T, K, N)
        x = ly.band(x).reshape(B, T, K, N).permute(0, 2, 1, 3)
        return x, st

    def _process(self, x):
        h, n, m = self.h, self.n, self.m
        inp = torch.cat([self.buf, x])
        self.buf = inp[-(n - h):].clone()
        frames = inp.unfold(0, n, h)
        T = frames.shape[0]
        spec = torch.fft.rfft(frames * self.win, dim=-1).T[None]
        if self.fixed_scale is not None:
            scale = float(self.fixed_scale)
        else:
            p = float(x.pow(2).mean())
            self.pow = p if self.pow is None else self.pow + self.alpha * (p - self.pow)
            scale = max(self.pow ** 0.5, 1e-4)
        if self.onnx_session is not None:
            names = [i.name for i in self.onnx_session.get_inputs()]
            feeds = {
                names[0]: torch.view_as_real(spec / scale).cpu().numpy(),
                names[1]: self.spk.cpu().numpy(),
            }
            feeds.update({name: value for name, value in zip(names[2:], self.onnx_states)})
            result = self.onnx_session.run(None, feeds)
            mask_ri = torch.from_numpy(result[0]).to(self.dev)
            self.onnx_states = result[1:]
        else:
            split = self.batched_ops.band_split if self.batched_ops is not None else m.band_split
            hid = split(spec / scale) + m.spk_add(self.spk)[:, None, None]
            for i, ly in enumerate(m.layers):
                hid, self.states[i] = self._layer(ly, hid, self.states[i])
            decode = self.batched_ops.decoder if self.batched_ops is not None else m.decoder
            mask_ri = decode(hid).float().contiguous()
        mask = torch.view_as_complex(mask_ri.float().contiguous())
        if self.max_mask > 0:
            mask = mask * torch.clamp(self.max_mask / (mask.abs() + 1e-8), max=1.0)
        y = torch.fft.irfft((spec * mask)[0].T, n=n, dim=-1) * self.win
        out = F.fold(
            y.T[None], output_size=(1, (T + 1) * h), kernel_size=(1, n), stride=(1, h)
        ).reshape(-1)
        out[:h] += self.tail
        self.tail = out[T * h:].clone()
        return (out[:T * h] / self.env).cpu().numpy()

    @torch.inference_mode()
    def push(self, x):
        self.pending = np.concatenate([self.pending, np.asarray(x, dtype=np.float32)])
        step, outs = self.h * self.cf, []
        while self.pending.size >= step:
            chunk, self.pending = self.pending[:step], self.pending[step:]
            t0 = time.perf_counter()
            outs.append(self._process(torch.from_numpy(np.ascontiguousarray(chunk)).to(self.dev)))
            self.last_step_ms = (time.perf_counter() - t0) * 1e3
        return np.concatenate(outs) if outs else np.zeros(0, np.float32)
