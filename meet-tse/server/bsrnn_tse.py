"""Speaker-conditioned Band-Split RNN (BSRNN) target speaker extractor.

Ported verbatim from the training library `tselib.py` (TSEBSRNN, DualPathLayer,
MaskDecoder, BandSplit, RNNBlock, make_bands) so that checkpoints saved by the
training code load and run with identical semantics:

  * waveform is divided by its std (level-independent), output multiplied back
  * 512-point Hann STFT, hop 128 -> 257 bins -> 28 non-uniform bands
  * per band: LayerNorm([Re;Im]) -> Linear(2*bins, N)
  * speaker: L2-normalised 192-d ECAPA -> 192->spk_hidden->spk_hidden (PReLU)
  * speaker vector added once to the band features, then FiLM before every
    layer's time block; parameters are "gamma, beta" applied as x*(1+g)+b
  * per layer: time BiLSTM over T, then band BiLSTM over the 28 bands, both
    with a residual projection
  * mask decoder per band: LayerNorm, Linear(N,4N), Tanh, Linear(4N, 2*bins*2),
    GLU -> interleaved (Re, Im) mask per bin, unbounded complex mask

This module contains no training/eval/data code; it is inference only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000

DEFAULT_MODEL_CFG = {
    "n_fft": 512, "hop": 128, "N": 128, "hidden": 256, "layers": 8,
    "emb_dim": 192, "spk_hidden": 256,
}


def make_bands(n_fft, sr=SR, plan=((1000, 125), (4000, 250), (8000, 500))):
    """Band edges in STFT bins: 125 Hz bands below 1 kHz, 250 Hz up to 4 kHz,
    500 Hz above, and the last band takes all leftover bins (same as training)."""
    n_bins = n_fft // 2 + 1
    hz = sr / n_fft
    bands, cur = [], 0
    for hi_hz, w_hz in plan:
        hi, w = min(int(round(hi_hz / hz)), n_bins), max(1, int(round(w_hz / hz)))
        while cur < hi:
            end = min(cur + w, hi)
            bands.append([cur, end])
            cur = end
    if cur < n_bins:
        bands[-1][1] = n_bins
    return [tuple(b) for b in bands]


class BandSplit(nn.Module):
    def __init__(self, bands, N):
        super().__init__()
        self.bands = bands
        self.norms = nn.ModuleList([nn.LayerNorm(2 * (e - s)) for s, e in bands])
        self.fcs = nn.ModuleList([nn.Linear(2 * (e - s), N) for s, e in bands])

    def forward(self, spec):  # spec [B,F,T] complex -> [B,K,T,N]
        x = torch.view_as_real(spec)  # B,F,T,2
        B, _, T, _ = x.shape
        out = []
        for (s, e), ln, fc in zip(self.bands, self.norms, self.fcs):
            z = x[:, s:e].permute(0, 2, 1, 3).reshape(B, T, -1)
            out.append(fc(ln(z)))
        return torch.stack(out, 1)


class RNNBlock(nn.Module):
    """LayerNorm -> BiLSTM -> linear projection, added as a residual."""

    def __init__(self, N, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(N)
        self.rnn = nn.LSTM(N, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * hidden, N)

    def forward(self, x):
        y, _ = self.rnn(self.norm(x))
        return x + self.proj(y)


class DualPathLayer(nn.Module):
    def __init__(self, N, hidden, spk_dim):
        super().__init__()
        self.time, self.band = RNNBlock(N, hidden), RNNBlock(N, hidden)
        self.film = nn.Linear(spk_dim, 2 * N)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, spk):  # x [B,K,T,N], spk [B,S]
        B, K, T, N = x.shape
        g, b = self.film(spk).chunk(2, -1)
        x = x * (1 + g[:, None, None]) + b[:, None, None]
        x = self.time(x.reshape(B * K, T, N)).reshape(B, K, T, N)
        x = x.permute(0, 2, 1, 3).reshape(B * T, K, N)
        x = self.band(x).reshape(B, T, K, N).permute(0, 2, 1, 3)
        return x


class MaskDecoder(nn.Module):
    def __init__(self, bands, N):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(N) for _ in bands])
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(N, 4 * N), nn.Tanh(), nn.Linear(4 * N, 4 * (e - s)), nn.GLU(dim=-1))
            for s, e in bands])
        self.bands = bands

    def forward(self, x):  # [B,K,T,N] -> [B,F,T,2] real (interleaved Re/Im)
        outs = []
        for k, ((s, e), ln, mlp) in enumerate(zip(self.bands, self.norms, self.mlps)):
            m = mlp(ln(x[:, k]))  # B,T,2*bw
            B, T, _ = m.shape
            outs.append(m.reshape(B, T, e - s, 2).permute(0, 2, 1, 3))
        return torch.cat(outs, 1)


class TSEBSRNN(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        self.n_fft, self.hop = mcfg["n_fft"], mcfg["hop"]
        N, H = mcfg["N"], mcfg["hidden"]
        self.bands = make_bands(self.n_fft)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.band_split = BandSplit(self.bands, N)
        S = mcfg["spk_hidden"]
        self.spk_enc = nn.Sequential(nn.Linear(mcfg["emb_dim"], S), nn.PReLU(), nn.Linear(S, S))
        self.spk_add = nn.Linear(S, N)
        self.layers = nn.ModuleList([DualPathLayer(N, H, S) for _ in range(mcfg["layers"])])
        self.decoder = MaskDecoder(self.bands, N)

    def forward(self, mix, emb):
        """mix [B,L] waveform, emb [B,192] ECAPA embedding -> target estimate [B,L]."""
        L = mix.shape[-1]
        scale = mix.float().std(-1, keepdim=True) + 1e-5
        x = mix.float() / scale
        with torch.autocast(device_type=mix.device.type, enabled=False):
            spec = torch.stft(x, self.n_fft, self.hop, window=self.window, return_complex=True)
        spk = self.spk_enc(F.normalize(emb.float(), dim=-1))
        h = self.band_split(spec) + self.spk_add(spk)[:, None, None]
        for layer in self.layers:
            h = layer(h, spk)
        m = self.decoder(h)
        with torch.autocast(device_type=mix.device.type, enabled=False):
            est = spec * torch.view_as_complex(m.float().contiguous())
            wav = torch.istft(est, self.n_fft, self.hop, window=self.window, length=L)
        return wav * scale


def build_model(mcfg=None):
    """Construct the trained architecture. `mcfg` is checkpoint['cfg']['model']."""
    cfg = dict(DEFAULT_MODEL_CFG)
    cfg.update(mcfg or {})
    return TSEBSRNN(cfg)
