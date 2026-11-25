"""Frozen SpeechBrain ECAPA-TDNN speaker encoder (192-d).

Training used the raw ECAPA output and let `TSEBSRNN` L2-normalise it, so we do
the same here. Enrollment audio is VAD-trimmed exactly as in `tselib.py` before
it reaches ECAPA. SpeechBrain is only imported when this class is instantiated,
so `--ckpt dummy` never needs it.
"""
import numpy as np

SR = 16000

# defaults match cfg["enroll"]["vad"] used during training
VAD_CFG = {"frame_ms": 25, "hop_ms": 10, "rel_db": 30.0, "abs_db": -55.0, "hang_frames": 10}


def energy_vad(x, sr=SR, frame_ms=25, hop_ms=10, rel_db=30.0, abs_db=-55.0, hang_frames=10):
    """Sample-level speech mask from frame energy (ported from tselib.py)."""
    fl, hl = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.ones(len(x), bool)
    n = 1 + (len(x) - fl) // hl
    idx = np.arange(fl)[None, :] + hl * np.arange(n)[:, None]
    e = 10 * np.log10((x[idx] ** 2).mean(1) + 1e-12)
    thr = max(abs_db, np.percentile(e, 95) - rel_db)
    act = (e > thr).astype(np.float32)
    act = np.convolve(act, np.ones(2 * hang_frames + 1), mode="same") > 0
    mask = np.zeros(len(x), bool)
    m = np.repeat(act, hl)[: len(x)]
    mask[: len(m)] = m
    return mask


class SpeakerEncoder:
    def __init__(self, device="cpu", savedir=None):
        import os
        import torch
        self.torch = torch
        self.device = device
        savedir = savedir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_ecapa")
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception:  # older speechbrain
            from speechbrain.pretrained import EncoderClassifier
        self.m = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir=savedir,
            run_opts={"device": device})
        self.m.eval()

    def embed(self, wav, vad_cfg=None, max_s=20.0):
        """Return a raw 192-d float32 ECAPA embedding for a mono 16 kHz waveform."""
        wav = np.asarray(wav, np.float32).reshape(-1)
        cfg = dict(VAD_CFG)
        cfg.update(vad_cfg or {})
        v = wav[energy_vad(wav, **cfg)]
        if len(v) < 0.5 * SR:
            raise ValueError(
                f"only {len(v) / SR:.2f}s of speech found in the enrollment clip; record a longer one")
        v = v[: int(max_s * SR)]
        with self.torch.inference_mode():
            t = self.torch.from_numpy(v)[None].to(self.device)
            e = self.m.encode_batch(t).reshape(-1).float().cpu().numpy()
        return e
