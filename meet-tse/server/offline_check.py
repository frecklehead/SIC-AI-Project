"""Compare whole-file and server-style sliding-window inference.

The streaming estimate is produced by the same resampler/SlidingWindowTSE path
used by ``server.py``.  It is intentionally an offline diagnostic, not a second
inference implementation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from model_adapter import Runner
from resampler import StreamResampler
from stream_engine import SlidingWindowTSE


def read_mono(path: Path, sr: int) -> np.ndarray:
    wav, got = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if got != sr:
        wav = StreamResampler(got, sr).process(wav)
    return np.asarray(wav, np.float32)


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    n = min(len(est), len(ref))
    est, ref = est[:n].astype(np.float64), ref[:n].astype(np.float64)
    est -= est.mean()
    ref -= ref.mean()
    proj = (est @ ref) / (ref @ ref + 1e-8) * ref
    err = est - proj
    return float(10 * np.log10((proj @ proj) / (err @ err + 1e-8) + 1e-8))


def align_score(stream: np.ndarray, whole: np.ndarray, sr: int, max_s: float = 2.5):
    """Return best SI-SDR and lag, allowing the streaming warm-up offset."""
    best, best_lag = -np.inf, 0
    max_lag = min(int(max_s * sr), max(0, len(whole) - 1))
    for lag in range(0, max_lag + 1, max(1, sr // 100)):
        n = min(len(stream), len(whole) - lag)
        if n < sr // 2:
            continue
        score = si_sdr(stream[:n], whole[lag:lag + n])
        if score > best:
            best, best_lag = score, lag
    return best, best_lag


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--mix", type=Path, required=True)
    p.add_argument("--enroll", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("streamed.wav"))
    p.add_argument("--whole-out", type=Path, default=Path("whole.wav"))
    p.add_argument("--device", default="auto")
    p.add_argument("--half", action="store_true")
    p.add_argument("--window", type=float, default=2.0)
    p.add_argument("--hop", type=float, default=0.08)
    p.add_argument("--lookahead", type=float, default=0.16)
    p.add_argument("--xfade-ms", type=float, default=8.0)
    args = p.parse_args()

    import torch
    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sr = int((obj.get("cfg") or {}).get("audio", {}).get("sr", 16000))
    # The training checkpoint stores the authoritative audio rate.
    runner = Runner(str(args.ckpt), args.device, args.half)
    mix = read_mono(args.mix, sr)
    enrollment = read_mono(args.enroll, sr)
    runner.set_enrollment(enrollment)

    whole = runner(mix)
    engine = SlidingWindowTSE(runner, sr, args.window, args.hop, args.lookahead, args.xfade_ms)
    streamed = engine.push(mix)
    # The engine intentionally does not invent a flush protocol. Save the
    # emitted stream as-is and compare it against the matching whole-file slice.
    sf.write(args.out, streamed, sr)
    sf.write(args.whole_out, whole, sr)
    score, lag = align_score(streamed, whole, sr)
    print(f"sample rate: {sr} Hz")
    print(f"whole samples: {len(whole)}  streamed samples: {len(streamed)}")
    print(f"best aligned streaming-vs-whole SI-SDR: {score:.2f} dB at lag {lag / sr:.3f} s")
    print(f"wrote: {args.out} and {args.whole_out}")


if __name__ == "__main__":
    main()
