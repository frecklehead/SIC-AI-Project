"""Compare whole-file and server-style sliding-window inference.

The streaming estimate is produced by the same resampler/SlidingWindowTSE path
used by ``server.py``; ``Runner`` is the same adapter the server uses.  It can
either consume a ready-made mixture or synthesize one from a clean target and
an interferer at several SNRs, and it reports quality against the clean target
(SI-SDR / SI-SDRi) as well as streaming-vs-whole consistency.

Examples
--------
# existing mixture
python offline_check.py --ckpt best.pt --enroll me.wav --mix meeting.wav --target me.wav

# synthesized mixtures at several SNRs + lookahead sweep
python offline_check.py --ckpt best.pt --enroll me.wav \
    --target me.wav --interferer other.wav --snr 5 0 -5 --lookahead-sweep 0 0.08 0.16 0.24
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


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2) + 1e-12))


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    n = min(len(est), len(ref))
    est, ref = est[:n].astype(np.float64), ref[:n].astype(np.float64)
    est -= est.mean()
    ref -= ref.mean()
    proj = (est @ ref) / (ref @ ref + 1e-8) * ref
    err = est - proj
    return float(10 * np.log10((proj @ proj) / (err @ err + 1e-8) + 1e-8))


def align_score(stream: np.ndarray, ref: np.ndarray, sr: int, max_s: float = 2.5):
    """Best SI-SDR allowing the stream to be delayed *or* advanced.

    SlidingWindowTSE emits audio that lags the input by (lookahead + xfade), so
    the stream must be shifted forward before comparing with the clean target.
    """
    best, best_lag = -np.inf, 0
    max_lag = min(int(max_s * sr), max(0, len(stream) - 1))
    step = max(1, sr // 1000)  # 1 ms resolution
    for lag in range(-max_lag, max_lag + 1, step):
        if lag >= 0:
            n = min(len(stream) - lag, len(ref))
            if n < sr // 2:
                continue
            score = si_sdr(stream[lag:lag + n], ref[:n])
        else:
            k = -lag
            n = min(len(stream), len(ref) - k)
            if n < sr // 2:
                continue
            score = si_sdr(stream[:n], ref[k:k + n])
        if score > best:
            best, best_lag = score, lag
    return best, best_lag


def mix_at_snr(target: np.ndarray, interferer: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(target), len(interferer))
    t, i = target[:n].copy(), interferer[:n].copy()
    i *= rms(t) / (rms(i) + 1e-12) / (10 ** (snr_db / 20.0))
    m = t + i
    peak = float(np.abs(m).max())
    if peak > 0.99:
        g = 0.99 / peak
        m, t = m * g, t * g
    return m.astype(np.float32), t.astype(np.float32)


def run_engine(fn, mix: np.ndarray, sr: int, window: float, hop: float, lookahead: float, xfade: float):
    engine = SlidingWindowTSE(fn, sr, window, hop, lookahead, xfade)
    return engine.push(mix)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--enroll", type=Path, required=True)
    p.add_argument("--mix", type=Path, help="ready-made mixture (optional)")
    p.add_argument("--target", type=Path, help="clean target, for SI-SDRi (optional)")
    p.add_argument("--interferer", type=Path, help="interferer used to synthesize mixtures")
    p.add_argument("--snr", type=float, nargs="+", default=[5.0, 0.0, -5.0],
                   help="SNRs (dB) for synthesized mixtures")
    p.add_argument("--lookahead-sweep", type=float, nargs="+", default=None,
                   help="evaluate these lookaheads (s), e.g. 0 0.08 0.16 0.24")
    p.add_argument("--out", type=Path, default=Path("streamed.wav"))
    p.add_argument("--whole-out", type=Path, default=Path("whole.wav"))
    p.add_argument("--device", default="auto")
    p.add_argument("--half", action="store_true")
    p.add_argument("--window", type=float, default=2.0)
    p.add_argument("--hop", type=float, default=0.08)
    p.add_argument("--lookahead", type=float, default=0.16)
    p.add_argument("--xfade-ms", type=float, default=8.0)
    p.add_argument("--seconds", type=float, default=None,
                   help="truncate mixture/target/interferer to N seconds (enrollment untouched)")
    args = p.parse_args()

    import torch
    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sr = int((obj.get("cfg") or {}).get("audio", {}).get("sr", 16000))
    runner = Runner(str(args.ckpt), args.device, args.half)
    enrollment = read_mono(args.enroll, sr)
    runner.set_enrollment(enrollment)

    def load_cond(path: Path) -> np.ndarray:
        x = read_mono(path, sr)
        return x[: int(args.seconds * sr)] if args.seconds else x

    if args.mix:
        conditions = [("given mixture", load_cond(args.mix), load_cond(args.target) if args.target else None)]
    elif args.target and args.interferer:
        tgt = load_cond(args.target)
        intf = load_cond(args.interferer)
        conditions = [(f"SNR {s:+.0f} dB", *mix_at_snr(tgt, intf, s)) for s in args.snr]
    else:
        raise SystemExit("provide --mix, or --target together with --interferer")

    lookaheads = args.lookahead_sweep or [args.lookahead]
    print(f"sample rate: {sr} Hz | window {args.window}s hop {args.hop}s "
          f"xfade {args.xfade_ms}ms | lookaheads {lookaheads}")

    print("\nwhole-file (oracle context):")
    print(f"{'condition':16s} {'mix SI-SDR':>11s} {'out SI-SDR':>11s} {'SI-SDRi':>9s}")
    prepared = []
    for name, mix, tgt in conditions:
        whole = runner(mix)
        prepared.append((name, mix, tgt, whole))
        if tgt is None:
            print(f"{name:16s} {'-':>11s} {'-':>11s} {'-':>9s}")
        else:
            base, out = si_sdr(mix, tgt), si_sdr(whole, tgt)
            print(f"{name:16s} {base:11.2f} {out:11.2f} {out - base:+9.2f}")

    print("\nstreaming (SlidingWindowTSE, same path as the server):")
    print(f"{'condition':16s} {'lookahead':>9s} {'out SI-SDR':>11s} {'SI-SDRi':>9s} {'vs whole':>9s} {'lag ms':>7s}")
    first = True
    for la in lookaheads:
        for name, mix, tgt, whole in prepared:
            streamed = run_engine(runner, mix, sr, args.window, args.hop, la, args.xfade_ms)
            if first:
                sf.write(args.out, streamed, sr)
                sf.write(args.whole_out, prepared[0][3], sr)
                first = False
            parity, lag = align_score(streamed, whole, sr)
            if tgt is None:
                print(f"{name:16s} {la:9.2f} {'-':>11s} {'-':>9s} {parity:9.2f} {1000 * lag / sr:7.1f}")
            else:
                # align to the clean target so algorithmic latency does not count as error
                out, _ = align_score(streamed, tgt, sr)
                base = si_sdr(mix, tgt)
                print(f"{name:16s} {la:9.2f} {out:11.2f} {out - base:+9.2f} {parity:9.2f} {1000 * lag / sr:7.1f}")
    print(f"\nwrote: {args.out} and {args.whole_out}")


if __name__ == "__main__":
    main()
