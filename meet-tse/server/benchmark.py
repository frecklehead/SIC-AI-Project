"""Benchmark BSRNN inference across streaming window/hop choices."""
from __future__ import annotations

import argparse
import time

import numpy as np

from model_adapter import Runner


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--half", action="store_true")
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()
    runner = Runner(args.ckpt, args.device, args.half)
    sr = 16000
    runner.cond_t = runner.torch.zeros(1, 192, device=runner.device)
    print(f"device={runner.device} half={runner.half}")
    print("window_s hop_s avg_step_ms RTF approx_latency_ms")
    for window in (1.0, 1.5, 2.0):
        for hop in (0.04, 0.08, 0.16):
            n = int(window * sr)
            x = np.zeros(n, np.float32)
            runner.warmup(n)
            times = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                runner(x)
                if runner.device == "cuda":
                    runner.torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
            avg = float(np.mean(times))
            rtf = avg / (hop * 1000)
            latency = 1000 * (hop + 0.16 + 0.008)
            print(f"{window:8.2f} {hop:6.2f} {avg:12.1f} {rtf:5.2f} {latency:17.1f}")


if __name__ == "__main__":
    main()
