"""CPU inference benchmark with hygiene checks.

Prints median/p95 step time and RTF for each streaming window/hop.  RTF is
step_time / hop_time; real-time needs RTF < 1 and we target < 0.7.

Run:
    python bench.py --ckpt ../../src/checkpoints/tse_bsrnn_big_best.pt --threads 6
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from model_adapter import Runner


def mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return float("nan")


def load_avg() -> float:
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def hygiene():
    cores = os.cpu_count() or 1
    mem = mem_available_gb()
    load = load_avg()
    print(f"system: cores={cores}  available_ram={mem:.1f} GB  load1={load:.2f}")
    if mem < 4.0:
        print(f"WARNING: only {mem:.1f} GB RAM available (< 4 GB); timings will be noisy", file=sys.stderr)
    if load > cores:
        print(f"WARNING: load1 {load:.2f} > cores {cores}; timings will be noisy", file=sys.stderr)
    return mem, load, cores


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--half", action="store_true")
    p.add_argument("--threads", type=int, default=0,
                   help="torch intra-op threads; 0 leaves the current value")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--windows", type=float, nargs="+", default=[1.0, 1.5])
    p.add_argument("--hops", type=float, nargs="+", default=[0.16, 0.32, 0.5])
    p.add_argument("--lookahead", type=float, default=0.16)
    args = p.parse_args()

    hygiene()
    import torch
    if args.threads > 0:
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(1)  # one DAG at a time; avoid oversubscription
        except RuntimeError:
            pass
    print(f"torch={torch.__version__}  intra_threads={torch.get_num_threads()}  "
          f"interop_threads={torch.get_num_interop_threads()}")

    runner = Runner(args.ckpt, args.device, args.half)
    sr = 16000
    runner.cond_t = torch.zeros(1, 192, device=runner.device)
    runner.model.requires_grad_(False)
    print(f"device={runner.device} half={runner.half} training={runner.model.training} "
          f"requires_grad={any(q.requires_grad for q in runner.model.parameters())}")

    rng = np.random.default_rng(0)
    print(f"steps={args.steps} warmup={args.warmup}  (times in ms)")
    print("window  hop    median    p95    RTF_med RTF_p95  latency_ms")
    for window in args.windows:
        n = int(round(window * sr))
        x = (rng.standard_normal(n).astype(np.float32) * 0.1)
        # A step's cost depends on the window, not on how often the server
        # calls it.  Time 30 steps once per window and derive RTF per hop.
        for _ in range(args.warmup):
            runner(x)
        times = []
        for _ in range(args.steps):
            t0 = time.perf_counter()
            runner(x)
            times.append((time.perf_counter() - t0) * 1e3)
        med = float(np.median(times))
        p95 = float(np.percentile(times, 95))
        for hop in args.hops:
            budget = hop * 1000.0
            print(f"{window:5.2f} {hop:5.2f} {med:9.1f} {p95:7.1f} "
                  f"{med / budget:7.2f} {p95 / budget:7.2f} "
                  f"{1000 * (hop + args.lookahead + 0.008):9.1f}")
    print(f"inference_mode_last_call={runner.last_inference_mode}")
    assert runner.last_inference_mode, "Runner did not use torch.inference_mode()"
    assert not any(q.requires_grad for q in runner.model.parameters()), "autograd is enabled"


if __name__ == "__main__":
    main()
