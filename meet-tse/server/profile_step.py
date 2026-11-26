"""Profile one CPU StatefulTSE step and report operator counts/self time."""
import argparse

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from model_adapter import Runner
from stateful_engine import StatefulTSE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    runner = Runner(args.ckpt, device="cpu")
    if not runner.causal:
        raise RuntimeError("checkpoint is not causal")
    runner.cond_t = torch.zeros(1, 192)
    engine = StatefulTSE(runner, chunk_frames=4)
    audio = np.zeros(4 * runner.model.hop, np.float32)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        engine.push(audio)
    events = sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)
    print("Top operators by self CPU time:")
    for event in events[:10]:
        print(f"{event.key:40s} self_ms={event.self_cpu_time_total / 1000:8.3f} "
              f"calls={event.count}")
    print(f"unique operator entries={len(events)} total operator calls={sum(e.count for e in events)}")


if __name__ == "__main__":
    main()
