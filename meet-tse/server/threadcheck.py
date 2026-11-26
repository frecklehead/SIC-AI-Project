"""Compare ONNX Runtime thread counts with the server's CPU session options."""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("onnx_step", nargs="?", default="stateful_step.onnx")
args = parser.parse_args()
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "4")

import time
import numpy as np
import onnxruntime as ort


def options(threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_cpu_mem_arena = True
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return opts


def input_shape(arg):
    return tuple(dim if isinstance(dim, int) and dim > 0 else 1 for dim in arg.shape)


for threads in (1, 2, 4):
    session = ort.InferenceSession(args.onnx_step, sess_options=options(threads),
                                   providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    feeds = {item.name: np.zeros(input_shape(item), dtype=np.float32) for item in inputs}
    # Carry explicit recurrent state between calls, matching the live step.
    for _ in range(5):
        values = session.run(None, feeds)
        for item, value in zip(inputs[2:], values[1:]):
            feeds[item.name] = value
    durations = []
    for _ in range(30):
        start = time.perf_counter()
        values = session.run(None, feeds)
        durations.append((time.perf_counter() - start) * 1000.0)
        for item, value in zip(inputs[2:], values[1:]):
            feeds[item.name] = value
    print(f"threads={threads} intra={threads} inter=1 "
          f"mean_ms={np.mean(durations):.2f} p95_ms={np.percentile(durations, 95):.2f} "
          f"max_ms={max(durations):.2f} providers={session.get_providers()}")
