"""Benchmark StatefulTSE on CPU for the requested chunk/thread combinations."""
import argparse
import platform
import sys

import numpy as np

from model_adapter import Runner
from stateful_engine import StatefulTSE


def cpu_model():
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            with key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--interop-threads", type=int, default=0)
    parser.add_argument("--flush-denormal", action="store_true")
    parser.add_argument("--int8", action="store_true", help="dynamically quantize nn.Linear to qint8")
    parser.add_argument("--onnx-step", help="ONNX causal-step model with explicit LSTM states")
    args = parser.parse_args()

    import torch

    if args.interop_threads:
        torch.set_num_interop_threads(args.interop_threads)
    if args.flush_denormal:
        torch.set_flush_denormal(True)

    if not torch.cuda.is_available():
        print("device=CPU")
    else:
        print("device=CPU (CPU benchmark requested)")
    print(f"cpu={cpu_model()} torch={torch.__version__}")
    runner = Runner(args.ckpt, device="cpu")
    if not runner.causal:
        raise RuntimeError("checkpoint is not marked causal")
    # Conditioning is only used to make a valid forward pass; its value does not
    # affect the amount of work performed by the streaming network.
    runner.cond_t = torch.zeros(1, 192)
    if args.onnx_step and args.int8:
        parser.error("--onnx-step and --int8 are separate measurements")
    if args.int8:
        runner.model = torch.ao.quantization.quantize_dynamic(
            runner.model, {torch.nn.Linear}, dtype=torch.qint8
        ).eval()
        runner.batched_ops = None
        print("dynamic_int8=enabled; using original per-band module path")
    rng = np.random.default_rng(2026)
    audio = (rng.standard_normal(5 * 16000).astype(np.float32) * 0.1)

    print("threads chunk avg_ms p95_ms budget_ms RTF_avg RTF_p95")
    chunk_choices = (4,) if args.onnx_step else (2, 4, 8)
    for threads in (1, 2, 4):
        torch.set_num_threads(threads)
        runner.onnx_session = None
        if args.onnx_step:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
            runner.onnx_session = ort.InferenceSession(
                args.onnx_step, sess_options=opts, providers=["CPUExecutionProvider"]
            )
        for chunk_frames in chunk_choices:
            engine = StatefulTSE(runner, chunk_frames=chunk_frames)
            block = chunk_frames * runner.model.hop
            warm_samples = 16000
            samples = 0
            times = []
            for start in range(0, len(audio) - block + 1, block):
                engine.push(audio[start:start + block])
                samples += block
                if samples - block >= warm_samples:
                    times.append(engine.last_step_ms)
            budget = 1000.0 * block / 16000
            avg = float(np.mean(times))
            p95 = float(np.percentile(times, 95))
            print(f"{threads:7d} {chunk_frames:6d} {avg:7.2f} {p95:7.2f} "
                  f"{budget:9.2f} {avg / budget:7.3f} {p95 / budget:7.3f}")


if __name__ == "__main__":
    main()
