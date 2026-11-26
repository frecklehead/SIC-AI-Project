"""Compare causal streaming output with offline Runner output on a mixture."""
import argparse

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from model_adapter import Runner
from stateful_engine import StatefulTSE


def si_sdr(reference, estimate):
    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)
    reference -= reference.mean()
    estimate -= estimate.mean()
    projection = np.dot(estimate, reference) * reference / (np.dot(reference, reference) + 1e-12)
    noise = estimate - projection
    return 10.0 * np.log10((np.dot(projection, projection) + 1e-12) /
                           (np.dot(noise, noise) + 1e-12))


def read_mono_16k(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 16000:
        divisor = np.gcd(sr, 16000)
        wav = resample_poly(wav, 16000 // divisor, sr // divisor).astype(np.float32)
    return wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--enroll", required=True)
    parser.add_argument("--mix", required=True, help="20-second mono 16 kHz mixture WAV")
    parser.add_argument("--int8", action="store_true", help="quantize nn.Linear after the fp32 reference pass")
    parser.add_argument("--onnx-step", help="run the ONNX chunk step with explicit LSTM states")
    args = parser.parse_args()

    runner = Runner(args.ckpt, device="cpu")
    if not runner.causal:
        raise RuntimeError("checkpoint is not marked causal")
    runner.set_enrollment(read_mono_16k(args.enroll))
    wav = read_mono_16k(args.mix)
    if len(wav) < 20 * 16000:
        raise ValueError(f"expected at least 20 seconds of mixture, got {len(wav) / 16000:.2f}s")
    wav = wav[:20 * 16000]

    ref = runner(wav)
    scale = float(wav.std() + 1e-5)
    if args.int8 and args.onnx_step:
        parser.error("--int8 and --onnx-step cannot be combined")
    if args.int8:
        import torch
        runner.model = torch.ao.quantization.quantize_dynamic(
            runner.model, {torch.nn.Linear}, dtype=torch.qint8
        ).eval()
        runner.batched_ops = None
        print("dynamic_int8=enabled; comparing against the fp32 offline reference")
    if args.onnx_step:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        runner.onnx_session = ort.InferenceSession(
            args.onnx_step, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        print("ONNX step enabled; comparing against the fp32 offline reference")
    engine = StatefulTSE(runner, chunk_frames=4 if args.onnx_step else 2, fixed_scale=scale)
    blocks = [engine.push(wav[i:i + 512]) for i in range(0, len(wav), 512)]
    stream = np.concatenate(blocks)
    # The streaming engine's frame convention produces one hop of algorithmic delay.
    aligned = stream[256:]
    count = min(len(aligned), len(ref))
    if count <= 0:
        raise RuntimeError("stream produced no aligned audio")
    sf.write("offline.wav", ref, 16000)
    sf.write("stateful.wav", stream, 16000)
    start = int(0.5 * 16000)
    score = si_sdr(ref[start:count], aligned[start:count])
    print(f"SI-SDR(stream, offline)={score:.2f} dB (required >= 30 dB)")
    print(f"aligned samples={count}, fixed_scale={scale:.8f}")
    print("wrote offline.wav and stateful.wav")
    if score < 30.0:
        raise SystemExit("parity gate failed; do not integrate the stateful engine")


if __name__ == "__main__":
    main()
