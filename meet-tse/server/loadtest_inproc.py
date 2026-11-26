"""Exercise Session and the ONNX StatefulTSE without sockets or a server process."""
import argparse
import asyncio
import faulthandler
faulthandler.dump_traceback_later(20, repeat=True)
import math
import multiprocessing as mp
import statistics
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from model_adapter import Runner
from server import MAX_LAG_S, Session, _feed_measured, _load_enrollment


def _burn_cpu(stop_event, ready_event):
    ready_event.set()
    x = 0x12345678
    while not stop_event.is_set():
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF


def start_stressor():
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    ready = [ctx.Event(), ctx.Event()]
    workers = [ctx.Process(target=_burn_cpu, args=(stop, event), daemon=False) for event in ready]
    for worker in workers:
        worker.start()
    if not all(event.wait(10) for event in ready):
        stop.set()
        for worker in workers:
            worker.join(timeout=5)
        raise RuntimeError("CPU stressor processes failed to start")
    return stop, workers


def stop_stressor(stressor):
    if stressor is None:
        return
    stop, workers = stressor
    stop.set()
    for worker in workers:
        worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)


def read_audio(path, duration_s):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 48000:
        div = math.gcd(sr, 48000)
        wav = resample_poly(wav, 48000 // div, sr // div).astype(np.float32)
    total = int(round(duration_s * 48000))
    if not wav.size:
        raise ValueError("mixture WAV is empty")
    return np.tile(wav, math.ceil(total / wav.size))[:total]


def describe(values):
    if not values:
        return (0.0, 0.0, 0.0, 0.0)
    return (statistics.mean(values), max(values), float(np.percentile(values, 95)),
            float(np.percentile(values, 99)))


async def run_case(runner, pcm, chunk_samples, duration, stress):
    args = SimpleNamespace(
        sr=16000, window=2.0, hop=0.064, lookahead=0.0, xfade_ms=8.0,
        sliding=False, chunk_frames=4, max_mask=0.0, no_bypass=False,
        debug_timing=True,
    )
    state = SimpleNamespace(args=args, runner=runner, sr=16000,
                            pool=ThreadPoolExecutor(max_workers=1),
                            processing=False, bypass=False, last_rtf=None)
    sess = Session(state, 48000)
    queue = asyncio.Queue()
    sent_marks = deque()
    rows = []
    latency_ms = []
    input_samples = output_samples = messages = 0
    start = time.monotonic()
    loop = asyncio.get_running_loop()
    audio = memoryview(pcm).cast("B")

    async def producer():
        nonlocal input_samples, messages
        total_samples = len(pcm) // 2
        for sample0 in range(0, total_samples, chunk_samples):
            deadline = start + sample0 / 48000.0
            delay = deadline - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            n = min(chunk_samples, total_samples - sample0)
            payload = bytes(audio[sample0 * 2:(sample0 + n) * 2])
            sent_at = time.monotonic()
            input_samples += n
            messages += 1
            sent_marks.append((input_samples, sent_at))
            await queue.put((payload, sent_at))
        await queue.put(None)

    async def consumer():
        nonlocal output_samples
        while True:
            item = await queue.get()
            if item is None:
                break
            payload, sent_at = item
            sess.audio_s += len(payload) / 2 / sess.sr_in
            lag = (time.monotonic() - sess.t0) - sess.audio_s
            if lag < 0:
                sess.t0 = time.monotonic() - sess.audio_s
                lag = 0.0
            sess.current_lag = max(0.0, lag)
            sess.max_lag_window = max(sess.max_lag_window, sess.current_lag)
            if not args.no_bypass and lag > MAX_LAG_S and not sess.bypass:
                sess.bypass = True
                state.bypass = True
            submitted = time.perf_counter()
            state.processing = True
            try:
                out, queue_ms, worker_ms = await loop.run_in_executor(
                    state.pool, _feed_measured, sess, payload, submitted
                )
            finally:
                state.processing = False
            returned_at = time.monotonic()
            output_samples += len(out) // 2
            while sent_marks and sent_marks[0][0] <= output_samples:
                _, sample_sent_at = sent_marks.popleft()
                latency_ms.append((returned_at - sample_sent_at) * 1e3)
            row = dict(sess.last_parts_ms)
            row.update({"queue_wait": queue_ms, "worker_ms": worker_ms,
                        "server_added": queue_ms + worker_ms,
                        "chunk_samples": len(payload) // 2})
            rows.append(row)
            sess.log_window()

    await asyncio.gather(producer(), consumer())
    sess.log_window(time.monotonic() + 5.0)
    state.pool.shutdown(wait=True)
    elapsed = time.monotonic() - start
    fields = ("decode", "resample_down", "engine_push", "resample_up", "int16", "queue_wait", "server_added")
    parts = {name: describe([row.get(name, 0.0) for row in rows]) for name in fields}
    result = {
        "stressor": stress, "chunk_samples": chunk_samples,
        "messages_per_s": messages / duration,
        "chunk_avg": float(np.mean([r["chunk_samples"] for r in rows])),
        "chunk_max": max(r["chunk_samples"] for r in rows),
        "parts_ms_avg_max_p95": parts,
        "worker_busy_fraction": min(1.0, sum(r["worker_ms"] for r in rows) / (elapsed * 1000)),
        "lag_max_s": max((r["lag"] for r in rows), default=0.0),
        "lag_final_s": sess.current_lag,
        "bypass": sess.bypass,
        "input_samples": input_samples,
        "output_samples": output_samples,
        "output_ratio": output_samples / max(input_samples, 1),
        "e2e_latency_ms_avg_p95_max": describe(latency_ms)[:3] +
            ((max(latency_ms) if latency_ms else 0.0),),
        "e2e_latency_count": len(latency_ms),
        "uncounted_chunk_marks": len(sent_marks),
    }
    return result


def load_runner(args):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_flush_denormal(True)
    runner = Runner(args.ckpt, device="cpu")
    import pathlib
    enroll_bytes = pathlib.Path(args.enroll).read_bytes()
    runner.set_enrollment(_load_enrollment(enroll_bytes, 16000, 15.0))
    if args.onnx_step:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        runner.onnx_session = ort.InferenceSession(
            args.onnx_step, sess_options=opts, providers=["CPUExecutionProvider"]
        )
    return runner


async def main_async(args):
    print("loading runner", flush=True)
    runner = load_runner(args)
    print("runner ready", flush=True)
    audio = read_audio(args.mix, args.duration)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
    results = []
    for stress in (False, True):
        stressor = start_stressor() if stress else None
        try:
            for chunk_size in args.chunk_samples:
                print(f"starting stress={stress} chunk={chunk_size}", flush=True)
                result = await run_case(runner, pcm, chunk_size, args.duration, stress)
                results.append(result)
                print(result, flush=True)
        finally:
            stop_stressor(stressor)
    print("RESULTS_JSON=" + __import__("json").dumps(results), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=r"C:\Users\Thinkpad\Downloads\tse_causal_v2_best.pt")
    p.add_argument("--enroll", default=r"C:\Users\Thinkpad\Downloads\Recording.wav")
    p.add_argument("--mix", default=r"C:\Users\Thinkpad\Downloads\Record (online-voice-recorder.com).wav")
    p.add_argument("--onnx-step", default="stateful_step.onnx")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--chunk-samples", type=int, nargs="+", default=[128, 480, 960, 2048])
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
