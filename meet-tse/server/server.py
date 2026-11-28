#!/usr/bin/env python3
"""Local real-time TSE server for the Meet extension.

    python server.py --ckpt /path/tse_bsrnn_big_best.pt --enroll my_voice.wav
    python server.py --ckpt dummy            # pipeline test, no model

Protocol (ws://127.0.0.1:8765/ws), one websocket per processed mic track:
  client -> {"type":"start","sampleRate":48000}      text
  server -> {"type":"ready"} | {"type":"error","msg":...}
  client -> int16-LE mono PCM chunks                  binary
  server -> int16-LE mono PCM chunks (same rate)      binary
HTTP: GET /status, POST /enroll (body = audio file: wav/flac/ogg).

If --token is set, /ws requires ?token=... and /enroll requires either
"Authorization: Bearer <token>" or "X-TSE-Token: <token>".
"""
import argparse
import asyncio
import collections
import hmac
import io
import json
import logging
import os
import re
import sys
import statistics
import threading
import time

# Torch, BLAS and ORT read these during import. Parse just the thread controls
# before importing any numerical or ML package.
_thread_parser = argparse.ArgumentParser(add_help=False)
_thread_parser.add_argument("--threads", type=int, default=1)
_thread_parser.add_argument("--interop-threads", type=int, default=1)
_thread_boot, _ = _thread_parser.parse_known_args(sys.argv[1:])
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, str(max(1, _thread_boot.threads)))

import torch
torch.set_num_threads(max(1, _thread_boot.threads))
torch.set_num_interop_threads(1)

import numpy as np
from aiohttp import web, WSMsgType, ClientConnectionResetError

from model_adapter import Runner
from backlog_policy import BacklogPolicy
from resampler import make_stream_resampler
from stream_engine import SlidingWindowTSE
from stateful_engine import StatefulTSE

log = logging.getLogger("tse")
HERE = os.path.dirname(os.path.abspath(__file__))
ENROLL_FILE = os.path.join(HERE, "enrollment.wav")


def make_ort_options(ort, threads, ort_spin=False):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(threads)
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_cpu_mem_arena = True
    opts.add_session_config_entry("session.intra_op.allow_spinning", "1" if ort_spin else "0")
    return opts


def onnx_step_mismatch(session, runner):
    """Return why an exported ONNX step does not match this checkpoint."""
    layers = len(runner.model.layers)
    expected_inputs = ["spec_ri", "spk"]
    expected_outputs = ["mask_ri"]
    for i in range(layers):
        expected_inputs.extend((f"h{i}", f"c{i}"))
        expected_outputs.extend((f"hout{i}", f"cout{i}"))
    inputs, outputs = session.get_inputs(), session.get_outputs()
    actual_inputs = [item.name for item in inputs]
    actual_outputs = [item.name for item in outputs]
    if actual_inputs != expected_inputs:
        return (f"ONNX inputs {actual_inputs} do not match this checkpoint's expected "
                f"inputs {expected_inputs}")
    if actual_outputs != expected_outputs:
        return (f"ONNX outputs {actual_outputs} do not match this checkpoint's expected "
                f"outputs {expected_outputs}")
    bands = len(runner.model.bands)
    for i, layer in enumerate(runner.model.layers):
        expected = (1, bands, layer.time.rnn.hidden_size)
        for name in (f"h{i}", f"c{i}"):
            shape = inputs[actual_inputs.index(name)].shape
            if len(shape) != len(expected) or any(
                isinstance(got, int) and got > 0 and got != want
                for got, want in zip(shape, expected)
            ):
                return f"ONNX input {name} has shape {shape}; checkpoint expects {expected}"
    return None


def runtime_thread_info():
    try:
        import psutil
        return f"process_threads={psutil.Process().num_threads()}"
    except ImportError:
        return f"threading.active_count={threading.active_count()} (psutil unavailable)"


def backend_summary(runner, args):
    backend = "onnx" if getattr(runner, "onnx_session", None) is not None else "torch"
    if backend == "onnx":
        import onnxruntime as ort
        version, providers = ort.__version__, runner.onnx_session.get_providers()
    else:
        version, providers = "unloaded", []
    return (f"backend={backend} onnxruntime={version} providers={providers} "
            f"intra={args.threads} inter=1 ort_spin={int(args.ort_spin)} "
            f"cpu_mem_arena=1 execution=sequential graph_opt=all "
            f"torch_intra={torch.get_num_threads()} torch_interop=1 "
            f"OMP={os.environ.get('OMP_NUM_THREADS')} MKL={os.environ.get('MKL_NUM_THREADS')} "
            f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} "
            f"NUMEXPR={os.environ.get('NUMEXPR_NUM_THREADS')} {runtime_thread_info()}")


class RedactTokenFilter(logging.Filter):
    """Keep ?token=... out of aiohttp access logs."""

    _token_re = re.compile(r"(token=)[^&\s\"]+")

    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if "token=" in msg:
            record.msg = self._token_re.sub(r"\1***", msg)
            record.args = ()
        return True


def _expected_token(args) -> str:
    """Token configured on the CLI, or an empty string when auth is disabled."""
    return str(getattr(args, "token", "") or "")


def _token_match(supplied, expected: str) -> bool:
    # Encode so tokens containing non-ASCII characters are handled safely.
    return hmac.compare_digest(str(supplied).encode("utf-8"), expected.encode("utf-8"))


def _ws_token_ok(request, args) -> bool:
    expected = _expected_token(args)
    if not expected:
        return True
    supplied = request.query.get("token", "")
    return _token_match(supplied, expected)


def _enroll_token_ok(request, args) -> bool:
    expected = _expected_token(args)
    if not expected:
        return True
    auth = request.headers.get("Authorization", "")
    supplied = auth[7:].strip() if auth[:7].lower() == "bearer " else ""
    if not supplied:
        supplied = request.headers.get("X-TSE-Token", "").strip()
    return _token_match(supplied, expected)


class State:
    def __init__(self, args, runner):
        self.args, self.runner, self.sr = args, runner, args.sr
        self.sessions = set()
        self.next_session_id = 1
        self.last_rtf = None
        try:
            import psutil
            self.process = psutil.Process()
            self.process.cpu_percent(None)  # prime the interval counter
        except ImportError:
            self.process = None


class Session:
    def __init__(self, st: State, sr_in: int, sid=0, port="?"):
        self.st, self.sr_in = st, sr_in
        self.id, self.port = sid, port
        self.down = make_stream_resampler(sr_in, st.sr)
        self.up = make_stream_resampler(st.sr, sr_in)
        self.reset_engine()
        self.bypass = False
        self.passthrough = False
        self.processing = False
        self.policy = BacklogPolicy(st.args.max_lag)
        self._condition = threading.Condition()
        self._input = collections.deque()
        self._input_samples = 0
        self.total_input_samples = 0
        self._consumed_samples = 0
        self._output_samples = 0
        self._startup_output = True
        self._stop = False
        self._loop = None
        self._outq = None
        self._worker = None
        self._last_drop_log = 0.0
        self.samples_dropped = 0
        self.last_log = time.monotonic()
        self.debug_timing = bool(getattr(st.args, "debug_timing", False))
        self.timing_rows = collections.deque() if self.debug_timing else None
        self.timing_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        self.window_input_samples = 0
        self.window_output_samples = 0
        self.window_message_sizes = []
        self.message_sizes = []
        self.window_step_times = []
        self.window_engine_input_samples = 0
        self.last_parts_ms = None
        self.last_step_ran = False
        self.dump_dir = getattr(st.args, "dump_audio", None) if sid > 0 else None
        self._dump_lock = threading.Lock()
        self._dump_files = None
        self._dump_paths = {}
        self._dump_totals = {"input": 0, "engine_input": 0, "output": 0}
        self._last_dump_flush = time.monotonic()
        if self.dump_dir:
            self._open_audio_dump()

    def reset_engine(self):
        a = self.st.args
        r = self.st.runner
        self.stateful = bool(r.causal and not r.dummy and not a.sliding)
        if self.stateful:
            self.eng = StatefulTSE(r, a.chunk_frames, a.max_mask)
            log.info("session engine: StatefulTSE (chunk_frames=%d)", a.chunk_frames)
        else:
            self.eng = SlidingWindowTSE(r, self.st.sr, a.window, a.hop, a.lookahead, a.xfade_ms)
            log.info("session engine: SlidingWindowTSE")

    def start_worker(self, loop, outq):
        self._loop, self._outq = loop, outq
        self._worker = threading.Thread(target=self._run_worker,
                                        name=f"tse-session-{self.id}", daemon=True)
        self._worker.start()

    def append(self, pcm: bytes, arrival=None):
        pcm = pcm[:len(pcm) // 2 * 2]
        if not pcm:
            return
        with self._condition:
            self._input.append([pcm, time.monotonic() if arrival is None else arrival])
            sample_count = len(pcm) // 2
            self._input_samples += sample_count
            self.total_input_samples += sample_count
            self._condition.notify()
        with self.metrics_lock:
            self.window_input_samples += sample_count
            self.window_message_sizes.append(sample_count)
            self.message_sizes.append(sample_count)
        self._dump_input(pcm)

    def _open_audio_dump(self):
        opened = {}
        try:
            import soundfile as sf
            os.makedirs(self.dump_dir, exist_ok=True)
            prefix = os.path.join(self.dump_dir, f"session{self.id}")
            self._dump_paths = {
                "input": prefix + "_input_as_received.wav",
                "engine_input": prefix + "_engine_input_16k.wav",
                "output": prefix + "_output.wav",
            }
            opened["input"] = sf.SoundFile(self._dump_paths["input"], mode="w",
                                            samplerate=self.sr_in, channels=1, format="WAV",
                                            subtype="PCM_16")
            opened["engine_input"] = sf.SoundFile(
                self._dump_paths["engine_input"], mode="w", samplerate=self.st.sr,
                channels=1, format="WAV", subtype="FLOAT")
            opened["output"] = sf.SoundFile(self._dump_paths["output"], mode="w",
                                            samplerate=self.sr_in, channels=1, format="WAV",
                                            subtype="PCM_16")
            self._dump_files = opened
        except Exception:
            for audio_file in opened.values():
                try:
                    audio_file.close()
                except Exception:
                    pass
            self._dump_files = None
            log.exception("session id=%d could not open audio dump files in %s",
                          self.id, self.dump_dir)

    def _dump_write(self, name, values):
        if self._dump_files is None or not len(values):
            return
        with self._dump_lock:
            if self._dump_files is None:
                return
            try:
                self._dump_files[name].write(values)
                self._dump_totals[name] += len(values)
                now = time.monotonic()
                if now - self._last_dump_flush >= 30.0:
                    for audio_file in self._dump_files.values():
                        audio_file.flush()
                    self._last_dump_flush = now
            except Exception:
                log.exception("session id=%d audio dump write failed (%s)", self.id, name)

    def _dump_input(self, pcm):
        if self._dump_files is not None:
            self._dump_write("input", np.frombuffer(pcm, dtype="<i2"))

    def _dump_engine_input(self, audio):
        if audio.size:
            self._dump_write("engine_input", np.asarray(audio, dtype=np.float32))
            with self.metrics_lock:
                self.window_engine_input_samples += int(audio.size)

    def record_output_sent(self, pcm):
        if not pcm:
            return
        sample_count = len(pcm) // 2
        with self.metrics_lock:
            self.window_output_samples += sample_count
        if self._dump_files is not None:
            self._dump_write("output", np.frombuffer(pcm, dtype="<i2"))

    def close_audio_dump(self):
        if not self.dump_dir:
            return
        with self._dump_lock:
            files, self._dump_files = self._dump_files, None
            if files is not None:
                try:
                    for audio_file in files.values():
                        audio_file.flush()
                        audio_file.close()
                except Exception:
                    log.exception("session id=%d failed to finalize audio dumps", self.id)
        median_samples = statistics.median(self.message_sizes) if self.message_sizes else 0
        log.info("audio dump session id=%d declared_sampleRate=%d "
                 "median_message_samples=%.1f input_samples=%d "
                 "engine_input_16k_samples=%d output_samples=%d files=%s",
                 self.id, self.sr_in, median_samples,
                 self._dump_totals["input"], self._dump_totals["engine_input"],
                 self._dump_totals["output"], self._dump_paths)

    def set_passthrough(self):
        with self._condition:
            self.passthrough = True
            self._condition.notify()

    @property
    def buffer_fill_ms(self):
        with self._condition:
            return self._input_samples * 1000.0 / max(1, self.sr_in)

    def stop(self):
        with self._condition:
            self._stop = True
            self._condition.notify_all()

    def _pop_samples(self, count):
        parts = []
        while count and self._input:
            data, arrival = self._input[0]
            n = len(data) // 2
            take = min(count, n)
            parts.append((data[:take * 2], arrival))
            if take == n:
                self._input.popleft()
            else:
                self._input[0] = [data[take * 2:], arrival + take / self.sr_in]
            self._input_samples -= take
            count -= take
        return parts

    def _queue_output(self, data):
        if data:
            self._output_samples += len(data) // 2
            self._loop.call_soon_threadsafe(self._outq.put_nowait, data)

    def _queue_aligned(self, data):
        remaining = max(0, self._consumed_samples - self._output_samples)
        data = data[:remaining * 2]
        self._queue_output(data)

    def _run_worker(self):
        if self.stateful:
            step_samples = max(1, int(round(self.eng.cf * self.st.runner.model.hop *
                                            self.sr_in / self.st.sr)))
        else:
            step_samples = max(1, int(round(self.st.args.hop * self.sr_in)))
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._stop or
                                          ((self.passthrough or self.bypass) and self._input_samples > 0) or
                                          self._input_samples >= step_samples)
                if self._stop and not self._input:
                    break
                if not self._input:
                    continue
                now = time.monotonic()
                if self._input:
                    oldest = self._input[0][1]
                    n_drop = (self.policy.samples_to_drop(oldest, self._input_samples, self.sr_in)
                              if not self.st.args.no_bypass and not self.passthrough else 0)
                else:
                    n_drop = 0
                if n_drop:
                    remaining_drop = n_drop
                    self._consumed_samples += n_drop
                    self.samples_dropped += n_drop
                    if self.stateful:
                        self.eng.tail.zero_()
                    while remaining_drop:
                        drop_batch = self._pop_samples(min(remaining_drop, step_samples))
                        raw = b"".join(part for part, _ in drop_batch)
                        self._queue_aligned(raw)
                        remaining_drop -= len(raw) // 2
                    if now - self._last_drop_log >= 5.0:
                        log.warning("session id=%d dropped %d input samples to recover backlog",
                                    self.id, n_drop)
                        self._last_drop_log = now
                if self._input_samples < step_samples and not self._stop:
                    continue
                passthrough = self.passthrough or self.bypass
                n_take = min(self._input_samples, step_samples)
                parts = self._pop_samples(n_take)
                self._consumed_samples += n_take
                batch = b"".join(part for part, _ in parts)
            try:
                self.processing = True
                if passthrough:
                    out = batch
                else:
                    result = self.feed(batch)
                    if self.debug_timing:
                        out, parts = result
                        if self.last_step_ran:
                            with self.timing_lock:
                                self.timing_rows.append(parts)
                    else:
                        out = result
                    if self.stateful and not passthrough:
                        produced = len(out) // 2
                        if getattr(self, "_startup_output", True):
                            if produced == 0:
                                out = bytes(len(batch))
                            else:
                                if produced < n_take:
                                    out = bytes((n_take - produced) * 2) + out
                                elif produced > n_take:
                                    out = out[:n_take * 2]
                                self._startup_output = False
                self._queue_aligned(out)
            except Exception:
                log.exception("session id=%d inference failed; using raw passthrough", self.id)
                self.bypass = True
                self._startup_output = False
                self._queue_aligned(batch)
            finally:
                self.processing = False
        if not self.bypass and not self.passthrough:
            for _ in range(4):
                missing = self.total_input_samples - self._output_samples
                if missing <= 0:
                    break
                flush_samples = max(1, step_samples)
                try:
                    result = self.feed(bytes(flush_samples * 2))
                    out = result[0] if self.debug_timing else result
                    self._queue_output(out[:missing * 2])
                except Exception:
                    break
        missing = self.total_input_samples - self._output_samples
        if missing > 0:
            self._queue_output(bytes(missing * 2))
        self._loop.call_soon_threadsafe(self._outq.put_nowait, None)

    def feed(self, pcm: bytes) -> bytes:
        pcm = pcm[: len(pcm) // 2 * 2]
        self.last_step_ran = False
        if self.bypass:
            return pcm
        if self.debug_timing:
            t0 = time.perf_counter()
            x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            t1 = time.perf_counter()
            x16 = self.down.process(x)
            t2 = time.perf_counter()
            y16 = self.eng.push(x16)
            t3 = time.perf_counter()
            if y16.size and self.eng.last_step_ms:
                self.last_step_ran = True
                with self.metrics_lock:
                    self.window_step_times.append(self.eng.last_step_ms)
            y = self.up.process(y16)
            t4 = time.perf_counter()
            out = (np.clip(y, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            t5 = time.perf_counter()
            self._dump_engine_input(x16)
            self.last_parts_ms = {
                "decode": (t1 - t0) * 1e3,
                "resample_down": (t2 - t1) * 1e3,
                "engine_push": (t3 - t2) * 1e3,
                "resample_up": (t4 - t3) * 1e3,
                "int16": (t5 - t4) * 1e3,
            }
            return out, self.last_parts_ms
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        x16 = self.down.process(x)
        y16 = self.eng.push(x16)
        if y16.size and self.eng.last_step_ms:
            self.last_step_ran = True
            with self.metrics_lock:
                self.window_step_times.append(self.eng.last_step_ms)
        self._dump_engine_input(x16)
        y = self.up.process(y16)
        return (np.clip(y, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    def log_window(self, now=None):
        now = time.monotonic() if now is None else now
        elapsed = now - self.last_log
        if elapsed < 5.0:
            return
        with self.metrics_lock:
            steps = self.window_step_times
            self.window_step_times = []
            input_samples, self.window_input_samples = self.window_input_samples, 0
            output_samples, self.window_output_samples = self.window_output_samples, 0
            engine_samples, self.window_engine_input_samples = self.window_engine_input_samples, 0
            message_sizes, self.window_message_sizes = self.window_message_sizes, []
            dropped, self.samples_dropped = self.samples_dropped, 0
        messages = len(message_sizes)
        median_message = statistics.median(message_sizes) if message_sizes else 0.0
        input_rate = input_samples / max(self.sr_in * elapsed, 1e-9)
        output_rate = output_samples / max(self.sr_in * elapsed, 1e-9)
        observed_rate = input_samples / max(elapsed, 1e-9)
        rate_anomaly = not 0.9 <= input_rate <= 1.1
        if steps:
            avg, mx = float(np.mean(steps)), max(steps)
            p95 = float(np.percentile(steps, 95))
            budget = (self.eng.cf * self.st.runner.model.hop / self.st.sr * 1e3
                      if self.stateful else self.st.args.hop * 1e3)
            self.st.last_rtf = avg / max(budget, 1e-9)
            log.info("model step session=%d avg %.1f ms / p95 %.1f ms / max %.1f ms "
                     "(budget %.1f ms) RTF=%.2f",
                     self.id, avg, p95, mx, budget, self.st.last_rtf)
        if self.debug_timing:
            with self.timing_lock:
                rows = list(self.timing_rows)
                self.timing_rows.clear()
            values = {key: [row[key] for row in rows if key in row]
                      for key in ("decode", "resample_down", "engine_push", "resample_up", "int16")}
            def stat(key):
                data = values[key]
                return f"{np.mean(data):.2f}/{max(data):.2f}ms" if data else "n/a"
            log.info("debug timing id=%d wall_s=%.1f steps/s=%.2f "
                     "step_avg/p95/max_ms=%s/%s/%s "
                     "input_audio_s_per_wall_s=%.3f output_audio_s_per_wall_s=%.3f "
                     "declared_sampleRate=%d observed_effective_sampleRate=%.1f "
                     "INPUT_RATE_ANOMALY=%s median_message_samples=%.1f messages/s=%.1f "
                     "engine_input_16k_samples=%d dropped_samples=%d "
                     "decode/down/engine/up/int16_avg/max=%s/%s/%s/%s/%s",
                     self.id, elapsed, len(steps) / elapsed,
                     f"{np.mean(steps):.2f}" if steps else "n/a",
                     f"{np.percentile(steps,95):.2f}" if steps else "n/a",
                     f"{max(steps):.2f}" if steps else "n/a",
                     input_rate, output_rate, self.sr_in, observed_rate,
                     int(rate_anomaly), median_message, messages / elapsed,
                     engine_samples, dropped,
                     stat("decode"), stat("resample_down"), stat("engine_push"),
                     stat("resample_up"), stat("int16"))
        self.last_log = now


@web.middleware
async def cors(request, handler):
    resp = web.Response() if request.method == "OPTIONS" else await handler(request)
    resp.headers.update({
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Private-Network": "true",
    })
    return resp


async def status(request):
    st: State = request.app["st"]
    a = st.args
    return web.json_response({
        "ok": True, "enrolled": st.runner.ready, "device": st.runner.device, "dummy": st.runner.dummy,
        "processing": any(s.processing for s in st.sessions),
        "bypass": any(s.bypass for s in st.sessions), "rtf": st.last_rtf,
        "sr": st.sr, "window_s": a.window, "hop_s": a.hop, "lookahead_s": a.lookahead,
        "engine": "stateful" if (st.runner.causal and not st.runner.dummy and not a.sliding) else "sliding",
        "chunk_frames": a.chunk_frames,
        "backend": "onnx" if getattr(st.runner, "onnx_session", None) is not None else "torch",
        "active_sessions": len(st.sessions),
        "approx_latency_ms": round(16 * a.chunk_frames + 16, 1)
        if (st.runner.causal and not st.runner.dummy and not a.sliding)
        else round(1000 * (a.hop + a.lookahead + a.xfade_ms / 1000), 1),
    })


def _load_enrollment(data: bytes, sr_model: int, max_s: float) -> np.ndarray:
    import soundfile as sf
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    wav = make_stream_resampler(sr, sr_model).process(wav)
    return wav[: int(max_s * sr_model)]


async def enroll(request):
    st: State = request.app["st"]
    if not _enroll_token_ok(request, st.args):
        log.warning("rejected enrollment upload: bad token")
        return web.json_response({"ok": False, "msg": "unauthorized"}, status=401)
    data = await request.read()
    try:
        wav = _load_enrollment(data, st.sr, st.args.enroll_max)
    except Exception as e:  # noqa
        return web.json_response({"ok": False, "msg": f"could not decode audio: {e}"}, status=400)
    if wav.size < st.sr * 2:
        return web.json_response({"ok": False, "msg": "enrollment too short (need >= 2 s, ideally 5-15 s)"}, status=400)
    try:
        await asyncio.to_thread(st.runner.set_enrollment, wav)
    except Exception as e:  # noqa: broad-except: surface encoder/download errors in the popup
        log.exception("enrollment failed")
        return web.json_response({"ok": False, "msg": f"speaker encoder failed: {e}"}, status=500)
    import soundfile as sf
    sf.write(ENROLL_FILE, wav, st.sr)
    log.info("enrolled %.1f s of audio", wav.size / st.sr)
    return web.json_response({"ok": True, "seconds": round(wav.size / st.sr, 1)})


async def ws_handler(request):
    st: State = request.app["st"]
    if not _ws_token_ok(request, st.args):
        log.warning("rejected websocket: bad token")
        return web.Response(status=401, text="unauthorized")
    ws = web.WebSocketResponse(max_msg_size=2 ** 22)
    await ws.prepare(request)
    loop = asyncio.get_running_loop()
    sess = None
    writer = None

    async def write_output(session, outq):
        while True:
            chunk = await outq.get()
            if chunk is None:
                return
            try:
                await ws.send_bytes(chunk)
                session.record_output_sent(chunk)
            except (ConnectionResetError, ClientConnectionResetError, RuntimeError):
                session.stop()
                return

    async def finish_session(session, writer_task):
        session.stop()
        if session._worker:
            await asyncio.to_thread(session._worker.join)
        if writer_task:
            await asyncio.gather(writer_task, return_exceptions=True)
        session.close_audio_dump()
        st.sessions.discard(session)
        log.info("session closed id=%d", session.id)

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    cfg = json.loads(msg.data)
                except (json.JSONDecodeError, TypeError):
                    log.warning("ignoring non-JSON text frame (%d bytes)", len(msg.data))
                    continue
                if cfg.get("type") == "start":
                    if not st.runner.ready:
                        await ws.send_json({"type": "error", "msg": "no enrollment audio uploaded yet"})
                        continue
                    if sess is not None:
                        await finish_session(sess, writer)
                        sess, writer = None, None
                    peer = request.transport.get_extra_info("peername")
                    remote_port = peer[1] if peer and len(peer) > 1 else "?"
                    sid = st.next_session_id
                    st.next_session_id += 1
                    if st.sessions:
                        log.warning("second websocket session starting; active=%d", len(st.sessions))
                        if st.args.single_session:
                            for old in list(st.sessions):
                                old.set_passthrough()
                    sess = Session(st, int(cfg["sampleRate"]), sid, remote_port)
                    st.sessions.add(sess)
                    outq = asyncio.Queue()
                    sess.start_worker(loop, outq)
                    writer = asyncio.create_task(write_output(sess, outq))
                    log.info("session started id=%d port=%s sampleRate=%d active=%d %s",
                             sid, remote_port, sess.sr_in, len(st.sessions),
                             backend_summary(st.runner, st.args))
                    await ws.send_json({"type": "ready"})
            elif msg.type == WSMsgType.BINARY and sess is not None:
                sess.append(msg.data, time.monotonic())
                sess.log_window()
    finally:
        if sess is not None:
            await finish_session(sess, writer)
    return ws


def make_app(args, runner):
    app = web.Application(middlewares=[cors])
    app["st"] = State(args, runner)
    app.add_routes([web.get("/ws", ws_handler), web.get("/status", status),
                    web.post("/enroll", enroll), web.route("OPTIONS", "/{tail:.*}", status)])
    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to .pt checkpoint, or 'dummy'")
    p.add_argument("--enroll", help="enrollment audio of the target speaker (5-15 s)")
    p.add_argument("--sr", type=int, default=16000, help="model sample rate")
    p.add_argument("--window", type=float, default=2.0, help="context fed to the model (s)")
    p.add_argument("--hop", type=float, default=0.08, help="how often the model runs (s)")
    p.add_argument("--sliding", action="store_true", help="use the legacy sliding-window engine")
    p.add_argument("--chunk-frames", type=int, default=2, help="stateful engine frames per step")
    p.add_argument("--max-mask", type=float, default=0.0, help="optional complex-mask magnitude cap; 0 disables")
    p.add_argument("--threads", type=int, default=1, help="CPU intra-op threads for Torch and ONNX Runtime")
    p.add_argument("--interop-threads", type=int, default=1, help="torch inter-op CPU threads")
    p.add_argument("--flush-denormal", action="store_true", help="flush CPU denormal values")
    p.add_argument("--onnx-step", help="ONNX Runtime causal chunk model with explicit LSTM states")
    p.add_argument("--lookahead", type=float, default=None,
                   help="future context before emitting (s); defaults to 0 for causal and 0.16 otherwise")
    p.add_argument("--xfade-ms", type=float, default=8.0)
    p.add_argument("--enroll-max", type=float, default=15.0)
    p.add_argument("--no-bypass", action="store_true",
                   help="disable backlog dropping; may cause growing audio delay")
    p.add_argument("--max-lag", type=float, default=0.6,
                   help="maximum queued-input age before old samples are dropped")
    p.add_argument("--ort-spin", action="store_true",
                   help="allow ONNX Runtime intra-op worker spinning (default: disabled)")
    p.add_argument("--single-session", action="store_true",
                   help="make older active sessions raw passthrough when a new one starts")
    p.add_argument("--debug-timing", action="store_true",
                   help="log per-session audio rates and engine timings every 5 s")
    p.add_argument("--dump-audio", metavar="DIR",
                   help="write per-session received, engine-input, and returned WAV files")
    p.add_argument("--device", default="auto")
    p.add_argument("--half", action="store_true", help="fp16 on CUDA")
    p.add_argument("--token", default="",
                   help="shared secret: /ws?token=... and Authorization: Bearer ... on /enroll")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    if args.threads < 1 or args.max_lag <= 0 or args.interop_threads != 1:
        p.error("--threads and --max-lag must be positive; --interop-threads is fixed at 1")
    if args.flush_denormal:
        torch.set_flush_denormal(True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("aiohttp.access").addFilter(RedactTokenFilter())

    runner = Runner(args.ckpt, args.device, args.half)
    if args.lookahead is None:
        args.lookahead = 0.0 if runner.causal else 0.16
    src = args.enroll or (ENROLL_FILE if os.path.exists(ENROLL_FILE) else None)
    if src and not runner.dummy:
        with open(src, "rb") as f:
            runner.set_enrollment(_load_enrollment(f.read(), args.sr, args.enroll_max))
        log.info("loaded enrollment from %s", src)
    if args.onnx_step:
        if args.sliding or runner.dummy or not runner.causal:
            p.error("--onnx-step requires a non-dummy causal stateful model")
        if args.chunk_frames != 4:
            p.error("the exported ONNX step requires --chunk-frames 4")
        import onnxruntime as ort
        opts = make_ort_options(ort, args.threads, args.ort_spin)
        runner.onnx_session = ort.InferenceSession(
            args.onnx_step, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        mismatch = onnx_step_mismatch(runner.onnx_session, runner)
        if mismatch:
            log.warning(
                "ONNX step %s is incompatible with checkpoint %s: %s. "
                "Falling back to the checkpoint-compatible PyTorch stateful engine. "
                "Re-export the ONNX step for this checkpoint for ONNX acceleration.",
                args.onnx_step, args.ckpt, mismatch,
            )
            runner.onnx_session = None
        else:
            log.info("loaded checkpoint-compatible ONNX Runtime step from %s", args.onnx_step)
    else:
        runner.onnx_session = None
    use_stateful = runner.causal and not runner.dummy and runner.ready and not args.sliding
    if use_stateful:
        if args.chunk_frames < 1:
            p.error("--chunk-frames must be at least 1")
        warm_state = State(args, runner)
        warm_session = Session(warm_state, 48000)
        warm_chunk = np.zeros(args.chunk_frames * runner.model.hop * 3, dtype="<i2").tobytes()
        warm_times = []
        for _ in range(12):
            warm_session.feed(warm_chunk)
            if warm_session.last_step_ran:
                warm_times.append(warm_session.eng.last_step_ms)
        steady = float(np.mean(warm_times[2:])) if len(warm_times) > 2 else float("inf")
        budget = args.chunk_frames * runner.model.hop / args.sr * 1e3
        log.info("stateful startup warm-up: steady step %.1f ms / budget %.1f ms (RTF=%.2f)",
                 steady, budget, steady / budget)
        if steady > 25.0:
            log.warning("idle steady step exceeds 25 ms; %s", backend_summary(runner, args))
    log.info("device=%s dummy=%s enrolled=%s %s", runner.device, runner.dummy,
             runner.ready, backend_summary(runner, args))
    web.run_app(make_app(args, runner), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
