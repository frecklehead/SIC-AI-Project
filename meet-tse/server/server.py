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
"""
import argparse
import asyncio
import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from aiohttp import web, WSMsgType

from model_adapter import Runner
from resampler import StreamResampler
from stream_engine import SlidingWindowTSE

log = logging.getLogger("tse")
MAX_LAG_S = 0.6
HERE = os.path.dirname(os.path.abspath(__file__))
ENROLL_FILE = os.path.join(HERE, "enrollment.wav")


class State:
    def __init__(self, args, runner):
        self.args, self.runner, self.sr = args, runner, args.sr
        self.pool = ThreadPoolExecutor(max_workers=1)  # one model, one thread
        self.processing = False
        self.bypass = False
        self.last_rtf = None


class Session:
    def __init__(self, st: State, sr_in: int):
        self.st, self.sr_in = st, sr_in
        self.down = StreamResampler(sr_in, st.sr)
        self.up = StreamResampler(st.sr, sr_in)
        self.reset_engine()
        self.t0, self.audio_s, self.bypass = time.monotonic(), 0.0, False
        self.last_log, self.steps = time.monotonic(), []

    def reset_engine(self):
        a = self.st.args
        self.eng = SlidingWindowTSE(self.st.runner, self.st.sr, a.window, a.hop, a.lookahead, a.xfade_ms)

    def feed(self, pcm: bytes) -> bytes:
        pcm = pcm[: len(pcm) // 2 * 2]
        if self.bypass:
            return pcm
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        y16 = self.eng.push(self.down.process(x))
        if y16.size and self.eng.last_step_ms:
            self.steps.append(self.eng.last_step_ms)
        y = self.up.process(y16)
        return (np.clip(y, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


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
        "processing": st.processing, "bypass": st.bypass, "rtf": st.last_rtf,
        "sr": st.sr, "window_s": a.window, "hop_s": a.hop, "lookahead_s": a.lookahead,
        "approx_latency_ms": round(1000 * (a.hop + a.lookahead + a.xfade_ms / 1000), 1),
    })


def _load_enrollment(data: bytes, sr_model: int, max_s: float) -> np.ndarray:
    import soundfile as sf
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    wav = StreamResampler(sr, sr_model).process(wav)
    return wav[: int(max_s * sr_model)]


async def enroll(request):
    st: State = request.app["st"]
    data = await request.read()
    try:
        wav = _load_enrollment(data, st.sr, st.args.enroll_max)
    except Exception as e:  # noqa
        return web.json_response({"ok": False, "msg": f"could not decode audio: {e}"}, status=400)
    if wav.size < st.sr * 2:
        return web.json_response({"ok": False, "msg": "enrollment too short (need >= 2 s, ideally 5-15 s)"}, status=400)
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(st.pool, st.runner.set_enrollment, wav)
    except Exception as e:  # noqa: broad-except: surface encoder/download errors in the popup
        log.exception("enrollment failed")
        return web.json_response({"ok": False, "msg": f"speaker encoder failed: {e}"}, status=500)
    import soundfile as sf
    sf.write(ENROLL_FILE, wav, st.sr)
    log.info("enrolled %.1f s of audio", wav.size / st.sr)
    return web.json_response({"ok": True, "seconds": round(wav.size / st.sr, 1)})


async def ws_handler(request):
    st: State = request.app["st"]
    ws = web.WebSocketResponse(max_msg_size=2 ** 22)
    await ws.prepare(request)
    loop = asyncio.get_running_loop()
    sess = None
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
                sess = Session(st, int(cfg["sampleRate"]))
                st.bypass = False
                st.last_rtf = None
                await ws.send_json({"type": "ready"})
                log.info("session started @ %d Hz", sess.sr_in)
        elif msg.type == WSMsgType.BINARY and sess is not None:
            sess.audio_s += len(msg.data) / 2 / sess.sr_in
            lag = (time.monotonic() - sess.t0) - sess.audio_s
            if lag < 0:                       # client was idle: re-anchor
                sess.t0 = time.monotonic() - sess.audio_s
                lag = 0.0
            if not st.args.no_bypass and lag > MAX_LAG_S and not sess.bypass:
                # Sticky for the rest of the session: re-enabling a model that is
                # still slower than real time causes periodic stalls (worse than
                # the raw mic). Reload the Meet tab to try again.
                sess.bypass = True
                st.bypass = True
                log.warning("server too slow (lag %.2fs): bypassing model for this session. "
                            "Increase hop / shrink window / use a GPU.", lag)
            st.processing = True
            try:
                out = await loop.run_in_executor(st.pool, sess.feed, msg.data)
            except Exception:  # noqa: broad-except - keep the call alive, fail open
                log.exception("inference failed; bypassing model for this session")
                sess.bypass = True
                st.bypass = True
                out = msg.data
            finally:
                st.processing = False
            if out:
                await ws.send_bytes(out)
            if time.monotonic() - sess.last_log > 5 and sess.steps:
                a = st.args
                avg, mx = float(np.mean(sess.steps)), float(np.max(sess.steps))
                log.info("model step avg %.1f ms / max %.1f ms  (budget %.0f ms)  RTF=%.2f",
                         avg, mx, a.hop * 1e3, avg / (a.hop * 1e3))
                st.last_rtf = avg / (a.hop * 1e3)
                sess.steps.clear()
                sess.last_log = time.monotonic()
    log.info("session closed")
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
    p.add_argument("--lookahead", type=float, default=0.16, help="future context before emitting (s); 0 for causal models")
    p.add_argument("--xfade-ms", type=float, default=8.0)
    p.add_argument("--enroll-max", type=float, default=15.0)
    p.add_argument("--no-bypass", action="store_true",
                   help="disable the real-time lag bypass (offline quality tests only)")
    p.add_argument("--device", default="auto")
    p.add_argument("--half", action="store_true", help="fp16 on CUDA")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    runner = Runner(args.ckpt, args.device, args.half)
    src = args.enroll or (ENROLL_FILE if os.path.exists(ENROLL_FILE) else None)
    if src and not runner.dummy:
        with open(src, "rb") as f:
            runner.set_enrollment(_load_enrollment(f.read(), args.sr, args.enroll_max))
        log.info("loaded enrollment from %s", src)
    runner.warmup(int(args.window * args.sr))
    log.info("device=%s dummy=%s enrolled=%s", runner.device, runner.dummy, runner.ready)
    web.run_app(make_app(args, runner), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
