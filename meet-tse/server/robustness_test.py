"""Robustness checks for the websocket server (no model, no browser).

Covers:
  * irregular chunk sizes / bursts (the client is not required to send 10 ms frames)
  * a malformed text frame must not kill the session
  * a model slower than real time must trigger the lag bypass and keep audio flowing

Run: python robustness_test.py
"""
import argparse
import asyncio
import json
import time

import numpy as np
from aiohttp import web, WSMsgType

from server import make_app

SR = 48000


def make_args():
    return argparse.Namespace(sr=16000, window=1.0, hop=0.08, lookahead=0.0,
                              xfade_ms=8.0, enroll_max=15.0, no_bypass=False)


def sine(seconds):
    t = np.arange(int(SR * seconds)) / SR
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class DummyRunner:
    dummy = True
    ready = True
    device = "cpu"

    def set_enrollment(self, wav):
        pass

    def __call__(self, mix):
        return mix


class SlowRunner:
    """Stands in for a model that cannot keep up (400 ms per 80 ms step)."""
    dummy = False
    device = "cpu"
    ready = True

    def __init__(self, step_s=0.4):
        self.step_s = step_s
        self.calls = 0

    def set_enrollment(self, wav):
        pass

    def __call__(self, mix):
        time.sleep(self.step_s)
        self.calls += 1
        return mix


class CrashRunner(DummyRunner):
    """Raises on every call, like an OOM or a bad checkpoint."""

    def __call__(self, mix):
        raise RuntimeError("simulated inference failure")


async def start_app(runner, port):
    ar = web.AppRunner(make_app(make_args(), runner))
    await ar.setup()
    await web.TCPSite(ar, "127.0.0.1", port).start()
    return ar


async def open_ws(port):
    import aiohttp
    s = aiohttp.ClientSession()
    ws = await s.ws_connect(f"http://127.0.0.1:{port}/ws")
    await ws.send_str(json.dumps({"type": "start", "sampleRate": SR}))
    msg = await ws.receive()
    assert msg.type == WSMsgType.TEXT and json.loads(msg.data)["type"] == "ready", msg
    return s, ws


async def burst_test(port):
    s, ws = await open_ws(port)
    pcm = (sine(2.0) * 32767).astype("<i2")
    sizes = [4800, 48, 960, 17, 2400, 1, 480]  # deliberately irregular
    got = []
    i = 0
    while i < len(pcm):
        n = sizes[(i // 480) % len(sizes)]
        await ws.send_bytes(pcm[i:i + n].tobytes())
        i += n
    try:
        while sum(len(g) for g in got) < len(pcm) * 0.9:
            msg = await asyncio.wait_for(ws.receive(), timeout=3)
            if msg.type == WSMsgType.BINARY:
                got.append(np.frombuffer(msg.data, "<i2"))
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                break
    except asyncio.TimeoutError:
        pass
    except Exception:
        pass
    await ws.close(); await s.close()
    out = sum(len(g) for g in got)
    print(f"bursts: sent {len(pcm)} samples, got {out}  {'OK' if out > 0.8 * len(pcm) else 'FAIL'}")
    return out > 0.8 * len(pcm)


async def malformed_test(port):
    s, ws = await open_ws(port)
    await ws.send_str("this is not json")
    await ws.send_bytes((sine(0.5) * 32767).astype("<i2").tobytes())
    got = 0
    try:
        while got < 0.4 * SR:
            msg = await asyncio.wait_for(ws.receive(), timeout=3)
            if msg.type == WSMsgType.BINARY:
                got += len(np.frombuffer(msg.data, "<i2"))
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                break
    except asyncio.TimeoutError:
        pass
    except Exception:
        pass
    await ws.close(); await s.close()
    print(f"malformed text frame: got {got} samples after it  {'OK' if got else 'FAIL'}")
    return got > 0


async def slow_test(port):
    s, ws = await open_ws(port)
    pcm = (sine(4.0) * 32767).astype("<i2")
    got, bypass_seen = [], False
    t0 = time.monotonic()

    async def rx():
        nonlocal bypass_seen
        import aiohttp
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=0.2)
            except asyncio.TimeoutError:
                msg = None
            if msg is not None and msg.type == WSMsgType.BINARY:
                got.append(np.frombuffer(msg.data, "<i2"))
            try:
                async with aiohttp.ClientSession() as cs:
                    async with cs.get(f"http://127.0.0.1:{port}/status") as r:
                        if (await r.json()).get("bypass"):
                            bypass_seen = True
            except Exception:
                pass
            if time.monotonic() - t0 > 8:
                return

    task = asyncio.create_task(rx())
    for i in range(0, len(pcm), 480):          # send in real time
        await ws.send_bytes(pcm[i:i + 480].tobytes())
        await asyncio.sleep(max(0, t0 + (i + 480) / SR - time.monotonic()))
    await asyncio.sleep(1)
    await ws.close(); task.cancel(); await s.close()
    out = sum(len(g) for g in got)
    print(f"slow model: bypass={'yes' if bypass_seen else 'NO'}, output={out} samples  "
          f"{'OK' if bypass_seen and out else 'FAIL'}")
    return bypass_seen and out > 0


async def crash_test(port):
    """A model exception must bypass, not break the websocket/audio."""
    import aiohttp
    s, ws = await open_ws(port)
    pcm = (sine(1.5) * 32767).astype("<i2")
    for i in range(0, len(pcm), 480):
        await ws.send_bytes(pcm[i:i + 480].tobytes())
    got = 0
    try:
        while got < 0.8 * len(pcm):
            msg = await asyncio.wait_for(ws.receive(), timeout=3)
            if msg.type == WSMsgType.BINARY:
                got += len(np.frombuffer(msg.data, "<i2"))
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                break
    except Exception:
        pass
    async with aiohttp.ClientSession() as cs:
        async with cs.get(f"http://127.0.0.1:{port}/status") as r:
            st = await r.json()
    await ws.close(); await s.close()
    print(f"crash: bypass={'yes' if st.get('bypass') else 'NO'}, output={got} samples  "
          f"{'OK' if st.get('bypass') and got else 'FAIL'}")
    return bool(st.get("bypass")) and got > 0


async def main():
    ok = True
    ar = await start_app(DummyRunner(), 8798)
    ok &= await burst_test(8798)
    ok &= await malformed_test(8798)
    await ar.cleanup()

    ar = await start_app(SlowRunner(), 8797)
    ok &= await slow_test(8797)
    await ar.cleanup()

    ar = await start_app(CrashRunner(), 8796)
    ok &= await crash_test(8796)
    await ar.cleanup()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
