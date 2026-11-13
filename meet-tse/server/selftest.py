"""End-to-end pipeline test with the dummy (pass-through) model.
Sends a real-time 48 kHz test signal through the websocket server and checks
that what comes back is the same signal, delayed by the expected latency."""
import asyncio, argparse, time, json
import numpy as np
import aiohttp
from aiohttp import web
from server import make_app
from model_adapter import Runner

SR, PORT = 48000, 8799


async def main():
    args = argparse.Namespace(sr=16000, window=2.0, hop=0.08, lookahead=0.16, xfade_ms=8.0, enroll_max=15.0)
    runner = Runner("dummy")
    ar = web.AppRunner(make_app(args, runner)); await ar.setup()
    await web.TCPSite(ar, "127.0.0.1", PORT).start()

    secs = 4
    t = np.arange(SR * secs) / SR
    sig = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 2300 * t * (1 + 0.1 * t))).astype(np.float32)
    pcm = (sig * 32767).astype("<i2")
    got, first_out = [], None
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(f"http://127.0.0.1:{PORT}/ws") as ws:
            await ws.send_str(json.dumps({"type": "start", "sampleRate": SR}))
            assert json.loads((await ws.receive()).data)["type"] == "ready"

            async def rx():
                nonlocal first_out
                async for m in ws:
                    if m.type == aiohttp.WSMsgType.BINARY:
                        if first_out is None: first_out = time.monotonic()
                        got.append(np.frombuffer(m.data, "<i2"))
            task = asyncio.create_task(rx())
            t0 = time.monotonic()
            for i in range(0, len(pcm), 480):
                await ws.send_bytes(pcm[i:i + 480].tobytes())
                await asyncio.sleep(max(0, t0 + (i + 480) / SR - time.monotonic()))
            await asyncio.sleep(0.5); await ws.close(); await task
    out = np.concatenate(got).astype(np.float32) / 32767
    n = min(len(out), len(sig))
    # estimate delay by cross-correlation on a middle segment
    seg = sig[SR:2 * SR]; best, lag = -1, 0
    for d in range(0, int(0.6 * SR), 8):
        c = np.dot(seg, out[SR + d: 2 * SR + d]) if 2 * SR + d <= len(out) else 0
        if c > best: best, lag = c, d
    a, b = sig[SR:3 * SR], out[SR + lag: 3 * SR + lag]
    m = min(len(a), len(b))
    err = np.sqrt(np.mean((a[:m] - b[:m]) ** 2)) / np.sqrt(np.mean(a[:m] ** 2))
    print(f"samples in={len(sig)} out={len(out)}  | estimated delay = {1000*lag/SR:.1f} ms | "
          f"first output after {1000*(first_out-t0):.0f} ms | relative error after alignment = {err:.4f}")
    await ar.cleanup()
    assert err < 0.05, "pipeline distorts the signal"
    print("PASS")

asyncio.run(main())
