"""Auth checks for the HTTP/WebSocket server.

The tests use the dummy runner (no model) and verify that --token rejects
missing/wrong credentials on both the websocket and enrollment endpoints while
the same requests succeed with the token.
"""
import argparse
import asyncio
import json
import logging

import aiohttp
from aiohttp import web, WSMsgType

from model_adapter import Runner
from server import RedactTokenFilter, make_app

PORT = 8795
TOKEN = "correct horse battery staple"


def make_args():
    return argparse.Namespace(sr=16000, window=1.0, hop=0.08, lookahead=0.0,
                              xfade_ms=8.0, enroll_max=15.0, no_bypass=False,
                              token=TOKEN)


async def start_app():
    runner = Runner("dummy")
    ar = web.AppRunner(make_app(make_args(), runner))
    await ar.setup()
    await web.TCPSite(ar, "127.0.0.1", PORT).start()
    return ar


async def expect_ws_401(session, url):
    try:
        await session.ws_connect(url)
    except aiohttp.WSServerHandshakeError as e:
        return e.status == 401, e.status
    return False, 200


async def main():
    ar = await start_app()
    ok = True
    record = logging.LogRecord("aiohttp.access", logging.INFO, __file__, 0,
                               'GET /ws?token=secret-value HTTP/1.1', (), None)
    RedactTokenFilter().filter(record)
    redacted = "secret-value" not in record.getMessage() and "token=***" in record.getMessage()
    print(f"access-log token redaction: {record.getMessage()!r}  {'OK' if redacted else 'FAIL'}")
    ok &= redacted
    async with aiohttp.ClientSession() as s:
        base = f"http://127.0.0.1:{PORT}"
        for label, url in [("missing", f"{base}/ws"),
                           ("wrong", f"{base}/ws?token=nope")]:
            good, status = await expect_ws_401(s, url)
            print(f"ws {label}: HTTP {status}  {'OK' if good else 'FAIL'}")
            ok &= good

        async with s.ws_connect(f"{base}/ws?token={TOKEN}") as ws:
            await ws.send_str(json.dumps({"type": "start", "sampleRate": 48000}))
            msg = await ws.receive()
            good = msg.type == WSMsgType.TEXT and json.loads(msg.data)["type"] == "ready"
            print(f"ws valid token: {json.loads(msg.data)}  {'OK' if good else 'FAIL'}")
            ok &= good

        for label, headers in [
            ("missing", {}),
            ("wrong", {"Authorization": "Bearer nope"}),
        ]:
            async with s.post(f"{base}/enroll", data=b"not audio", headers=headers) as r:
                good = r.status == 401
                print(f"enroll {label}: HTTP {r.status}  {'OK' if good else 'FAIL'}")
                ok &= good

        for label, headers in [
            ("bearer", {"Authorization": f"Bearer {TOKEN}"}),
            ("x-header", {"X-TSE-Token": TOKEN}),
        ]:
            async with s.post(f"{base}/enroll", data=b"not audio", headers=headers) as r:
                # Auth passed: the handler reaches audio decoding and returns 400.
                good = r.status == 400
                print(f"enroll {label}: HTTP {r.status} (auth passed)  {'OK' if good else 'FAIL'}")
                ok &= good

    await ar.cleanup()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
