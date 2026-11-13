"""Record a short target-speaker enrollment clip and upload it to the server."""
from __future__ import annotations

import argparse
import io
import urllib.request
import wave

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--rate", type=int, default=16000)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--url", default="http://127.0.0.1:8765/enroll")
    args = p.parse_args()
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit("Install sounddevice to record from a microphone: pip install sounddevice") from exc
    print(f"Recording {args.seconds:.1f}s at {args.rate} Hz; speak naturally after the countdown.")
    audio = sd.rec(int(args.seconds * args.rate), samplerate=args.rate, channels=1,
                   dtype="float32", device=args.device)
    sd.wait()
    pcm = np.clip(audio[:, 0], -1, 1)
    raw = (pcm * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(args.rate); f.writeframes(raw)
    req = urllib.request.Request(args.url, data=buf.getvalue(), method="POST",
                                 headers={"Content-Type": "audio/wav"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        print(resp.read().decode())


if __name__ == "__main__":
    main()
