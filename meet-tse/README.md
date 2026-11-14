# Meet TSE — real-time target speaker extraction for Google Meet

```
 mic ─▶ getUserMedia (wrapped, inject.js, page world)
          └▶ MediaStreamTrackProcessor ─▶ int16 PCM ─▶ bridge.js (content script)
                                                          │ runtime port (base64 PCM)
                                                          ▼
                                              background.js (MV3 service worker) ─ws─▶ server.py
                                                                                      127.0.0.1:8765
                                                                                      48k→16k → BSRNN-TSE → 16k→48k
          ◀─ MediaStreamTrackGenerator ◀─ jitter buffer ◀──────────────────────────────┘
 Meet/WebRTC sends the generated track → others hear only the enrolled speaker.
```
Fail-open: server offline / not enrolled / too slow → your normal mic passes through.

The websocket lives in the service worker, not the page content script: recent Chrome applies the
Local Network Access check to content-script connections to `127.0.0.1` and blocks them
(`ERR_BLOCKED_BY_LOCAL_NETWORK_ACCESS_CHECKS`). Extension-origin contexts with the localhost
host permission are allowed. Verified on Chrome 156 with no flags.

## Run
1. `cd server && python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
2. Pipeline check, no model needed: `python selftest.py` (must print `PASS`).
3. Start the trained model with the best checkpoint:
   `python server.py --ckpt /path/tse_bsrnn_big_best.pt --enroll my_voice.wav`
   The bundled adapter is matched to the project checkpoint (16 kHz, 512-point STFT, hop 128,
   16.83M parameters, frozen 192-d ECAPA enrollment conditioning). Use `--half` only on CUDA.
   First start downloads the frozen ECAPA weights into `server/pretrained_ecapa/` and warms up the
   model; wait for the `device=... enrolled=True` log line (1–2 min on a loaded CPU, seconds on GPU).
4. Chrome → `chrome://extensions` → Developer mode → Load unpacked → `extension/`
5. Popup: upload enrollment audio (or pass `--enroll`), keep both checkboxes, reload the Meet tab.
   In Meet: Settings → Audio → turn OFF "Noise cancellation" (avoid double processing).

## Measured quality (this checkpoint)
LibriSpeech dev-clean two-speaker mixture, 4 s, 16 kHz, enrollment 4.8 s. Streaming uses the same
`SlidingWindowTSE` path as the server (`window 2.0 s`, `hop 0.08 s`):

| path | SI-SDR | SI-SDRi | algorithmic delay |
|---|---|---|---|
| mixture | −0.16 dB | — | — |
| whole-file (oracle context) | 13.03 dB | +13.19 dB | 0 |
| streaming, lookahead 0 | 7.83 dB | +7.99 dB | 8 ms |
| streaming, lookahead 0.08 | 11.75 dB | +11.91 dB | 88 ms |
| streaming, lookahead 0.16 (default) | 12.25 dB | +12.41 dB | 168 ms |
| streaming, lookahead 0.24 | 12.54 dB | +12.70 dB | 248 ms |

The wrong speaker is rejected (−28 dB). The model is non-causal (bidirectional LSTMs), so lookahead
matters: 80 ms recovers almost all of the whole-file quality. Use `--lookahead 0.08` if 88 ms of
algorithmic delay matters more than the extra ~0.5 dB.

End-to-end through the real server (1.5 s clip, `hop 0.16`, `lookahead 0.08`, including the
48k↔16k resamplers and int16 websocket PCM): mixture −1.37 dB → output **10.95 dB**
(**+12.32 dB SI-SDRi**) at the measured 88 ms delay. `offline_check.py` on the 4 s clip gives the
slightly higher numbers above because it stays at 16 kHz and uses a smaller hop.

## Performance (why the server may bypass)
`benchmark.py` on the development laptop (i7-1255U, 12 threads, **no CUDA**). The machine was
memory-thrashing; these are best-case snapshots, not stable numbers (under heavier load a 2 s
window took 5–11 s per 0.16 s step, RTF 33–70):

| window | hop 0.04 | hop 0.08 | hop 0.16 |
|---|---|---|---|
| 1.0 s | 764 ms / RTF 19.1 | 577 ms / RTF 7.2 | 632 ms / RTF 4.0 |
| 1.5 s | 2477 ms / RTF 61.9 | 1291 ms / RTF 16.1 | 1129 ms / RTF 7.1 |
| 2.0 s | 1519 ms / RTF 38.0 | 1365 ms / RTF 17.1 | 1433 ms / RTF 9.0 |

RTF is compute per audio-second; real time needs < 1 and < 0.7 for headroom. No CPU configuration
met that on this laptop. If the server falls more than 0.6 s behind it bypasses the model for the
rest of the session (raw mic keeps flowing, popup shows `BYPASS`) instead of stalling the call.
Use a CUDA GPU (`--device cuda --half`) for actual extraction, or a smaller model.
`--no-bypass` disables the safety valve for offline quality tests (where audio is intentionally
fed faster than real time).

## Tuning (watch the `RTF` line the server logs every 5 s; keep it < 0.7)
| flag | effect |
|---|---|
| `--hop` (0.08) | how often the model runs. Larger = cheaper, more latency |
| `--window` (2.0) | context per run. Larger = better speaker tracking, costs compute |
| `--lookahead` (0.16) | future context before emitting. Lower = less delay, slightly worse |
| `--half` | fp16 on CUDA |
Latency ≈ lookahead + xfade in steady state (measured 168 ms at the default 0.16 s); the first output
arrives after ~hop + one model step. Add network, client prebuffer (150 ms) and jitter. The popup's
`latency ≈` figure uses hop + lookahead + xfade as a conservative estimate.

## Diagnostics
```bash
python benchmark.py --ckpt /path/tse_bsrnn_big_best.pt
python offline_check.py --ckpt /path/tse_bsrnn_big_best.pt --enroll my_voice.wav \
  --mix mixture.wav --target me.wav                     # existing mixture
python offline_check.py --ckpt /path/tse_bsrnn_big_best.pt --enroll my_voice.wav \
  --target me.wav --interferer other.wav --snr 5 0 -5 --seconds 2 \
  --lookahead-sweep 0 0.08 0.16 0.24                       # synthesized mixtures
python record_enrollment.py --seconds 10
python robustness_test.py                                  # bursts, bad frames, slow/crashing model
```

`offline_check.py` compares the same sliding-window path used by the server with whole-file
inference and reports SI-SDR / SI-SDRi against the clean target plus streaming-vs-whole parity.
`record_enrollment.py` requires the optional `sounddevice` package and posts the recording to
localhost `/enroll`. `robustness_test.py` starts the server with fake runners and prints `PASS`.

## Known limits
* Targets your OWN mic (what others hear). Extracting a remote speaker from incoming audio is a separate phase.
* Chrome only (needs MediaStreamTrackProcessor/Generator). Server must be on localhost.
* The first ~0.3–0.6 s of the extracted stream is near-silent while the 2 s model window fills.
  The extension passes the raw mic until its 150 ms jitter buffer is ready, but then plays that quiet
  head; this can clip the first word of a call. A raw-mic warmup for the first model window (or a
  server-side prefill) would fix it.
* Verified: real checkpoint extraction, streaming parity, websocket server, and the extension's
  getUserMedia wrapper on a `meet.google.com` test page with a fake mic (including server
  restart/reconnect and fail-open). **Not yet verified inside a live Meet call** (needs a real
  meeting/login); Meet's own audio pipeline may still require adjustments.
* CPU RTF is far above 1 on the development laptop; expect bypass unless a CUDA GPU is used.
