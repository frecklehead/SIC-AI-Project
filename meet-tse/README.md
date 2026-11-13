# Meet TSE — real-time target speaker extraction for Google Meet

```
 mic ─▶ getUserMedia (wrapped, inject.js, page world)
          └▶ MediaStreamTrackProcessor ─▶ int16 PCM ─▶ bridge.js ─ws─▶ server.py (127.0.0.1:8765)
                                                                         48k→16k → sliding-window BSRNN-TSE → 16k→48k
          ◀─ MediaStreamTrackGenerator ◀─ jitter buffer ◀─────────────────┘
 Meet/WebRTC sends the generated track → others hear only the enrolled speaker.
```
Fail-open: server offline / not enrolled → your normal mic passes through.

## Run
1. `cd server && python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
2. Pipeline check, no model needed: `python selftest.py` (must print `PASS`).
3. Start the trained model with the best checkpoint:
   `python server.py --ckpt /path/tse_bsrnn_big_best.pt --enroll my_voice.wav`
   The bundled adapter is matched to the project checkpoint (16 kHz, 512-point STFT, hop 128,
   16.83M parameters, frozen 192-d ECAPA enrollment conditioning). Use `--half` only on CUDA.
4. Chrome → `chrome://extensions` → Developer mode → Load unpacked → `extension/`
5. Popup: upload enrollment audio (or pass `--enroll`), keep both checkboxes, reload the Meet tab.
   In Meet: Settings → Audio → turn OFF "Noise cancellation" (avoid double processing).

## Tuning (watch the `RTF` line the server logs every 5 s; keep it < 0.7)
| flag | effect |
|---|---|
| `--hop` (0.08) | how often the model runs. Larger = cheaper, more latency |
| `--window` (2.0) | context per run. Larger = better speaker tracking, costs compute |
| `--lookahead` (0.16) | future context before emitting. 0 for a causal model |
| `--half` | fp16 on CUDA |
Latency ≈ hop + lookahead + xfade + model time + prebuffer (client, default 150 ms).

Diagnostics:

```bash
python benchmark.py --ckpt /path/tse_bsrnn_big_best.pt
python offline_check.py --ckpt /path/tse_bsrnn_big_best.pt \
  --mix mixture.wav --enroll my_voice.wav
python record_enrollment.py --seconds 10
```

`benchmark.py` measures all requested window/hop pairs. `offline_check.py` compares the same
sliding-window path used by the server with whole-file inference. `record_enrollment.py` requires
the optional `sounddevice` package and posts the recording to localhost `/enroll`.

## Known limits
* Targets your OWN mic (what others hear). Extracting a remote speaker from incoming audio is a separate phase.
* Chrome only (needs MediaStreamTrackProcessor/Generator). Server must be on localhost.
* Not yet tested against a live Meet call or your real checkpoint.
