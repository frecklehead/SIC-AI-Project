"""Glue between the streaming server and the trained BSRNN-TSE checkpoint.

This is the only model integration point.  `bsrnn_tse.py` holds the network
(ported from the training library) and `speaker_encoder.py` holds the frozen
ECAPA-TDNN encoder; this file loads the checkpoint, caches the enrollment
embedding once, and calls `model(mix, embedding)`.
"""
import numpy as np

from bsrnn_tse import DEFAULT_MODEL_CFG, build_model


class Runner:
    def __init__(self, ckpt: str, device: str = "auto", half: bool = False):
        self.dummy = ckpt == "dummy"
        self.enroll = None
        self.cond = None
        self.cond_t = None
        self.enroll_max = 20.0
        self.vad_cfg = None
        self.device = "cpu"
        self.model = None
        self.half = False
        self.torch = None
        self.speaker_encoder = None
        if self.dummy:
            return

        import torch
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(obj, torch.nn.Module):
            model, mcfg, meta = obj, None, {}
        else:
            sd, cfg, meta = obj, {}, {}
            if isinstance(obj, dict):
                cfg = obj.get("cfg") or {}
                meta = {k: obj[k] for k in ("epoch", "val_si_sdri", "best") if k in obj}
                for k in ("ema", "model", "model_state_dict", "state_dict"):
                    if k in obj and isinstance(obj[k], dict):
                        sd = obj[k]
                        break
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            mcfg = dict(DEFAULT_MODEL_CFG)
            mcfg.update(cfg.get("model") or {})
            model = build_model(mcfg)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"checkpoint wiring mismatch: missing={missing}, unexpected={unexpected}"
                )
            print("[adapter] checkpoint keys: missing=0 unexpected=0")

        self.model = model.to(self.device).eval()
        self.half = bool(half) and self.device == "cuda"
        if self.half:  # autocast, not .half(): STFT/iSTFT must stay fp32
            self.autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            self.autocast = torch.autocast(device_type=self.device, enabled=False)

        if cfg:
            self.vad_cfg = (cfg.get("enroll") or {}).get("vad")
            self.enroll_max = float((cfg.get("enroll") or {}).get("max_s", 20.0))
        n = sum(p.numel() for p in self.model.parameters()) / 1e6
        extra = " ".join(f"{k}={v}" for k, v in meta.items())
        print(f"[adapter] loaded {ckpt}  ({n:.2f} M params) {extra}".rstrip())

    @property
    def ready(self):
        return self.dummy or self.enroll is not None

    def set_enrollment(self, wav: np.ndarray):
        """wav: float32 mono at the model sample rate; VAD-trimmed like training."""
        self.enroll = wav.astype(np.float32)
        if self.dummy:
            return
        if self.speaker_encoder is None:
            from speaker_encoder import SpeakerEncoder
            self.speaker_encoder = SpeakerEncoder(device=self.device)
        emb = self.speaker_encoder.embed(self.enroll, vad_cfg=self.vad_cfg, max_s=self.enroll_max)
        self.cond = emb
        self.cond_t = self.torch.from_numpy(emb)[None].to(self.device)
        print(f"[adapter] enrollment: {len(self.enroll) / 16000:.1f}s -> embedding norm {np.linalg.norm(emb):.1f}")

    def _forward(self, mix_t):
        """mix_t: [1, T] tensor -> [1, T] estimate of the enrolled speaker."""
        with self.autocast:
            return self.model(mix_t, self.cond_t)

    def __call__(self, mix: np.ndarray) -> np.ndarray:
        if self.dummy:
            return mix
        torch = self.torch
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(mix, dtype=np.float32))[None].to(self.device)
            out = self._forward(t)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = out.reshape(-1).float().cpu().numpy()
        n = mix.shape[0]
        if out.shape[0] < n:
            out = np.pad(out, (0, n - out.shape[0]))
        return out[:n]

    def warmup(self, n_samples: int):
        if self.dummy or self.enroll is None:
            return
        for _ in range(3):
            self(np.zeros(n_samples, np.float32))
