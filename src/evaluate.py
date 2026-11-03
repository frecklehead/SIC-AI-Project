"""Evaluate the selected BSRNN checkpoint and write test_metrics_final.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pesq import pesq
from pystoi import stoi
from tqdm.auto import tqdm


def si_sdr(est, ref, eps=1e-8):
    est, ref = est - est.mean(), ref - ref.mean()
    projection = (est @ ref) / (ref @ ref + eps) * ref
    error = est - projection
    return 10 * np.log10((projection @ projection) / (error @ error + eps) + eps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sys.path.insert(0, str(args.project_dir))
    try:
        import tselib as T
    except ImportError as exc:
        raise SystemExit("Missing tselib.py from the private tse-project directory") from exc

    device = torch.device(args.device)
    cfg = T.load_cfg(str(args.project_dir / "config.yaml"))
    cfg_big = T._merge(cfg, {"model": {"N": 128, "hidden": 256, "layers": 8}})
    dirs = T.proj_dirs(cfg)
    manifest = pd.read_csv(dirs["manifests"] + "/test_mix.csv")
    bank = T.load_bank(f"{dirs['embeddings']}/test")
    dataset = T.FixedDataset(
        manifest, bank[1], args.data_dir,
        cfg_big["audio"]["eval_seconds"], cfg_big["mix"]["peak_limit"]
    )

    checkpoint_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint_data.get("model", checkpoint_data)
    state = {key.replace("module.", "", 1): value for key, value in state.items()}
    model = T.TSEBSRNN(cfg_big["model"])
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    rows = []
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            mix, target, embedding, *_ = dataset[idx]
            mix_t = torch.as_tensor(mix).reshape(1, -1).to(device)
            emb_t = torch.as_tensor(embedding).reshape(1, -1).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                estimate = model(mix_t, emb_t)
            estimate = estimate.float().reshape(-1).cpu().numpy()
            n = min(len(mix), len(target), len(estimate))
            mix, target, estimate = mix[:n], target[:n], estimate[:n]
            try:
                pesq_out = pesq(16000, target, estimate, "wb")
            except Exception:
                pesq_out = np.nan
            rows.append({
                "idx": idx,
                "sisdr_in": si_sdr(mix, target),
                "sisdr_out": si_sdr(estimate, target),
                "stoi_in": stoi(target, mix, 16000),
                "stoi_out": stoi(target, estimate, 16000),
                "pesq_out": pesq_out,
            })

    result = pd.DataFrame(rows)
    result["sisdri"] = result.sisdr_out - result.sisdr_in
    output = args.project_dir / "test_metrics_final.csv"
    result.to_csv(output, index=False)
    print(f"checkpoint: {args.checkpoint}")
    print(f"wrote {output} ({len(result)} rows)")


if __name__ == "__main__":
    main()
