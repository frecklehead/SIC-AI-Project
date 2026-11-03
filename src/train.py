"""Train the BSRNN using the project module from the original notebook.

The actual model, sampler, datasets, loss and training loop remain in the
Drive-only ``tselib.py`` module and are imported rather than re-created.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.utils.checkpoint as checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sys.path.insert(0, str(args.project_dir))
    try:
        import tselib as T
    except ImportError as exc:
        raise SystemExit("Missing tselib.py from the private tse-project directory") from exc

    device = torch.device(args.device)
    cfg = T.load_cfg(str(args.project_dir / "config.yaml"))
    dirs = T.proj_dirs(cfg)
    libri = pd.read_csv(dirs["manifests"] + "/librispeech.csv")
    noise = pd.read_csv(dirs["manifests"] + "/noise.csv")
    banks = {role: T.load_bank(f"{dirs['embeddings']}/{role}")
             for role in ("train", "val", "test")}

    segments = {"train": cfg["audio"]["train_seconds"],
                "val": cfg["audio"]["eval_seconds"],
                "test": cfg["audio"]["eval_seconds"]}
    samplers = {
        role: T.SceneSampler(T.role_df(libri, cfg, role), noise, banks[role][0],
                             cfg, role, segments[role])
        for role in segments
    }
    val_manifest = pd.read_csv(dirs["manifests"] + "/val_mix.csv")

    cfg_big = T._merge(cfg, {
        "model": {"N": 128, "hidden": 256, "layers": 8},
        "train": {"epochs": 100, "steps_per_epoch": 1000, "batch_size": 8,
                  "lr": 5e-4, "warmup_steps": 2000, "num_workers": 2},
    })
    T.save_cfg(cfg_big, str(args.project_dir / "config_big.yaml"))

    if not hasattr(T.DualPathLayer, "_orig_forward"):
        T.DualPathLayer._orig_forward = T.DualPathLayer.forward

        def checkpointed_forward(layer, x, spk):
            if layer.training and torch.is_grad_enabled():
                return checkpoint.checkpoint(
                    T.DualPathLayer._orig_forward, layer, x, spk, use_reentrant=False
                )
            return T.DualPathLayer._orig_forward(layer, x, spk)

        T.DualPathLayer.forward = checkpointed_forward

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_cfg = cfg_big["train"]
    model = T.TSEBSRNN(cfg_big["model"])
    train_ds = T.TSEDataset(
        samplers["train"], banks["train"][1],
        train_cfg["batch_size"] * train_cfg["steps_per_epoch"], args.data_dir
    )
    val_ds = T.FixedDataset(
        val_manifest, banks["val"][1], args.data_dir,
        cfg_big["audio"]["eval_seconds"], cfg_big["mix"]["peak_limit"]
    )
    T.train(cfg_big, model, train_ds, val_ds, dirs["checkpoints"], dirs["logs"],
            device, tag="tse_bsrnn_big")


if __name__ == "__main__":
    main()
