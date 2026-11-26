"""Numerically compare grouped band ops with the original checkpoint modules."""
import argparse

import torch

from model_adapter import Runner


def report(name, ref, got):
    d = (ref - got).abs()
    print(f"{name}: max_abs={d.max().item():.8g} mean_abs={d.mean().item():.8g} "
          f"allclose={torch.allclose(ref, got, rtol=1e-5, atol=1e-6)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    args = p.parse_args()
    torch.manual_seed(7)
    runner = Runner(args.ckpt, device="cpu")
    m, ops = runner.model, runner.batched_ops
    spec = torch.randn(1, m.n_fft // 2 + 1, 4, dtype=torch.complex64)
    hidden = torch.randn(1, len(m.bands), 4, m.band_split.fcs[0].out_features)
    report("band_split", m.band_split(spec), ops.band_split(spec))
    report("decoder", m.decoder(hidden), ops.decoder(hidden))
    print("width groups:", {w: len(ix) for w, ix in ops.width_groups.items()})


if __name__ == "__main__":
    main()
