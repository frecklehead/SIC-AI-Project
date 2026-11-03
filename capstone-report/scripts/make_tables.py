"""Generate reproducible LaTeX table bodies from final evaluation CSV files.

Example:
    python scripts/make_tables.py --metrics test_metrics_final.csv \
        --manifest test_mix.csv --out-dir generated_tables

The metrics CSV must contain sisdr_in, sisdr_out, stoi_in, stoi_out and
pesq_out. The manifest supplies scenario, sir, snr and noise_path metadata.
Bootstrap settings match the analysis notebook: 5,000 resamples, seed 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {"sisdr_in", "sisdr_out", "stoi_in", "stoi_out", "pesq_out"}
METRICS = ["sisdr_in", "sisdr_out", "sisdri", "stoi_in", "stoi_out", "pesq_out"]
ROW = " " + chr(92) * 2


def boot_ci(values: pd.Series, n: int = 5000, seed: int = 0):
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def fmt(value, digits=2):
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}"


def headline(df):
    lines = []
    for col in METRICS:
        mean, lo, hi = boot_ci(df[col])
        digits = 3 if col.startswith("stoi") else 2
        lines.append(f"{col} & {fmt(mean, digits)} [{fmt(lo, digits)}, {fmt(hi, digits)}]" + ROW)
    return "\n".join(lines)


def scenario_body(df):
    lines = []
    for name, group in df.groupby("scenario", sort=True):
        mean, lo, hi = boot_ci(group["sisdri"])
        label = str(name).replace("_", r"\_")
        lines.append(
            f"{label} & {len(group)} & {fmt(group.sisdr_in.mean())} & "
            f"{fmt(group.sisdr_out.mean())} & {fmt(mean)} [{fmt(lo)}, {fmt(hi)}] & "
            f"{fmt(group.stoi_out.mean())} & {fmt(group.pesq_out.mean())}" + ROW
        )
    return "\n".join(lines)


def difficulty_body(df, column):
    data = df.dropna(subset=[column]).copy()
    data["bin"] = pd.cut(data[column], bins=6)
    lines = []
    for interval, group in data.groupby("bin", observed=True):
        mean, _, _ = boot_ci(group["sisdri"])
        label = f"[{interval.left:.1f},{interval.right:.1f}]"
        lines.append(f"${label}$ & {len(group)} & {fmt(mean)} & {fmt(group.pesq_out.mean())}" + ROW)
    return "\n".join(lines)


def noise_body(df):
    lines = []
    for name, group in df[df.noise_env != "none"].groupby("noise_env", sort=True):
        mean, _, _ = boot_ci(group["sisdri"])
        lines.append(
            f"{name} & {len(group)} & {fmt(mean)} & {fmt(group.stoi_out.mean())} & "
            f"{fmt(group.pesq_out.mean())}" + ROW
        )
    return "\n".join(lines)


def load(args):
    df = pd.read_csv(args.metrics)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise SystemExit("metrics CSV is missing: " + ", ".join(sorted(missing)))
    if "sisdri" not in df:
        df["sisdri"] = df["sisdr_out"] - df["sisdr_in"]
    if args.manifest:
        manifest = pd.read_csv(args.manifest)
        if "idx" in df and "idx" in manifest:
            df = df.merge(manifest, on="idx", how="left", validate="one_to_one")
        elif len(df) == len(manifest):
            df = pd.concat([df.reset_index(drop=True), manifest.reset_index(drop=True)], axis=1)
        else:
            raise SystemExit("metrics and manifest need a shared idx or equal row counts")
    if "noise_env" not in df and "noise_path" in df:
        df["noise_env"] = (df["noise_path"].astype("string")
                            .str.extract(r"DEMAND/([^/]+)/", expand=False).fillna("none"))
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, help="test_mix.csv; required for breakdowns")
    parser.add_argument("--out-dir", type=Path, default=Path("generated_tables"))
    args = parser.parse_args()
    df = load(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    nonclean = df[df.scenario != "clean"] if "scenario" in df else df
    (args.out_dir / "headline_nonclean.tex").write_text(headline(nonclean) + "\n")
    print(f"wrote headline_nonclean.tex ({len(nonclean)} rows)")
    if "scenario" not in df or "sir" not in df or "snr" not in df:
        raise SystemExit("manifest metadata is required for scenario/SIR/SNR tables")
    (args.out_dir / "by_scenario.tex").write_text(scenario_body(df) + "\n")
    (args.out_dir / "by_sir.tex").write_text(difficulty_body(df, "sir") + "\n")
    (args.out_dir / "by_snr.tex").write_text(difficulty_body(df, "snr") + "\n")
    if "noise_env" not in df:
        raise SystemExit("manifest must contain noise_path or noise_env")
    (args.out_dir / "by_noise_environment.tex").write_text(noise_body(df) + "\n")
    print(f"wrote breakdown tables to {args.out_dir}")


if __name__ == "__main__":
    main()
