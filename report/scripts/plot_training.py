"""Regenerate figures/training_curves.pdf and figures/lr_schedule.pdf.
Append new epochs to data/training_log.csv (same columns) and re-run:
    python scripts/plot_training.py
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8.5, "axes.grid": True, "grid.alpha": .25,
                     "grid.linewidth": .5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight",
                     "pdf.fonttype": 42})
df = pd.read_csv("data/training_log.csv")
# The final run uses explicit SI-SDR column names; retain aliases for older logs.
df = df.rename(columns={
    "train_si_sdr": "train_sisdr",
    "val_si_sdr": "val_sisdr",
    "val_si_sdri": "val_sisdri",
    "train_loss": "loss",
})
required = {"epoch", "loss", "train_sisdr", "val_sisdr", "val_sisdri"}
missing = required - set(df.columns)
if missing:
    raise SystemExit("training log is missing columns: " + ", ".join(sorted(missing)))

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.45))
ax[0].plot(df.epoch, df.train_sisdr, "-", lw=1.4, color="0.10",
           marker="o", ms=2.6, markevery=4, label="train SI-SDR")
ax[0].plot(df.epoch, df.val_sisdr, "--", lw=1.2, color="0.35",
           marker="s", ms=2.6, markevery=4, label="val SI-SDR")
ax[0].plot(df.epoch, df.val_sisdri, "-.", lw=1.2, color="0.60",
           marker="^", ms=3.0, markevery=4, label="val SI-SDRi")
b = df.loc[df.val_sisdri.idxmax()]
ax[0].plot(b.epoch, b.val_sisdri, "*", ms=9, color="0.0", zorder=5)
ax[0].annotate(f"{b.val_sisdri:.2f} dB\n(epoch {int(b.epoch)})",
               xy=(b.epoch, b.val_sisdri), xytext=(-40, 10),
               textcoords="offset points", fontsize=7,
               arrowprops=dict(arrowstyle="-", lw=.6, color="0.3"))
ax[0].set(xlabel="epoch", ylabel="SI-SDR (dB)")
ax[0].set_xlim(df.epoch.min() - 1, df.epoch.max() + 1)
ax[0].legend(frameon=False, fontsize=7.2, loc="lower right", handlelength=2.4)
ax[1].plot(df.epoch, df.loss, "-", lw=1.3, color="0.20",
           marker="o", ms=2.6, markevery=4)
ax[1].set(xlabel="epoch", ylabel="training loss")
ax[1].set_xlim(df.epoch.min() - 1, df.epoch.max() + 1)
plt.tight_layout(pad=0.4); plt.savefig("figures/training_curves.pdf"); plt.close()

# learning-rate schedule: 2000-step linear warm-up to 5e-4, cosine decay to 1e-5 over 100k steps
total, warm, lr_max, lr_min = 100_000, 2_000, 5e-4, 1e-5
s = np.arange(total)
lr = np.where(s < warm, lr_max * s / warm,
              lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * (s - warm) / (total - warm))))
cur = int(df.epoch.max()) * 1000
fig, ax = plt.subplots(figsize=(3.6, 2.4))
ax.plot(s / 1000, lr, color="0.2", lw=1.2); ax.axvline(cur / 1000, ls="--", color="0.5")
ax.text(cur / 1000 + 1, lr_max * .8, f"epoch {int(df.epoch.max())}", fontsize=8)
ax.set(xlabel="epoch (1000 steps each)", ylabel="learning rate", title="LR schedule")
plt.tight_layout(); plt.savefig("figures/lr_schedule.pdf"); plt.close()
print("ok, lr at epoch", df.epoch.max(), "=", lr[min(cur, total-1)])
