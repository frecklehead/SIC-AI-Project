"""Regenerate figures/training_curves.pdf and figures/lr_schedule.pdf.
Append new epochs to data/training_log.csv (same columns) and re-run:
    python scripts/plot_training.py
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})
df = pd.read_csv("data/training_log.csv")

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax[0].plot(df.epoch, df.train_sisdr, "o-", ms=3, label="train SI-SDR")
ax[0].plot(df.epoch, df.val_sisdr, "s-", ms=3, label="val SI-SDR")
ax[0].plot(df.epoch, df.val_sisdri, "^-", ms=3, label="val SI-SDRi")
b = df.loc[df.val_sisdri.idxmax()]
ax[0].plot(b.epoch, b.val_sisdri, "k*", ms=9, label=f"best val SI-SDRi ({b.val_sisdri:.2f} dB, ep. {int(b.epoch)})")
ax[0].set(xlabel="epoch", ylabel="dB", title="SI-SDR progress"); ax[0].legend(frameon=False, fontsize=7)
ax[1].plot(df.epoch, df.loss, "o-", ms=3, color="tab:red")
ax[1].set(xlabel="epoch", ylabel="training loss", title="Combined loss")
plt.tight_layout(); plt.savefig("figures/training_curves.pdf"); plt.close()

# learning-rate schedule: 2000-step linear warm-up to 5e-4, cosine decay to 1e-5 over 100k steps
total, warm, lr_max, lr_min = 100_000, 2_000, 5e-4, 1e-5
s = np.arange(total)
lr = np.where(s < warm, lr_max * s / warm,
              lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * (s - warm) / (total - warm))))
cur = int(df.epoch.max()) * 1000
fig, ax = plt.subplots(figsize=(3.6, 2.4))
ax.plot(s / 1000, lr); ax.axvline(cur / 1000, ls="--", color="gray")
ax.text(cur / 1000 + 1, lr_max * .8, f"epoch {int(df.epoch.max())}", fontsize=8)
ax.set(xlabel="epoch (1000 steps each)", ylabel="learning rate", title="LR schedule")
plt.tight_layout(); plt.savefig("figures/lr_schedule.pdf"); plt.close()
print("ok, lr at epoch", df.epoch.max(), "=", lr[min(cur, total-1)])
