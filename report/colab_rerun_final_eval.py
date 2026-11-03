# ---------------------------------------------------------------------------
# Run this in the TRAINING notebook, AFTER the cell that builds `cache`
# (cache = outputs of the FINAL best checkpoint). It replaces the stale
# test_metrics.csv that the later analysis cells were reading.
# Then change the cell that does  df = pd.read_csv(f"{PROJ}/test_metrics.csv")
# to read test_metrics_final.csv, and re-run the breakdown / bootstrap / scenario /
# speaker-confusion cells. Paste the new tables into main.tex (or give them to your agent).
# ---------------------------------------------------------------------------
from pystoi import stoi
from pesq import pesq
rows = []
for i in tqdm(range(len(test_ds))):
    mix, tgt, est = cache[i]
    def _pesq(x):
        try: return pesq(SR, tgt, x, "wb")
        except Exception: return np.nan
    rows.append(dict(idx=i,
        sisdr_in=si_sdr(mix, tgt), sisdr_out=si_sdr(est, tgt),
        stoi_in=stoi(tgt, mix, SR), stoi_out=stoi(tgt, est, SR),
        pesq_in=_pesq(mix), pesq_out=_pesq(est)))
df = pd.DataFrame(rows); df["sisdri"] = df.sisdr_out - df.sisdr_in
df.to_csv(f"{PROJ}/test_metrics_final.csv", index=False)
print("checkpoint epoch:", ck.get("epoch"), "| val SI-SDRi stored:", ck.get("val_si_sdri"))
print(df[df.index.isin(range(len(df)))].describe().loc[["mean","std"]].round(2))
