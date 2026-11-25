# Context for AI coding agents

This repository documents the SIC-AI capstone project
"Speaker-Aware Noise Cancellation and Target Speech Enhancement" (Pulchowk Campus, Tribhuvan
University).

## Repository layout

- `docs/` — compiled mid-term report, proposal and study notes.
- `notebooks/01-bsrn-target-extraction.ipynb` — main BSRNN experiments and saved outputs.
- `notebooks/02-conv-tasnet-baseline.ipynb` — earlier causal Conv-TasNet baseline.
- `report/` — LaTeX source, bibliography, figures, training log and report scripts.
- `src/` — training/evaluation code skeleton.
- `README.md` — public project overview, setup instructions and results summary.

## Report files

- `report/main.tex` — the whole report.
- `report/references.bib` — bibliography (`ieeetr`, natbib).
- `report/figures/` — report figures, including the TU logo.
- `report/data/training_log.csv` — per-epoch training log.
- `report/scripts/plot_training.py` — regenerates `figures/training_curves.pdf` and
  `figures/lr_schedule.pdf` from the CSV.
- `report/scripts/make_tables.py` — generates tables from a final-checkpoint metrics CSV.
- `report/scripts/colab_rerun_final_eval.py` — final-checkpoint evaluation rerun.

## Rules

1. **Never invent numbers or claims.** Results come only from the provided notebooks, logs,
   configuration or report sources. If a value is missing, describe it as an explicit limitation
   or pending measurement; do not silently fill it with an estimate.
2. Keep architecture claims consistent with the code: BSRNN, `N=128`, hidden 256, 8 layers,
   28 bands, 16.83 M parameters, `n_fft=512` / hop 128, frozen ECAPA-TDNN (192-d), FiLM in every
   layer, loss = `-SI-SDR` (clamped at 30 dB) + 5×MR-STFT.
3. The model is **non-causal** because it uses bidirectional LSTMs. Do not describe it as
   real-time or streaming.
4. The test SI-SDR/STOI/PESQ tables use the **epoch-97 checkpoint** and include bootstrap
   intervals. Training finished at 100 epochs with the best checkpoint at epoch 97 (validation
   SI-SDRi 11.84 dB). Speaker confusion, WER, efficiency and qualitative examples use the same
   checkpoint. The private per-mixture CSV is not distributed.
5. Clean-scenario items make aggregate SI-SDRi misleading because the input SI-SDR is about
   105 dB. Headline numbers exclude the `clean` scenario.
6. The Conv-TasNet baseline reached 5.90 dB validation SI-SDRi at epoch 21; a 20-clip check gave
   3.88 dB on training voices and 3.60 dB on held-out voices/noise. It was dropped after BSRNN
   reached 11.84 dB. Do not revert this to an unqualified "not trained" statement.
7. The report has no active TODO-marker system. Source-backed limitations are written explicitly in
   the text rather than hidden behind conditional TODO macros.
8. Build with `cd report && latexmk -pdf -interaction=nonstopmode main.tex`. The compiled submission
   copy is `docs/midterm-progress-report.pdf`; refresh it after a successful build.

## Typical tasks

- Append new epochs to `report/data/training_log.csv`, run `python scripts/plot_training.py` from
  `report/`, and update the training table/text.
- Regenerate the test tables if a revised checkpoint or evaluation protocol is introduced.
- Keep the README, `CITATION.cff`, notebook index and report results consistent when numbers change.
