# Context for AI coding agents (Antigravity / Codex / OpenCode / Claude Code)

This repo is a LaTeX mid-term progress report for the SIC-AI capstone
"Speaker-Aware Noise Cancellation and Target Speech Enhancement" (Pulchowk Campus).

## Files
- `main.tex` — the whole report. Every unverified item is marked `\todo{...}` (red in the PDF).
- `references.bib` — bibliography (BibTeX, `ieeetr` style, natbib).
- `figures/` — spectrogram PNGs from the test notebook, plus generated PDFs.
- `data/training_log.csv` — per-epoch training log (epochs 34–100).
- `scripts/plot_training.py` — regenerates `figures/training_curves.pdf` and `figures/lr_schedule.pdf` from the CSV.

## Rules
1. **Never invent numbers.** Results come only from the notebooks/logs the authors provide.
   If a value is missing, leave or add a `\todo{...}`.
2. Keep claims consistent with the code: BSRNN, N=128, hidden 256, 8 layers, 28 bands, 16.8M params,
   n_fft 512 / hop 128, frozen ECAPA-TDNN (192-d), FiLM in every layer, loss = -SI-SDR(clamped 30 dB) + 5*MR-STFT.
3. The model is **non-causal** (bidirectional LSTMs). Do not describe it as real-time.
4. The test SI-SDR/STOI/PESQ tables (test, scenario, difficulty, noise-env) come from an EARLIER checkpoint (stale test_metrics.csv).
   Training finished at 100 epochs; best checkpoint = epoch 97 (val SI-SDRi 11.84 dB). Speaker confusion, WER, efficiency and the
   qualitative examples are from the final checkpoint. After `scripts/colab_rerun_final_eval.py` is run, replace the stale tables and
   update the 'evaluation status' wording (Table evalstatus, captions, abstract, conclusion).
5. Clean-scenario items make aggregate SI-SDRi misleading (input SI-SDR ~105 dB); headline numbers exclude them.
6. Build with `latexmk -pdf main.tex`. Before submission set `\showtodofalse` in `main.tex`.

## Typical tasks
- Append new epochs to `data/training_log.csv`, run `python scripts/plot_training.py`, update the training table/text.
- Replace the test table after re-evaluating on non-clean items.
- Fill in Conv-TasNet baseline results in the "Changes from the Proposal" section.
