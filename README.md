# Speaker-Aware Noise Cancellation and Target Speech Enhancement

**SIC-AI Capstone Project**<br>
Tribhuvan University, Pulchowk Campus<br>
November 2026

**Authors:** Austina Aryal (079BEI010), Prashamsa Aryal (079BEI024), Sostika Shrestha (079BEI041)<br>
**Supervisors:** Arun Timilsena and Suresh Pokharel

[Read the mid-term report](docs/midterm-progress-report.pdf) · [Proposal](docs/proposal.pdf) · [Study notes](docs/study-notes-and-viva-prep.docx) · [Notebooks](notebooks/)

---

## Overview

Target speaker extraction (TSE) keeps one enrolled voice while removing competing talkers and
background noise. Unlike conventional noise suppression, which treats every voice as signal, TSE
uses a short enrollment recording to decide which speaker should be preserved.

This project studies a single-channel, speaker-conditioned **Band-Split RNN (BSRNN)** for TSE. A
frozen ECAPA-TDNN speaker encoder produces a 192-dimensional embedding, which is added once to the
band features and injected through FiLM in each of eight dual-path layers. The model is trained on
on-the-fly mixtures of LibriSpeech speech and DEMAND noise with a combined SI-SDR and
multi-resolution STFT loss.

The repository contains the research notebooks, the LaTeX report source, the compiled report, and
a training/evaluation code skeleton. It is intended to document the work honestly and make the
experimental pipeline inspectable; private data, manifests, embedding banks and checkpoints are not
distributed here.

## Research question

Can a compact speaker-conditioned separator preserve the enrolled speaker while removing both a
competing talker and environmental noise, and how does it fail when the wrong speaker is selected?

## Method at a glance

```text
enrollment x ──> ECAPA-TDNN ──> speaker embedding e
                                      │
mixture y ──> STFT ──> 28 band features ──> 8 dual-path BSRNN layers ──> masks ──> iSTFT ──> estimate
                                      ▲
                                FiLM(e) in every layer
```

- **Model:** BSRNN, `N=128`, hidden size 256, 8 dual-path layers, 28 bands.
- **Speaker encoder:** frozen ECAPA-TDNN, 192-dimensional embedding.
- **Audio:** 16 kHz, `n_fft=512`, hop length 128, 4 s training segments.
- **Training:** 100 epochs, AdamW, learning rate `5e-4` with 2000-step warmup then cosine decay.
- **Loss:** clamped SI-SDR plus 5× multi-resolution STFT loss.
- **Causality:** the model uses bidirectional LSTMs and is **non-causal**; it is not a streaming or
  real-time system.

## Results

### Model selection

| Model | Causal | Best validation SI-SDRi | Outcome |
|---|---|---:|---|
| Conv-TasNet | yes | 5.90 dB at epoch 21 | Kept as a baseline; dropped after BSRNN outperformed it |
| BSRNN | no | **11.84 dB at epoch 97** | Final model |

The Conv-TasNet run used `N=512`, `H=512`, film2 conditioning, 3 s segments, batch 3 with gradient
accumulation 2, and learning rate `1e-3`, with 80 epochs planned. A 20-clip check gave 3.88 dB on
training voices and 3.60 dB on held-out voices and noise.

### Test metrics

The table below is from the **earlier checkpoint** evaluation. The non-clean column is the headline
result; the 65 clean-scenario items are excluded because their input SI-SDR is about 105 dB.

| Metric | Non-clean (`n=1135`) | All items (`n=1200`) |
|---|---:|---:|
| SI-SDR input | 0.28 dB | 5.95 dB |
| SI-SDR output | 11.06 dB | 12.72 dB |
| **SI-SDRi** | **10.78 dB [10.34, 11.21]** | 6.77 dB [5.73, 7.76] |
| STOI | 0.772 → 0.872 | 0.785 → 0.879 |
| PESQ (wideband) | 2.20 | 2.33 |

### Final-checkpoint checks

| Metric | Result |
|---|---:|
| Speaker confusion | **7.7%** (65 of 845 interferer mixtures) |
| Whisper WER, 300 mixtures | **38.3% → 18.1%** |
| Parameters | 16.83 M |
| Compute | 23.87 GMACs per 6 s clip |
| Offline RTF on NVIDIA L4, fp32 | 0.0275 |

The RTF is an offline GPU measurement for a non-causal model. It does not establish real-time
latency. CPU speed has not been measured.

## Repository layout

```text
.
├── docs/                 # Compiled report, proposal and study notes
├── notebooks/            # BSRNN and Conv-TasNet notebooks with saved outputs
├── report/               # LaTeX source, bibliography, figures, data and report scripts
├── src/                  # Training and evaluation code skeleton
├── CITATION.cff
├── README.md
└── .gitignore
```

- `docs/midterm-progress-report.pdf` is the compiled mid-term report.
- `report/main.tex` is the report source. `report/figures/`, `report/data/` and `report/scripts/`
  contain the assets needed to rebuild it.
- `notebooks/01-bsrn-target-extraction.ipynb` contains the main BSRNN training, evaluation,
  confusion, WER, efficiency and qualitative-example outputs.
- `notebooks/02-conv-tasnet-baseline.ipynb` contains the earlier Conv-TasNet baseline run.
- `src/` contains the cleaned training and evaluation entry points extracted from the notebook.

## Setup

### 1. Clone the repository

```bash
git clone git@github.com:frecklehead/SIC-AI-Project.git
cd SIC-AI-Project
```

### 2. Create a Python environment

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
```

### 3. Add the private project artifacts

The public repository does **not** include:

- LibriSpeech or DEMAND audio;
- mixture manifests and cached ECAPA-TDNN embedding banks;
- the project-specific `tselib.py` implementation;
- the epoch-97 checkpoint `tse_bsrnn_big_best.pt`.

The scripts in `src/` intentionally fail with a clear message when `tselib.py` is missing. This is
not a one-command reproduction of the paper; it is a transparent code skeleton plus the analysis
notebooks. See [`src/README.md`](src/README.md) for the expected directory layout and commands.

### 4. Open the notebooks

```bash
jupyter lab notebooks/
```

The notebooks retain their saved outputs, so the reported tables and figures can be inspected
without rerunning training.

### 5. Build the report

TeX Live and `latexmk` are required.

```bash
cd report
latexmk -pdf -interaction=nonstopmode main.tex
```

The build writes `report/main.pdf`. The committed submission copy is
`docs/midterm-progress-report.pdf`; refresh it after a successful build with:

```bash
cp report/main.pdf ../docs/midterm-progress-report.pdf
```

For Overleaf, upload the `report/` directory as a project, select **pdfLaTeX** and set `main.tex`
as the main document.

## Reproducibility notes

- The best validation checkpoint is epoch 97 with **11.84 dB** SI-SDRi on unseen speakers.
- The test SI-SDR/STOI/PESQ tables come from an earlier checkpoint. The final-checkpoint rerun is
  documented in the report as a remaining task; `test_metrics_final.csv` is not present in this
  repository.
- Speaker confusion, Whisper WER, efficiency and the six qualitative examples use the final epoch-97
  checkpoint.
- The DEMAND recordings are split by time, not by unseen noise environment. The test set therefore
  shares the same 16 environments as training.
- The model is non-causal, has no target-absent training data, and is evaluated without
  reverberation.
- Clean-scenario SI-SDRi is a metric artifact because the input SI-SDR is about 105 dB; headline
  numbers exclude those 65 items.

## Limitations and next steps

- Re-evaluate the final epoch-97 checkpoint on the full test set and replace the earlier-checkpoint
  tables with bootstrap confidence intervals.
- Build a causal or low-latency variant if the deployment target requires streaming.
- Add target-absent training and evaluation, unseen-noise tests, and reverberant conditions.
- Reduce speaker confusion, especially for 2–3 s enrollments, where the error rate is highest.
- Measure CPU inference cost and memory, not only offline GPU throughput.

## Citation

```bibtex
@techreport{aryal2026speakeraware,
  author      = {Austina Aryal and Prashamsa Aryal and Sostika Shrestha},
  title       = {Speaker-Aware Noise Cancellation and Target Speech Enhancement:
                 Mid-Term Progress Report},
  institution = {Tribhuvan University, Pulchowk Campus},
  year        = {2026},
  type        = {SIC-AI Capstone Mid-Term Report},
  url         = {https://github.com/frecklehead/SIC-AI-Project}
}
```

## Acknowledgements

This work was carried out under the SIC-AI Capstone Project at Pulchowk Campus, Tribhuvan
University. We thank our supervisors, Arun Timilsena and Suresh Pokharel, for their guidance.
