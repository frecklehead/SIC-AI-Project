# Training and evaluation code

This directory is the cleaned code skeleton extracted from
[`notebooks/01-bsrn-target-extraction.ipynb`](../notebooks/01-bsrn-target-extraction.ipynb).

It contains the command-line entry points for training and final evaluation, but **not** the
project-specific `tselib.py`, `config.yaml`, manifests, embedding banks, audio corpora or
checkpoints. Those files were loaded from the private project directory during the capstone and
are not distributed here. The scripts fail with a useful message when `tselib.py` is missing; no
hidden implementation has been invented.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Required artifacts

Place these in the project directory passed to the scripts:

- `tselib.py`;
- the project `config.yaml`;
- LibriSpeech and DEMAND audio;
- train/validation/test mixture manifests and cached ECAPA-TDNN embedding banks;
- the epoch-97 checkpoint `tse_bsrnn_big_best.pt` (for evaluation).

## Training

The defaults match the final notebook configuration: `N=128`, hidden size 256, 8 dual-path layers,
batch size 8, 1000 steps per epoch, and 100 epochs.

```bash
PYTHONPATH=/path/to/tse-project python src/train.py \
  --project-dir /path/to/tse-project \
  --data-dir /path/to/tse-project/DATA
```

The model uses bidirectional LSTMs and is non-causal.

## Final evaluation

```bash
PYTHONPATH=/path/to/tse-project python src/evaluate.py \
  --project-dir /path/to/tse-project \
  --data-dir /path/to/tse-project/DATA \
  --checkpoint /path/to/tse-project/checkpoints/tse_bsrnn_big_best.pt
```

`evaluate.py` writes `test_metrics_final.csv` in the current directory when the required data and
checkpoint are available. The report table generator expects that file:

```bash
python report/scripts/make_tables.py \
  --metrics test_metrics_final.csv \
  --manifest /path/to/tse-project/manifests/test_mix.csv \
  --out-dir generated_tables
```

The orchestration is based on notebook cells 4–8, 14–18 and 20–21.

## Notes

- `test_metrics_final.csv` is not committed because the final-checkpoint rerun has not been
  completed on the available artifacts.
- The test tables in the report therefore come from the earlier checkpoint; the report states this
  limitation explicitly.
- `report/scripts/plot_training.py` regenerates the training-curve figure from
  `report/data/training_log.csv`.
