# Speaker-aware noise cancellation with BSRNN

This folder is the GitHub-ready code skeleton extracted from
`sources/BSRNN_SOSTIKA_RUNNING.ipynb`.

The notebook does not contain the implementation of `tselib.py`,
`config.yaml`, manifests, embedding banks, or checkpoints. Those files are
loaded from the private Google Drive project directory and must be added
before training or evaluation can run. The scripts below intentionally fail
with a useful message if `tselib.py` is absent; no hidden implementation has
been invented.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Provide `tselib.py`, `config.yaml`, the LibriSpeech/DEMAND data, manifests,
cached ECAPA-TDNN embedding banks, and the epoch-97 checkpoint
`tse_bsrnn_big_best.pt`.

## Training

```bash
PYTHONPATH=/path/to/project python src/train.py \
  --project-dir /path/to/tse-project \
  --data-dir /path/to/tse-project/DATA
```

The defaults match the notebook: `N=128`, hidden size `256`, `8` dual-path
layers, batch size `8`, `1000` steps per epoch, and `100` epochs. The model
uses bidirectional LSTMs and is non-causal.

## Final evaluation

```bash
PYTHONPATH=/path/to/project python src/evaluate.py \
  --project-dir /path/to/tse-project \
  --data-dir /path/to/tse-project/DATA \
  --checkpoint /path/to/tse-project/checkpoints/tse_bsrnn_big_best.pt
```

Then generate report tables:

```bash
python ../scripts/make_tables.py \
  --metrics test_metrics_final.csv \
  --manifest /path/to/tse-project/manifests/test_mix.csv \
  --out-dir generated_tables
```

The orchestration is based on notebook cells 4--8, 14--18 and 20--21.
