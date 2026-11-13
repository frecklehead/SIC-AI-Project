# Notebooks

These notebooks are the primary experimental record for the project. Their saved outputs are
retained so the reported tables, plots and qualitative examples can be inspected without rerunning
training.

## `01-bsrn-target-extraction.ipynb`

Main BSRNN run:

- data preparation and manifest checks;
- training with the final configuration (`N=128`, hidden 256, 8 layers, 100 epochs);
- validation and epoch-97 test evaluation;
- speaker-confusion analysis on the final checkpoint;
- Whisper WER check;
- efficiency measurement on an NVIDIA L4;
- six qualitative test examples and spectrograms.

The notebook depends on private data, cached ECAPA-TDNN embedding banks, `tselib.py` and the
epoch-97 checkpoint. Those artifacts are not part of this repository.

## `02-conv-tasnet-baseline.ipynb`

Earlier causal Conv-TasNet baseline:

- `N=512`, `H=512`, film2 conditioning, 3 s segments;
- batch 3 with gradient accumulation 2, learning rate `1e-3`;
- 80 epochs planned, saved best checkpoint at epoch 21;
- validation SI-SDRi 5.90 dB;
- 20-clip check: 3.88 dB on training voices and 3.60 dB on held-out voices and noise.

This run was dropped after the speaker-conditioned BSRNN reached 11.84 dB validation SI-SDRi. The
notebook is included for provenance and to document the model-selection decision.

## Reproducing the notebooks

The public repository does not contain LibriSpeech, DEMAND, manifests, embedding banks,
`tselib.py` or checkpoints. See [`../src/README.md`](../src/README.md) for the expected project
layout. Running the notebooks from scratch therefore requires the private project directory used
during the capstone.
