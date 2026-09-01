# DyBraSS

Official PyTorch implementation of **DyBraSS: A Unified Spatiotemporal
State-Space Model for Dynamic Brain State Analysis in Resting-State fMRI**.

## Installation

```bash
conda create -n dybrass python=3.10
conda activate dybrass
pip install -r requirements.txt
```

## Repository structure

```text
conf/            Hydra configurations
dataset/         Data loading utilities
models/dybrass.py  DyBraSS implementation
training/        Training and evaluation code
utils/           Learning-rate scheduler
scripts/         Preprocessing and launch scripts
main.py          Training and evaluation entry point
```

## Data preparation

Prepare a NumPy archive containing:

- `dfc`: an object array of length `N`; each element has shape `[L_i, R, R]`
- `label`: an integer array of shape `[N]`

To generate dFC sequences from ROI time series:

```bash
python scripts/prepare_dfc.py \
  --input /path/to/roi_timeseries.npz \
  --output /path/to/dfc.npz
```

Split files use archive row indices. To generate a new five-fold split:

```bash
python scripts/make_splits.py \
  --data /path/to/dfc.npz \
  --output splits/custom/seed_42.json \
  --seed 42
```

## Training

```bash
python main.py \
  experiment=abide \
  dataset.data_path=/path/to/abide_dfc.npz \
  dataset.split_path=/path/to/seed_42.json
```

Use `experiment=adhd` or `experiment=cobre` for the other datasets. To run all predefined seeds:

```bash
bash scripts/run_all_seeds.sh \
  abide \
  /path/to/abide_dfc.npz \
  /path/to/abide_splits
```
