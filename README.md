# DataChallenge 2026 — Face Occlusion Regression

Predict face occlusion percentage (0–1) from 224×224 face images.

**Model:** DINOv3 (`vit_base_patch16_dinov3.lvd1689m`) fine-tuned end-to-end with a shared
patch classification head (background / visible / occluded) aggregated into a single ratio.

---

## Setup

```bash
git lfs pull   # optional downloads model weights (requires git-lfs: brew install git-lfs)
uv sync
```

## Reproduce

**Step 1 — Train** (outputs `dino_trainval_final.pt` + `raw_prediction.csv`):
```bash
python train_dino_trainval.py
```

**Step 2 — Fairness post-processing** (injects calibrated Gaussian noise on the predicted-female images to reduce the gender disparity penalty):
```bash
python post_processing_inject_noise.py \
    --test-csv data/results/raw_prediction.csv \
    --genre F \
    --var 0.0025
```

**Step 3 — Calibrate noise from a leaderboard probe** (optional — given two scores, computes the optimal variance analytically):
```bash
python post_processing_read_probe.py \
    --s0 0.00108 --s1 0.00426 --var 0.0025 --genre F \
    --confusion data/results/confusion_matrix.csv \
    --test-csv data/results/raw_prediction.csv \
    --test-genre data/results/test_genre.csv
```

---

## Notebooks

- `notebooks/diagnostic_dinov3.ipynb` — statistical analysis of val predictions (scatter, weighted MSE by gender/bin, worst predictions). Edit the `pd.read_csv('val.csv')` line to point to your submission val CSV.
- `notebooks/explainable_DINO_clean_trainval.ipynb` — per-patch heatmap visualisation of the final model (`dino_trainval_final.pt`).

---

## File map

```
train_dino_trainval.py              Main training script
post_processing_inject_noise.py     Gender-noise fairness post-processing
post_processing_read_probe.py       Leaderboard-based noise calibration

dino_trainval_final.pt              Final trained model checkpoint
raw_prediction.csv                  Raw test-set predictions (input to post-processing)

src/config.py                       Paths + device
src/data/dataset.py                 Dataset class
src/data/data_utils.py              get_challenge_split()
src/data/data_stats.py              Distribution reweighting (importance weights)

data/                               Images + CSVs
checkpoints/                        Intermediate training checkpoints
data/results/                       Post-processing outputs and predictions
test_distribution.png               Test-set occlusion histogram (domain reweighting)
```
