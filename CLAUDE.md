# DataChallenge2026 — Project Guide for Agents

## Challenge Overview

**Task:** Predict the percentage of face occlusion in an image as a single regression value (0–1).

- **Provider:** Idemia / Telecom Paris
- **Deadline:** June 12, 23:59
- **Submissions:** Max 10; each overwrites the previous; last submission is final
- **Report:** ≤ 2 pages PDF submitted to Moodle; must include a link to the code repository and reproduction steps

**Data:**
- ~100k training images (224×224 WEBP) at `data/Crop_224_5fp_100K/`
- Training labels + gender: `data/occlusion_datasets/train.csv` (columns: `filename`, `FaceOcclusion`, `gender`)
- Test set (~30k samples): `data/occlusion_datasets/test_students.csv` (column: `filename`)
- Submission CSV columns: `filename`, `FaceOcclusion`, `gender` — set `gender='x'` for the test set

---

## Evaluation Metric

**Weighted MSE per gender group:**
```
Err = Σ(w_i · (p_i − GT_i)²) / Σ(w_i)    where w_i = 1/30 + GT_i
```
Weights increase with occlusion level, so heavily occluded faces matter more.

**Final score (lower is better):**
```
Score = (Err_Female + Err_Male) / 2  +  |Err_Female − Err_Male|
```
The second term penalises gender disparity — the model must perform similarly on both genders.

---

## Architecture

```
train_dino_trainval.py     # Main training script (self-contained, no config YAMLs)
post_processing_inject_noise.py             # Fairness post-processing: Gaussian noise on estimated gender
post_processing_read_probe.py               # Leaderboard-based noise calibration utility

src/
  config.py                 # Absolute paths (IMG_DIR, CSV_DIR, SUBMISSION_DIR, …),
                            #   DEVICE (MPS → CUDA → CPU), NUM_WORKERS
  data/
    dataset.py              # Dataset — torch Dataset: returns (image, label, gender,
                            #   filename, iw, pi) for training; (image, filename) for test
    data_utils.py           # get_challenge_split(): loads CSVs, 80/20 split,
                            #   applies distribution reweighting via importance weights
    data_stats.py           # distribution_adaptation_reweight(): KL-divergence ratio
                            #   get_test_distribution_from_screenshot(): reads test
                            #   distribution from test_distribution.png at repo root

notebooks/
  diagnostic_dinov3.ipynb              # Val prediction analysis (scatter, weighted MSE,
                                       #   worst predictions, gender balance)
  explainable_DINO_clean_trainval.ipynb  # Per-patch heatmap visualisation of final model

data/
  Crop_224_5fp_100K/        # Training images (224×224 WEBP)
  occlusion_datasets/
    train.csv               # filename, FaceOcclusion, gender
    test_students.csv       # filename (no labels)

dino_trainval_final.pt      # Final trained model checkpoint
raw_prediction.csv          # Raw test-set predictions (input to post-processing)
checkpoints/                # Intermediate training checkpoints
submission/                 # postproc/ subfolder for noise-injection outputs
test_distribution.png       # Test-set occlusion histogram (used for domain reweighting)
```

---

## Model

**Backbone:** `vit_base_patch16_dinov3.lvd1689m` (timm, pretrained)
**Head:** shared 2-layer MLP (768 → 128 → 3) applied independently to each patch token

Forward pass:
```
backbone.forward_features(x)          → (B, N, 768)   [CLS, reg×4, patch×196]
patch_tokens = features[:, num_prefix:, :]             → (B, 196, 768)
logits  = head(patch_tokens)                           → (B, 196, 3)  [bg / visible / occluded]
probs   = softmax(logits)
ratio   = Σ p_occ / (Σ p_vis + Σ p_occ)  ∈ [0, 1]   → (B, 1)
```

---

## Pipeline

### Step 1 — Training (`train_dino_trainval.py`)

Two-phase training, no early stopping, fixed epoch counts:

| Phase | Backbone | Head LR | Backbone LR | Epochs |
|---|---|---|---|---|
| 1 — Warmup | frozen | 1e-3 | — | 5 |
| 2 — Fine-tune | unfrozen | 5e-4 | 2e-5 | 6 |

- **Dataset:** train + val combined (~100k images), 17 bad-annotation images excluded (`EXCLUDE_FILES`)
- **Loss:** weighted MSE: `Σ(iw × pi × (pred − gt)²) / Σ(iw × pi)` with `pi = 1/30 + gt`
- **Augmentation:** random horizontal flip, rotation (±15°), color jitter, grayscale, Gaussian blur — no `RandomErasing`
- **Outputs:** `checkpoints/clean_trainval_phase1.pt`, `dino_trainval_final.pt`, `raw_prediction.csv`

### Step 2 — Fairness post-processing (`post_processing_inject_noise.py`)

Estimates gender on the test set via DINOv3 CLS tokens + logistic regression (trained on train split), then injects Gaussian noise `N(0, var)` on the predictions for images estimated as the target gender. CLS tokens are cached in `cache_cls/` as parquet files.

CLI args: `--test-csv`, `--genre {F,M}`, `--var`

Outputs: `data/results/test_genre.csv`, `data/results/confusion_matrix.csv`, `data/results/test_{genre}_{var}.csv`

### Step 3 — Noise calibration (`post_processing_read_probe.py`)

Given two leaderboard scores (before/after noise injection) and the confusion matrix from Step 2, estimates the optimal noise variance `v* = G / b` analytically and regenerates a submission at that variance.

CLI args: `--s0`, `--s1`, `--var`, `--genre`, `--confusion`, `--test-csv`, `--test-genre`

---

## How to Run

```bash
# Step 1 — Train (outputs checkpoint + test.csv)
python train_dino_trainval.py

# Step 2 — Fairness post-processing
python post_processing_inject_noise.py \
    --test-csv raw_prediction.csv \
    --genre F \
    --var 0.0025

# Step 3 — Calibrate noise from leaderboard probe (optional)
python post_processing_read_probe.py \
    --s0 0.00108 --s1 0.00426 --var 0.0025 --genre F \
    --confusion data/results/confusion_matrix.csv \
    --test-csv raw_prediction.csv \
    --test-genre data/results/test_genre.csv
```

---

## How to Modify / Extend

### Change the training loss
Edit `weighted_mse()` in [train_dino_trainval.py](train_dino_trainval.py) directly.

### Change the model head
Edit `PatchHead` and `PatchOcclusionModel` in [train_dino_trainval.py](train_dino_trainval.py).

### Change importance weights or sampling distribution
Edit `distribution_adaptation_reweight()` in [src/data/data_stats.py](src/data/data_stats.py).
`N_BINS = 20` is hardcoded there and in [src/data/data_utils.py](src/data/data_utils.py).

### Change the train/val split
Edit `get_challenge_split()` in [src/data/data_utils.py](src/data/data_utils.py).

---

## Maintaining This File

Update this file whenever:
- A source file is added, renamed, or deleted → update the Architecture section
- The pipeline steps change → update the Pipeline section
- Challenge rules or submission format change → update Challenge Overview / Evaluation
