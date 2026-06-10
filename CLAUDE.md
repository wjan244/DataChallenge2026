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

Implemented in [src/metrics.py](src/metrics.py): `error_fn()` and `metric_fn()`.

---

## Code Architecture

```
main.py                         # Runs the 3-stage pipeline, scratch, CNN finetuning, or DINOv3 pipeline
test.py                         # Single-batch smoke test (monkey-patches DataLoader)

src/
  config.py                     # Absolute paths (IMG_DIR, CSV_DIR, CHECKPOINT_DIR…),
                                #   DEVICE (MPS → CUDA → CPU), NUM_WORKERS
  config_utils.py               # load_config(): deep-merges pipeline_default.yaml
                                #   with a model-specific YAML

  metrics.py                    # error_fn(), metric_fn() — the competition metric

  data/
    dataset.py                  # Dataset, ChallengeTrain, CelebA — torch Dataset classes
    data_loader.py              # Factory functions returning DataLoaders
                                #   num_workers=NUM_WORKERS, pin_memory (CUDA only),
                                #   persistent_workers enabled by default
    data_utils.py               # get_challenge_split(): loads CSV, 80/20 split,
                                #   applies distribution reweighting
    data_stats.py               # distribution_adaptation_reweight(): computes iw/pi columns
                                #   get_test_distribution_from_screenshot(): reads test
                                #   distribution from data/test_distribution.png
    transforms.py               # torchvision.transforms.v2 augmentation pipelines

  models/
    models.py                   # OcclusionModel (timm backbone + Sigmoid head)
                                #   get_model(): factory; applies finetuning method
                                #   CUSTOM_MODELS dict maps name → class (bypasses timm):
                                #   "convnet" → ConvNet, "resnet18" → ResNet18, "efficientnet" → EfficientNet
    scratch_cnn.py              # ConvNet: 4-layer CNN trained from scratch
                                #   _ConvBlock(Conv2d→BN→ReLU→MaxPool), AdaptiveAvgPool, Linear head
                                # ResNet18: custom 18-layer ResNet (stem + 4 layer groups of ResBlocks)
                                # EfficientNet: thin wrapper around timm efficientnet_b0 (pretrained=False)
    finetuning.py               # inject_lora_transformer(): replaces qkv/query/key/value
                                #   Linear layers with LoRALinear
                                #   inject_linear_mlp_probing(): replaces model head
    lora.py                     # LoRALinear: frozen base + trainable low-rank A·B
    loss.py                     # WeightedMSELoss (nMSE), WeightedLiteMSELoss (nLiteMSE),
                                #   PWGLoss, PWGLossRegularized, HuberPWGLossRegularized — gender-weighted losses
                                #   CompoundLoss — meta-loss with exponent params gamma (iw), kappa (gw),
                                #     alpha (gender disparity), beta (Huber delta); interpolates between
                                #     PLoss (gamma=kappa=0) and full PWGHuber (gamma=kappa=1)
                                #   PWScore — competition metric as nn.Module (used for val early stopping)
                                #   PScore — like PWScore but uses only pi weights (no iw)
                                #   UniversalLossWrapper — routes by loss_name string via LOSS_MAPPING

  pipeline/
    train.py                    # run_train(): Adam + CosineAnnealingLR, early stopping,
                                #   MLflow logging (incl. method_kwargs + epoch/total time),
                                #   checkpoint saving
    evaluation.py               # run_evaluation(index=): gender-split metric; F1/acc for
                                #   domain-adaptation stage; index suffix avoids MLflow
                                #   metric overwrite when called twice in the same run
    test.py                     # run_test(): loads best checkpoint → submission CSV
    run_domain_adaptation.py    # Stage 1 entry point
    run_probing.py              # Stage 2 entry point
    run_lora.py                 # Stage 3 entry point
    run_scratch.py              # Scratch entry point (single stage, no checkpoint chaining)
    run_cnn_ft.py               # CNN finetuning entry point: head warmup → progressive block
                                #   unfreezing; all phases in a single MLflow run
                                #   _train_phase(): shared train loop used by run_scratch too;
                                #     optimizer threaded across phases (Adam momentum preserved);
                                #     val_score computed globally over full val set via PWScore;
                                #     logs val_err_female + val_err_male per epoch

config/
  pipeline_default.yaml         # Global defaults: SEED=42, BATCH_SIZE=32, N_SAMPLE=5000,
                                #   PATIENCE=5, N_BINS=20, NUM_CLASSES=1
  models/
    beit3_base_patch16_224.yaml        # Per-model hyperparameter overrides
    vit_tiny_patch16_224.yaml
    vit_tiny_patch16_224_no_celeba.yaml  # Minimal run — skips CelebA stage
    cnn_4l.yaml                        # 4-layer ConvNet trained from scratch
    efficient_net.yaml                 # EfficientNetV2-S finetuned via CNN finetuning pipeline
    dino.yaml                          # DINOv3 LP config (type: dino_lp)
    dino_cnn.yaml                      # DINOv3 PatchCNN config (type: dino_cnn)
    dino_cnn_ft.yaml                   # DINOv3 end-to-end unfreeze config (type: dino_unfreeze)
  optuna/                        # Optuna result CSVs saved here per run

src/dino/
  embed.py                      # Step 1: extract DINOv3 embeddings → disk
                                #   Saves {split}_cls.pt, {split}_patch_mean.pt,
                                #   {split}_meta.csv, {split}_patches.bin (numpy memmap fp16)
                                #   Noisy samples (validation_noisy.csv) always split 80/20
                                #   and appended to both train and val
  utils.py                      # Shared: EmbeddingDataset (LP), PatchDataset (CNN),
                                #   ImageDataset (unfreeze — loads raw images from IMG_DIR,
                                #     reads {split}_meta.csv from embedding_dir for same split),
                                #   compute_laplacian_iw(), eval_epoch / eval_epoch_cnn,
                                #   eval_epoch_image (image loader variant for unfreeze),
                                #   eval_final_cnn (PScore without iw),
                                #   save_submission / save_submission_cnn, load_config
  run_lp.py                     # Linear probe on frozen embeddings: LinearProbe, train_lp,
                                #   build_loss, run_lp
  run_cnn.py                    # PatchCNN on full patch embeddings: PatchCNN, train_cnn,
                                #   run_cnn; logs val_Pscore after training
  run_optuna.py                 # Optuna search: objective_lp (discrete loss search),
                                #   objective_cnn (CompoundLoss param search), run_optuna
  run_unfreeze.py               # End-to-end finetuning of DINOv3 backbone + PatchCNN head:
                                #   DinoFinetuneModel: HF backbone + PatchCNN head;
                                #     forward() extracts CLS + patch grid from last_hidden_state,
                                #     skips register tokens via patch_start = 1 + n_reg
                                #   _get_layers_and_norm(): auto-detects layer path across HF
                                #     wrapper depths (DINOv3Model outer vs DINOv3ViTModel inner)
                                #   build_dino_ft_model(): freeze all → load head checkpoint →
                                #     selectively unfreeze top n_blocks: attention + mlp +
                                #     layer_scale1/2 only; norm1/norm2/final norm stay frozen
                                #   _build_image_transform(): uses AutoImageProcessor for
                                #     exact model mean/std
                                #   _build_optimizer(): AdamW with two param groups
                                #     (learning_rate_head / learning_rate_backbone)
                                #   _make_scheduler(): LambdaLR per group; optional linear
                                #     warmup on backbone only (head already finetuned)
                                #   train_unfreeze(): early stopping on CLEAN val (no noisy),
                                #     grad norm clipping + per-epoch averaged MLflow logging
                                #   save_submission_unfreeze(): test CSV + final val scoring
                                #     on NOISY val after best model is reloaded

scripts/
  run_cluster.sbatch            # SLURM job script for cluster training
  run_cluster_l_embed.sbatch    # SLURM job script for embedding extraction
  mean_norm.py                  # One-shot script: computes per-channel mean & std of the
                                #   training set (averaged over batches); use the output to
                                #   replace the ImageNet defaults in _get_transform() for
                                #   scratch CNN models
```

---

## Pipelines

`main.py` selects the pipeline based on the config:
- `type: dino_lp` → DINOv3 linear probe (with Optuna if `optuna_n_trials` is set)
- `type: dino_cnn` → DINOv3 PatchCNN (with Optuna if `optuna_n_trials` is set)
- `type: dino_unfreeze` → DINOv3 end-to-end finetuning (backbone + PatchCNN head)
- `scratch_training.run_execution: True` → scratch pipeline
- `cnn_ft_training.run_execution: True` → CNN finetuning pipeline
- otherwise → 3-stage transformer pipeline

### 3-Stage Pipeline (pretrained ViT/BEiT)

| Stage | Script | Dataset | What it trains |
|---|---|---|---|
| 1 — Domain Adaptation | `run_domain_adaptation.py` | CelebA | Full backbone or LoRA, adapts to face domain |
| 2 — Probing | `run_probing.py` | Challenge | Frozen backbone + new MLP head |
| 3 — LoRA Finetuning | `run_lora.py` | Challenge | LoRA weights + head |

Run IDs are chained: each stage passes `precedent_run_id` so the next stage loads the previous checkpoint via MLflow.

### Scratch Pipeline (CNN from scratch)

Single stage: `run_scratch.py` trains any `CUSTOM_MODELS` entry (e.g. `ConvNet`, `ResNet18`, `EfficientNet`) from random init with all parameters unfrozen. No checkpoint chaining. Triggered by `scratch_training.run_execution: True` in the config. Per-epoch competition metrics (`val_err_female`, `val_err_male`, `val_score`) are logged to MLflow during training by reusing val-loop predictions (no extra forward pass).

### CNN Finetuning Pipeline (pretrained CNN — e.g. EfficientNet)

Single MLflow run, multiple internal phases in `run_cnn_ft.py`. Triggered by `cnn_ft_training.run_execution: True`.

| Phase | What happens |
|---|---|
| 0 — Head warmup | Backbone fully frozen, only MLP head trains at `learning_rate` |
| 1..n_phases | Progressively unfreeze top blocks (`ceil(total_blocks / n_phases)` per phase), LR = `learning_rate × lr_decay_factor^phase` |

Key config keys (under `cnn_ft_training`): `learning_rate`, `num_epoch_head`, `num_epoch_per_phase`, `n_phases`, `lr_decay_factor`. Block count per phase is computed automatically from the model. Requires `method_kwargs.probing_type` and `method_kwargs.hidden_size` for the MLP head. Optional `loss_alpha` scales the gender-disparity term in `PWGLossRegularized` (default 1.0 if omitted); applies to `scratch_training` the same way.

**Optimizer continuity:** the Adam optimizer is threaded across all phases. Newly unfrozen blocks are appended as a new param group via `add_param_group`; all existing groups have their LR lowered to the current phase LR. Adam's `m`/`v` moment estimates are preserved for the head and any blocks already training.

**Validation scoring:** `val_score` (used for early stopping and checkpointing) is computed by accumulating all val-set predictions into a single tensor and calling `PWScore` once — not averaged per batch. `val_err_female` and `val_err_male` are also logged each epoch at no extra cost.

Run everything: `python main.py`  
Run scratch CNN: `python main.py --config cnn_4l.yaml`  
Run CNN finetuning: `python main.py --config efficient_net.yaml`

### DINOv3 Pipeline (frozen ViT-H+ backbone)

Two-step pipeline: embed once, train many times.

**Step 1 — Embedding extraction** (`src/dino/embed.py`):
```bash
python src/dino/embed.py --config dino_cnn.yaml
# or on cluster:
sbatch scripts/run_cluster_l_embed.sbatch
```
Loads `facebook/dinov3-vith16plus-pretrain-lvd1689m` (840M params, 1280-dim).
Layout of `last_hidden_state`: `[CLS, reg×4, patch×196]` — `patch_start = 1 + n_reg` (read from `model.config.num_register_tokens`, not hardcoded).

Saves per split to `data/{embedding_dir}/`:
| File | Shape | Description |
|------|-------|-------------|
| `{split}_cls.pt` | `[N, 1280]` fp16 | CLS token |
| `{split}_patch_mean.pt` | `[N, 1280]` fp16 | Mean of 196 patch tokens |
| `{split}_meta.csv` | — | Row-aligned metadata (filename, FaceOcclusion, gender) |
| `{split}_patches.bin` | `[N, 196, 1280]` fp16 | Full patch grid as numpy memmap (only when `save_patches: true`) |

`validation_noisy.csv` (16 noisy-label samples) is always split 80/20 and appended to both train and val at embed time. At training time, inclusion is controlled by `train_use_noisy` / `val_use_noisy` flags in the config (default: `true`).

**Step 2a — Linear Probe** (`type: dino_lp`):
```bash
python main.py --config dino.yaml
```
`EmbeddingDataset` loads CLS / patch_mean / concat (controlled by `lp_embedding`). `LinearProbe`: 2-layer MLP. `compute_laplacian_iw()` computes importance weights with Laplacian smoothing: `ratio = (test_dist + alpha/N) / (train_dist + alpha/N)`.

**Step 2b — PatchCNN** (`type: dino_cnn`):
```bash
python main.py --config dino_cnn.yaml
```
`PatchDataset` lazily loads `{split}_patches.bin` via numpy memmap (no RAM accumulation). `__getitem__` remaps indices via `keep_idx` to handle noisy-sample filtering correctly.

`PatchCNN` architecture — input `[B, 1280, 14, 14]`:
```
Conv2d(1280→512, 1×1) → GroupNorm → GELU    # channel reduction
Conv2d(512→256, 3×3)  → GroupNorm → GELU    # spatial mixing 14×14
Conv2d(256→64,  3×3)  → GroupNorm → GELU    # spatial mixing 14×14
Conv2d(64→8,   3×3, stride=2) → GroupNorm → GELU   # downsample → 7×7
Flatten → [B, 392]
optional: CLS projected to 256-d and concatenated → [B, 648]
Linear(392/648 → 256) → GELU → Dropout → Linear(256→64) → GELU → Dropout → Linear(64→1) → Sigmoid
```
After training, `eval_final_cnn` computes and logs `val_Pscore` — the competition score using only `pi` weights (no importance weighting), as a distribution-free reference.

**Optuna hyperparameter search:**
Uncomment `optuna_n_trials` / `optuna_epochs` / `optuna_patience` in the config to activate.
- LP Optuna (`objective_lp`): searches over lr, hidden, dropout, weight_decay, loss type, smooth_alpha
- CNN Optuna (`objective_cnn`): searches over lr, dropout, weight_decay, and all four `CompoundLoss` params (alpha, beta, gamma, kappa)
Results saved to `config/optuna/{timestamp}_optuna_results.csv` and logged to MLflow.

**Step 2c — End-to-end Unfreeze** (`type: dino_unfreeze`):
```bash
python main.py --config dino_cnn_ft.yaml
```
Loads a pretrained PatchCNN head checkpoint and plugs it back onto the live DINOv3 backbone. Processes raw images (no pre-computed embeddings). Requires `{split}_meta.csv` files from a prior `embed.py` run in `embedding_dir` to guarantee the same train/val split.

**DINOv3 backbone structure** (`facebook/dinov3-vitb16-pretrain-lvd1689m`, 768-dim, 12 layers):
```
backbone
├── embeddings       — patch projection + position encoding  [always frozen]
├── rope_embeddings  — rotary position embeddings            [always frozen]
├── model (DINOv3ViTEncoder)
│   └── layer [ModuleList, 12 × DINOv3ViTLayer]
│       Each block:
│       ├── norm1        — LayerNorm pre-attention  [frozen — advised by Enzo]
│       ├── attention    — q/k/v/o projections      [unfrozen]
│       ├── layer_scale1 — per-channel learned gate  [unfrozen]
│       ├── norm2        — LayerNorm pre-MLP        [frozen — advised by Enzo]
│       ├── mlp          — 768→3072→768 feed-forward [unfrozen]
│       └── layer_scale2 — per-channel learned gate  [unfrozen]
└── norm             — final LayerNorm               [frozen — advised by Enzo]
```
`_get_layers_and_norm()` auto-detects the layer path across HF wrapper depths so the code works regardless of which class `AutoModel` returns.

**Validation strategy:**
- During training: early stopping on **clean val** (`val_use_noisy` hardcoded `False`)
- After training (best checkpoint reloaded): final scoring on **noisy val** (`val_use_noisy` hardcoded `True`), logged as `final_val_score_noisy`

**Key config keys** (dino_cnn_ft.yaml / dino_unfreeze):
| Key | Description |
|-----|-------------|
| `model_name` | HuggingFace model id for the backbone |
| `embedding_dir` | Dir under `data/` containing `{split}_meta.csv` from embed step |
| `head_checkpoint` | Absolute path to a saved `PatchCNN` `.pt` file |
| `n_blocks` | Number of top transformer blocks to unfreeze (max 12 for base) |
| `learning_rate_head` | LR for the PatchCNN head (e.g. 1e-4) |
| `learning_rate_backbone` | LR for unfrozen backbone layers (e.g. 1e-5) |
| `weight_decay` | AdamW weight decay for both param groups |
| `warmup_epochs` | Linear warmup epochs for backbone LR only (0 = off) |
| `augmentation` | Whether to apply augmentation to train images |
| `lp_loss` / `lp_loss_alpha/beta` | Loss function — same keys as `dino_lp` / `dino_cnn` |
| `lp_epochs` / `lp_patience` / `lp_batch_size` | Training loop settings |
| `smooth_alpha` | Laplacian smoothing for importance weights |

**Key config keys** (dino_cnn.yaml / dino.yaml):
| Key | Description |
|-----|-------------|
| `type` | `dino_lp` or `dino_cnn` |
| `embedding_dir` | Subfolder under `data/` where embeddings are stored |
| `save_patches` | Whether embed.py saves the full patch memmap |
| `lp_embedding` | `cls` / `patch_mean` / `concat` (LP only) |
| `patch_use_cls` | Concatenate CLS projection to CNN features |
| `train_use_noisy` / `val_use_noisy` | Include noisy samples in train/val at training time |
| `smooth_alpha` | Laplacian smoothing strength for iw (count-based: alpha/N added per bin) |
| `lp_loss` | Loss name — use `CompoundLoss` for CNN Optuna |
| `lp_loss_alpha/beta/gamma/kappa` | CompoundLoss params |
| `optuna_n_trials` | If set, routes to Optuna search instead of single run |

---

## How to Modify / Extend Components

### Change the loss function
1. Add a new `nn.Module` class in [src/models/loss.py](src/models/loss.py)
2. Add an `elif loss_name == "your_name":` branch in `UniversalLossWrapper.__init__()`
3. Set `loss_name: your_name` in the relevant section of `config/pipeline_default.yaml` or a model YAML

### Change importance weights or sampling distribution
- Weights (`iw`, `pi`) are computed in `distribution_adaptation_reweight()` in [src/data/data_stats.py](src/data/data_stats.py)
- `iw` = importance weight matching the test distribution; `pi` = label-based weight
- Modify the KL-divergence ratio computation there to change the reweighting scheme
- `WeightedRandomSampler` in [src/data/data_loader.py](src/data/data_loader.py) consumes these weights

### Add a new model
1. Copy an existing file in `config/models/` as `config/models/<timm_model_name>.yaml`
2. Set `num_epoch`, `learning_rate`, `loss_name`, `augmentation` etc. for each stage
3. Pass the model name to `get_model()` in [src/models/models.py](src/models/models.py)
4. If the backbone uses non-standard attention layer names, add them to the target list in `inject_lora_transformer()` in [src/models/finetuning.py](src/models/finetuning.py) (currently targets: `"qkv"`, `"query"`, `"key"`, `"value"`)

### Add a new training stage
1. Create `src/pipeline/run_<stage>.py` following the pattern of `run_lora.py`
2. Call it from `main.py`, passing the `precedent_run_id` from the previous stage

### Add a new scratch CNN architecture
1. Add a new `nn.Module` class in `src/models/scratch_cnn.py` (or a new file); output shape must be `[B, num_classes]`
2. Register it in `src/models/__init__.py`: add the import and add `"your_name": YourClass` to `CUSTOM_MODELS`
3. Create `config/models/your_name.yaml` with `model: "your_name"` and `scratch_training.run_execution: True`
4. Run with `python main.py --config your_name.yaml`

**Data loading for custom models — two things to watch:**
- `CUSTOM_MODELS` (from `src.models`) is checked in `_get_transform()` in [src/data/data_loader.py](src/data/data_loader.py) to skip timm's transform resolution, which would crash on unknown model names
- The scratch transform is minimal: `ToImage() → ToDtype(float32, scale=True) → Normalize(ImageNet stats)` — no resize/crop because challenge images are already 224×224. `ToImage()` is required to convert PIL→tensor before `Normalize` can run
- Augmentation (random flips, crops etc.) is applied separately by the existing `augmentation_transform` pipeline in the loader — do not add random transforms inside `_get_transform` or they will be doubled

### Change hyperparameters
- Global defaults: `config/pipeline_default.yaml`
- Per-model overrides: `config/models/<model_name>.yaml` (deep-merged on top of defaults)
- `config_utils.load_config(model_name, stage)` returns the merged dict for a given model and stage

---

## Running the Project

```bash
# Full pipeline (default config)
python main.py

# Choose a specific model config
python main.py --config vit_tiny_patch16_224_no_celeba.yaml

# Single-batch smoke test (fast sanity check, no GPU needed)
python test.py

# Cluster (SLURM)
sbatch scripts/run_cluster.sbatch
```

---

## Visualisation App

An interactive Streamlit app for exploring dataset images and model predictions.

**Run:**
```bash
uv run streamlit run src/viz/app.py
```

**Location:** `src/viz/app.py` — all viz code lives here.  
**Stars:** `src/viz/stars.json` — persists starred images across runs (not tracked by git).

### Sidebar controls
| Control | Description |
|---------|-------------|
| Model run | Selects a submission dir: `submission/{timestamp}_submission_{model_tag}/` |
| Split | `train` / `val` / `test` — controls which `{split}.csv` is loaded |
| Gender | Filter by `Female (0)` / `Male (1)` / `All` (train/val only) |
| Occlusion interval | Range slider [0 %, 100 %] applied to GT (train/val) or pred (test) |
| Losses to compare | Multi-select, auto-populated from all `nn.Module` subclasses in `src/models/loss.py` |

### Tabs
- **Statistics** — overlaid histogram for all available splits, competition score + loss table for the filtered interval (train/val only)
- **Picture** — random image viewer with GT, prediction, delta, and star/unstar button
- **Stars** — tile grid of starred images filtered by current sidebar criteria

### CSV format written by the pipeline
After each training stage, `save_split_predictions()` (`src/pipeline/test.py`) saves:

| File | Columns |
|------|---------|
| `submission/{run}/train.csv` | `filename, FaceOcclusion (GT), pred, gender, iw` |
| `submission/{run}/val.csv`   | `filename, FaceOcclusion (GT), pred, gender, iw` |
| `submission/{run}/test.csv`  | `filename, FaceOcclusion (pred, submission format)` |

### Add a new loss
Just add a new `nn.Module` subclass to `src/models/loss.py` — it appears automatically in the loss selector without any changes to the app.

---

## Maintaining This File

Update this file whenever:
- A source file is added, renamed, or deleted → update the Architecture section
- A new extension point is added (new loss, new stage, new config key) → add a recipe
- Challenge rules or submission format change → update Challenge Overview / Evaluation
- After any significant refactor, re-run `python test.py` to verify the smoke test still passes
