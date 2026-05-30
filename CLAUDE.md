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
main.py                         # Runs the full 3-stage pipeline sequentially
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
                                #   ⚠ num_workers defaults to 0 everywhere (see TODO)
    data_utils.py               # get_challenge_split(): loads CSV, 80/20 split,
                                #   applies distribution reweighting
    data_stats.py               # distribution_adaptation_reweight(): computes iw/pi columns
                                #   get_test_distribution_from_screenshot(): reads test
                                #   distribution from data/test_distribution.png
    transforms.py               # torchvision.transforms.v2 augmentation pipelines

  models/
    models.py                   # OcclusionModel (timm backbone + Sigmoid head)
                                #   get_model(): factory; applies finetuning method
    finetuning.py               # inject_lora_transformer(): replaces qkv/query/key/value
                                #   Linear layers with LoRALinear
                                #   inject_linear_mlp_probing(): replaces model head
    lora.py                     # LoRALinear: frozen base + trainable low-rank A·B
    loss.py                     # WeightedMSELoss (nMSE), WeightedLiteMSELoss (nLiteMSE),
                                #   UniversalLossWrapper — routes by loss_name string
                                #   ⚠ WeightedLiteMSELoss has a missing return (see TODO)

  pipeline/
    train.py                    # run_train(): Adam + CosineAnnealingLR, early stopping,
                                #   MLflow logging, checkpoint saving
    evaluation.py               # run_evaluation(): gender-split metric; F1/acc for
                                #   domain-adaptation stage
    test.py                     # run_test(): loads best checkpoint → submission CSV
    run_domain_adaptation.py    # Stage 1 entry point
    run_probing.py              # Stage 2 entry point
    run_lora.py                 # Stage 3 entry point

config/
  pipeline_default.yaml         # Global defaults: SEED=42, BATCH_SIZE=32, N_SAMPLE=5000,
                                #   PATIENCE=5, N_BINS=20, NUM_CLASSES=1
  models/
    beit3_base_patch16_224.yaml # Per-model hyperparameter overrides
    vit_tiny_patch16_224.yaml

scripts/
  run_cluster.sbatch            # SLURM job script for cluster training
```

---

## 3-Stage Pipeline

The pipeline is designed to progressively adapt a pretrained ViT/BEiT:

| Stage | Script | Dataset | What it trains |
|---|---|---|---|
| 1 — Domain Adaptation | `run_domain_adaptation.py` | CelebA | Full backbone or LoRA, adapts to face domain |
| 2 — Probing | `run_probing.py` | Challenge | Frozen backbone + new MLP head |
| 3 — LoRA Finetuning | `run_lora.py` | Challenge | LoRA weights + head |

Run IDs are chained: each stage passes `precedent_run_id` so the next stage loads the previous checkpoint via MLflow.

Run everything: `python main.py`

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

### Change hyperparameters
- Global defaults: `config/pipeline_default.yaml`
- Per-model overrides: `config/models/<model_name>.yaml` (deep-merged on top of defaults)
- `config_utils.load_config(model_name, stage)` returns the merged dict for a given model and stage

---

## Running the Project

```bash
# Full pipeline
python main.py

# Single-batch smoke test (fast sanity check, no GPU needed)
python test.py

# Cluster (SLURM)
sbatch scripts/run_cluster.sbatch
```

---

## Maintaining This File

Update this file whenever:
- A source file is added, renamed, or deleted → update the Architecture section
- A new extension point is added (new loss, new stage, new config key) → add a recipe
- Challenge rules or submission format change → update Challenge Overview / Evaluation
- After any significant refactor, re-run `python test.py` to verify the smoke test still passes
