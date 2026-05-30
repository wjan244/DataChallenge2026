# TODO — DataChallenge2026

Checklist of bugs, performance issues, and improvement opportunities found during code review.
Check off items as they are resolved.

---

## 🔴 Critical Bugs (fix before any training run)

- [x] **`WeightedLiteMSELoss` never returns its loss**
  - File: [src/models/loss.py](src/models/loss.py) ~line 29
  - The loss value is computed but the function has no `return` statement — it silently returns `None`
  - This means training with `loss_name: nLiteMSE` produces no gradient and appears to work
  - Fix: add `return loss` at the end of `WeightedLiteMSELoss.forward()`
**DONE** — `return (iw * (y_pred - y_true) ** 2).mean()` added inside the `try` block

---

## 🟠 High-Priority Performance (significant impact on training speed)

- [x] **DataLoaders use `num_workers=0` (single-threaded)**
  - All five factory functions in [src/data/data_loader.py](src/data/data_loader.py) now default to `NUM_WORKERS` from config
  - `NUM_WORKERS` uses `os.sched_getaffinity(0)` on the cluster (respects SLURM allocation) and `os.cpu_count()` on Mac
  - Also removed the unused `size_augmentation` parameter from `get_challenge_train_loader`
  - `persistent_workers=True` also added — workers stay alive between epochs instead of being killed and respawned, saving ~10–30s of process startup per epoch

- [x] **No `pin_memory` in DataLoaders**
  - All five loaders now pass `pin_memory=_PIN` (True only on CUDA) and `persistent_workers=_PW`
  - `pin_memory` pre-allocates batches in page-locked RAM so the CPU→GPU transfer uses DMA and can run asynchronously — avoids the extra pageable→pinned copy step, ~10–20% faster data transfer on CUDA

- [ ] **No mixed-precision training (AMP)**
  - File: [src/pipeline/train.py](src/pipeline/train.py) — training and validation loops
  - On CUDA, `torch.amp.autocast` + `GradScaler` typically gives 1.5–2× speedup and halves GPU memory usage
  - Fix: wrap forward pass in `with torch.amp.autocast(device_type="cuda"):` and scale gradients

---

## 🟡 Medium-Priority Improvements

- [ ] **CSV loaded at module import time**
  - File: [src/data/data_utils.py](src/data/data_utils.py) ~lines 20–21
  - `pd.read_csv()` is called at module level — blocks on import even if the loader is never used
  - Fix: move CSV reads inside the `get_challenge_split()` function body

- [ ] **Double (unused) augmentation pipeline**
  - File: [src/data/data_loader.py](src/data/data_loader.py) ~lines 20–29
  - `get_augmentation_finetuning_transforms()` is composed into the pipeline but never applied — dead code
  - Fix: remove the second augmentation composition or document why it exists

- [ ] **Test distribution screenshot reprocessed on every call**
  - File: [src/data/data_stats.py](src/data/data_stats.py) — `get_test_distribution_from_screenshot()`
  - The PNG is read and processed via numpy on every call
  - Fix: add `@functools.lru_cache(maxsize=None)` or cache the result in a module-level variable

- [ ] **All augmentations run on CPU**
  - File: [src/data/transforms.py](src/data/transforms.py)
  - `torchvision.transforms.v2` runs on CPU; on GPU-heavy workloads this can become a bottleneck
  - Fix (medium effort): move compatible transforms after the `.to(device)` call, or consider NVIDIA DALI

---

## 🔵 Code Quality / Correctness

- [ ] **`iw`/`pi` tensors not moved to DEVICE when loss changes**
  - File: [src/pipeline/train.py](src/pipeline/train.py)
  - `iw` and `pi` are only extracted and moved to DEVICE inside `if loss_name == "nMSE":` — if the loss name changes, they remain on CPU and will cause a device mismatch error at runtime
  - Fix: always extract `iw`/`pi` from the batch and move them to DEVICE, regardless of loss name

- [ ] **No gradient clipping**
  - File: [src/pipeline/train.py](src/pipeline/train.py) — inside the training loop
  - Transformer fine-tuning is prone to gradient explosions, especially in early epochs
  - Fix: add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`

- [ ] **`test.py` uses brittle monkey-patches**
  - File: [test.py](test.py)
  - Patches `DataLoader.__iter__` and `run_evaluation` at the class/module level — fragile across refactors
  - Fix: add a `max_batches` parameter to `run_train()` and a `dry_run` flag to `run_evaluation()`

- [ ] **`torch.compile()` not used**
  - File: [src/models/models.py](src/models/models.py) — after model is constructed
  - `torch.compile()` (PyTorch ≥ 2.0) provides free speed on CUDA and MPS with one line
  - Fix: add `model = torch.compile(model)` in `get_model()`, guarded by a config flag

---

## 🟢 Experiments / Strategy

- [ ] **Increase `BATCH_SIZE` beyond 32**
  - With mixed precision the GPU can likely handle 64–128; larger batches stabilise training and speed up epochs
  - Config: `config/pipeline_default.yaml` → `globaux.BATCH_SIZE`

- [ ] **Try stronger backbones: DINOv2, CLIP (via timm)**
  - DINOv2 ViT-B/14 or CLIP ViT-L/14 likely have richer face features than BEiT3-base
  - Add a new model YAML in `config/models/` to try them (see CLAUDE.md for how)

- [ ] **Test-Time Augmentation (TTA) at inference**
  - Average predictions over horizontally-flipped / brightness-shifted variants
  - Implement in [src/pipeline/test.py](src/pipeline/test.py)

- [ ] **Ensemble / SWA (Stochastic Weight Averaging)**
  - Average the last N checkpoints or use `torch.optim.swa_utils.AveragedModel`
  - Low effort, often gives 0.5–1% metric improvement

- [ ] **Stratified sampling by gender during training**
  - The metric penalises gender disparity — ensuring balanced gender batches may help
  - Implement a `WeightedRandomSampler` that balances gender classes alongside the existing distribution weights in [src/data/data_loader.py](src/data/data_loader.py)

- [ ] **Learning-rate warm-up**
  - Add a linear warm-up phase before `CosineAnnealingLR` kicks in (use `torch.optim.lr_scheduler.SequentialLR`)
  - Helps stability in the first few epochs of LoRA fine-tuning
