# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Standalone implementation of the RECAP value function for z02 robot data (references RLinf's RECAP flow, but has no runtime dependency on RLinf). The repo covers data adaptation, value-model training, independent test-set evaluation, and offline advantage-label export. Documentation and user-facing messages are in Chinese; code and comments are mostly English. See `README.md` for method background and `docs/usage.md` for the full input/output contract of every command.

Current status: the new Gemma3 architecture has only passed smoke tests — there is no formally trained checkpoint yet. `checkpoints/optimized/best_model.pt` and `docs/results.md` belong to the old averaged-feature baseline and must not be treated as results for the current architecture.

## Commands

All commands run from the repo root inside the `value_function` conda environment (`conda activate value_function`). Raw data and SigLIP2/Gemma3 weights under `models/` are not committed.

```bash
# Data checks / regeneration (config: config/z02_data.yaml)
python scripts/prepare_data.py --analyze          # read-only audit
python scripts/prepare_data.py --all              # regenerate returns + splits (updates meta/ and data/splits/)

# Training (config: config/train_value.yaml)
python scripts/train.py --smoke_test              # small-sample smoke run → artifacts/performance/smoke-recap/
python scripts/train.py                           # full run → checkpoints/recap_patch/best_model.pt
bash run_train.sh [--smoke_test] [--dry-run]      # prepare_data + train sequence

# Evaluation (needs CUDA/BF16 for the new model)
bash run_test.sh [--dry-run] [--checkpoint PATH]  # check_cache → evaluate → render
python scripts/calculate_advantage.py --split test --output artifacts/advantage/test-run

# Tests (fast, CPU-only, unittest — no pytest config)
python -m unittest discover -s tests -v           # full suite
python -m unittest tests.test_recap_model -v      # single module
python -m unittest tests.test_training.SomeClass.test_name -v   # single test
```

There is no linter/formatter configuration in the repo.

## Architecture

### Two model architectures, dispatched by checkpoint identity

The codebase contains two value-model architectures. Dispatch happens on the `architecture` field — in `config/train_value.yaml` for training, and in the saved checkpoint payload for evaluation/advantage:

- **`recap_patch_gemma_expert` (current)** — `submodules/recap_model.py` defines `RecapValueModel`: SigLIP2 patch tokens → projection → prefixed with task-text embeddings → Gemma3 backbone → independent Gemma value expert reads the prefix via a learnable CLS → 201-atom distribution over `[-1,0]` trained with two-hot cross entropy (`two_hot_loss`), continuous value = probability-weighted expectation. `submodules/recap_workflow.py` holds the training loop (`train_recap`), checkpoint load/verify (`load_recap_checkpoint`), and inference (`predict_recap_dataset`).
- **`siglip_mean_patch_categorical` (legacy)** — averaged patch features, `submodules/model.py` + precomputed feature caches (`cache.py`, `feature_loader.py`). `--prepare-cache` flags and cache verification apply only to this path.

Dispatch points: `training_workflow.train()` (line ~147) branches on config; `evaluation_workflow.py`, `scripts/calculate_advantage.py`, and `check_cache.py` branch on `payload['architecture']`. Any change touching one architecture should check whether the other path needs the parallel change.

### Checkpoints embed identity hashes

`best_model.pt` stores SHA256 hashes of pretrained weights, tokenizer, data-contract/return sidecars, and split manifests. Loading re-verifies them and fails if data, splits, or pretrained weights changed — regenerating data or touching `models/` invalidates existing checkpoints. Frozen-backbone checkpoints omit those weights, and load validates the resulting missing-keys pattern. Checkpoints also record `smoke_test` status.

### Import-order rule (critical)

`submodules/__init__.py` sets `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS=1` to work around a BLAS threading race in this conda env that segfaults scipy/transformers/matplotlib imports. Every entry point must `import submodules` **before** importing numpy/torch/cv2/transformers/matplotlib, immediately after the `sys.path.insert(0, PROJECT_ROOT)` boilerplate. Tests do the same. Breaking this order causes intermittent `-11` crashes that look like repo bugs.

### Data flow

- `config/z02_data.yaml` describes the four raw batches under `data/raw/`, camera `cam2`, the 22-dim joint convention, result overrides (the `2026.09.16_error` batch is all-failure), and train/val/test splits. `scripts/prepare_data.py` (→ `submodules/data_workflow.py`) audits and writes `meta/returns_<tag>.parquet` sidecars plus `data/splits/{train,val,test}.json`; it never modifies raw frame parquets or videos.
- Frames are decoded online from videos during training (no image-feature cache in the new architecture). A frame is identified by `(dataset_id, episode_index, frame_index)`.
- Reward contract: non-terminal `-1`, success `0`, failure `-2000`; returns scaled by `/4000` into `[-1,0]`. Joint states and actions are stored but never enter the value model.
- `config/train_value.yaml` is the main training config: three independent freeze switches and learning rates (`vision_lr`, `gemma_lr`, `expert_lr`), BF16, gradient checkpointing, gradient accumulation, `critic_expert_variant` sizes (`gemma_1m` default … `gemma_2b`). Relative paths resolve against the repo root via `submodules.contracts.resolve_path`.

### Outputs

- `checkpoints/recap_patch/` — formal training checkpoint + per-epoch `metrics.json` (overwritten by full runs; use `--save_dir` for experiments).
- `artifacts/` (gitignored) — `performance/` smoke artifacts, `evaluation/` test reports (`predictions.npz`, `metrics.json`, `protocol.json`, HTML via `render.py`), `advantage/` scoring results (only `manifest.status=complete` dirs are valid; `--reuse-values` recomputes labels without re-running the model).

### Tests

`tests/` is unittest-based and CPU-fast; pretrained tokenizer files under `models/` are used directly (`local_files_only`). `tests/test_cli.py` asserts every entry script's `--help` exits 0 — add any new entry script to its `SCRIPTS_DIR` map.
