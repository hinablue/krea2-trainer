# Krea2 Trainer Runpod Training Workflow Implementation Plan

> **For Hermes:** Execute this plan task-by-task with tests first. Preserve unrelated working-tree changes and commit each task independently.

**Goal:** Add a reproducible Runpod workflow where users prepare only datasets, captions, a dataset TOML, and optionally a TQD score manifest; the Runpod launcher receives settings through environment variables, downloads missing model files from Hugging Face, generates all missing caches by default, then starts Krea2 LoRA training.

**Architecture:** Package the trainer in a CUDA 13.0 / PyTorch 2.9.1 Runpod image, mount one persistent network volume at `/workspace`, and use a portable launcher shared by local and Runpod wrappers. The Runpod wrapper consumes process environment variables (from Runpod Template/API/MCP) plus an optional non-executable dotenv file, performs preflight validation, downloads missing Krea2 assets through `huggingface_hub`, runs latent and text caches sequentially with `--skip_existing`, then launches standard or TQD training. TQD manifests use original image filenames as the canonical index while retaining legacy `cache_file` compatibility.

**Tech Stack:** Bash, Python 3.10–3.12, PyTorch 2.9.1 with CUDA 13.0, uv, Accelerate, TOML, JSONL, Docker/Runpod Pod, safetensors.

---

## Confirmed repository facts

- `run_training.sh` and `run_training_tqd.sh` already expose most training settings as environment variables, but default to machine-specific `/home/hina/...` paths.
- Both launchers default `RUN_CACHE_LATENTS=0` and `RUN_CACHE_TEXT=0`; cache execution is currently opt-in.
- Model files currently occupy about **32.98 GiB** locally:
  - RAW DiT: **24.48 GiB**
  - Qwen3-VL text encoder: **8.27 GiB**
  - VAE: **0.24 GiB**
- The cache commands support `--skip_existing`, so the default Runpod workflow can run both cache phases idempotently without recomputing valid files.
- The current TQD JSONL schema requires `cache_file`, and score attachment happens only after latent/text caches are discovered for training.
- Cache paths are derived from source image basename plus resolution, e.g. `foo.png` → `foo_1024x1536_krea2.safetensors`.
- `pyproject.toml` supports CUDA 13.0 via `uv sync --extra cu130`, selecting `torch>=2.9.1` and `torchvision>=0.24.1` from the PyTorch cu130 index.
- The stable Runpod image tag `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404` exists on Docker Hub and its manifest was verified on 2026-07-20.
- The three public Comfy-Org/Krea-2 Hugging Face assets were verified reachable on 2026-07-20. Their response sizes are 26,283,332,608 bytes (RAW DiT), 8,875,719,384 bytes (text encoder), and 253,806,246 bytes (VAE).

---

## Runpod machine design

### Recommended production profile

| Item | Recommendation | Reason |
| --- | --- | --- |
| Preferred GPU | **1× RTX PRO 6000 Blackwell 96GB** | Provides broad headroom for FP8 base loading, optimizer state, 1024px activations, cache phases, and `torch.compile`, without requiring datacenter-class A100/H100 hardware. |
| Standard 48GB profile | **RTX A6000 48GB**, A40 48GB, L40/L40S 48GB, or RTX PRO 5000 48GB | The existing launcher already uses `--fp8_base`, `--fp8_scaled`, gradient checkpointing, and small image batches. Cache phases run sequentially and do not coexist with DiT training in VRAM, so 48GB is a supported design target. Default to batch size 1 and retain the low-memory settings. |
| Upgrade alternatives | A100/H100 80GB | Optional when faster throughput, larger batches, or extra compile headroom matters; not a minimum requirement. |
| Cost snapshot | RTX PRO 6000 96GB: **$1.69/hr Community / $1.99/hr Secure**; RTX A6000 48GB: **$0.33/hr Community / $0.49/hr Secure**; A40 48GB Secure: **$0.44/hr** | Queried from Runpod MCP on 2026-07-20; prices/availability are not contractual. |
| CPU/RAM target | At least 16 vCPU / 64GB RAM; prefer 96GB+ RAM | Dataset preprocessing, compile workers, and temporary model loading need host headroom. Actual CPU/RAM is coupled to the selected Runpod host. |
| Container disk | 50GB | Holds OS, uv environment, CUDA/Python packages, and source; do not store datasets/models/results here. |
| Network volume | 200GB minimum; 500GB recommended | Persists ~33GB models, datasets, latent/text caches, checkpoints, and logs across Pod replacement. |
| Volume mount | `/workspace` | Stable paths across templates and replacement Pods. |
| Ports | `22/tcp`; optional `8888/http` | SSH plus optional Jupyter. Training itself needs no public HTTP port. |
| Base image | `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404` | Provides CUDA 13.0, PyTorch 2.9.1, and Ubuntu 24.04, matching the repository's `cu130` dependency path. |

### Persistent volume layout

```text
/workspace/
  krea2-trainer/                # repository checkout or image-mounted source
  models/
    diffusion_models/krea2_raw_bf16.safetensors
    text_encoders/qwen3vl_4b_bf16.safetensors
    vae/qwen_image_vae.safetensors
  datasets/
    my_dataset/
      images/
        0001.png
        0001.txt
      cache/                    # generated automatically
      scores.tqd.jsonl          # only for TQD
  configs/
    dataset.toml
    train.env                   # optional; chmod 600
  output/
    checkpoints/
    logs/
```

### Pod lifecycle

1. Create/mount the persistent network volume in the same data center as the Pod.
2. Upload images/captions, dataset TOML, and optional TQD manifest to the volume. Model upload is optional because the launcher downloads missing public assets from Hugging Face.
3. Create the Pod with non-secret settings in the Runpod Template and secrets such as `WANDB_API_KEY` in Runpod environment variables.
4. Run `runpod/train.sh`. Do **not** auto-start expensive training merely because a Pod boots; explicit launcher invocation remains the spending boundary.
5. Launcher validates inputs, downloads missing models, generates all missing caches, starts training, and writes models/checkpoints/logs to the network volume.
6. Stop/delete the Pod after completion; keep the volume for artifacts and resume runs.

---

## User-provided inputs

### Required

1. One or more image directories.
2. One caption file per image using the configured extension, normally `.txt`.
3. Dataset TOML using Runpod paths under `/workspace`, never local `/home/hina/...` paths.
4. Training identifiers/settings through Runpod environment variables or `/workspace/configs/train.env`.

### Automatically downloaded by default

The Runpod launcher downloads these public files only when their target paths are missing:

```text
https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_raw_bf16.safetensors
https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_bf16.safetensors
https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors
```

They remain on the persistent network volume, so replacement Pods do not redownload them. Users may still pre-upload files or override the repository/revision/paths for mirrors and pinned deployments.

### Required only for TQD

- One JSONL score manifest per dataset, indexed by source image filename.
- Every source image used by that dataset must have exactly one score record.
- `structure_score` and `detail_score` must both be within `[0, 1]`.

### Dataset TOML example

```toml
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "/workspace/datasets/my_dataset/images"
cache_directory = "/workspace/datasets/my_dataset/cache"
num_repeats = 1
# TQD only:
# tqd_score_file = "/workspace/datasets/my_dataset/scores.tqd.jsonl"
```

---

## Environment-variable contract

### Configuration sources and precedence

```text
CLI arguments > Runpod process environment > optional ENV_FILE > launcher defaults
```

`ENV_FILE` must be parsed as plain `KEY=VALUE` data, not sourced as shell code. Existing process variables must not be overwritten by the dotenv file.

### Core path variables

```dotenv
PROJECT_DIR=/workspace/krea2-trainer
DATASET_CONFIG=/workspace/configs/dataset.toml
MODEL_DIR=/workspace/models
RAW_DIT=/workspace/models/diffusion_models/krea2_raw_bf16.safetensors
VAE=/workspace/models/vae/qwen_image_vae.safetensors
TEXT_ENCODER=/workspace/models/text_encoders/qwen3vl_4b_bf16.safetensors
OUTPUT_DIR=/workspace/output/checkpoints
LOGGING_DIR=/workspace/output/logs
```

### Workflow variables

```dotenv
TRAIN_MODE=standard             # standard | tqd
MODEL_FETCH=if_missing          # if_missing | never | force
HF_MODEL_REPO=Comfy-Org/Krea-2
HF_MODEL_REVISION=main          # allow pinning to a commit SHA in production
CACHE_MODE=all                  # all | latents | text | none
CACHE_SKIP_EXISTING=1           # default idempotent behavior
FORCE_REBUILD_CACHE=0           # explicit override; mutually exclusive with skip
ENABLE_COMPILE=1
OUTPUT_NAME=my_krea2_lora
MAX_TRAIN_EPOCHS=5
SAVE_EVERY_N_EPOCHS=1
MAX_DATA_LOADER_N_WORKERS=8
SEED=17415
```

### Secrets

- `WANDB_API_KEY` and future Hugging Face tokens must be supplied as Runpod environment secrets.
- The default Comfy-Org/Krea-2 downloads are public and require no token. `HF_TOKEN` remains optional for rate limits, private mirrors, or future gated revisions.
- The launcher must never print secret values or enable `set -x`.
- The example dotenv file must contain placeholders only and be safe to commit.

---

## TQD image-filename index decision

**Decision: yes, use the original image filename as the canonical TQD index.** This fixes the current bootstrap problem because scores can be prepared before any cache exists.

Canonical format:

```json
{"image_file":"0001.png","structure_score":0.91,"detail_score":0.84}
{"image_file":"0002.webp","structure_score":0.88,"detail_score":0.42}
```

Lookup strategy:

1. Normalize `image_file` to basename and then to its extension-free stem (`0001.png` → `0001`).
2. During training, use `ItemInfo.item_key`, which is already reconstructed from the latent cache basename, as the normalized lookup key.
3. Keep accepting the legacy `cache_file` field for one deprecation cycle. Normalize `0001_1024x1536_krea2.safetensors` to the same stem `0001`.
4. Reject records containing both `image_file` and `cache_file`, duplicate normalized stems, paths instead of basenames, missing scores, and values outside `[0, 1]`.
5. Reject image sets containing two files with the same stem but different extensions because the current cache naming scheme would collide too (`0001.png` and `0001.webp`).

This does not require a cache file during score authoring and preserves existing manifests during migration.

---

## Task 1: Add failing tests for image-indexed TQD manifests

**Files:**
- Modify: `tests/test_tqd_dataset.py`

**Steps:**
1. Add a test that loads valid `image_file` records before a cache directory/cache file exists.
2. Add a test proving `image_file: sample.png` attaches to `sample_1024x1024_krea2.safetensors`.
3. Add compatibility coverage for existing `cache_file` records.
4. Add rejection tests for both keys present, neither key present, nested/absolute paths, duplicate normalized stems, same-stem/different-extension collision, missing coverage, malformed scores, and out-of-range scores.
5. Run the test and confirm the new canonical-schema tests fail before implementation.

**Verification:**

```bash
uv run python -m unittest tests.test_tqd_dataset -v
```

**Commit:** `test: define image-indexed TQD manifest behavior`

---

## Task 2: Implement canonical TQD image indexing

**Files:**
- Modify: `src/krea2_trainer/dataset/image_video_dataset.py`
- Modify: `README.md`

**Steps:**
1. Add one small helper that normalizes either canonical `image_file` or legacy `cache_file` to the source-image stem.
2. Make `_load_tqd_scores()` accept exactly one index field and store scores by normalized image stem.
3. Make `attach_tqd_scores()` look up by `ItemInfo.item_key` rather than requiring a pre-known cache filename.
4. Preserve full-coverage validation and clear errors.
5. Update README examples to lead with `image_file`, mark `cache_file` as legacy compatibility, and remove the requirement to generate the manifest from completed caches.
6. Run targeted and full tests.

**Verification:**

```bash
uv run python -m unittest tests.test_tqd_dataset -v
uv run python -m unittest discover -s tests -v
uv run python -m compileall src
```

**Commit:** `feat: index TQD scores by source image filename`

---

## Task 3: Add resumable Hugging Face model provisioning

**Files:**
- Create: `src/krea2_trainer/scripts/download_models.py`
- Modify: `pyproject.toml`
- Create: `tests/test_model_download.py`

**Steps:**
1. Add a `krea2-download-models` CLI backed by the existing `huggingface-hub` dependency.
2. Define the default repository as `Comfy-Org/Krea-2` and download these repository paths into a stable `local_dir` layout:
   - `diffusion_models/krea2_raw_bf16.safetensors`
   - `text_encoders/qwen3vl_4b_bf16.safetensors`
   - `vae/qwen_image_vae.safetensors`
3. Support `MODEL_FETCH=if_missing|never|force`, `HF_MODEL_REPO`, `HF_MODEL_REVISION`, `MODEL_DIR`, and optional `HF_TOKEN`.
4. Use `hf_hub_download(..., local_dir=MODEL_DIR)` so Hugging Face handles redirects, ETag validation, partial-download resume, file locking, and atomic completion. Never implement a raw one-shot 33GB downloader in Bash.
5. Check free disk space before starting missing downloads and report required/available bytes clearly. Reserve at least 40 GiB when all three assets are absent.
6. Treat target files as ready only after Hugging Face completes validation; interrupted temporary files must never satisfy launcher preflight.
7. Test default path mapping, missing-file skip behavior, force mode, revision forwarding, optional token forwarding without logging it, download failure propagation, and low-disk failure using mocks and temporary directories.

**Verification:**

```bash
uv run python -m unittest tests.test_model_download -v
uv run krea2-download-models --help
```

**Commit:** `feat: download missing Krea2 models from Hugging Face`

---

## Task 4: Extract a portable shared training launcher

**Files:**
- Create: `scripts/train_from_env.sh`
- Modify: `run_training.sh`
- Modify: `run_training_tqd.sh`
- Create: `tests/test_train_from_env.py`

**Steps:**
1. Move common validation, cache commands, training argument construction, compile settings, W&B handling, and Accelerate launch into `scripts/train_from_env.sh`.
2. Add `TRAIN_MODE=standard|tqd` and select the current standard/TQD argument sets without `eval`.
3. Add `CACHE_MODE=all|latents|text|none`, `CACHE_SKIP_EXISTING`, and `FORCE_REBUILD_CACHE`.
4. Execute the post-provisioning phases in fixed order: latent cache first, text encoder cache second, training third.
5. Pass `--skip_existing` by default when caching; force mode recomputes rather than skips.
6. Keep arbitrary trailing training CLI arguments as an array so quoting is preserved.
7. Keep root local wrappers thin and backward compatible with their existing local defaults. Their cache default may remain `none`; the Runpod wrapper introduced next will default to `all`.
8. Test with fake `python`/`accelerate` executables so command ordering, variable mapping, secret redaction, and failure propagation are verified without loading models.

**Verification:**

```bash
bash -n scripts/train_from_env.sh run_training.sh run_training_tqd.sh
uv run python -m unittest tests.test_train_from_env -v
```

**Commit:** `refactor: share environment-driven training launcher`

---

## Task 5: Add the Runpod wrapper and environment contract

**Files:**
- Create: `runpod/train.sh`
- Create: `runpod/train.env.example`
- Create: `runpod/dataset.example.toml`
- Extend: `tests/test_train_from_env.py`

**Steps:**
1. Implement a safe dotenv parser that accepts only simple `KEY=VALUE` lines, ignores blank/comment lines, rejects invalid names, does not execute shell syntax, and does not overwrite process environment variables.
2. Set Runpod path defaults under `/workspace`.
3. Set `CACHE_MODE=all` and `CACHE_SKIP_EXISTING=1` by default.
4. Run the model provisioning CLI before model-file preflight; `MODEL_FETCH=if_missing` is the Runpod default.
5. Validate dataset TOML, completed model files, output/cache writability, allowed enum values, and incompatible cache flags before any expensive GPU work begins.
6. Print a redacted configuration summary without secrets.
7. Delegate to `scripts/train_from_env.sh` with correctly quoted passthrough arguments.
8. Test precedence: CLI > process env > dotenv > defaults and verify the command order model download → latent cache → text cache → training.

**Verification:**

```bash
bash -n runpod/train.sh
uv run python -m unittest tests.test_train_from_env -v
```

**Commit:** `feat: add Runpod environment launcher`

---

## Task 6: Add a reproducible Runpod container

**Files:**
- Create: `runpod/Dockerfile`
- Create: `runpod/.dockerignore`
- Create: `runpod/entrypoint.sh`

**Steps:**
1. Base the image on the verified stable tag `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404`.
2. Install uv and sync the locked project with `uv sync --extra cu130` so the environment uses the PyTorch CUDA 13.0 wheels.
3. Copy only source, scripts, metadata, and docs; never copy local models, datasets, outputs, `.env`, or secrets.
4. Create a non-destructive entrypoint that verifies `/workspace`, prints the documented launch command, and keeps the Pod usable for SSH/Jupyter. Do not auto-spend by launching training on container boot.
5. Add an explicit optional `AUTO_START_TRAINING=1` path only if product requirements later demand unattended jobs; keep it off by default.
6. Add a startup/preflight assertion that CUDA is available, `torch.version.cuda` reports 13.0, and the selected GPU is visible before model download/cache/training begins.
7. Build the image and run lightweight CLI/help and CUDA-version smoke tests.

**Verification:**

```bash
docker build -f runpod/Dockerfile -t krea2-trainer:runpod .
docker run --rm krea2-trainer:runpod python -m krea2_trainer.scripts.train_lora --help
docker run --rm krea2-trainer:runpod python -m krea2_trainer.scripts.cache_latents --help
# Run on a GPU-enabled host/Pod:
docker run --rm --gpus all krea2-trainer:runpod python -c 'import torch; assert torch.cuda.is_available(); assert torch.version.cuda.startswith("13.0"); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
```

**Commit:** `build: add Runpod CUDA 13.0 image`

---

## Task 7: Document Pod creation and end-to-end use

**Files:**
- Create: `docs/runpod-training.md`
- Modify: `README.md`

**Steps:**
1. Document RTX PRO 6000 96GB as the preferred profile and 48GB GPUs as the standard economy profile, with batch size 1 and the current FP8/checkpointing defaults.
2. Document network volume creation, same-data-center placement, mount layout, automatic model downloads, optional pre-upload/offline mode, dataset upload, dataset TOML, TQD manifest, environment variables, Pod creation, launch, monitoring, artifact retrieval, and shutdown.
3. Provide a Runpod Template example using:
   - image built from `runpod/Dockerfile`
   - 50GB container disk
   - 200–500GB network volume at `/workspace`
   - `22/tcp` and optional `8888/http`
   - secret environment variables without literal secret values
4. Include both Runpod UI and MCP/API environment-object examples, but do not commit credentials.
5. Include resume/retry behavior: rerunning the launcher skips existing caches and writes to persistent output.
6. Link the guide from README.

**Verification:**
- Follow the guide using a temporary local directory mapped to `/workspace` and confirm preflight/path behavior.
- Review every example for absence of `/home/hina` paths and literal secrets.

**Commit:** `docs: add Runpod training workflow`

---

## Task 8: Real Runpod smoke test and sizing record

**Files:**
- Create: `docs/runpod-benchmarks.md`
- Modify only if measurements reveal a real issue: launcher/container files from earlier tasks

**Steps:**
1. Create one RTX A6000/A40/L40S-class 48GB Pod with the persistent volume; this is the minimum target that must pass.
2. Start with an empty model directory and verify all three public assets download successfully to the persistent volume.
3. Use a tiny 2–4 image dataset and `MAX_TRAIN_STEPS=1`.
4. Start with empty cache and confirm the observed order is model provisioning → latent cache → text cache → training.
5. Run the launcher a second time and confirm model download and both cache phases skip existing files.
6. Run one TQD smoke test using `image_file` records created before caches exist.
7. Record peak VRAM, host RAM, model download time, cache time, compile time, first-step time, and disk growth.
8. Record whether compile can remain enabled on 48GB. If compile causes pressure or instability, make the documented 48GB default `ENABLE_COMPILE=0` while keeping RTX PRO 6000 96GB at `ENABLE_COMPILE=1`.
9. Stop/delete the Pod and verify models and training artifacts remain on the volume.

**Verification:**

```text
PASS: preflight succeeds
PASS: all three Hugging Face models download and survive Pod replacement
PASS: both caches are generated from an empty cache directory
PASS: standard one-step training completes
PASS: TQD one-step training resolves image_file scores
PASS: second launch skips caches
PASS: output survives Pod deletion
```

**Commit:** `docs: record Runpod smoke-test measurements`

---

## Acceptance criteria

- A new user only needs to prepare images/captions, dataset TOML, and optional image-indexed TQD JSONL; the three public model files are downloaded automatically when absent.
- Runpod process environment variables are sufficient to configure the run; an optional dotenv file is supported without executing shell code.
- Invoking the Runpod launcher with no cache override runs both latent and text cache phases before training.
- Existing cache files are skipped by default; an explicit force mode rebuilds them.
- Existing model files are reused by default; downloads are resumable, validated, and persisted on the network volume.
- Existing `cache_file` TQD manifests remain accepted during migration.
- No committed file contains local `/home/hina` paths, API keys, tokens, datasets, model weights, caches, or outputs.
- The documented 48GB path completes a real empty-cache one-step standard and TQD smoke test; RTX PRO 6000 96GB is the preferred higher-headroom profile.
- GPU support claims are backed by measured peak VRAM rather than inference from model file size alone.
