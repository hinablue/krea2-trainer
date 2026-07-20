#!/usr/bin/env bash
set -Eeuo pipefail

load_env_file() {
  local file="$1" line key value
  [[ -f "${file}" ]] || { echo "[ERROR] ENV_FILE not found: ${file}" >&2; return 1; }
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
    [[ "${line}" == *=* ]] || { echo "[ERROR] Invalid ENV_FILE line (expected KEY=VALUE)" >&2; return 1; }
    key="${line%%=*}"; value="${line#*=}"
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || { echo "[ERROR] Invalid ENV_FILE variable name: ${key}" >&2; return 1; }
    if [[ ! -v "${key}" ]]; then
      printf -v "${key}" '%s' "${value}"
      export "${key}"
    fi
  done < "${file}"
}

[[ -n "${ENV_FILE:-}" ]] && load_env_file "${ENV_FILE}"
export PROJECT_DIR="${PROJECT_DIR:-/opt/krea2-trainer}"
export DATASET_CONFIG="${DATASET_CONFIG:-/workspace/configs/dataset.toml}"
export MODEL_DIR="${MODEL_DIR:-/workspace/models}"
export RAW_DIT="${RAW_DIT:-${MODEL_DIR}/diffusion_models/krea2_raw_bf16.safetensors}"
export TEXT_ENCODER="${TEXT_ENCODER:-${MODEL_DIR}/text_encoders/qwen3vl_4b_bf16.safetensors}"
export VAE="${VAE:-${MODEL_DIR}/vae/qwen_image_vae.safetensors}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/checkpoints}"
export LOGGING_DIR="${LOGGING_DIR:-/workspace/output/logs}"
export TRAIN_MODE="${TRAIN_MODE:-standard}"
export MODEL_FETCH="${MODEL_FETCH:-if_missing}"
export CACHE_MODE="${CACHE_MODE:-all}"
export CACHE_SKIP_EXISTING="${CACHE_SKIP_EXISTING:-1}"
export FORCE_REBUILD_CACHE="${FORCE_REBUILD_CACHE:-0}"
export OUTPUT_NAME="${OUTPUT_NAME:-krea2_lora}"
export HF_MODEL_REPO="${HF_MODEL_REPO:-Comfy-Org/Krea-2}"
export HF_MODEL_REVISION="${HF_MODEL_REVISION:-main}"

[[ -d "${PROJECT_DIR}/.venv" ]] && export PATH="${PROJECT_DIR}/.venv/bin:${PATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Validate cheap inputs and writable persistent paths before invoking CUDA or downloads.
[[ "${TRAIN_MODE}" =~ ^(standard|tqd)$ ]] || { echo "[ERROR] Invalid TRAIN_MODE: ${TRAIN_MODE}" >&2; exit 1; }
[[ "${CACHE_MODE}" =~ ^(all|latents|text|none)$ ]] || { echo "[ERROR] Invalid CACHE_MODE: ${CACHE_MODE}" >&2; exit 1; }
[[ "${MODEL_FETCH}" =~ ^(if_missing|never|force)$ ]] || { echo "[ERROR] Invalid MODEL_FETCH: ${MODEL_FETCH}" >&2; exit 1; }
[[ "${CACHE_SKIP_EXISTING}" == 1 && "${FORCE_REBUILD_CACHE}" == 1 ]] && { echo "[ERROR] Incompatible cache flags: skip=1 and force=1" >&2; exit 1; }
[[ -f "${DATASET_CONFIG}" ]] || { echo "[ERROR] Dataset TOML not found: ${DATASET_CONFIG}" >&2; exit 1; }

mkdir -p "${MODEL_DIR}" "${OUTPUT_DIR}" "${LOGGING_DIR}" || { echo "[ERROR] Persistent directories are not writable" >&2; exit 1; }
for writable_dir in "${MODEL_DIR}" "${OUTPUT_DIR}" "${LOGGING_DIR}"; do
  [[ -w "${writable_dir}" ]] || { echo "[ERROR] Directory not writable: ${writable_dir}" >&2; exit 1; }
done
[[ -x "${PROJECT_DIR}/scripts/train_from_env.sh" ]] || { echo "[ERROR] Shared launcher not executable: ${PROJECT_DIR}/scripts/train_from_env.sh" >&2; exit 1; }

if [[ "${SKIP_CUDA_CHECK:-0}" != 1 ]]; then
  "${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[ERROR] CUDA GPU is not available")
if not (torch.version.cuda or "").startswith("13.0"):
    raise SystemExit(f"[ERROR] CUDA 13.0 is required, found {torch.version.cuda}")
print(f"[preflight] torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name()}")
PY
fi

printf '[config] project=%s\n[config] dataset=%s\n[config] models=%s\n[config] output=%s\n[config] mode=%s cache=%s fetch=%s revision=%s\n' \
  "${PROJECT_DIR}" "${DATASET_CONFIG}" "${MODEL_DIR}" "${OUTPUT_DIR}" "${TRAIN_MODE}" "${CACHE_MODE}" "${MODEL_FETCH}" "${HF_MODEL_REVISION}"

"${PYTHON_BIN}" -m krea2_trainer.scripts.download_models --model-dir "${MODEL_DIR}" --mode "${MODEL_FETCH}" --repo "${HF_MODEL_REPO}" --revision "${HF_MODEL_REVISION}"

# We check model files after download
[[ -f "${RAW_DIT}" ]] || { echo "[ERROR] Missing RAW_DIT: ${RAW_DIT}" >&2; exit 1; }
[[ -f "${TEXT_ENCODER}" ]] || { echo "[ERROR] Missing TEXT_ENCODER: ${TEXT_ENCODER}" >&2; exit 1; }
[[ -f "${VAE}" ]] || { echo "[ERROR] Missing VAE: ${VAE}" >&2; exit 1; }

exec "${PROJECT_DIR}/scripts/train_from_env.sh" "$@"
