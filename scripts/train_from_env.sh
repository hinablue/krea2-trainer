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
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET_CONFIG="${DATASET_CONFIG:?DATASET_CONFIG is required}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/models}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output}"
LOGGING_DIR="${LOGGING_DIR:-${OUTPUT_DIR}/logs}"
RAW_DIT="${RAW_DIT:-${MODEL_DIR}/diffusion_models/krea2_raw_bf16.safetensors}"
VAE="${VAE:-${MODEL_DIR}/vae/qwen_image_vae.safetensors}"
TEXT_ENCODER="${TEXT_ENCODER:-${MODEL_DIR}/text_encoders/qwen3vl_4b_bf16.safetensors}"
TRAIN_MODE="${TRAIN_MODE:-standard}"
CACHE_MODE="${CACHE_MODE:-none}"
CACHE_SKIP_EXISTING="${CACHE_SKIP_EXISTING:-1}"
FORCE_REBUILD_CACHE="${FORCE_REBUILD_CACHE:-0}"
OUTPUT_NAME="${OUTPUT_NAME:-krea2_lora}"
MAX_TRAIN_EPOCHS="${MAX_TRAIN_EPOCHS:-5}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}"
NUM_CPU_THREADS_PER_PROCESS="${NUM_CPU_THREADS_PER_PROCESS:-1}"
MAX_DATA_LOADER_N_WORKERS="${MAX_DATA_LOADER_N_WORKERS:-4}"
SEED="${SEED:-17415}"
ENABLE_COMPILE="${ENABLE_COMPILE:-0}"
COMPILE_MODE="${COMPILE_MODE:-max-autotune-no-cudagraphs}"
COMPILE_DYNAMIC="${COMPILE_DYNAMIC:-auto}"
COMPILE_CACHE_SIZE_LIMIT="${COMPILE_CACHE_SIZE_LIMIT:-32}"
REFRESH_TEXT_CACHE_EVERY_EPOCH="${REFRESH_TEXT_CACHE_EVERY_EPOCH:-0}"
CACHE_TE_CAPTION_DROPOUT_RATE="${CACHE_TE_CAPTION_DROPOUT_RATE:-0.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

case "${TRAIN_MODE}" in standard|tqd) ;; *) echo "[ERROR] TRAIN_MODE must be standard or tqd" >&2; exit 2;; esac
case "${CACHE_MODE}" in all|latents|text|none) ;; *) echo "[ERROR] CACHE_MODE must be all, latents, text, or none" >&2; exit 2;; esac
if [[ "${CACHE_SKIP_EXISTING}" == "1" && "${FORCE_REBUILD_CACHE}" == "1" ]]; then
  echo "[ERROR] CACHE_SKIP_EXISTING and FORCE_REBUILD_CACHE are mutually exclusive" >&2; exit 2
fi

require_file() { [[ -f "$1" ]] || { echo "[ERROR] Missing $2: $1" >&2; exit 1; }; }
require_file "${DATASET_CONFIG}" "dataset config"
require_file "${RAW_DIT}" "Krea2 raw DiT"
require_file "${VAE}" "Qwen Image VAE"
require_file "${TEXT_ENCODER}" "Qwen3-VL text encoder"
mkdir -p "${OUTPUT_DIR}" "${LOGGING_DIR}"
cd "${PROJECT_DIR}"
[[ -d .venv ]] && export PATH="${PROJECT_DIR}/.venv/bin:${PATH}"
command -v "${PYTHON_BIN}" >/dev/null || { echo "[ERROR] Python not found" >&2; exit 1; }

if ! command -v "${ACCELERATE_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] accelerate not found." >&2
  exit 1
fi

if [[ "${FORCE_REBUILD_CACHE}" == "1" ]]; then
  CACHE_SKIP_EXISTING="0"
fi

cache_args=("--dataset_config" "${DATASET_CONFIG}" "--device" "cuda")
if [[ "${CACHE_SKIP_EXISTING}" == "1" ]]; then
  cache_args+=("--skip_existing")
fi

if [[ "${CACHE_MODE}" == "all" || "${CACHE_MODE}" == "latents" ]]; then
  echo "[INFO] Caching latents..."
  "${PYTHON_BIN}" -m krea2_trainer.scripts.cache_latents "${cache_args[@]}" --vae "${VAE}"
fi

if [[ "${CACHE_MODE}" == "all" || "${CACHE_MODE}" == "text" ]]; then
  echo "[INFO] Caching text encoder outputs..."
  "${PYTHON_BIN}" -m krea2_trainer.scripts.cache_text_encoder "${cache_args[@]}" --text_encoder "${TEXT_ENCODER}"
fi

TRAIN_ARGS=(
  --raw_dit "${RAW_DIT}" --vae "${VAE}" --dataset_config "${DATASET_CONFIG}"
  --sdpa --mixed_precision bf16 --save_precision bf16 --weighting_scheme none
  --gradient_checkpointing --network_module krea2_trainer.networks.lora_krea2
  --network_dim 32 --network_alpha 16 --disable_numpy_memmap --fp8_base --fp8_scaled
  --learning_rate 5e-5 --lr_scheduler constant_with_warmup
  --optimizer_type Adopt_adv --optimizer_args cautious_wd=true kourkoutas_beta=true use_atan2=true weight_decay=0.01
  --max_train_epochs "${MAX_TRAIN_EPOCHS}" --save_every_n_epochs "${SAVE_EVERY_N_EPOCHS}"
  --output_dir "${OUTPUT_DIR}" --output_name "${OUTPUT_NAME}" --logging_dir "${LOGGING_DIR}"
  --seed "${SEED}" --max_data_loader_n_workers "${MAX_DATA_LOADER_N_WORKERS}"
  --persistent_data_loader_workers --cuda_allow_tf32 --cuda_cudnn_benchmark
)
if [[ "${TRAIN_MODE}" == tqd ]]; then
  TRAIN_ARGS+=(--timestep_sampling tqd_krea2_shift --tqd_kappa_base "${TQD_KAPPA_BASE:-2}" --tqd_kappa_max "${TQD_KAPPA_MAX:-8}" --tqd_quality_weighting --lr_warmup_steps "${LR_WARMUP_STEPS:-200}")
  LOG_PREFIX="${LOG_PREFIX:-tqd_rosie_}"
  LOG_TRACKER_NAME="${LOG_TRACKER_NAME:-krea2-trainer-tqd}"
  WANDB_RUN_NAME="${WANDB_RUN_NAME:-tqd-lora-run}"
else
  TRAIN_ARGS+=(--timestep_sampling krea2_shift --lr_warmup_steps "${LR_WARMUP_STEPS:-500}")
  LOG_PREFIX="${LOG_PREFIX:-myKrea2Lora_}"
  LOG_TRACKER_NAME="${LOG_TRACKER_NAME:-krea2-trainer}"
  WANDB_RUN_NAME="${WANDB_RUN_NAME:-my-krea2-run}"
fi
LOG_WITH="${LOG_WITH:-wandb}"
if [[ -n "${LOG_WITH}" ]]; then
  TRAIN_ARGS+=(--log_prefix "${LOG_PREFIX}" --log_with "${LOG_WITH}" --log_config --log_tracker_name "${LOG_TRACKER_NAME}" --wandb_run_name "${WANDB_RUN_NAME}")
fi
if [[ "${ENABLE_COMPILE}" == 1 ]]; then
  TRAIN_ARGS+=(--compile --compile_mode "${COMPILE_MODE}" --compile_dynamic "${COMPILE_DYNAMIC}" --compile_cache_size_limit "${COMPILE_CACHE_SIZE_LIMIT}")
fi
# WANDB_API_KEY stays in the process environment; never copy secrets into argv or logs.

if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
  TRAIN_ARGS+=(--max_train_steps "${MAX_TRAIN_STEPS}")
fi

if [[ "${REFRESH_TEXT_CACHE_EVERY_EPOCH:-0}" == "1" ]]; then
  TRAIN_ARGS+=(
    --text_encoder "${TEXT_ENCODER}"
    --cache_te_every_epoch
    --cache_te_shuffle_caption
    --cache_te_caption_dropout_rate "${CACHE_TE_CAPTION_DROPOUT_RATE:-0.0}"
    --cache_te_keep_tokens 1
    --cache_te_device cuda
    --cache_te_dtype bfloat16
  )
fi

TRAIN_ARGS+=("$@")

echo "[INFO] Launching training..."
"${ACCELERATE_BIN:-accelerate}" launch --num_cpu_threads_per_process "${NUM_CPU_THREADS_PER_PROCESS:-1}" \
  -m krea2_trainer.scripts.train_lora \
  "${TRAIN_ARGS[@]}"
