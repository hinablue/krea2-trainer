#!/usr/bin/env bash
set -Eeuo pipefail

# Rosie Krea2 LoRA training launcher (TQD Version)
# Usage:
#   ./run_training_tqd.sh [extra train_lora args...]
#
# TQD (Temporal Quality Distribution) Sampling enables more robust learning 
# by diversifying sample quality/complexity across training steps.

PROJECT_DIR="${PROJECT_DIR:-/home/hina/Workspace/krea2-trainer}"
DATASET_CONFIG="${DATASET_CONFIG:-${PROJECT_DIR}/configs/asian_tqd.toml}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/models}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output}"
LOGGING_DIR="${LOGGING_DIR:-${OUTPUT_DIR}/logs}"

RAW_DIT="${RAW_DIT:-${MODEL_DIR}/krea2-raw.safetensors}"
VAE="${VAE:-${MODEL_DIR}/qwen_image_vae.safetensors}"
TEXT_ENCODER="${TEXT_ENCODER:-${MODEL_DIR}/Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors}"

OUTPUT_NAME="${OUTPUT_NAME:-hina_krea2_lora_tqd_v1}"
MAX_TRAIN_EPOCHS="${MAX_TRAIN_EPOCHS:-5}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}"
NUM_CPU_THREADS_PER_PROCESS="${NUM_CPU_THREADS_PER_PROCESS:-1}"
MAX_DATA_LOADER_N_WORKERS="${MAX_DATA_LOADER_N_WORKERS:-8}"
SEED="${SEED:-17415}"
ENABLE_COMPILE="${ENABLE_COMPILE:-1}"
COMPILE_MODE="${COMPILE_MODE:-max-autotune-no-cudagraphs}"
COMPILE_DYNAMIC="${COMPILE_DYNAMIC:-auto}"
COMPILE_CACHE_SIZE_LIMIT="${COMPILE_CACHE_SIZE_LIMIT:-32}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

RUN_CACHE_LATENTS="${RUN_CACHE_LATENTS:-0}"
RUN_CACHE_TEXT="${RUN_CACHE_TEXT:-0}"
REFRESH_TEXT_CACHE_EVERY_EPOCH="${REFRESH_TEXT_CACHE_EVERY_EPOCH:-0}"
CACHE_TE_CAPTION_DROPOUT_RATE="${CACHE_TE_CAPTION_DROPOUT_RATE:-0.0}"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}" "${LOGGING_DIR}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_file "${DATASET_CONFIG}" "dataset config"
require_file "${RAW_DIT}" "Krea2 raw DiT"
require_file "${VAE}" "Qwen Image VAE"
require_file "${TEXT_ENCODER}" "Qwen3-VL text encoder"

if [[ -d "${PROJECT_DIR}/.venv" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/.venv/bin/activate"
fi

if ! command -v accelerate >/dev/null 2>&1; then
  echo "[ERROR] accelerate not found. Run: cd ${PROJECT_DIR} && uv sync --extra cu128" >&2
  exit 1
fi

echo "[INFO] TQD Training Project: ${PROJECT_DIR}"
echo "[INFO] Dataset config: ${DATASET_CONFIG}"
echo "[INFO] Output name:    ${OUTPUT_NAME}"
echo "[INFO] TQD Enabled:    Yes"

if [[ "${RUN_CACHE_LATENTS}" == "1" ]]; then
  echo "[INFO] Caching latents..."
  python -m krea2_trainer.scripts.cache_latents \
    --dataset_config "${DATASET_CONFIG}" \
    --vae "${VAE}" \
    --device cuda
fi

if [[ "${RUN_CACHE_TEXT}" == "1" ]]; then
  echo "[INFO] Caching text encoder outputs..."
  python -m krea2_trainer.scripts.cache_text_encoder \
    --dataset_config "${DATASET_CONFIG}" \
    --text_encoder "${TEXT_ENCODER}" \
    --device cuda
fi

TRAIN_ARGS=(
  --raw_dit "${RAW_DIT}"
  --vae "${VAE}"
  --dataset_config "${DATASET_CONFIG}"
  --sdpa
  --mixed_precision bf16
  --save_precision bf16
  --timestep_sampling tqd_krea2_shift
  --weighting_scheme none
  --gradient_checkpointing
  --network_module krea2_trainer.networks.lora_krea2
  --network_dim 32
  --network_alpha 16
  --disable_numpy_memmap
  --fp8_base
  --fp8_scaled
  # TQD Specific Settings
  --tqd_kappa_base 2
  --tqd_kappa_max 8
  --tqd_quality_weighting
  # Hyperparameters
  --learning_rate 5e-5
  --lr_scheduler constant_with_warmup
  --lr_warmup_steps 200
  --optimizer_type Adopt_adv
  --optimizer_args
    cautious_wd=true
    kourkoutas_beta=true
    use_atan2=true
    weight_decay=0.01
  --max_train_epochs "${MAX_TRAIN_EPOCHS}"
  --save_every_n_epochs "${SAVE_EVERY_N_EPOCHS}"
  --output_dir "${OUTPUT_DIR}"
  --output_name "${OUTPUT_NAME}"
  --logging_dir "${LOGGING_DIR}"
  --log_prefix "tqd_rosie_"
  --log_with wandb
  --log_config
  --log_tracker_name krea2-trainer-tqd
  --wandb_run_name tqd-lora-run
  --seed "${SEED}"
  --max_data_loader_n_workers "${MAX_DATA_LOADER_N_WORKERS}"
  --persistent_data_loader_workers
  --cuda_allow_tf32
  --cuda_cudnn_benchmark
)

if [[ "${ENABLE_COMPILE}" == "1" ]]; then
  TRAIN_ARGS+=(
    --compile
    --compile_mode "${COMPILE_MODE}"
    --compile_dynamic "${COMPILE_DYNAMIC}"
    --compile_cache_size_limit "${COMPILE_CACHE_SIZE_LIMIT}"
  )
fi

if [[ -n "${WANDB_API_KEY}" ]]; then
  TRAIN_ARGS+=(--wandb_api_key "${WANDB_API_KEY}")
fi

if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
  TRAIN_ARGS+=(--max_train_steps "${MAX_TRAIN_STEPS}")
fi

if [[ "${REFRESH_TEXT_CACHE_EVERY_EPOCH}" == "1" ]]; then
  TRAIN_ARGS+=(
    --text_encoder "${TEXT_ENCODER}"
    --cache_te_every_epoch
    --cache_te_shuffle_caption
    --cache_te_caption_dropout_rate "${CACHE_TE_CAPTION_DROPOUT_RATE}"
    --cache_te_keep_tokens 1
    --cache_te_device cuda
    --cache_te_dtype bfloat16
  )
fi

TRAIN_ARGS+=("$@")

echo "[INFO] Launching TQD training..."
accelerate launch --num_cpu_threads_per_process "${NUM_CPU_THREADS_PER_PROCESS}" \
  -m krea2_trainer.scripts.train_lora \
  "${TRAIN_ARGS[@]}"
