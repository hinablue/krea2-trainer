#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export PROJECT_DIR
export TRAIN_MODE="${TRAIN_MODE:-tqd}"
export DATASET_CONFIG="${DATASET_CONFIG:-${PROJECT_DIR}/configs/asian_tqd.toml}"
export MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/models}"
export RAW_DIT="${RAW_DIT:-${MODEL_DIR}/krea2_raw_bf16.safetensors}"
export VAE="${VAE:-${MODEL_DIR}/qwen_image_vae.safetensors}"
export TEXT_ENCODER="${TEXT_ENCODER:-${MODEL_DIR}/Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors}"
export OUTPUT_NAME="${OUTPUT_NAME:-hina_krea2_tqd_lora_v1}"
if [[ -n "${RUN_CACHE_LATENTS:-}" || -n "${RUN_CACHE_TEXT:-}" ]]; then
  if [[ "${RUN_CACHE_LATENTS:-0}" == 1 && "${RUN_CACHE_TEXT:-0}" == 1 ]]; then export CACHE_MODE=all
  elif [[ "${RUN_CACHE_LATENTS:-0}" == 1 ]]; then export CACHE_MODE=latents
  elif [[ "${RUN_CACHE_TEXT:-0}" == 1 ]]; then export CACHE_MODE=text
  else export CACHE_MODE=none; fi
fi
export CACHE_MODE="${CACHE_MODE:-none}"
exec "${PROJECT_DIR}/scripts/train_from_env.sh" "$@"
