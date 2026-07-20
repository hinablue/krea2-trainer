#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/opt/krea2-trainer}"
TRAIN_COMMAND="${PROJECT_DIR}/runpod/train.sh"

if [[ "${AUTO_START_TRAINING:-0}" == "1" ]]; then
  echo "[entrypoint] AUTO_START_TRAINING=1; starting explicit training workflow"
  exec "${TRAIN_COMMAND}"
fi

if [[ "$#" -gt 0 && "$1" != "sleep" ]]; then
  exec "$@"
fi

cat <<EOF
Krea2 trainer container is ready. No GPU training was started.
Mount the persistent volume at /workspace, then run:
  ${TRAIN_COMMAND}
Set ENV_FILE=/workspace/configs/train.env when using a dotenv file.
EOF
if [[ "$#" -gt 0 ]]; then
  exec "$@"
fi
exec sleep infinity
