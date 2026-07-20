# Local CUDA 13 training image. The NGC base is multi-arch (amd64/arm64),
# unlike the amd64-only Runpod image used by runpod/Dockerfile.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/krea2-trainer/.venv \
    PROJECT_DIR=/opt/krea2-trainer \
    HF_HOME=/workspace/.cache/huggingface \
    PATH=/root/.local/bin:/opt/krea2-trainer/.venv/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /opt/krea2-trainer
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
COPY scripts/ scripts/
COPY runpod/ runpod/
COPY docs/ docs/

RUN chmod +x scripts/train_from_env.sh runpod/train.sh runpod/entrypoint.sh \
    && uv sync --frozen --extra cu130

WORKDIR /workspace
ENTRYPOINT ["/opt/krea2-trainer/runpod/entrypoint.sh"]
CMD ["sleep", "infinity"]
