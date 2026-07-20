FROM runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PROJECT_DIR=/opt/krea2-trainer

RUN apt-get update && apt-get install -y --no-install-recommends curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/opt/krea2-trainer/.venv/bin:${PATH}"

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
