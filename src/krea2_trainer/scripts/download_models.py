from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Callable

from huggingface_hub import hf_hub_download

DEFAULT_REPO = "Comfy-Org/Krea-2"
ASSETS: dict[str, int] = {
    "diffusion_models/krea2_raw_bf16.safetensors": 26_283_332_608,
    "text_encoders/qwen3vl_4b_bf16.safetensors": 8_875_719_384,
    "vae/qwen_image_vae.safetensors": 253_806_246,
}
MIN_FULL_DOWNLOAD_FREE_BYTES = 40 * 1024**3
DISK_RESERVE_BYTES = 2 * 1024**3


def provision_models(
    model_dir: Path,
    *,
    mode: str = "if_missing",
    repo_id: str = DEFAULT_REPO,
    revision: str = "main",
    token: str | None = None,
    downloader: Callable[..., str] = hf_hub_download,
) -> list[Path]:
    if mode not in {"if_missing", "never", "force"}:
        raise ValueError(f"Unsupported MODEL_FETCH mode: {mode}")
    model_dir = model_dir.expanduser().resolve()
    missing = [name for name in ASSETS if not (model_dir / name).is_file()]
    if mode == "never":
        if missing:
            raise FileNotFoundError("MODEL_FETCH=never and model files are missing: " + ", ".join(missing))
        return [model_dir / name for name in ASSETS]

    required = sum(ASSETS[name] for name in (ASSETS if mode == "force" else missing))
    minimum_free = required + DISK_RESERVE_BYTES
    if len(missing) == len(ASSETS) or mode == "force":
        minimum_free = max(minimum_free, MIN_FULL_DOWNLOAD_FREE_BYTES)
    if required:
        model_dir.mkdir(parents=True, exist_ok=True)
        available = shutil.disk_usage(model_dir).free
        if available < minimum_free:
            raise OSError(
                f"Insufficient disk space for Krea2 models: required={minimum_free} available={available}"
            )

    for filename in ASSETS:
        target = model_dir / filename
        if mode == "if_missing" and target.is_file():
            print(f"[model] reuse {target}")
            continue
        print(f"[model] fetch {repo_id}/{filename} @ {revision}")
        downloader(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=str(model_dir),
            token=token,
            force_download=mode == "force",
        )
        if not target.is_file():
            raise RuntimeError(f"Hugging Face download completed without expected file: {target}")
    return [model_dir / name for name in ASSETS]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Provision the public Krea 2 model files into a stable directory")
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "/workspace/models"))
    parser.add_argument("--mode", choices=("if_missing", "never", "force"), default=os.environ.get("MODEL_FETCH", "if_missing"))
    parser.add_argument("--repo", default=os.environ.get("HF_MODEL_REPO", DEFAULT_REPO))
    parser.add_argument("--revision", default=os.environ.get("HF_MODEL_REVISION", "main"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    provision_models(
        Path(args.model_dir),
        mode=args.mode,
        repo_id=args.repo,
        revision=args.revision,
        token=os.environ.get("HF_TOKEN"),
    )


if __name__ == "__main__":
    main()
