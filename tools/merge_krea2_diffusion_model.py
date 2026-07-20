#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import torch
from diffusers import Krea2Transformer2DModel
from diffusers.loaders import Krea2LoraLoaderMixin
from safetensors import safe_open
from safetensors.torch import save_file

from convert_fused_diffusers_to_comfy_krea2 import map_key as diffusers_to_comfy_key
from merge_krea2_lora import krea2_musubi_to_diffusion_key


def bake_network_alpha(
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Bake alpha/rank scaling into LoRA B/up tensors.

    Supports both legacy Musubi/Kohya ``lora_down/lora_up + .alpha``
    checkpoints and ai-toolkit Krea-2 checkpoints using native
    ``diffusion_model.*.lora_A/lora_B`` keys with ``ss_network_alpha`` in
    safetensors metadata.
    """
    metadata = metadata or {}
    metadata_alpha = metadata.get("ss_network_alpha")
    metadata_alpha_value = float(metadata_alpha) if metadata_alpha is not None else None
    output = {}
    factors = {}
    for key, value in state_dict.items():
        if key.endswith(".alpha"):
            continue
        if key.endswith(".lora_up.weight") or key.endswith(".lora_B.weight"):
            if key.endswith(".lora_up.weight"):
                stem = key[: -len(".lora_up.weight")]
                down_key = f"{stem}.lora_down.weight"
            else:
                stem = key[: -len(".lora_B.weight")]
                down_key = f"{stem}.lora_A.weight"
            alpha_key = f"{stem}.alpha"
            if down_key not in state_dict:
                raise KeyError(f"Incomplete LoRA A/B pair for module: {stem}")
            rank = state_dict[down_key].shape[0]
            if alpha_key in state_dict:
                alpha = float(state_dict[alpha_key].item())
            elif metadata_alpha_value is not None:
                alpha = metadata_alpha_value
            else:
                # PEFT's default scaling is alpha=rank. Preserve that behavior
                # when neither tensor nor metadata carries an explicit alpha.
                alpha = float(rank)
            factor = alpha / rank
            value = (value.float() * factor).to(value.dtype)
            factors[stem] = factor
        output[key] = value
    return output, factors


def read_safetensors_metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.metadata() or {}


def save_single_comfyui_file(
    transformer: Krea2Transformer2DModel,
    output_path: Path,
    merge_info: dict,
) -> None:
    """Save one ComfyUI-native Krea-2 safetensors artifact atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output_path) + ".tmp")
    temporary.unlink(missing_ok=True)

    output = {}
    source_count = 0
    for key, tensor in transformer.state_dict().items():
        new_key, flatten = diffusers_to_comfy_key(key)
        if new_key in output:
            raise RuntimeError(f"Duplicate ComfyUI key after conversion: {new_key}")
        tensor = tensor.detach().cpu()
        if flatten:
            tensor = tensor.reshape(-1)
        output[new_key] = tensor.contiguous()
        source_count += 1

    if source_count != 430 or len(output) != 430:
        raise RuntimeError(
            f"Expected 430 Krea-2 tensors, source={source_count}, mapped={len(output)}"
        )
    required = {
        "txtfusion.projector.weight",
        "first.weight",
        "blocks.0.attn.wq.weight",
        "blocks.27.mlp.down.weight",
        "last.linear.weight",
    }
    missing = required - set(output)
    if missing:
        raise RuntimeError(f"Missing ComfyUI Krea-2 signatures: {sorted(missing)}")

    dtype_counts: dict[str, int] = {}
    for tensor in output.values():
        name = str(tensor.dtype).removeprefix("torch.")
        dtype_counts[name] = dtype_counts.get(name, 0) + 1

    metadata = {
        "key_layout": "comfyui_krea2",
        "converted_from": "diffusers_krea2",
        "base_model": str(merge_info["base_model"]),
        "merge_info": json.dumps(merge_info, separators=(",", ":")),
        "dtype_counts": json.dumps(dtype_counts, sort_keys=True),
    }
    try:
        save_file(output, str(temporary), metadata=metadata)
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Single-file tensor dtypes: {json.dumps(dtype_counts, sort_keys=True)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequentially fuse Krea-2 LoRAs into Krea-2-Turbo and save the merged transformer."
    )
    parser.add_argument(
        "--base",
        default="krea/Krea-2-Turbo",
        help="Base Krea-2-Turbo model ID or local Diffusers directory.",
    )
    parser.add_argument(
        "--lora",
        action="append",
        nargs=2,
        required=True,
        metavar=("PATH", "SCALE"),
        help="LoRA path and scale. Repeat this option to fuse adapters in order.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output directory for a sharded Diffusers transformer, or a .safetensors "
            "path for one ComfyUI-native BF16 transformer file."
        ),
    )
    parser.add_argument(
        "--adapter-name",
        default="merged_lora",
        help="Prefix used for the internal adapter names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_target = Path(args.output).expanduser().resolve()
    single_file = output_target.suffix == ".safetensors"
    output_dir = output_target.parent if single_file else output_target
    transformer_dir = output_dir / "transformer"
    if not single_file:
        transformer_dir.mkdir(parents=True, exist_ok=True)

    loras = []
    for index, (raw_path, raw_scale) in enumerate(args.lora, start=1):
        lora_path = Path(raw_path).expanduser().resolve()
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA path does not exist: {lora_path}")
        try:
            scale = float(raw_scale)
        except ValueError as exc:
            raise ValueError(f"Invalid scale for {lora_path}: {raw_scale}") from exc
        loras.append(
            {
                "path": lora_path,
                "scale": scale,
                "adapter_name": f"{args.adapter_name}_{index}",
            }
        )

    print(f"Loading base model: {args.base}")

    # Only the transformer is needed. Loading the full pipeline would also
    # allocate the text encoder and VAE, and requires scheduler/tokenizer files.
    transformer = Krea2Transformer2DModel.from_pretrained(
        args.base,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    for index, lora in enumerate(loras, start=1):
        lora_path = lora["path"]
        adapter_name = lora["adapter_name"]
        scale = lora["scale"]
        print(f"[{index}/{len(loras)}] Loading LoRA: {lora_path}")

        if lora_path.is_file():
            state_dict, metadata = Krea2LoraLoaderMixin.lora_state_dict(
                str(lora_path.parent),
                weight_name=lora_path.name,
                return_lora_metadata=True,
            )
        else:
            state_dict, metadata = Krea2LoraLoaderMixin.lora_state_dict(
                str(lora_path),
                return_lora_metadata=True,
            )

        # These training outputs use Musubi/Kohya `lora_unet_*` keys. Convert
        # them to Krea-2 diffusion keys, then let Diffusers perform its native
        # conversion to PEFT keys.
        file_metadata = read_safetensors_metadata(lora_path) if lora_path.is_file() else metadata
        state_dict, alpha_factors = bake_network_alpha(state_dict, file_metadata)
        unique_alpha_factors = sorted(set(alpha_factors.values()))
        print(f"Baked alpha/rank factors into LoRA B tensors: {unique_alpha_factors}")
        state_dict = {
            krea2_musubi_to_diffusion_key(key): value
            for key, value in state_dict.items()
        }
        state_dict = Krea2LoraLoaderMixin.lora_state_dict(state_dict)
        Krea2LoraLoaderMixin.load_lora_into_transformer(
            state_dict,
            transformer=transformer,
            adapter_name=adapter_name,
            low_cpu_mem_usage=True,
            metadata=metadata,
        )

        loaded_adapters = sorted(transformer.peft_config)
        print("Loaded transformer adapters:")
        print(json.dumps(loaded_adapters, indent=2))
        if adapter_name not in loaded_adapters:
            raise RuntimeError(
                "LoRA was not loaded into the Krea-2 transformer. "
                "Check whether this is actually a Krea-2-compatible LoRA."
            )

        print(
            f"Fusing adapter '{adapter_name}' into transformer "
            f"with scale={scale}"
        )
        transformer.fuse_lora(
            lora_scale=scale,
            safe_fusing=True,
            adapter_names=[adapter_name],
        )

        # Remove the PEFT wrapper before loading the next adapter. The fused
        # delta remains baked into the transformer weights.
        transformer.unload_lora()

    transformer.to(device="cpu")

    merge_info = {
        "base_model": args.base,
        "loras": [
            {
                "path": str(lora["path"]),
                "scale": lora["scale"],
                "adapter_name": lora["adapter_name"],
            }
            for lora in loras
        ],
        "fusion_order": "sequential",
        "dtype": "BF16 weights with Krea-2-required FP32 norm modules",
        "component": "transformer",
        "key_layout": "comfyui_krea2" if single_file else "diffusers_krea2",
    }

    if single_file:
        print(f"Saving fused transformer as one ComfyUI file: {output_target}")
        save_single_comfyui_file(transformer, output_target, merge_info)
    else:
        print(f"Saving fused transformer to: {transformer_dir}")
        transformer.save_pretrained(
            transformer_dir,
            safe_serialization=True,
            max_shard_size="4GB",
        )
        (output_dir / "merge_info.json").write_text(
            json.dumps(merge_info, indent=2),
            encoding="utf-8",
        )

    print("Fusion completed successfully.")
    print(f"Output: {output_target if single_file else transformer_dir}")


if __name__ == "__main__":
    main()
