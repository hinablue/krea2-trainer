"""Full training-loop fixtures: real tiny Krea2 DiT/LoRA/cache/optimizer/save.

Only checkpoint loading substitutes a dimension-reduced SingleStreamDiT. No
production checkpoints, source pictures or external services are accessed.
Each child process has an isolated CPU Accelerator state.
"""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


def run_fixture(mode, directory, encoding="varlen"):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from krea2_trainer.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT
    from krea2_trainer.krea2 import krea2_utils
    from krea2_trainer.krea2_train_network import main as training_main
    from krea2_trainer.krea2_post_training import Krea2PostTrainingTrainer
    from krea2_trainer.networks.lora_krea2 import create_arch_network

    torch.set_num_threads(1)
    torch.manual_seed(41)
    directory = Path(directory)
    images, cache, output = [directory / name for name in ("images", "cache", "output")]
    for path in (images, cache, output):
        path.mkdir()
    config = SingleMMDiTConfig(
        features=16, tdim=8, txtdim=8, heads=1, multiplier=1, layers=1, patch=2, channels=2, txtlayers=2, txtheads=1, txtkvheads=1
    )
    model = SingleStreamDiT(config)
    dit = directory / "tiny-base.safetensors"
    save_file(model.state_dict(), str(dit))
    stage1_model = copy.deepcopy(model)
    stage1 = create_arch_network(1.0, 2, 2.0, None, None, stage1_model)
    stage1.apply_to(None, stage1_model, apply_text_encoder=False, apply_unet=True)
    with torch.no_grad():
        for module in stage1.unet_loras:
            module.lora_up.weight.normal_(std=0.01)
    stage1_path = directory / "stage1.safetensors"
    stage1.save_weights(str(stage1_path), torch.float32, {})
    text = torch.randn(3, 2, 8)
    for name in ("winner", "loser"):
        save_file({"latents_1x4x4_float32": torch.randn(2, 1, 4, 4)}, str(cache / f"{name}_0032x0032_kr2.safetensors"))
        save_file(
            {f"{'varlen_' if encoding == 'varlen' else ''}krea2_vl_embed_float32": text},
            str(cache / f"{name}_kr2_te.safetensors"),
            metadata={"caption1": "fixture portrait"},
        )
    manifest = directory / "pairs.jsonl"
    manifest.write_text(
        json.dumps(
            dict(pair_id="fixture-pair", prompt="fixture portrait", chosen="winner.png", rejected="loser.png", split="train")
        )
        + "\n"
    )
    dataset = directory / "dataset.toml"
    dataset.write_text(
        "[general]\nresolution=[32,32]\nbatch_size=1\nenable_bucket=false\n"
        f'[[datasets]]\nimage_directory="{images}"\ncache_directory="{cache}"\nnum_repeats=1\n'
    )
    cli_args = [
        "--post_training",
        mode,
        "--preference_manifest",
        str(manifest),
        "--dataset_config",
        str(dataset),
        "--dit",
        str(dit),
        "--reference_lora",
        str(stage1_path),
        "--preference_batch_size",
        "1",
        "--network_module",
        "krea2_trainer.networks.lora_krea2",
        "--network_dim",
        "2",
        "--network_alpha",
        "2",
        "--mixed_precision",
        "bf16",
        "--sdpa",
        "--gradient_checkpointing",
        "--timestep_sampling",
        "uniform",
        "--weighting_scheme",
        "none",
        "--optimizer_type",
        "AdamW",
        "--learning_rate",
        "0.001",
        "--max_train_steps",
        "2",
        "--max_data_loader_n_workers",
        "0",
        "--seed",
        "41",
        "--output_dir",
        str(output),
        "--output_name",
        "fixture",
        "--save_state",
    ]
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    trainer = Krea2PostTrainingTrainer()

    # Fixture replaces only model loading; real dataset/trainer/DiT forward,
    # gradient checkpointing, AdamW, Accelerator and checkpoint saving all run.
    def load_tiny(*unused, **kwargs):
        return model.to(device=kwargs.get("device", "cpu"), dtype=kwargs.get("dtype", torch.float32))

    with (
        patch.object(krea2_utils, "load_krea2_dit", side_effect=load_tiny),
        patch("krea2_trainer.krea2_post_training.Krea2PostTrainingTrainer", return_value=trainer),
        patch.object(sys, "argv", ["krea2-train-lora", *cli_args]),
    ):
        training_main()
    saved = output / "fixture.safetensors"
    if not saved.is_file():
        raise AssertionError("Full trainer failed to write final incremental checkpoint")
    weights = load_file(str(saved))
    if not all(torch.isfinite(value).all() for value in weights.values()):
        raise AssertionError("Checkpoint contains nonfinite tensors")
    updates = sum(bool(torch.count_nonzero(value)) for key, value in weights.items() if "lora_up.weight" in key)
    if updates == 0:
        raise AssertionError("Optimizer failed to update zero-initialized incremental adapter")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), initial[key].to(value.dtype), rtol=0, atol=0)
    reference = trainer.reference_network
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value.cpu(), load_file(str(stage1_path))[key], rtol=0, atol=0)
    if any(parameter.grad is not None or parameter.requires_grad for parameter in reference.parameters()):
        raise AssertionError("Reference adapter was not frozen")
    with safe_open(str(saved), framework="pt") as checkpoint:
        metadata = checkpoint.metadata()
    if metadata["ss_post_training"] != mode or metadata["ss_steps"] != "2":
        raise AssertionError("Post-training checkpoint provenance or completed steps missing")
    if metadata["ss_mixed_precision"] != "bf16":
        raise AssertionError("CLI precision was overridden by the environment")
    state_files = list(output.glob("**/krea2_post_training_state.json"))
    if not state_files:
        raise AssertionError("Accelerator state omitted fixed-reference provenance sidecar")
    # Resume through the real Accelerator hooks with a fresh base+stage1 stack.
    # max_train_steps follows the existing trainer's additional-run step counter.
    original_contract = trainer.reference_contract
    model = SingleStreamDiT(config)
    model.load_state_dict(initial)
    cli_args += ["--resume", str(state_files[0].parent), "--max_train_steps", "1", "--output_name", "fixture_resumed"]
    trainer = Krea2PostTrainingTrainer()
    with (
        patch.object(krea2_utils, "load_krea2_dit", side_effect=load_tiny),
        patch("krea2_trainer.krea2_post_training.Krea2PostTrainingTrainer", return_value=trainer),
        patch.object(sys, "argv", ["krea2-train-lora", *cli_args]),
    ):
        training_main()
    resumed = load_file(str(output / "fixture_resumed.safetensors"))
    if not any(not torch.equal(weights[key], resumed[key]) for key in weights):
        raise AssertionError("Resumed optimizer made no additional update")
    if trainer.reference_contract != original_contract:
        raise AssertionError("Resume changed the original reference contract")
    for key, value in trainer.reference_network.state_dict().items():
        torch.testing.assert_close(value.cpu(), load_file(str(stage1_path))[key], rtol=0, atol=0)
    print(
        "FIXTURE_RESULT="
        + json.dumps(
            dict(
                mode=mode,
                conditioning=encoding,
                steps=2,
                updated_up_tensors=updates,
                checkpoint=str(saved),
                frozen_base=True,
                frozen_reference=True,
                resumed_optimizer_update=True,
                reference_state_sidecars=len(state_files),
            )
        )
    )


class PostTrainingIntegrationTests(unittest.TestCase):
    def test_real_tiny_krea2_training_loops(self):
        self._training_loops("varlen")

    def test_real_tiny_krea2_dense_training_loops(self):
        self._training_loops("dense")

    def _training_loops(self, encoding):
        for mode in ("rft", "flow_dpo"):
            with self.subTest(mode=mode, encoding=encoding), tempfile.TemporaryDirectory() as directory:
                environment = dict(
                    os.environ,
                    ACCELERATE_USE_CPU="true",
                    ACCELERATE_MIXED_PRECISION="fp16",
                    ACCELERATE_DYNAMO_BACKEND="NO",
                    CUDA_VISIBLE_DEVICES="",
                    WANDB_MODE="disabled",
                    TOKENIZERS_PARALLELISM="false",
                )
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--fixture", mode, directory, encoding],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=150,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                records = [
                    json.loads(line.split("=", 1)[1]) for line in result.stdout.splitlines() if line.startswith("FIXTURE_RESULT=")
                ]
                self.assertEqual(len(records), 1, result.stdout)
                self.assertEqual(records[0]["mode"], mode)
                self.assertEqual(records[0]["conditioning"], encoding)
                self.assertEqual(records[0]["steps"], 2)
                self.assertGreater(records[0]["updated_up_tensors"], 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fixture":
        run_fixture(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "varlen")
    else:
        unittest.main()
