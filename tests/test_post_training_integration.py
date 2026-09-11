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
    from krea2_trainer.krea2_train_network import main as training_main, Krea2NetworkTrainer
    from krea2_trainer.krea2_post_training import Krea2PostTrainingTrainer
    from krea2_trainer.networks.lora_krea2 import create_arch_network

    if mode == "flow_cpo":
        from krea2_trainer.krea2_flow_cpo import Krea2FlowCPOTrainer
        from krea2_trainer.training.flow_cpo import FLOW_CPO_EMA_STATE

        trainer_class = Krea2FlowCPOTrainer
        trainer_patch = "krea2_trainer.krea2_flow_cpo.Krea2FlowCPOTrainer"
    elif mode == "tqd":
        trainer_class = Krea2NetworkTrainer
        trainer_patch = "krea2_trainer.krea2_train_network.Krea2NetworkTrainer"
    else:
        trainer_class = Krea2PostTrainingTrainer
        trainer_patch = "krea2_trainer.krea2_post_training.Krea2PostTrainingTrainer"

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
    # Two distinct pairs allow an actual unsynchronized accumulation microstep.
    names = ("winner", "loser", "winner2", "loser2") if mode == "flow_cpo" else ("winner", "loser")
    for name in names:
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
    if mode == "flow_cpo":
        with manifest.open("a") as handle:
            handle.write(
                json.dumps(
                    dict(
                        pair_id="fixture-pair-2",
                        prompt="fixture portrait",
                        chosen="winner2.png",
                        rejected="loser2.png",
                        split="train",
                    )
                )
                + "\n"
            )
    dataset = directory / "dataset.toml"
    scores = directory / "scores.jsonl"
    scores.write_text("".join(json.dumps(dict(image_file=f"{name}.png", structure_score=0.8, detail_score=0.3)) + "\n" for name in names))
    dataset.write_text(
        "[general]\nresolution=[32,32]\nbatch_size=1\nenable_bucket=false\n"
        f'[[datasets]]\nimage_directory="{images}"\ncache_directory="{cache}"\nnum_repeats=1\n'
        + (f'tqd_score_file="{scores}"\n' if mode == "tqd" else "")
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
    if mode == "tqd":
        for flag in ("--post_training", "--preference_manifest", "--preference_batch_size", "--reference_lora"):
            index = cli_args.index(flag)
            del cli_args[index:index + 2]
        cli_args[cli_args.index("--timestep_sampling") + 1] = "tqd_krea2_shift"
        cli_args += ["--tqd_quality_weighting"]
    if mode == "flow_cpo":
        cli_args += [
            "--gradient_accumulation_steps",
            "2",
            "--flow_cpo_beta",
            "0.5",
            "--flow_cpo_lambda",
            "1",
            "--flow_cpo_ema_decay",
            "0.99",
        ]
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    trainer = trainer_class()

    def instrument_ema(candidate, expected_updates, expected_ema=None):
        events = []
        if mode != "flow_cpo":
            return events
        original_start = candidate.on_train_start
        original_step = candidate.on_post_optimizer_step

        def checked_start(*args, **kwargs):
            original_start(*args, **kwargs)
            if candidate.ema_updates != expected_updates:
                raise AssertionError("EMA update counter reset or failed to resume")
            if expected_ema is not None:
                for key, value in candidate.ema_network.named_parameters():
                    torch.testing.assert_close(value.cpu(), expected_ema[key], rtol=0, atol=0)

        def checked_step(args, accelerator, network, transformer, sync_gradients, global_step):
            before = {key: value.detach().clone() for key, value in candidate.ema_network.named_parameters()}
            count = candidate.ema_updates
            should_update = sync_gradients and accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped
            original_step(args, accelerator, network, transformer, sync_gradients, global_step)
            if candidate.ema_updates != count + int(should_update):
                raise AssertionError("EMA updated at the wrong accumulation boundary")
            policy_parameters = dict(accelerator.unwrap_model(network).named_parameters())
            for key, value in candidate.ema_network.named_parameters():
                expected = before[key]
                if should_update:
                    expected = expected * candidate.ema.decay + policy_parameters[key].detach() * (1 - candidate.ema.decay)
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
                if value.requires_grad or value.grad is not None:
                    raise AssertionError("EMA acquired gradients")
            events.append(bool(should_update))

        candidate.on_train_start = checked_start
        candidate.on_post_optimizer_step = checked_step
        return events

    ema_events = instrument_ema(trainer, 0)

    # Fixture replaces only model loading; real dataset/trainer/DiT forward,
    # gradient checkpointing, AdamW, Accelerator and checkpoint saving all run.
    def load_tiny(*unused, **kwargs):
        return model.to(device=kwargs.get("device", "cpu"), dtype=kwargs.get("dtype", torch.float32))

    with (
        patch.object(krea2_utils, "load_krea2_dit", side_effect=load_tiny),
        patch(trainer_patch, return_value=trainer),
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
    reference = getattr(trainer, "reference_network", None)
    if reference is not None:
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(value.cpu(), load_file(str(stage1_path))[key], rtol=0, atol=0)
        if any(parameter.grad is not None or parameter.requires_grad for parameter in reference.parameters()):
            raise AssertionError("Reference adapter was not frozen")
    with safe_open(str(saved), framework="pt") as checkpoint:
        metadata = checkpoint.metadata()
    if (mode != "tqd" and metadata["ss_post_training"] != mode) or metadata["ss_steps"] != "2":
        raise AssertionError("Post-training checkpoint provenance or completed steps missing")
    if metadata["ss_mixed_precision"] != "bf16":
        raise AssertionError("CLI precision was overridden by the environment")
    state_files = list(output.glob("**/optimizer.bin" if mode == "tqd" else "**/krea2_post_training_state.json"))
    if not state_files:
        raise AssertionError("Accelerator state omitted fixed-reference provenance sidecar")
    # Resume through the real Accelerator hooks with a fresh base+stage1 stack.
    # max_train_steps follows the existing trainer's additional-run step counter.
    saved_ema = None
    if mode == "flow_cpo":
        if ema_events != [False, True, False, True] or trainer.ema_updates != 2:
            raise AssertionError(f"Unexpected EMA update schedule: {ema_events}")
        ema_path = state_files[0].parent / FLOW_CPO_EMA_STATE
        saved_ema = load_file(str(ema_path))
        with safe_open(str(ema_path), framework="pt") as handle:
            if handle.metadata()["updates"] != "2":
                raise AssertionError("EMA saved count differs from completed steps")
        for key, value in trainer.ema_network.named_parameters():
            torch.testing.assert_close(value.cpu(), saved_ema[key], rtol=0, atol=0)
        if not any(not torch.equal(weights[key], saved_ema[key]) for key in saved_ema):
            raise AssertionError("EMA state unexpectedly equals policy rather than tracking its history")
        if set(weights) != set(trainer.ema_network.state_dict()):
            raise AssertionError("Export contains non-policy adapter tensors")
    if mode == "tqd" and not trainer._tqd_score_cache._entries:
        raise AssertionError("TQD CLI did not use prepared score parameters")
    original_contract = getattr(trainer, "reference_contract", None)
    model = SingleStreamDiT(config)
    model.load_state_dict(initial)
    cli_args += ["--resume", str(state_files[0].parent), "--max_train_steps", "1", "--output_name", "fixture_resumed"]
    trainer = trainer_class()
    resumed_ema_events = instrument_ema(trainer, 2, saved_ema)
    with (
        patch.object(krea2_utils, "load_krea2_dit", side_effect=load_tiny),
        patch(trainer_patch, return_value=trainer),
        patch.object(sys, "argv", ["krea2-train-lora", *cli_args]),
    ):
        training_main()
    resumed = load_file(str(output / "fixture_resumed.safetensors"))
    if not any(not torch.equal(weights[key], resumed[key]) for key in weights):
        raise AssertionError("Resumed optimizer made no additional update")
    if getattr(trainer, "reference_contract", None) != original_contract:
        raise AssertionError("Resume changed the original reference contract")
    if mode != "tqd":
        for key, value in trainer.reference_network.state_dict().items():
            torch.testing.assert_close(value.cpu(), load_file(str(stage1_path))[key], rtol=0, atol=0)
    if mode == "flow_cpo":
        if resumed_ema_events != [False, True] or trainer.ema_updates != 3:
            raise AssertionError("Resume failed to continue EMA history across accumulation")
        resumed_states = []
        for path in output.glob("**/" + FLOW_CPO_EMA_STATE):
            with safe_open(str(path), framework="pt") as handle:
                if handle.metadata()["updates"] == "3":
                    resumed_states.append(path)
        if len(resumed_states) != 1:
            raise AssertionError("Resume omitted its updated EMA state")
        resumed_ema = load_file(str(resumed_states[0]))
        for key, value in trainer.ema_network.named_parameters():
            torch.testing.assert_close(value.cpu(), resumed_ema[key], rtol=0, atol=0)
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
                ema_updates=trainer.ema_updates if mode == "flow_cpo" else None,
                ema_accumulation_checked=mode == "flow_cpo",
            )
        )
    )


class PostTrainingIntegrationTests(unittest.TestCase):
    def test_real_tiny_krea2_training_loops(self):
        self._training_loops("varlen")

    def test_real_tiny_krea2_dense_training_loops(self):
        self._training_loops("dense")

    def _training_loops(self, encoding):
        for mode in ("rft", "flow_dpo", "flow_cpo", "tqd"):
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
                if mode == "flow_cpo":
                    self.assertEqual(records[0]["ema_updates"], 3)
                    self.assertTrue(records[0]["ema_accumulation_checked"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fixture":
        run_fixture(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "varlen")
    else:
        unittest.main()
