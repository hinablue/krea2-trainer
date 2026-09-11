"""CPU contract tests: no pretrained checkpoints, image corpus, or CUDA needed."""

from contextlib import nullcontext, redirect_stderr
import copy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from accelerate import Accelerator
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from krea2_trainer.krea2_post_training import Krea2PostTrainingTrainer
from krea2_trainer.krea2_train_network import Krea2NetworkTrainer, apply_krea2_preset, krea2_setup_parser, main
from krea2_trainer.networks import lora_krea2
from krea2_trainer.training.parser_common import read_config_from_file, setup_parser_common
from krea2_trainer.training.preference import (
    POST_TRAINING_STATE,
    build_reference_contract,
    file_fingerprint,
    flow_dpo_loss,
    per_image_mse,
    reference_adapter_context,
    validate_post_training_args,
    validate_resume_contract,
)


def parser():
    return krea2_setup_parser(setup_parser_common())


def arguments(mode="flow_dpo", *options):
    return parser().parse_args(
        [
            "--post_training",
            mode,
            "--preference_manifest",
            "pairs.jsonl",
            "--network_module",
            "krea2_trainer.networks.lora_krea2",
            "--network_dim",
            "2",
            "--network_alpha",
            "2",
            "--timestep_sampling",
            "uniform",
            "--sdpa",
            "--mixed_precision",
            "bf16",
            *options,
        ]
    )


class PostTrainingParserTests(unittest.TestCase):
    def test_real_parser_defaults_leave_ordinary_training_unchanged(self):
        args = parser().parse_args([])
        self.assertEqual(args.post_training, "none")
        self.assertEqual(args.preference_batch_size, 1)
        self.assertEqual(args.flow_dpo_beta, 100.0)
        self.assertIsNone(args.reference_lora)
        self.assertEqual(args.timestep_sampling, "sigma")
        self.assertIsNone(args.mixed_precision)
        validate_post_training_args(args)

    def test_post_training_requires_explicit_supported_precision_before_loading(self):
        for mode in ("rft", "flow_dpo"):
            for environment in ("bf16", "fp16"):
                for precision in (None, "no"):
                    args = arguments(mode)
                    args.mixed_precision = precision
                    with (
                        self.subTest(mode=mode, environment=environment, precision=precision),
                        patch.dict("os.environ", {"ACCELERATE_MIXED_PRECISION": environment}),
                        patch.object(Krea2NetworkTrainer, "_validate_args_and_init") as parent,
                        patch("krea2_trainer.krea2_post_training.build_reference_contract", return_value={}) as fingerprint,
                    ):
                        with self.assertRaisesRegex(ValueError, "mixed_precision.*bf16.*fp16"):
                            Krea2PostTrainingTrainer()._validate_args_and_init(args)
                        parent.assert_not_called()
                        fingerprint.assert_not_called()
                for precision in ("bf16", "fp16"):
                    with patch.dict("os.environ", {"ACCELERATE_MIXED_PRECISION": environment}):
                        validate_post_training_args(arguments(mode, "--mixed_precision", precision))

    def test_explicit_default_valued_post_options_rejected_in_ordinary_mode(self):
        for options in (
            ["--preference_batch_size", "1"],
            ["--flow_dpo_beta", "100"],
            ["--preference_manifest", "pairs.jsonl"],
            ["--reference_lora", "stage1.safetensors"],
        ):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "Post-training-only"):
                validate_post_training_args(parser().parse_args(options))

    def test_positive_finite_parser_values(self):
        for flag, values in (("--flow_dpo_beta", ("0", "-1", "nan", "inf")), ("--preference_batch_size", ("0", "-2", "1.2"))):
            for value in values:
                with self.subTest(flag=flag, value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser().parse_args([flag, value])

    def test_config_values_validated_even_when_argparse_actions_are_bypassed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.toml"
            for text, error in (
                ("preference_batch_size = 1\n", "Post-training-only"),
                ('post_training = "bogus"\n', "must be none"),
                (
                    'post_training = "flow_dpo"\nmixed_precision = "bf16"\npreference_manifest = "p"\npreference_batch_size = -1\n',
                    "positive integer",
                ),
            ):
                path.write_text(text)
                argv = ["train", "--config_file", str(path)]
                with self.subTest(text=text), patch.object(sys, "argv", argv):
                    p = parser()
                    args = read_config_from_file(p.parse_args(), p)
                    with self.assertRaisesRegex(ValueError, error):
                        validate_post_training_args(args)

    def test_real_config_loader_and_cli_override_preserve_post_training_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.toml"
            path.write_text(
                '[training]\npost_training="flow_dpo"\npreference_manifest="pairs.jsonl"\n'
                'preference_batch_size=2\nflow_dpo_beta=80.0\ntimestep_sampling="uniform"\n'
                'network_module="krea2_trainer.networks.lora_krea2"\n'
                'mixed_precision="bf16"\n'
            )
            with patch.object(sys, "argv", ["train", "--config_file", str(path), "--flow_dpo_beta", "25"]):
                p = parser()
                args = read_config_from_file(p.parse_args(), p)
            validate_post_training_args(args)
            self.assertEqual(args.flow_dpo_beta, 25.0)
            self.assertEqual(args.preference_batch_size, 2)
            self.assertEqual(apply_krea2_preset(args).timestep_sampling, "uniform")

    def test_ordinary_preset_retained_but_post_training_preset_fails_without_mutation(self):
        args = apply_krea2_preset(parser().parse_args(["--preset", "lora-default"]))
        self.assertEqual(args.timestep_sampling, "shift")
        self.assertEqual(args.network_dim, 32)
        for mode in ("rft", "flow_dpo"):
            args = arguments(mode, "--preset", "lora-default")
            with self.assertRaisesRegex(ValueError, "preset"):
                apply_krea2_preset(args)
            self.assertEqual(args.timestep_sampling, "uniform")

    def test_invalid_combinations_fail_explicitly(self):
        cases = [
            ("--timestep_sampling", "sigma"),
            ("--timestep_sampling", "krea2_shift"),
            ("--weighting_scheme", "sigma_sqrt"),
            ("--min_timestep", "0"),
            ("--max_timestep", "1000"),
            ("--num_timestep_buckets", "1"),
            ("--preserve_distribution_shape",),
            ("--discrete_flow_shift", "3"),
            ("--sigmoid_scale", "2"),
            ("--logit_mean", "1"),
            ("--logit_std", "2"),
            ("--mode_scale", "2"),
            ("--compile",),
            ("--blocks_to_swap", "1"),
            ("--block_swap_h2d_only",),
            ("--dynamo_backend", "EAGER"),
            ("--gradient_checkpointing_cpu_offload",),
            ("--network_dropout", "0.2"),
            ("--network_args", "rank_dropout=0.2"),
            ("--network_args", "module_dropout=0.2"),
            ("--network_weights", "stage1.safetensors"),
            ("--dim_from_weights",),
            ("--no_metadata",),
            ("--cache_te_every_epoch",),
            ("--turbo_dit", "turbo"),
            ("--turbo_dit_cache",),
            ("--tqd_quality_weighting",),
            ("--timestep_sampling", "tqd_krea2_shift"),
            ("--tqd_kappa_base", "3"),
            ("--resume_from_huggingface",),
            ("--base_weights", "stage1", "--fp8_base"),
            ("--base_weights", "stage1", "--reference_lora", "stage2"),
            ("--network_module", "other.network"),
        ]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_post_training_args(arguments("flow_dpo", *options))

    def test_rft_retains_native_sampler_and_safe_fp8_stacking(self):
        for sampler in ("uniform", "krea2_shift"):
            args = arguments("rft", "--timestep_sampling", sampler, "--reference_lora", "stage1", "--fp8_base")
            validate_post_training_args(args)
            self.assertEqual(args.timestep_sampling, sampler)
        validate_post_training_args(arguments("flow_dpo", "--reference_lora", "stage1", "--fp8_base"))
        validate_post_training_args(arguments("flow_dpo", "--base_weights", "stage1"))
        with self.assertRaisesRegex(ValueError, "only meaningful"):
            validate_post_training_args(arguments("rft", "--flow_dpo_beta", "100"))

    def test_accelerator_environment_cannot_bypass_compile_guard(self):
        with patch.dict("os.environ", {"ACCELERATE_DYNAMO_BACKEND": "INDUCTOR"}):
            with self.assertRaisesRegex(ValueError, "Dynamo"):
                validate_post_training_args(arguments())

    def test_main_dispatch_and_invalid_mode_before_training(self):
        for mode in ("none", "rft", "flow_dpo"):
            argv = (
                ["train"]
                if mode == "none"
                else [
                    "train",
                    "--post_training",
                    mode,
                    "--preference_manifest",
                    "pairs.jsonl",
                    "--network_module",
                    "krea2_trainer.networks.lora_krea2",
                    "--timestep_sampling",
                    "uniform",
                    "--mixed_precision",
                    "bf16",
                ]
            )
            with (
                self.subTest(mode=mode),
                patch.object(sys, "argv", argv),
                patch("krea2_trainer.krea2_train_network.Krea2NetworkTrainer") as ordinary,
                patch("krea2_trainer.krea2_post_training.Krea2PostTrainingTrainer") as post,
            ):
                main()
                selected, other = (ordinary, post) if mode == "none" else (post, ordinary)
                selected.return_value.train.assert_called_once()
                other.assert_not_called()
        with (
            patch.object(sys, "argv", ["train", "--post_training", "flow_dpo"]),
            patch("krea2_trainer.krea2_post_training.Krea2PostTrainingTrainer") as post,
        ):
            with self.assertRaises(ValueError):
                main()
            post.assert_not_called()


class FlowDPOMathTests(unittest.TestCase):
    def test_identity_is_log_two(self):
        values = torch.tensor([0.2, 0.9])
        loss, metrics = flow_dpo_loss(values, values + 1, values, values + 1, 100.0)
        torch.testing.assert_close(loss, torch.tensor(math.log(2.0)))
        self.assertEqual(metrics["flow_dpo/margin"], 0)

    def test_sign_beta_half_factor_and_pair_mean(self):
        pw = torch.tensor([1.0, 2.0], requires_grad=True)
        pl = torch.tensor([3.0, 1.0], requires_grad=True)
        rw = torch.tensor([2.0, 2.0], requires_grad=True)
        rl = torch.tensor([2.0, 1.0], requires_grad=True)
        loss, _ = flow_dpo_loss(pw, pl, rw, rl, 2.0)
        expected = -torch.nn.functional.logsigmoid(torch.tensor([2.0, 0.0])).mean()
        torch.testing.assert_close(loss, expected)
        self.assertLess(loss.item(), math.log(2.0))
        loss.backward()
        self.assertTrue((pw.grad > 0).all())
        self.assertTrue((pl.grad < 0).all())
        self.assertIsNone(rw.grad)
        self.assertIsNone(rl.grad)

    def test_fp32_mse_mean_not_sum_and_shape_guard(self):
        pred = torch.full((2, 3, 1, 4, 4), 300.0, dtype=torch.float16)
        target = torch.zeros_like(pred)
        mse = per_image_mse(pred, target)
        self.assertEqual(mse.dtype, torch.float32)
        torch.testing.assert_close(mse, torch.full((2,), 90000.0))
        for left, right in (
            (pred, target[:1]),
            (torch.empty(0, 2), torch.empty(0, 2)),
            (torch.empty(2, 0), torch.empty(2, 0)),
            (torch.ones(2), torch.ones(2)),
        ):
            with self.assertRaises(ValueError):
                per_image_mse(left, right)

    def test_nonfinite_every_branch_beta_and_logits_rejected(self):
        for branch in range(4):
            for invalid in (float("nan"), float("inf"), float("-inf")):
                inputs = [torch.ones(2) for _ in range(4)]
                inputs[branch][0] = invalid
                with self.subTest(branch=branch, invalid=invalid), self.assertRaises(FloatingPointError):
                    flow_dpo_loss(*inputs, beta=100.0)
        for beta in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                flow_dpo_loss(*[torch.ones(2) for _ in range(4)], beta=beta)
        with self.assertRaises(FloatingPointError):
            flow_dpo_loss(torch.zeros(1), torch.full((1,), 1e38), torch.zeros(1), torch.zeros(1), 100.0)
        with self.assertRaises(FloatingPointError):
            per_image_mse(torch.full((1, 2), float("nan")), torch.zeros(1, 2))

    def test_vector_shapes_required(self):
        for shapes in (((), (), (), ()), ((0,),) * 4, ((2,), (1,), (2,), (2,)), ((2, 1),) * 4):
            with self.subTest(shapes=shapes), self.assertRaises(ValueError):
                flow_dpo_loss(*[torch.ones(shape) for shape in shapes], beta=1)


class TinyDiT(nn.Module):
    """Tiny model with native call_dit token/time/conditioning interfaces."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(patch=1)
        self.first = nn.Linear(2, 2)
        self.checkpointing = False
        self.calls = []

    def enable_gradient_checkpointing(self, cpu_offload=False):
        self.checkpointing = True

    def forward(self, img, context, t, pos, mask):
        self.calls.append(
            {
                "img": img.detach().clone(),
                "context": context.detach().clone(),
                "t": t.detach().clone(),
                "training": self.training,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        output = checkpoint(self.first, img, use_reentrant=False) if self.checkpointing and self.training else self.first(img)
        return output + 0.01 * context.mean((1, 2, 3))[:, None, None] + 0.01 * t[:, None, None]


def toy_accelerator():
    return SimpleNamespace(
        device=torch.device("cpu"), unwrap_model=lambda model: model, autocast=nullcontext, print=lambda *a, **kw: None
    )


def toy_batch():
    return {
        "latents": torch.arange(16, dtype=torch.float32).reshape(2, 2, 1, 2, 2) / 10,
        "rejected_latents": torch.arange(16, dtype=torch.float32).reshape(2, 2, 1, 2, 2) / -7,
        "krea2_vl_embed": [torch.ones(1, 1, 2), torch.full((2, 1, 2), 2.0)],
        "timesteps": None,
    }


class FlowDPOTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(321)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stage_path = Path(self.temp.name) / "stage1.safetensors"
        stage_model = TinyDiT()
        self.base_state = copy.deepcopy(stage_model.state_dict())
        stage = lora_krea2.create_arch_network(1.0, 2, 2, None, None, stage_model)
        stage.apply_to(None, stage_model, apply_text_encoder=False, apply_unet=True)
        with torch.no_grad():
            for module in stage.unet_loras:
                module.lora_up.weight.fill_(0.15)
        save_file(stage.state_dict(), str(self.stage_path))

    def invalid_reference_states(self):
        valid = load_file(str(self.stage_path))
        prefix = "lora_unet_first."
        return {
            "empty": {},
            "not-lora": {"not_a_lora": torch.ones(2)},
            "nan": dict(valid, **{prefix + "lora_down.weight": torch.full((2, 2), float("nan"))}),
            "inf-alpha": dict(valid, **{prefix + "alpha": torch.tensor(float("inf"))}),
            "missing-up": {key: value for key, value in valid.items() if not key.endswith("lora_up.weight")},
            "missing-alpha": {key: value for key, value in valid.items() if not key.endswith("alpha")},
            "wrong-model": {key.replace("first", "foreign"): value for key, value in valid.items()},
            "partial-match": dict(valid, **{key.replace("first", "foreign"): value.clone() for key, value in valid.items()}),
            "wrong-input": dict(valid, **{prefix + "lora_down.weight": torch.ones(2, 3)}),
            "wrong-output": dict(valid, **{prefix + "lora_up.weight": torch.ones(3, 2)}),
            "wrong-rank": dict(valid, **{prefix + "lora_up.weight": torch.ones(2, 3)}),
            "integer-weights": dict(valid, **{prefix + "lora_down.weight": torch.ones(2, 2, dtype=torch.int32)}),
            "zero-rank": dict(
                valid, **{prefix + "lora_down.weight": torch.empty(0, 2), prefix + "lora_up.weight": torch.empty(2, 0)}
            ),
            "vector-alpha": dict(valid, **{prefix + "alpha": torch.ones(2)}),
            "unexpected-key": dict(valid, **{prefix + "extra": torch.ones(1)}),
        }

    def test_invalid_base_weights_rejected_before_any_source_or_hook_mutates_base(self):
        invalid_path = Path(self.temp.name) / "invalid.safetensors"
        for case, state in self.invalid_reference_states().items():
            with self.subTest(case=case):
                save_file(state, str(invalid_path))
                # The first source is valid: rejecting only when merging the
                # second source is too late to preserve the reference model.
                args = arguments("flow_dpo", "--base_weights", str(self.stage_path), str(invalid_path))
                model = TinyDiT()
                before = copy.deepcopy(model.state_dict())
                original_forward = model.first.forward
                with self.assertRaisesRegex(ValueError, "base_weights"):
                    Krea2PostTrainingTrainer()._build_network(args, toy_accelerator(), model, None, torch.float32)
                self.assertEqual(model.first.forward, original_forward)
                for key, value in before.items():
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_invalid_stacked_reference_rejected_without_installing_hooks(self):
        invalid_path = Path(self.temp.name) / "invalid.safetensors"
        for case, state in self.invalid_reference_states().items():
            with self.subTest(case=case):
                save_file(state, str(invalid_path))
                args = arguments("flow_dpo", "--reference_lora", str(invalid_path))
                model = TinyDiT()
                before = copy.deepcopy(model.state_dict())
                original_forward = model.first.forward
                with self.assertRaisesRegex(ValueError, "reference_lora"):
                    Krea2PostTrainingTrainer()._build_network(args, toy_accelerator(), model, None, torch.float32)
                self.assertEqual(model.first.forward, original_forward)
                for key, value in before.items():
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def build(self, checkpointing=False, stage=True):
        args = arguments("flow_dpo", *(["--gradient_checkpointing"] if checkpointing else []))
        args.reference_lora = str(self.stage_path) if stage else None
        model = TinyDiT()
        model.load_state_dict(self.base_state)
        model.requires_grad_(False)
        model.train(checkpointing)
        trainer = Krea2PostTrainingTrainer()
        network = trainer._build_network(args, toy_accelerator(), model, None, torch.float32)
        return args, model, trainer, network

    @staticmethod
    def run_batch(args, model, trainer, network, batch=None):
        batch = toy_batch() if batch is None else batch
        noise = torch.full_like(batch["latents"], 0.3)
        return trainer.process_batch(
            args, toy_accelerator(), model, network, batch, batch["latents"], noise, None, torch.float32, torch.float32, None, 0
        )

    def test_actual_lora_chain_shared_inputs_initial_identity_and_update(self):
        args, model, trainer, network = self.build()
        base_before = copy.deepcopy(model.state_dict())
        reference_before = copy.deepcopy(trainer.reference_network.state_dict())
        optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
        trainer.on_train_start(args, toy_accelerator(), network, model, optimizer)
        batch = toy_batch()
        original = batch["latents"].clone()
        with patch("krea2_trainer.krea2_post_training.torch.rand", return_value=torch.tensor([0.2, 0.8])):
            loss, _ = self.run_batch(args, model, trainer, network, batch)
        torch.testing.assert_close(loss, torch.tensor(math.log(2.0)))
        ref, policy = model.calls
        torch.testing.assert_close(ref["img"], policy["img"])
        torch.testing.assert_close(ref["context"], policy["context"])
        torch.testing.assert_close(ref["t"], policy["t"])
        torch.testing.assert_close(ref["t"][:2], torch.tensor([0.201, 0.801]))
        torch.testing.assert_close(ref["t"][:2], ref["t"][2:])
        torch.testing.assert_close(ref["context"][:2], ref["context"][2:])
        self.assertFalse(ref["grad_enabled"])
        self.assertTrue(policy["grad_enabled"])
        t = torch.tensor([0.2, 0.8]).view(-1, 1, 1, 1, 1)
        expected = torch.cat(((1 - t) * batch["latents"] + t * 0.3, (1 - t) * batch["rejected_latents"] + t * 0.3))
        expected = expected.squeeze(2).flatten(2).transpose(1, 2)
        torch.testing.assert_close(ref["img"], expected)
        torch.testing.assert_close(batch["latents"], original)
        policy_before = copy.deepcopy(network.state_dict())
        probe = torch.randn(1, 3, 2)
        with reference_adapter_context(network, model):
            reference_output = model.first(probe).clone()
        loss.backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(value, network.state_dict()[key]) for key, value in policy_before.items()))
        for key, value in base_before.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        for key, value in reference_before.items():
            torch.testing.assert_close(trainer.reference_network.state_dict()[key], value, rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(
            all(parameter.grad is None and not parameter.requires_grad for parameter in trainer.reference_network.parameters())
        )
        with reference_adapter_context(network, model):
            torch.testing.assert_close(model.first(probe), reference_output, rtol=0, atol=0)
        with torch.no_grad():
            self.assertFalse(torch.equal(model.first(probe), reference_output))

    def dense_dataset_batch(self):
        from tests.test_preference_dataset import PreferenceDatasetTests

        fixture = PreferenceDatasetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for name, value in (("winner", 0.2), ("loser", -0.2), ("winner2", 0.4), ("loser2", -0.4)):
            fixture.item(name, value, varlen=False)
        return fixture.build([fixture.record(), fixture.record("winner2", "loser2", "pair-2")])[0]

    def test_dense_dataset_conditioning_reaches_loss_and_backward(self):
        batch = self.dense_dataset_batch()
        embeds = batch["krea2_vl_embed"]
        self.assertIsInstance(embeds, torch.Tensor)
        self.assertEqual(embeds.shape, (2, 3, 2, 4))
        original = embeds.clone()
        for checkpointing in (False, True):
            with self.subTest(checkpointing=checkpointing):
                args, model, trainer, network = self.build(checkpointing)
                with patch.object(trainer, "call_dit", wraps=trainer.call_dit) as forward:
                    loss, _ = self.run_batch(args, model, trainer, network, batch)
                torch.testing.assert_close(loss, torch.tensor(math.log(2.0)))
                self.assertEqual(forward.call_count, 2)
                paired = forward.call_args_list[0].args[4]["krea2_vl_embed"]
                self.assertIsInstance(paired, list)
                self.assertEqual(len(paired), 4)
                self.assertIs(paired, forward.call_args_list[1].args[4]["krea2_vl_embed"])
                for i in range(2):
                    self.assertIs(paired[i], paired[i + 2])
                    torch.testing.assert_close(paired[i], embeds[i], rtol=0, atol=0)
                reference, policy = model.calls
                torch.testing.assert_close(reference["context"], policy["context"], rtol=0, atol=0)
                torch.testing.assert_close(policy["context"], torch.cat((embeds, embeds)), rtol=0, atol=0)
                self.assertFalse(reference["grad_enabled"])
                self.assertTrue(policy["grad_enabled"])
                before = copy.deepcopy(network.state_dict())
                loss.backward()
                grads = [parameter.grad for parameter in network.parameters() if parameter.requires_grad]
                self.assertTrue(all(grad is not None and torch.isfinite(grad).all() for grad in grads))
                self.assertTrue(any(torch.count_nonzero(grad) for grad in grads))
                torch.optim.SGD(network.parameters(), lr=0.01).step()
                self.assertTrue(any(not torch.equal(value, network.state_dict()[key]) for key, value in before.items()))
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
                self.assertTrue(all(parameter.grad is None for parameter in trainer.reference_network.parameters()))
                self.assertIs(batch["krea2_vl_embed"], embeds)
                torch.testing.assert_close(embeds, original, rtol=0, atol=0)

    def test_dense_dataset_checkpointed_gradients_match_list_and_tuple_conditioning(self):
        batch = self.dense_dataset_batch()
        args, model, trainer, network = self.build(False)
        with torch.no_grad():
            for module in network.unet_loras:
                module.lora_up.weight.fill_(0.07)
        policy_state = copy.deepcopy(network.state_dict())
        torch.manual_seed(999)
        expected_loss, _ = self.run_batch(args, model, trainer, network, batch)
        expected_loss.backward()
        for encoding in ("dense", "list", "tuple"):
            with self.subTest(encoding=encoding):
                args2, model2, trainer2, network2 = self.build(True)
                network2.load_state_dict(policy_state, strict=True)
                embeds = batch["krea2_vl_embed"]
                if encoding != "dense":
                    embeds = list(embeds.unbind(0)) if encoding == "list" else tuple(embeds.unbind(0))
                torch.manual_seed(999)
                loss, _ = self.run_batch(args2, model2, trainer2, network2, dict(batch, krea2_vl_embed=embeds))
                loss.backward()
                torch.testing.assert_close(loss, expected_loss)
                for parameter, parameter2 in zip(network.parameters(), network2.parameters()):
                    torch.testing.assert_close(parameter2.grad, parameter.grad)
                self.assertFalse(model2.calls[0]["training"])
                self.assertTrue(model2.calls[1]["training"])
                self.assertTrue(all(module.multiplier == 1 for module in network2.unet_loras))

    def test_no_stage_one_uses_dit_itself_as_reference(self):
        args, model, trainer, network = self.build(stage=False)
        self.assertIsNone(trainer.reference_network)
        loss, _ = self.run_batch(args, model, trainer, network)
        torch.testing.assert_close(loss, torch.tensor(math.log(2.0)))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in network.parameters()))

    def test_non_fp8_base_weight_merge_becomes_immutable_reference(self):
        args = arguments("flow_dpo", "--base_weights", str(self.stage_path), "--base_weights_multiplier", "0.5")
        model = TinyDiT()
        model.load_state_dict(self.base_state)
        model.requires_grad_(False)
        model.eval()
        trainer = Krea2PostTrainingTrainer()
        network = trainer._build_network(args, toy_accelerator(), model, None, torch.float32)
        self.assertIsNone(trainer.reference_network)
        self.assertFalse(torch.equal(model.first.weight, self.base_state["first.weight"]))
        source = load_file(str(self.stage_path))
        delta = source["lora_unet_first.lora_up.weight"] @ source["lora_unet_first.lora_down.weight"]
        torch.testing.assert_close(model.first.weight, self.base_state["first.weight"] + 0.5 * delta, rtol=0, atol=0)
        self.assertEqual(args.base_weights, [str(self.stage_path)])
        merged = model.first.weight.detach().clone()
        loss, _ = self.run_batch(args, model, trainer, network)
        torch.testing.assert_close(loss, torch.tensor(math.log(2.0)))
        loss.backward()
        torch.optim.SGD(network.parameters(), lr=0.01).step()
        torch.testing.assert_close(model.first.weight, merged, rtol=0, atol=0)

    def test_multiple_valid_native_sources_merge_once_with_subset_and_default_multiplier(self):
        from krea2_trainer.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT

        model = SingleStreamDiT(
            SingleMMDiTConfig(
                features=16,
                tdim=8,
                txtdim=8,
                heads=1,
                multiplier=1,
                layers=1,
                patch=2,
                channels=2,
                txtlayers=2,
                txtheads=1,
                txtkvheads=1,
            )
        )
        before = copy.deepcopy(model.state_dict())
        source_model = copy.deepcopy(model)
        source = lora_krea2.create_arch_network(
            1.0,
            2,
            1.0,
            None,
            None,
            source_model,
            exclude_patterns=[".*"],
            include_patterns=["first"],
        )
        source.apply_to(None, source_model, apply_text_encoder=False, apply_unet=True)
        with torch.no_grad():
            source.unet_loras[0].lora_up.weight.fill_(0.2)
        path = Path(self.temp.name) / "subset.safetensors"
        save_file(source.state_dict(), str(path))
        args = arguments("flow_dpo", "--base_weights", str(path), str(path), "--base_weights_multiplier", "0.25")
        trainer = Krea2PostTrainingTrainer()
        network = trainer._build_network(args, toy_accelerator(), model, None, torch.bfloat16)
        # Native merge computes in FP32, rounds each source to weight_dtype,
        # then copies into the existing parameter dtype. Apply each source once.
        delta = source.unet_loras[0].lora_up.weight @ source.unet_loras[0].lora_down.weight
        expected = before["first.weight"]
        for multiplier in (0.25, 1.0):
            expected = (expected + multiplier * delta * 0.5).to(torch.bfloat16).float()
        torch.testing.assert_close(model.first.weight, expected, rtol=0, atol=0)
        for key, value in before.items():
            if key != "first.weight":
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(args.base_weights, [str(path), str(path)])
        self.assertEqual(args.base_weights_multiplier, [0.25])
        self.assertIsNone(trainer.reference_network)
        self.assertTrue(all(torch.count_nonzero(module.lora_up.weight) == 0 for module in network.unet_loras))

    def test_non_safetensors_reference_sources_rejected_before_mutation(self):
        invalid = Path(self.temp.name) / "invalid.safetensors"
        invalid.write_bytes(b"not a safetensors file")
        for option in ("base_weights", "reference_lora"):
            with self.subTest(option=option):
                args = arguments("flow_dpo", "--" + option, str(invalid))
                model = TinyDiT()
                before = copy.deepcopy(model.state_dict())
                original_forward = model.first.forward
                with self.assertRaisesRegex(ValueError, option):
                    Krea2PostTrainingTrainer()._build_network(args, toy_accelerator(), model, None, torch.float32)
                self.assertEqual(model.first.forward, original_forward)
                for key, value in before.items():
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_checkpointed_backward_matches_ordinary_with_nonzero_policy(self):
        args1, model1, trainer1, network1 = self.build(False)
        args2, model2, trainer2, network2 = self.build(True)
        with torch.no_grad():
            for module in network1.unet_loras:
                module.lora_up.weight.fill_(0.07)
        network2.load_state_dict(network1.state_dict(), strict=True)
        torch.manual_seed(999)
        loss1, _ = self.run_batch(args1, model1, trainer1, network1)
        loss1.backward()
        torch.manual_seed(999)
        loss2, _ = self.run_batch(args2, model2, trainer2, network2)
        loss2.backward()
        torch.testing.assert_close(loss1, loss2)
        self.assertFalse(model2.calls[0]["training"])
        self.assertTrue(model2.calls[1]["training"])
        for parameter1, parameter2 in zip(network1.parameters(), network2.parameters()):
            torch.testing.assert_close(parameter1.grad, parameter2.grad)
        self.assertTrue(all(module.multiplier == 1 for module in network2.unet_loras))

    def test_reference_context_restores_heterogeneous_flags_and_multipliers_on_error(self):
        _, model, trainer, network = self.build(True)
        network.multiplier = 0.7
        network.unet_loras[0].multiplier = 0.4
        network.unet_loras[0].lora_up.eval()
        flags = {id(module): module.training for module in [*network.modules(), *model.modules()]}
        with self.assertRaisesRegex(RuntimeError, "reference failed"):
            with reference_adapter_context(network, model):
                self.assertEqual(network.multiplier, 0)
                self.assertEqual(network.unet_loras[0].multiplier, 0)
                self.assertEqual(trainer.reference_network.multiplier, 1)
                raise RuntimeError("reference failed")
        self.assertEqual(network.multiplier, 0.7)
        self.assertEqual(network.unet_loras[0].multiplier, 0.4)
        self.assertEqual(flags, {id(module): module.training for module in [*network.modules(), *model.modules()]})

    def test_process_reference_exception_restores_policy(self):
        args, model, trainer, network = self.build(True)
        with patch.object(trainer, "call_dit", side_effect=RuntimeError("reference failed")):
            with self.assertRaisesRegex(RuntimeError, "reference failed"):
                self.run_batch(args, model, trainer, network)
        self.assertEqual(network.multiplier, 1)
        self.assertTrue(model.training)
        self.assertTrue(network.training)

    def test_reference_loading_is_strict_and_finite(self):
        for state in (
            {"not_a_lora": torch.ones(2)},
            {"lora_unet_first.lora_down.weight": torch.ones(2, 2)},
            {"lora_unet_first.lora_down.weight": torch.full((2, 2), float("nan"))},
        ):
            save_file(state, str(self.stage_path))
            with self.assertRaisesRegex(ValueError, "reference_lora"):
                self.build()

    def test_shape_conditioning_and_bucket_errors(self):
        args, model, trainer, network = self.build()
        for key, value in (
            ("rejected_latents", torch.ones(1, 2, 1, 2, 2)),
            ("krea2_vl_embed", []),
            ("timesteps", [0.2, 0.3]),
            ("tqd_quality_weight", torch.ones(2)),
        ):
            batch = toy_batch()
            batch[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.run_batch(args, model, trainer, network, batch)

    def test_dense_conditioning_rejects_invalid_shape_and_pair_count_before_forward(self):
        args, model, trainer, network = self.build()
        for shape in ((), (2, 3, 4), (2, 3, 1, 2, 4), (0, 3, 2, 4), (2, 0, 2, 4), (2, 3, 0, 4), (2, 3, 2, 0)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "dense krea2_vl_embed.*nonempty"):
                self.run_batch(args, model, trainer, network, dict(toy_batch(), krea2_vl_embed=torch.empty(shape)))
        for pairs in (1, 3):
            with self.subTest(pairs=pairs), self.assertRaisesRegex(ValueError, "one shared krea2_vl_embed per preference pair"):
                self.run_batch(args, model, trainer, network, dict(toy_batch(), krea2_vl_embed=torch.ones(pairs, 3, 2, 4)))
        self.assertEqual(model.calls, [])

    def test_variable_length_conditioning_rejects_invalid_items_before_forward(self):
        args, model, trainer, network = self.build()
        for container in (list, tuple):
            for invalid in (
                None,
                torch.ones(2, 4),
                torch.ones(1, 3, 2, 4),
                torch.empty(0, 2, 4),
                torch.empty(3, 0, 4),
                torch.empty(3, 2, 0),
            ):
                with (
                    self.subTest(container=container, invalid=invalid),
                    self.assertRaisesRegex(ValueError, "krea2_vl_embed.*nonempty.*tokens, layers, hidden"),
                ):
                    embeds = container((torch.ones(3, 2, 4), invalid))
                    self.run_batch(args, model, trainer, network, dict(toy_batch(), krea2_vl_embed=embeds))
        self.assertEqual(model.calls, [])

    def test_rft_delegates_ordinary_chosen_only_process(self):
        args, model, trainer, network = self.build()
        args.post_training = "rft"
        batch = toy_batch()
        del batch["rejected_latents"]
        sentinel = (torch.tensor(0.123), {"native": 1})
        with patch.object(Krea2NetworkTrainer, "process_batch", return_value=sentinel) as native:
            result = self.run_batch(args, model, trainer, network, batch)
        self.assertIs(result, sentinel)
        native.assert_called_once()
        self.assertIs(native.call_args.args[4], batch)
        self.assertEqual(model.calls, [])

    def test_rft_real_native_process_has_single_forward(self):
        args, model, trainer, network = self.build()
        args.post_training = "rft"
        batch = toy_batch()
        del batch["rejected_latents"]
        loss, _ = self.run_batch(args, model, trainer, network, batch)
        self.assertEqual(len(model.calls), 1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in network.parameters()))

    def test_dataset_tqd_rejected_before_parent_loading(self):
        args = arguments()
        args.dataset_config = "unused.toml"
        trainer = Krea2PostTrainingTrainer()
        with (
            patch(
                "krea2_trainer.krea2_post_training.config_utils.load_user_config",
                return_value={"datasets": [{"tqd_score_file": "missing.jsonl"}]},
            ),
            patch.object(Krea2NetworkTrainer, "_build_dataset") as base,
        ):
            with self.assertRaisesRegex(ValueError, "tqd_score_file"):
                trainer._build_dataset(args)
            base.assert_not_called()

    def contract(self, args):
        root = Path(self.temp.name)
        for name in ("toy-dit.bin", "pairs.jsonl", "dataset.toml"):
            (root / name).write_text("test fixture\n")
        args.dit = str(root / "toy-dit.bin")
        args.preference_manifest = str(root / "pairs.jsonl")
        args.dataset_config = str(root / "dataset.toml")
        return build_reference_contract(args)

    def test_metadata_and_source_identity_resume_guard(self):
        args, _, trainer, network = self.build()
        trainer.reference_contract = self.contract(args)
        metadata = trainer.extra_metadata(args)
        path = Path(self.temp.name) / "incremental.safetensors"
        network.save_weights(str(path), torch.float32, metadata)
        with safe_open(path, framework="pt") as handle:
            saved = handle.metadata()
            self.assertEqual(saved["ss_post_training"], "flow_dpo")
            self.assertEqual(saved["ss_reference_contract"], metadata["ss_reference_contract"])
        contract = trainer.reference_contract
        self.assertNotIn("sha256", contract["dit"])
        self.assertIn("stat-only", contract["dit"]["identity"])
        self.assertEqual(len(contract["reference_lora"]["sha256"]), 64)
        with self.assertRaisesRegex(ValueError, "metadata-less"):
            validate_resume_contract(self.temp.name, contract)
        (Path(self.temp.name) / POST_TRAINING_STATE).write_text(json.dumps(contract))
        validate_resume_contract(self.temp.name, contract)
        for field in ("dit", "reference_lora", "flow_dpo_beta", "preference_manifest", "post_training"):
            altered = copy.deepcopy(contract)
            altered[field] = "different"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Incompatible"):
                validate_resume_contract(self.temp.name, altered)
        self.assertEqual(file_fingerprint(args.reference_lora, content_hash=True), contract["reference_lora"])

    def test_actual_accelerator_state_roundtrip_and_guard_before_loading(self):
        args, model, trainer, network = self.build()
        trainer.reference_contract = self.contract(args)
        accelerator = Accelerator(cpu=True, mixed_precision="no")
        self.addCleanup(accelerator.free_memory)
        optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)
        model, network, optimizer = accelerator.prepare(model, network, optimizer)
        trainer._register_hooks_and_resume(args, accelerator, network)
        loss, _ = self.run_batch(args, model, trainer, network)
        accelerator.backward(loss)
        optimizer.step()
        before = copy.deepcopy(network.state_dict())
        root = Path(self.temp.name) / "state"
        accelerator.save_state(str(root))
        self.assertTrue((root / POST_TRAINING_STATE).is_file())
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.add_(1)
        accelerator.load_state(str(root))
        for key, value in before.items():
            torch.testing.assert_close(network.state_dict()[key], value, rtol=0, atol=0)
        saved = json.loads((root / POST_TRAINING_STATE).read_text())
        saved["reference_lora"] = None
        (root / POST_TRAINING_STATE).write_text(json.dumps(saved))
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.add_(2)
        changed = copy.deepcopy(network.state_dict())
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            accelerator.load_state(str(root))
        for key, value in changed.items():
            torch.testing.assert_close(network.state_dict()[key], value, rtol=0, atol=0)

    def test_explicit_precision_is_fingerprinted_and_resume_mismatch_rejected(self):
        args = arguments()
        self.contract(args)
        contracts = {}
        for precision in ("bf16", "fp16"):
            args.mixed_precision = precision
            trainer = Krea2PostTrainingTrainer()
            self.assertTrue(trainer._validate_args_and_init(args))
            contracts[precision] = trainer.reference_contract
            self.assertEqual(trainer.reference_contract["settings"]["mixed_precision"], precision)
        (Path(self.temp.name) / POST_TRAINING_STATE).write_text(json.dumps(contracts["bf16"]))
        args.resume = self.temp.name
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            Krea2PostTrainingTrainer()._validate_args_and_init(args)
        args.mixed_precision = "bf16"
        self.assertTrue(Krea2PostTrainingTrainer()._validate_args_and_init(args))


if __name__ == "__main__":
    unittest.main()
