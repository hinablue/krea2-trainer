"""CPU-only FlowCPO helper contracts; no pretrained weights or training data."""

from contextlib import ExitStack
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from krea2_trainer.networks import lora_krea2
from krea2_trainer.training.flow_cpo import AdapterEMA, FLOW_CPO_EMA_STATE, flow_cpo_loss, old_adapter_context
from tests.test_post_training import TinyDiT


DEV_ROOT = Path(__file__).resolve().parents[1]


def attach(model, multiplier=1.0, rank=2, alpha=2):
    network = lora_krea2.create_arch_network(multiplier, rank, alpha, None, None, model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    return network


def snapshot(network):
    return {key: value.detach().clone() for key, value in network.state_dict().items()}


def runtime_state(*networks):
    modules = list(dict.fromkeys(module for network in networks for module in network.modules()))
    return [(module, module.training, getattr(module, "multiplier", None)) for module in modules]


class FlowCPOMathTests(unittest.TestCase):
    def test_independent_scalar_oracle_and_all_nonbatch_dimensions(self):
        values = [
            torch.tensor(numbers).reshape(2, 2, 2)
            for numbers in (
                [1.0, -2, 3, 0, 4, 1, -1, 2],
                [3.0, 1, -2, 4, 2, 0, 1, -3],
                [0.5, -1, 2, 1, -2, 0, 3, 2],
                [1.0, 2, 0, -1, 4, 1, -2, 0],
                [0.0, 1, -1, 2, 1, 3, 0, -2],
                [-1.0, 0, 2, 1, 3, -2, 1, 4],
            )
        ]
        beta, loss_lambda = 0.4, 1.7
        rows = [value.flatten(1).tolist() for value in values]
        chosen, rejected = [], []
        for pair in range(2):
            chosen.append(
                sum(
                    ((1 - beta) * old + beta * policy - target) ** 2
                    for policy, old, target in zip(rows[0][pair], rows[2][pair], rows[4][pair])
                )
                / 4
            )
            rejected.append(
                sum(
                    ((1 + beta) * old - beta * policy - target) ** 2
                    for policy, old, target in zip(rows[1][pair], rows[3][pair], rows[5][pair])
                )
                / 4
            )
        expected = sum(winner + loss_lambda * loser for winner, loser in zip(chosen, rejected)) / 2
        loss, metrics = flow_cpo_loss(*values, beta, loss_lambda)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(loss.ndim, 0)
        self.assertAlmostEqual(loss.item(), expected, places=5)
        self.assertAlmostEqual(metrics["flow_cpo/loss"], expected, places=5)
        self.assertAlmostEqual(metrics["flow_cpo/chosen_mse"], sum(chosen) / 2, places=5)
        self.assertAlmostEqual(metrics["flow_cpo/rejected_mse"], sum(rejected) / 2, places=5)
        self.assertTrue(all(isinstance(value, float) and math.isfinite(value) for value in metrics.values()))

    def test_analytic_policy_gradients_and_old_stopgrad(self):
        pw = torch.tensor([[2.0, 4], [3, 1]], requires_grad=True)
        pl = torch.tensor([[3.0, 2], [4, 1]], requires_grad=True)
        ow = torch.tensor([[1.0, 2], [2, 0]], requires_grad=True)
        ol = torch.tensor([[4.0, 3], [3, 2]], requires_grad=True)
        tw, tl = torch.zeros_like(pw), torch.zeros_like(pl)
        beta, loss_lambda = 0.25, 2.0
        loss, _ = flow_cpo_loss(pw, pl, ow, ol, tw, tl, beta, loss_lambda)
        loss.backward()
        expected_w = torch.tensor(
            [
                [2 * beta * ((1 - beta) * o + beta * p) / 4 for p, o in zip(ps, os)]
                for ps, os in zip(pw.detach().tolist(), ow.detach().tolist())
            ]
        )
        expected_l = torch.tensor(
            [
                [-2 * beta * loss_lambda * ((1 + beta) * o - beta * p) / 4 for p, o in zip(ps, os)]
                for ps, os in zip(pl.detach().tolist(), ol.detach().tolist())
            ]
        )
        torch.testing.assert_close(pw.grad, expected_w)
        torch.testing.assert_close(pl.grad, expected_l)
        self.assertTrue(torch.all(pw.grad > 0))
        self.assertTrue(torch.all(pl.grad < 0))
        self.assertIsNone(ow.grad)
        self.assertIsNone(ol.grad)

    def test_beta_one_lambda_zero_is_exact_chosen_flow_matching(self):
        pw = torch.tensor([[2.0, -3], [1, 4]], requires_grad=True)
        pl = torch.full_like(pw, 7.0, requires_grad=True)
        ow = torch.full_like(pw, 13.0, requires_grad=True)
        ol = torch.full_like(pw, -2.0, requires_grad=True)
        tw, tl = torch.ones_like(pw), torch.zeros_like(pw)
        loss, _ = flow_cpo_loss(pw, pl, ow, ol, tw, tl, beta=1, loss_lambda=0)
        torch.testing.assert_close(loss, F.mse_loss(pw, tw), rtol=0, atol=0)
        loss.backward()
        torch.testing.assert_close(pw.grad, 2 * (pw.detach() - tw) / pw.numel(), rtol=0, atol=0)
        torch.testing.assert_close(pl.grad, torch.zeros_like(pl), rtol=0, atol=0)
        self.assertIsNone(ow.grad)
        self.assertIsNone(ol.grad)

    def test_mix_is_fp32_before_arithmetic_not_just_before_square(self):
        # In FP16, both 2*old and 2*policy overflow before cancellation.
        pw = torch.full((2, 2), 40000.0, dtype=torch.float16, requires_grad=True)
        pl = pw.detach().clone().requires_grad_()
        ow = pw.detach().clone()
        ol = pw.detach().clone()
        tw = torch.full((2, 2), 39990.0, dtype=torch.float32)
        tl = torch.full((2, 2), 39980.0, dtype=torch.float32)
        self.assertFalse(torch.isfinite(3 * ol - 2 * pl).all())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, metrics = flow_cpo_loss(pw, pl, ow, ol, tw, tl, beta=2, loss_lambda=0.5)
        torch.testing.assert_close(loss, torch.tensor(300.0), rtol=0, atol=0)
        self.assertEqual(metrics["flow_cpo/chosen_mse"], 100.0)
        self.assertEqual(metrics["flow_cpo/rejected_mse"], 400.0)
        loss.backward()
        torch.testing.assert_close(pw.grad, torch.full_like(pw, 10.0), rtol=0, atol=0)
        torch.testing.assert_close(pl.grad, torch.full_like(pl, -10.0), rtol=0, atol=0)

    def test_scalar_hyperparameters_reject_malformed_bool_and_nonfinite(self):
        values = [torch.ones(2, 2)] * 6
        invalid = (
            True,
            False,
            None,
            "1",
            "bad",
            [],
            {},
            complex(1, 0),
            torch.tensor(1.0),
            float("nan"),
            float("inf"),
            -float("inf"),
        )
        for name, bad_values in (("beta", (*invalid, 0, -1)), ("loss_lambda", (*invalid, -1))):
            for bad in bad_values:
                kwargs = {"beta": 1, "loss_lambda": 1, name: bad}
                with self.subTest(name=name, bad=repr(bad)), self.assertRaises(ValueError):
                    flow_cpo_loss(*values, **kwargs)

    def test_tensors_require_identical_nonempty_native_velocity_shapes(self):
        for shape in ((), (2,), (0, 2), (2, 0), (1, 0, 2)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                flow_cpo_loss(*[torch.ones(shape)] * 6, beta=1, loss_lambda=1)
        for index in range(6):
            for invalid in (
                torch.ones(1, 2),
                torch.ones(2, 2, dtype=torch.int64),
                torch.ones(2, 2, dtype=torch.bool),
                torch.ones(2, 2, dtype=torch.complex64),
                [[1.0, 1], [1, 1]],
            ):
                values = [torch.ones(2, 2)] * 6
                values[index] = invalid
                with self.subTest(index=index, invalid=repr(invalid)), self.assertRaises(ValueError):
                    flow_cpo_loss(*values, beta=1, loss_lambda=1)

    def test_every_input_branch_is_finite_even_in_ablation(self):
        for index in range(6):
            for bad in (float("nan"), float("inf"), -float("inf")):
                values = [torch.ones(2, 2)] * 6
                values[index] = torch.full((2, 2), bad)
                with self.subTest(index=index, bad=bad), self.assertRaises(FloatingPointError):
                    flow_cpo_loss(*values, beta=1, loss_lambda=0)

    def test_fp32_output_overflow_is_reported(self):
        values = [torch.ones(2, 2)] * 6
        values[0] = torch.full((2, 2), 1e30)
        with self.assertRaises(FloatingPointError):
            flow_cpo_loss(*values, beta=1, loss_lambda=1)


class AdapterEMAHelperTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.model = TinyDiT().requires_grad_(False)
        self.policy = attach(self.model)
        self.ema = attach(self.model, multiplier=0)
        with torch.no_grad():
            for index, parameter in enumerate(self.policy.parameters()):
                parameter.fill_(0.1234567 + index)
        self.temp = tempfile.TemporaryDirectory(prefix=".flow-cpo-test-", dir=DEV_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def assert_unchanged(self, network, before):
        self.assertEqual(set(network.state_dict()), set(before))
        for key, value in before.items():
            torch.testing.assert_close(network.state_dict()[key], value, rtol=0, atol=0, equal_nan=True)

    def test_constructor_exact_fp32_copy_freeze_eval_disable_and_independence(self):
        policy_before, base_before = snapshot(self.policy), snapshot(self.model)
        self.ema.set_multiplier(5)
        ema = AdapterEMA(self.policy, self.ema, decay=0.75)
        self.assertEqual(ema.updates, 0)
        self.assertIs(type(ema.updates), int)
        self.assertIs(type(ema.decay), float)
        self.assertEqual(ema.decay, 0.75)
        self.assertTrue(all(not module.training for module in self.ema.modules()))
        self.assertTrue(all(not p.requires_grad and p.grad is None and p.dtype == torch.float32 for p in self.ema.parameters()))
        self.assertTrue(all(module.multiplier == 0 for module in self.ema.modules() if hasattr(module, "multiplier")))
        self.assert_unchanged(self.policy, policy_before)
        self.assert_unchanged(self.ema, policy_before)
        self.assert_unchanged(self.model, base_before)
        for key, value in self.policy.named_parameters():
            self.assertNotEqual(value.data_ptr(), dict(self.ema.named_parameters())[key].data_ptr())

    def test_update_fp32_recurrence_only_matrices_never_alpha(self):
        ema = AdapterEMA(self.policy, self.ema, decay=0.75)
        expected = {key: value.detach().clone() for key, value in self.policy.named_parameters()}
        alpha_before = {key: value.clone() for key, value in self.ema.named_buffers()}
        with torch.no_grad():
            for parameter in self.policy.parameters():
                parameter.add_(0.2)
        policy_before = snapshot(self.policy)
        for step in range(3):
            expected = {
                key: value * 0.75 + dict(self.policy.named_parameters())[key].detach() * 0.25 for key, value in expected.items()
            }
            with torch.autocast("cpu", dtype=torch.bfloat16):
                ema.update(self.policy)
            self.assertEqual(ema.updates, step + 1)
            for key, value in self.ema.named_parameters():
                self.assertEqual(value.dtype, torch.float32)
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                self.assertIsNone(value.grad)
            for key, value in self.ema.named_buffers():
                torch.testing.assert_close(value, alpha_before[key], rtol=0, atol=0)
        self.assert_unchanged(self.policy, policy_before)

    def test_zero_decay_copies_policy_after_update(self):
        ema = AdapterEMA(self.policy, self.ema, decay=0)
        with torch.no_grad():
            for parameter in self.policy.parameters():
                parameter.fill_(5)
        ema.update(self.policy)
        self.assert_unchanged(self.ema, snapshot(self.policy))
        self.assertEqual(ema.updates, 1)

    def test_constructor_rejects_invalid_decay_before_mutation(self):
        before, flags = snapshot(self.ema), runtime_state(self.ema)
        for decay in (True, False, None, "0.9", "bad", [], torch.tensor(0.9), -0.1, 1, 1.1, float("nan"), float("inf")):
            with self.subTest(decay=repr(decay)), self.assertRaises(ValueError):
                AdapterEMA(self.policy, self.ema, decay)
            self.assert_unchanged(self.ema, before)
            self.assertEqual(runtime_state(self.ema), flags)

    def test_constructor_requires_fp32_policy_and_ema(self):
        for which in ("policy", "ema"):
            network = getattr(self, which)
            for dtype in (torch.float16, torch.bfloat16, torch.float64):
                network.to(dtype=dtype)
                before = snapshot(self.ema)
                with self.subTest(which=which, dtype=dtype), self.assertRaises(ValueError):
                    AdapterEMA(self.policy, self.ema, 0.5)
                self.assert_unchanged(self.ema, before)
            network.float()

    def test_constructor_rejects_key_shape_alpha_and_nonfinite_mismatches(self):
        candidates = []
        foreign_model = TinyDiT()
        foreign_model.second = torch.nn.Linear(2, 2)
        candidates.append(attach(foreign_model))
        candidates.append(attach(TinyDiT(), rank=1))
        candidates.append(attach(TinyDiT(), alpha=3))
        nonfinite = attach(TinyDiT())
        with torch.no_grad():
            list(nonfinite.parameters())[-1].fill_(float("nan"))
        candidates.append(nonfinite)
        for candidate in candidates:
            before, flags = snapshot(self.ema), runtime_state(self.ema)
            with self.subTest(candidate=list(candidate.state_dict())), self.assertRaises((ValueError, FloatingPointError)):
                AdapterEMA(candidate, self.ema, 0.5)
            self.assert_unchanged(self.ema, before)
            self.assertEqual(runtime_state(self.ema), flags)

    def test_constructor_rejects_shared_adapter_or_parameter_storage(self):
        with self.assertRaises(ValueError):
            AdapterEMA(self.policy, self.policy, 0.5)
        self.ema.unet_loras[0].lora_down.weight = self.policy.unet_loras[0].lora_down.weight
        with self.assertRaises(ValueError):
            AdapterEMA(self.policy, self.ema, 0.5)

    def test_invalid_update_is_preflighted_without_partial_mutation_or_count(self):
        ema = AdapterEMA(self.policy, self.ema, 0.5)
        before = snapshot(self.ema)
        original = snapshot(self.policy)
        for invalid in ("nonfinite", "alpha", "shape", "dtype", "keys"):
            self.policy.load_state_dict(original)
            with torch.no_grad():
                for parameter in self.policy.parameters():
                    parameter.add_(1)
            last = self.policy.unet_loras[-1]
            with ExitStack() as stack:
                if invalid == "nonfinite":
                    with torch.no_grad():
                        last.lora_up.weight.fill_(float("inf"))
                elif invalid == "alpha":
                    last.alpha.add_(1)
                elif invalid == "shape":
                    stack.enter_context(patch.object(last.lora_up, "weight", torch.nn.Parameter(torch.ones(3, 2))))
                elif invalid == "dtype":
                    stack.enter_context(patch.object(last.lora_up, "weight", torch.nn.Parameter(last.lora_up.weight.half())))
                else:
                    self.policy.register_parameter("foreign", torch.nn.Parameter(torch.ones(2, 2)))
                    stack.callback(lambda: self.policy._parameters.pop("foreign"))
                with self.subTest(invalid=invalid), self.assertRaises((ValueError, FloatingPointError)):
                    ema.update(self.policy)
            self.assert_unchanged(self.ema, before)
            self.assertEqual(ema.updates, 0)

    def test_save_validate_load_roundtrip_single_file_metadata_and_resume_recurrence(self):
        ema = AdapterEMA(self.policy, self.ema, 0.75)
        with torch.no_grad():
            for parameter in self.policy.parameters():
                parameter.add_(1)
        ema.update(self.policy)
        before = snapshot(self.ema)
        ema.save(self.directory)
        self.assertEqual(FLOW_CPO_EMA_STATE, "krea2_flow_cpo_ema.safetensors")
        self.assertEqual([path.name for path in self.directory.iterdir()], [FLOW_CPO_EMA_STATE])
        path = self.directory / FLOW_CPO_EMA_STATE
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            self.assertEqual(handle.metadata(), {"version": "1", "decay": "0.75", "updates": "1"})
            self.assertEqual(set(handle.keys()), set(dict(self.ema.named_parameters())))
            self.assertTrue(all(handle.get_tensor(key).dtype == torch.float32 for key in handle.keys()))
        fresh_model = TinyDiT()
        fresh_policy, fresh_old = attach(fresh_model), attach(fresh_model, multiplier=0)
        restored = AdapterEMA(fresh_policy, fresh_old, 0.75)
        fresh_before = snapshot(fresh_old)
        payload = restored.validate(self.directory)
        self.assertEqual(set(payload), {"version", "decay", "updates", "parameters"})
        self.assertEqual(payload["updates"], 1)
        self.assertEqual(payload["decay"], 0.75)
        self.assertEqual(payload["version"], "1")
        self.assertEqual(set(payload["parameters"]), set(dict(fresh_old.named_parameters())))
        self.assert_unchanged(fresh_old, fresh_before)
        self.assertEqual(restored.updates, 0)
        with patch("krea2_trainer.training.flow_cpo.safe_open", wraps=safe_open) as reader:
            restored.load(self.directory)
        self.assertEqual(reader.call_count, 1)
        self.assert_unchanged(fresh_old, before)
        self.assertEqual(restored.updates, 1)
        fresh_policy.load_state_dict(self.policy.state_dict())
        ema.update(self.policy)
        restored.update(fresh_policy)
        self.assert_unchanged(fresh_old, snapshot(self.ema))
        self.assertEqual(restored.updates, ema.updates)

    def test_checkpoint_metadata_rejects_malformed_state_before_load_mutation(self):
        ema = AdapterEMA(self.policy, self.ema, 0.75)
        ema.save(self.directory)
        path = self.directory / FLOW_CPO_EMA_STATE
        tensors, before = load_file(str(path)), snapshot(self.ema)
        valid = {"version": "1", "decay": "0.75", "updates": "2"}
        cases = [{}, {**valid, "extra": "x"}, {key: value for key, value in valid.items() if key != "updates"}]
        cases.extend({**valid, "version": value} for value in ("0", "2", "01", "1.0"))
        cases.extend({**valid, "decay": value} for value in ("0.5", "nan", "inf", "-inf", "true", "", "bad"))
        cases.extend({**valid, "updates": value} for value in ("-1", "+1", "01", "1.0", "1e2", " 1", "1 ", "true", "nan", "", "١"))
        for metadata in cases:
            save_file(tensors, str(path), metadata=metadata)
            for method in (ema.validate, ema.load):
                with self.subTest(metadata=metadata, method=method.__name__), self.assertRaises(ValueError):
                    method(self.directory)
                self.assert_unchanged(self.ema, before)
                self.assertEqual(ema.updates, 0)

    def test_checkpoint_tensor_schema_preflight_blocks_partial_mutation(self):
        ema = AdapterEMA(self.policy, self.ema, 0.75)
        ema.save(self.directory)
        path = self.directory / FLOW_CPO_EMA_STATE
        tensors, before = load_file(str(path)), snapshot(self.ema)
        keys = sorted(tensors)
        changed = {key: value + 1 for key, value in tensors.items()}
        cases = {
            "missing": {key: value for key, value in changed.items() if key != keys[-1]},
            "extra": {**changed, "foreign.lora_up.weight": torch.ones(2, 2)},
            "alpha": {**changed, "lora_unet_first.alpha": torch.tensor(2.0)},
            "shape": {**changed, keys[-1]: torch.ones(3, 2)},
            "fp16": {**changed, keys[-1]: changed[keys[-1]].half()},
            "fp64": {**changed, keys[-1]: changed[keys[-1]].double()},
            "integer": {**changed, keys[-1]: changed[keys[-1]].long()},
            "nan": {**changed, keys[-1]: torch.full_like(changed[keys[-1]], float("nan"))},
            "inf": {**changed, keys[-1]: torch.full_like(changed[keys[-1]], float("inf"))},
        }
        for name, bad in cases.items():
            save_file(bad, str(path), metadata={"version": "1", "decay": "0.75", "updates": "5"})
            with self.subTest(name=name), self.assertRaises((ValueError, FloatingPointError)):
                ema.load(self.directory)
            self.assert_unchanged(self.ema, before)
            self.assertEqual(ema.updates, 0)

    def test_missing_or_truncated_checkpoint_is_not_silently_reset(self):
        ema = AdapterEMA(self.policy, self.ema, 0.75)
        before = snapshot(self.ema)
        with self.assertRaises((FileNotFoundError, ValueError)):
            ema.load(self.directory)
        (self.directory / FLOW_CPO_EMA_STATE).write_bytes(b"not a safetensors file")
        with self.assertRaises(Exception):
            ema.load(self.directory)
        self.assert_unchanged(self.ema, before)
        self.assertEqual(ema.updates, 0)


class OldAdapterContextTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = TinyDiT().requires_grad_(False)
        self.stage = attach(self.model).requires_grad_(False).eval()
        self.policy = attach(self.model)
        self.ema = attach(self.model, multiplier=0)
        self.helper = AdapterEMA(self.policy, self.ema, 0.5)
        with torch.no_grad():
            for network, down, up in ((self.stage, 0.2, 0.3), (self.policy, 0.4, 0.5), (self.ema, 0.6, 0.7)):
                network.unet_loras[0].lora_down.weight.fill_(down)
                network.unet_loras[0].lora_up.weight.fill_(up)
        self.probe = torch.tensor([[[1.0, -0.5], [2, 3]]])

    def expected(self, selected):
        base = F.linear(self.probe, self.model.first.weight, self.model.first.bias)
        for network in (self.stage, selected):
            module = network.unet_loras[0]
            base = base + F.linear(F.linear(self.probe, module.lora_down.weight), module.lora_up.weight) * module.scale
        return base

    def test_native_base_stage_policy_ema_chain_selects_only_old_and_restores_policy(self):
        state_before = [snapshot(network) for network in (self.model, self.stage, self.policy, self.ema)]
        self.assertIs(self.model.first.forward.__self__, self.ema.unet_loras[0])
        self.assertIs(self.ema.unet_loras[0].org_forward.__self__, self.policy.unet_loras[0])
        self.assertIs(self.policy.unet_loras[0].org_forward.__self__, self.stage.unet_loras[0])
        versions = [
            tensor._version
            for network in (self.model, self.stage, self.policy, self.ema)
            for tensor in (*network.parameters(), *network.buffers())
        ]
        with patch.object(self.policy.unet_loras[0].lora_down, "forward", side_effect=AssertionError("disabled policy ran")):
            with old_adapter_context(self.policy, self.ema, self.model):
                self.assertFalse(torch.is_grad_enabled())
                self.assertTrue(
                    all(not module.training for root in (self.policy, self.ema, self.model) for module in root.modules())
                )
                self.assertEqual(self.policy.multiplier, 0)
                self.assertEqual(self.ema.multiplier, 1)
                output = self.model.first(self.probe)
                torch.testing.assert_close(output, self.expected(self.ema))
                self.assertFalse(output.requires_grad)
        self.assertTrue(torch.is_grad_enabled())
        self.assertEqual(self.ema.multiplier, 0)
        self.assertEqual(self.policy.multiplier, 1)
        output = self.model.first(self.probe)
        torch.testing.assert_close(output, self.expected(self.policy))
        output.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.policy.parameters()))
        self.assertTrue(all(p.grad is None for root in (self.model, self.stage, self.ema) for p in root.parameters()))
        for network, before in zip((self.model, self.stage, self.policy, self.ema), state_before):
            for key, value in network.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertEqual(
            versions,
            [
                tensor._version
                for network in (self.model, self.stage, self.policy, self.ema)
                for tensor in (*network.parameters(), *network.buffers())
            ],
        )

    def test_exception_restores_heterogeneous_flags_and_all_multipliers(self):
        self.model.train()
        self.model.first.eval()
        self.policy.train()
        self.policy.unet_loras[0].lora_down.eval()
        self.policy.multiplier = 0.75
        self.policy.unet_loras[0].multiplier = 1.25
        self.ema.unet_loras[0].lora_up.train()
        before = runtime_state(self.policy, self.ema, self.model, self.stage)
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with old_adapter_context(self.policy, self.ema, self.model):
                self.assertEqual(self.stage.multiplier, 1)
                raise RuntimeError("deliberate")
        self.assertEqual(runtime_state(self.policy, self.ema, self.model, self.stage), before)
        self.assertTrue(torch.is_grad_enabled())

    def test_nested_context_restores_outer_state_and_existing_no_grad(self):
        before = runtime_state(self.policy, self.ema, self.model)
        with torch.no_grad():
            with old_adapter_context(self.policy, self.ema, self.model):
                outer = runtime_state(self.policy, self.ema, self.model)
                with old_adapter_context(self.policy, self.ema, self.model):
                    torch.testing.assert_close(self.model.first(self.probe), self.expected(self.ema))
                self.assertEqual(runtime_state(self.policy, self.ema, self.model), outer)
            self.assertFalse(torch.is_grad_enabled())
        self.assertEqual(runtime_state(self.policy, self.ema, self.model), before)

    def test_setup_exception_restores_flags_and_multipliers(self):
        before = runtime_state(self.policy, self.ema, self.model)
        with patch.object(self.ema, "eval", side_effect=RuntimeError("eval failed")):
            with self.assertRaisesRegex(RuntimeError, "eval failed"):
                with old_adapter_context(self.policy, self.ema, self.model):
                    self.fail("context must not yield")
        self.assertEqual(runtime_state(self.policy, self.ema, self.model), before)

    def test_same_network_rejected_without_changing_policy(self):
        before = runtime_state(self.policy, self.model)
        with self.assertRaises(ValueError):
            with old_adapter_context(self.policy, self.policy, self.model):
                self.fail("context must not yield")
        self.assertEqual(runtime_state(self.policy, self.model), before)


class FlowCPOParserTests(unittest.TestCase):
    def test_defaults_and_dispatch(self):
        from tests.test_post_training import arguments
        from krea2_trainer.krea2_train_network import main
        from krea2_trainer.training.preference import validate_post_training_args
        import sys

        args = arguments("flow_cpo")
        self.assertEqual((args.flow_cpo_beta, args.flow_cpo_lambda, args.flow_cpo_ema_decay), (0.5, 1.0, 0.99))
        validate_post_training_args(args)
        with (
            patch.object(sys, "argv", ["train"]),
            patch("krea2_trainer.krea2_train_network.read_config_from_file", return_value=args),
            patch("krea2_trainer.krea2_flow_cpo.Krea2FlowCPOTrainer") as trainer,
        ):
            main()
            trainer.return_value.train.assert_called_once_with(args)

    def test_wrong_mode_explicit_defaults_and_invalid_values(self):
        from tests.test_post_training import arguments, parser
        from krea2_trainer.training.preference import validate_post_training_args
        from contextlib import redirect_stderr
        import io

        for flag, default in (("flow_cpo_beta", "0.5"), ("flow_cpo_lambda", "1"), ("flow_cpo_ema_decay", "0.99")):
            for mode in ("none", "rft", "flow_dpo"):
                args = parser().parse_args(["--" + flag, default]) if mode == "none" else arguments(mode, "--" + flag, default)
                with self.subTest(flag=flag, mode=mode), self.assertRaisesRegex(ValueError, "[Pp]ost.training|only meaningful"):
                    validate_post_training_args(args)
            for value in ("nan", "inf", "-inf", "-0.1") + (
                ("0",) if flag.endswith("beta") else ("1",) if flag.endswith("decay") else ()
            ):
                with self.subTest(flag=flag, value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    arguments("flow_cpo", "--" + flag, value)
        with self.assertRaisesRegex(ValueError, "flow_dpo_beta"):
            validate_post_training_args(arguments("flow_cpo", "--flow_dpo_beta", "100"))
        for flag in ("flow_cpo_beta", "flow_cpo_lambda", "flow_cpo_ema_decay"):
            for bad in (True, "0.5", None, float("nan"), float("inf")):
                args = arguments("flow_cpo")
                setattr(args, flag, bad)
                with self.subTest(flag=flag, bad=bad), self.assertRaises(ValueError):
                    validate_post_training_args(args)

    def test_real_config_loader_explicit_defaults_and_cli_override(self):
        from tests.test_post_training import parser
        from krea2_trainer.training.parser_common import read_config_from_file
        from krea2_trainer.training.preference import validate_post_training_args
        import sys

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            for flag, value in (("flow_cpo_beta", "0.5"), ("flow_cpo_lambda", "1.0"), ("flow_cpo_ema_decay", "0.99")):
                config.write_text(f"[training]\n{flag}={value}\n")
                with patch.object(sys, "argv", ["train", "--config_file", str(config)]):
                    p = parser()
                    args = read_config_from_file(p.parse_args(), p)
                with self.assertRaises(ValueError):
                    validate_post_training_args(args)
            config.write_text(
                '[training]\npost_training="flow_cpo"\npreference_manifest="pairs"\n'
                'network_module="krea2_trainer.networks.lora_krea2"\nmixed_precision="bf16"\n'
                'timestep_sampling="uniform"\nflow_cpo_beta=0.7\nflow_cpo_lambda=0.0\nflow_cpo_ema_decay=0.8\n'
            )
            with patch.object(sys, "argv", ["train", "--config_file", str(config), "--flow_cpo_beta", "0.4"]):
                p = parser()
                args = read_config_from_file(p.parse_args(), p)
            validate_post_training_args(args)
            self.assertEqual((args.flow_cpo_beta, args.flow_cpo_lambda, args.flow_cpo_ema_decay), (0.4, 0.0, 0.8))

    def test_incompatible_modes_rejected_and_checkpointing_retained(self):
        from tests.test_post_training import arguments
        from krea2_trainer.training.preference import validate_post_training_args

        for options in (
            ("--compile",),
            ("--blocks_to_swap", "1"),
            ("--gradient_checkpointing_cpu_offload",),
            ("--dynamo_backend", "EAGER"),
            ("--weighting_scheme", "sigma_sqrt"),
            ("--timestep_sampling", "sigma"),
            ("--min_timestep", "0"),
            ("--max_timestep", "1000"),
            ("--network_dropout", "0.1"),
            ("--network_args", "rank_dropout=0.1"),
            ("--optimizer_type", "schedulefree.AdamWScheduleFree"),
            ("--optimizer_type", "LOMO"),
            ("--optimizer_type", "custom.Optimizer"),
            ("--scale_weight_norms", "1"),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_post_training_args(arguments("flow_cpo", *options))
        with patch.dict("os.environ", {"ACCELERATE_DYNAMO_BACKEND": "INDUCTOR"}), self.assertRaises(ValueError):
            validate_post_training_args(arguments("flow_cpo"))
        for key in ("full_bf16", "full_fp16"):
            args = arguments("flow_cpo")
            setattr(args, key, True)
            with self.assertRaises(ValueError):
                validate_post_training_args(args)
        args = arguments("flow_cpo", "--gradient_checkpointing", "--flow_cpo_beta", "1", "--flow_cpo_lambda", "0")
        validate_post_training_args(args)
        self.assertTrue(args.gradient_checkpointing)


class FlowCPODataTests(unittest.TestCase):
    def fixture(self):
        from tests.test_preference_dataset import PreferenceDatasetTests

        fixture = PreferenceDatasetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_native_paired_dense_and_varlen_cache_contract(self):
        for varlen in (False, True):
            fixture = self.fixture()
            fixture.item("winner", 0.2, varlen=varlen)
            fixture.item("loser", -0.2, varlen=varlen)
            group = fixture.build(mode="flow_cpo")
            self.assertEqual(group.num_train_items, 1)
            batch = group[0]
            torch.testing.assert_close(batch["latents"], -batch["rejected_latents"])
            self.assertIsInstance(batch["krea2_vl_embed"], list if varlen else torch.Tensor)
            self.assertEqual(group.datasets[0].mode, "flow_cpo")

    def test_cpo_reuses_strict_caption_shape_and_conditioning_guards(self):
        for case, pattern in (
            ("caption", "caption1"),
            ("shape", "shape|bucket"),
            ("conditioning", "conditioning"),
            ("missing", "[Mm]issing"),
        ):
            fixture = self.fixture()
            fixture.item("winner", 0.2)
            loser = fixture.item("loser", -0.2, shape=(2, 1, 4, 8) if case == "shape" else (2, 1, 4, 4))
            if case in ("caption", "conditioning"):
                save_file(
                    {"varlen_krea2_vl_embed_float32": torch.ones(3, 2, 4)},
                    loser.text_encoder_output_cache_path,
                    metadata={"caption1": "wrong" if case == "caption" else "a portrait"},
                )
            if case == "missing":
                Path(loser.latent_cache_path).unlink()
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, pattern):
                fixture.build(mode="flow_cpo")


class FlowCPOTrainerTests(unittest.TestCase):
    def build(self, checkpointing=False):
        from tests.test_post_training import arguments, toy_accelerator
        from krea2_trainer.krea2_flow_cpo import Krea2FlowCPOTrainer

        torch.manual_seed(313)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        model = TinyDiT()
        bare = snapshot(model)
        stage = attach(model)
        with torch.no_grad():
            stage.unet_loras[0].lora_up.weight.fill_(0.13)
        path = Path(directory.name) / "stage1.safetensors"
        save_file(stage.state_dict(), str(path))
        model = TinyDiT()
        model.load_state_dict(bare)
        model.requires_grad_(False)
        model.train(checkpointing)
        args = arguments(
            "flow_cpo",
            "--reference_lora",
            str(path),
            "--flow_cpo_ema_decay",
            "0.5",
            *(["--gradient_checkpointing"] if checkpointing else []),
        )
        trainer = Krea2FlowCPOTrainer()
        network = trainer._build_network(args, toy_accelerator(), model, None, torch.float32)
        return args, model, trainer, network

    @staticmethod
    def run_batch(args, model, trainer, network, batch=None):
        from tests.test_post_training import toy_accelerator, toy_batch

        batch = toy_batch() if batch is None else batch
        return trainer.process_batch(
            args,
            toy_accelerator(),
            model,
            network,
            batch,
            batch["latents"],
            torch.full_like(batch["latents"], 0.3),
            None,
            torch.float32,
            torch.float32,
            None,
            0,
        )

    def test_initial_copy_native_chain_shared_inputs_and_post_step_ema(self):
        from tests.test_post_training import toy_accelerator, toy_batch
        from krea2_trainer.training.preference import reference_adapter_context

        args, model, trainer, network = self.build(True)
        base_before, stage_before = snapshot(model), snapshot(trainer.reference_network)
        ema_initial = snapshot(trainer.ema_network)
        for key, value in network.state_dict().items():
            torch.testing.assert_close(ema_initial[key], value, rtol=0, atol=0)
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
        accelerator = toy_accelerator()
        accelerator.sync_gradients = True
        accelerator.optimizer_step_was_skipped = False
        trainer.on_train_start(args, accelerator, network, model, optimizer)
        probe = torch.randn(1, 3, 2)
        with reference_adapter_context(network, model):
            fixed = model.first(probe).clone()
        with old_adapter_context(network, trainer.ema_network, model):
            torch.testing.assert_close(model.first(probe), fixed, rtol=0, atol=0)
        with patch("torch.rand", return_value=torch.tensor([0.2, 0.8])):
            loss, metrics = self.run_batch(args, model, trainer, network)
        old_call, policy_call = model.calls
        self.assertFalse(old_call["grad_enabled"])
        self.assertFalse(old_call["training"])
        self.assertTrue(policy_call["grad_enabled"])
        self.assertTrue(policy_call["training"])
        for key in ("img", "context", "t"):
            torch.testing.assert_close(old_call[key], policy_call[key], rtol=0, atol=0)
        torch.testing.assert_close(old_call["t"], torch.tensor([0.201, 0.801, 0.201, 0.801]))
        batch = toy_batch()
        t = torch.tensor([0.2, 0.8]).view(-1, 1, 1, 1, 1)
        expected = torch.cat(((1 - t) * batch["latents"] + t * 0.3, (1 - t) * batch["rejected_latents"] + t * 0.3))
        torch.testing.assert_close(old_call["img"], expected.squeeze(2).flatten(2).transpose(1, 2))
        self.assertIn("flow_cpo/loss", metrics)
        loss.backward()
        optimizer.step()
        trainer.on_post_optimizer_step(args, accelerator, network, model, True, 0)
        self.assertEqual(trainer.ema_updates, 1)
        for key, parameter in network.named_parameters():
            torch.testing.assert_close(trainer.ema_network.state_dict()[key], 0.5 * ema_initial[key] + 0.5 * parameter.detach())
        with old_adapter_context(network, trainer.ema_network, model):
            self.assertFalse(torch.equal(model.first(probe), fixed))
        for obj, before in ((model, base_before), (trainer.reference_network, stage_before)):
            for key, value in before.items():
                torch.testing.assert_close(obj.state_dict()[key], value, rtol=0, atol=0)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in obj.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in trainer.ema_network.parameters()))

    def test_checkpointed_nonzero_policy_and_ema_equivalence_dense_varlen(self):
        from tests.test_post_training import toy_batch

        for dense in (False, True):
            batch = toy_batch()
            if dense:
                batch["krea2_vl_embed"] = torch.ones(2, 2, 1, 2)
            results = []
            for checkpointing in (False, True):
                args, model, trainer, network = self.build(checkpointing)
                with torch.no_grad():
                    for module in network.unet_loras:
                        module.lora_up.weight.fill_(0.07)
                    for module in trainer.ema_network.unet_loras:
                        module.lora_up.weight.fill_(0.04)
                torch.manual_seed(93)
                loss, _ = self.run_batch(args, model, trainer, network, batch)
                flags = runtime_state(network, trainer.ema_network, model)
                loss.backward()
                self.assertEqual(flags, runtime_state(network, trainer.ema_network, model))
                results.append((loss.detach(), [p.grad.detach().clone() for p in network.parameters()]))
            torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
            for expected, actual in zip(results[0][1], results[1][1]):
                torch.testing.assert_close(expected, actual, rtol=1e-6, atol=1e-7)

    def test_old_forward_exception_restores_flags_without_policy_graph(self):
        args, model, trainer, network = self.build(True)
        before = runtime_state(network, trainer.ema_network, model)
        with patch.object(trainer, "call_dit", side_effect=RuntimeError("old failed")) as call:
            with self.assertRaisesRegex(RuntimeError, "old failed"):
                self.run_batch(args, model, trainer, network)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(before, runtime_state(network, trainer.ema_network, model))

    def test_skip_and_accumulation_never_advance_ema(self):
        from tests.test_post_training import toy_accelerator

        args, model, trainer, network = self.build()
        before = snapshot(trainer.ema_network)
        accelerator = toy_accelerator()
        for sync_argument, actual_sync, skipped in ((False, False, False), (True, False, False), (True, True, True)):
            accelerator.sync_gradients = actual_sync
            accelerator.optimizer_step_was_skipped = skipped
            trainer.on_post_optimizer_step(args, accelerator, network, model, sync_argument, 0)
        self.assertEqual(trainer.ema_updates, 0)
        for key, value in before.items():
            torch.testing.assert_close(trainer.ema_network.state_dict()[key], value, rtol=0, atol=0)

    def test_real_accelerate_accumulation_and_overflow_signal_lifecycle(self):
        from accelerate import Accelerator

        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
        self.addCleanup(accelerator.free_memory)
        args, model, trainer, network = self.build()
        network, optimizer = accelerator.prepare(network, torch.optim.AdamW(network.parameters(), lr=0.001))
        counts = []
        for _ in range(4):
            with accelerator.accumulate(network):
                loss, _ = self.run_batch(args, model, trainer, accelerator.unwrap_model(network))
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                trainer.on_post_optimizer_step(args, accelerator, network, model, accelerator.sync_gradients, 0)
                counts.append(trainer.ema_updates)
        self.assertEqual(counts, [0, 1, 1, 2])
        optimizer._is_overflow = True
        self.assertTrue(accelerator.optimizer_step_was_skipped)
        trainer.on_post_optimizer_step(args, accelerator, network, model, True, 0)
        self.assertEqual(trainer.ema_updates, 2)

    def test_runtime_rejects_optimizer_owned_frozen_adapters_and_mode_swappers(self):
        from tests.test_post_training import toy_accelerator

        args, model, trainer, network = self.build()
        for frozen in (trainer.ema_network, trainer.reference_network, model):
            optimizer = torch.optim.AdamW([*network.parameters(), *frozen.parameters()], lr=0.01)
            with self.subTest(frozen=type(frozen).__name__), self.assertRaises(ValueError):
                trainer.on_train_start(args, toy_accelerator(), network, model, optimizer)
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
        optimizer.train = lambda: None
        optimizer.eval = lambda: None
        with self.assertRaisesRegex(ValueError, "optimizer|Optimizer"):
            trainer.on_train_start(args, toy_accelerator(), network, model, optimizer)

    def test_resume_pre_hook_guard_and_ema_not_reset_on_train_start(self):
        from tests.test_post_training import toy_accelerator
        from krea2_trainer.krea2_train_network import Krea2NetworkTrainer
        from krea2_trainer.training.preference import POST_TRAINING_STATE
        import json

        args, model, trainer, network = self.build()
        trainer.reference_contract = {"post_training": "flow_cpo"}
        trainer.ema.update(network)
        accelerator = toy_accelerator()
        accelerator.is_main_process = True
        saves, loads = [], []
        accelerator.register_save_state_pre_hook = saves.append
        accelerator.register_load_state_pre_hook = loads.append
        with patch.object(Krea2NetworkTrainer, "_register_hooks_and_resume") as parent:
            trainer._register_hooks_and_resume(args, accelerator, network)
        parent.assert_called_once()
        with tempfile.TemporaryDirectory() as directory:
            for save in saves:
                save([], [], directory)
            self.assertTrue((Path(directory) / FLOW_CPO_EMA_STATE).exists())
            self.assertEqual(json.loads((Path(directory) / POST_TRAINING_STATE).read_text()), trainer.reference_contract)
            trainer.ema.update(network)
            for load in loads:
                load([], directory)
            self.assertEqual(trainer.ema_updates, 1)
            trainer.on_train_start(args, accelerator, network, model, torch.optim.AdamW(network.parameters(), lr=0.01))
            self.assertEqual(trainer.ema_updates, 1)
            (Path(directory) / FLOW_CPO_EMA_STATE).unlink()
            with self.assertRaises((ValueError, OSError)):
                loads[0]([], directory)

    def test_prepare_recopies_broadcast_policy_before_resume_only(self):
        from tests.test_post_training import toy_accelerator
        from krea2_trainer.krea2_train_network import Krea2NetworkTrainer

        args, model, trainer, network = self.build()
        with torch.no_grad():
            network.unet_loras[0].lora_down.weight.add_(0.25)
        prepared = (model, network, object(), object(), object(), network, torch.float32)
        with patch.object(Krea2NetworkTrainer, "_prepare_with_accelerator", return_value=prepared):
            result = trainer._prepare_with_accelerator(
                args, toy_accelerator(), model, network, None, None, None, torch.float32, torch.float32, torch.float32
            )
        self.assertIs(result, prepared)
        self.assertEqual(trainer.ema_updates, 0)
        for key, parameter in network.named_parameters():
            torch.testing.assert_close(trainer.ema_network.state_dict()[key], parameter, rtol=0, atol=0)

    def test_runtime_batch_guards_and_fixed_multiplier(self):
        from tests.test_post_training import toy_batch

        args, model, trainer, network = self.build()
        bad_batches = [
            dict(toy_batch(), timesteps=torch.ones(2)),
            dict(toy_batch(), tqd_quality_weight=torch.ones(2)),
            dict(toy_batch(), rejected_latents=torch.ones(1, 2, 1, 2, 2)),
            dict(toy_batch(), krea2_vl_embed=torch.ones(2, 2)),
            dict(toy_batch(), krea2_vl_embed=[torch.ones(1, 1, 2)]),
            dict(toy_batch(), krea2_vl_embed=[torch.empty(0, 1, 2)] * 2),
        ]
        for batch in bad_batches:
            with self.assertRaises(ValueError):
                self.run_batch(args, model, trainer, network, batch)
        for key in ("latents", "rejected_latents", "krea2_vl_embed"):
            for bad in (float("nan"), float("inf")):
                batch = toy_batch()
                value = batch[key][0] if isinstance(batch[key], list) else batch[key]
                value.flatten()[0] = bad
                with self.subTest(key=key, bad=bad), self.assertRaises(FloatingPointError):
                    self.run_batch(args, model, trainer, network, batch)
        for candidate, invalid, original in ((network, 0.5, 1.0), (trainer.ema_network, 1.0, 0.0)):
            candidate.unet_loras[0].multiplier = invalid
            with self.assertRaises(ValueError):
                self.run_batch(args, model, trainer, network)
            candidate.unet_loras[0].multiplier = original
        self.assertFalse(model.calls)

    def test_default_optimizer_and_native_policy_only_export(self):
        args, model, trainer, network = self.build()
        result = trainer.get_optimizer(args, list(network.parameters()))
        self.assertIsInstance(result[2], torch.optim.AdamW)
        self.assertEqual(args.optimizer_type, "")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.safetensors"
            network.save_weights(str(path), torch.float32, {"ss_post_training": "flow_cpo"})
            state = load_file(str(path))
        self.assertEqual(set(state), set(network.state_dict()))
        self.assertFalse(any("ema" in key or "reference" in key for key in state))

    def test_cpo_contract_hyperparameters_and_dpo_contract_unchanged(self):
        from tests.test_post_training import arguments
        from krea2_trainer.training.preference import build_reference_contract, validate_resume_contract, POST_TRAINING_STATE
        import json

        args, model, trainer, network = self.build()
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "fixture"
            file.write_text("fixture only")
            args.dit = args.dataset_config = args.preference_manifest = str(file)
            cpo = build_reference_contract(args)
            self.assertEqual(cpo["flow_cpo_beta"], 0.5)
            self.assertEqual(cpo["flow_cpo_lambda"], 1.0)
            self.assertEqual(cpo["flow_cpo_ema_decay"], 0.5)
            self.assertNotIn("flow_dpo_beta", cpo)
            self.assertIn("velocity", cpo["objective"])
            (Path(directory) / POST_TRAINING_STATE).write_text(json.dumps(cpo))
            validate_resume_contract(directory, cpo)
            for key in ("flow_cpo_beta", "flow_cpo_lambda", "flow_cpo_ema_decay", "objective"):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_resume_contract(directory, dict(cpo, **{key: "changed"}))
            dpo_args = arguments("flow_dpo")
            dpo_args.dit = dpo_args.dataset_config = dpo_args.preference_manifest = str(file)
            dpo = build_reference_contract(dpo_args)
            self.assertEqual(dpo["version"], 1)
            self.assertEqual(dpo["objective"], "mean-fp32-velocity-mse-beta-over-2-uniform")
            self.assertFalse(any(key.startswith("flow_cpo") for key in dpo))


if __name__ == "__main__":
    unittest.main()
