"""Disabled-adapter inference must skip projections, not the base chain."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

from krea2_trainer.networks.lora import LoRAModule


class ZeroMultiplierInferenceTests(unittest.TestCase):
    def test_disabled_eval_no_grad_skips_dense_split_and_conv_projections(self):
        cases = [
            (nn.Linear(4, 6), torch.randn(2, 3, 4), None),
            (nn.Linear(4, 6), torch.randn(2, 3, 4), [2, 4]),
            (nn.Conv2d(4, 6, 1), torch.randn(2, 4, 3, 3), None),
            (nn.Conv3d(4, 6, 1), torch.randn(2, 4, 1, 3, 3), None),
        ]
        for base, x, split in cases:
            with self.subTest(base=type(base).__name__, split=split):
                original = base.forward
                adapter = LoRAModule("test", base, multiplier=0.0, split_dims=split)
                adapter.apply_to()
                adapter.eval()
                down = adapter.lora_down if split is None else adapter.lora_down[0]
                up = adapter.lora_up if split is None else adapter.lora_up[0]
                with torch.no_grad(), patch.object(down, "forward", wraps=down.forward) as down_call, patch.object(
                    up, "forward", wraps=up.forward
                ) as up_call:
                    expected = original(x)
                    actual = base(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(down_call.call_count, 0)
                self.assertEqual(up_call.call_count, 0)

    def test_zero_multiplier_grad_enabled_keeps_zero_parameter_gradients(self):
        base = nn.Linear(4, 6)
        adapter = LoRAModule("test", base, multiplier=0.0)
        adapter.apply_to()
        adapter.eval()
        x = torch.randn(2, 4, requires_grad=True)
        base(x).sum().backward()
        for parameter in (adapter.lora_down.weight, adapter.lora_up.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)
        self.assertIsNotNone(x.grad)

    def test_frozen_zero_ema_skips_projections_without_changing_input_or_policy_gradients(self):
        torch.manual_seed(19)
        base = nn.Linear(4, 6).requires_grad_(False)
        policy = LoRAModule("policy", base, lora_dim=2)
        policy.apply_to()
        ema = LoRAModule("ema", base, multiplier=0.0, lora_dim=2)
        ema.apply_to()
        ema.eval().requires_grad_(False)
        with torch.no_grad():
            policy.lora_up.weight.normal_()
            ema.lora_up.weight.normal_()
        x = torch.randn(2, 3, 4, requires_grad=True)
        params = (x, *policy.parameters())
        expected = ema.org_forward(x) + ema.lora_up(ema.lora_down(x)) * 0.0
        expected_grads = torch.autograd.grad(expected.square().mean(), params)
        with patch.object(ema.lora_down, "forward", side_effect=AssertionError("inactive EMA ran")):
            actual = base(x)
            actual_grads = torch.autograd.grad(actual.square().mean(), params)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_training_no_grad_retains_dropout_rng_behavior(self):
        base = nn.Linear(4, 6)
        adapter = LoRAModule("test", base, multiplier=0.0, dropout=0.5)
        adapter.apply_to()
        adapter.train()
        with torch.no_grad(), patch.object(adapter.lora_down, "forward", wraps=adapter.lora_down.forward) as call:
            base(torch.randn(2, 4))
        self.assertEqual(call.call_count, 1)

    def test_active_adapter_no_grad_is_not_skipped(self):
        base = nn.Linear(4, 6)
        adapter = LoRAModule("test", base, multiplier=0.5)
        adapter.apply_to()
        adapter.eval()
        with torch.no_grad():
            adapter.lora_up.weight.fill_(0.1)
            x = torch.randn(2, 4)
            expected = adapter.org_forward(x) + adapter.lora_up(adapter.lora_down(x)) * adapter.multiplier * adapter.scale
            actual = base(x)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
