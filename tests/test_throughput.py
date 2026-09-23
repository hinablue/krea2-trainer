"""Throughput changes: effective caps, diagnostics and numerical contracts."""

from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from krea2_trainer.krea2_post_training import Krea2PostTrainingTrainer
from krea2_trainer.training.flow_cpo import flow_cpo_loss
from krea2_trainer.training.metrics import materialize_metrics
from krea2_trainer.training.preference import flow_dpo_loss
from krea2_trainer.training.profiling import TrainingStepProfiler, create_step_profiler
from tests.test_training_optimizations import ScalarReads, tiny_model


class MetricsTests(unittest.TestCase):
    def test_dpo_optional_metrics_preserve_loss_and_gradients(self):
        values = [torch.tensor([0.3, 0.7], requires_grad=i < 2) for i in range(4)]
        expected, floats = flow_dpo_loss(*values, beta=2)
        gradients = torch.autograd.grad(expected, values[:2])
        for collect in (False, True):
            with ScalarReads() as reads:
                actual, metrics = flow_dpo_loss(*values, beta=2, collect_metrics=collect, metrics_as_tensors=True)
            self.assertEqual(reads.count, 2)  # batched input finite check and logit finite check
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for a, b in zip(torch.autograd.grad(actual, values[:2]), gradients):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(materialize_metrics(metrics), floats if collect else {})
            if collect:
                self.assertTrue(all(not value.requires_grad for value in metrics.values()))
        with self.assertRaises(FloatingPointError):
            flow_dpo_loss(*([torch.tensor([float("nan")])] * 4), beta=1, collect_metrics=False)

    def test_cpo_optional_metrics_preserve_loss_gradients_and_guards(self):
        torch.manual_seed(51)
        values = [torch.randn(2, 3, requires_grad=i < 2) for i in range(6)]
        expected, floats = flow_cpo_loss(*values, 0.5, 1)
        gradients = torch.autograd.grad(expected, values[:2])
        for collect in (False, True):
            actual, metrics = flow_cpo_loss(*values, 0.5, 1, collect_metrics=collect, metrics_as_tensors=True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for a, b in zip(torch.autograd.grad(actual, values[:2]), gradients):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(materialize_metrics(metrics), floats if collect else {})
        values[0] = torch.full((2, 3), float("inf"))
        with self.assertRaises(FloatingPointError):
            flow_cpo_loss(*values, 0.5, 1, collect_metrics=False)


class PairedDPOTests(unittest.TestCase):
    def test_native_checkpointed_dpo_dense_varlen_singleton_and_multi_pair(self):
        for dtype in (torch.float32, torch.bfloat16):
            for lengths, dense in (((3,), False), ((3, 3), True), ((3, 5), False)):
                with self.subTest(dtype=dtype, lengths=lengths, dense=dense):
                    model, policy, _ = tiny_model(dtype=dtype, cpo=True)
                    trainer = Krea2PostTrainingTrainer()
                    accelerator = SimpleNamespace(
                        device=torch.device("cpu"), unwrap_model=lambda x: x, trackers=(object(),),
                        autocast=lambda: torch.autocast("cpu", dtype=torch.bfloat16) if dtype == torch.bfloat16 else nullcontext(),
                    )
                    args = SimpleNamespace(post_training="flow_dpo", flow_dpo_beta=1.0,
                                           network_args=None, network_dropout=None, compile=False)
                    torch.manual_seed(202)
                    latents = torch.randn(len(lengths), 2, 1, 4, 6)
                    rejected = torch.randn_like(latents)
                    embeds = [torch.randn(length, 2, 16) for length in lengths]
                    if dense:
                        embeds = torch.stack(embeds)
                    noise = torch.randn_like(latents)
                    original_call = trainer.call_dit
                    rows = []
                    for optimized in (False, True):
                        def call(*a, **kw):
                            if not optimized:
                                a[4].pop("_krea2_pair_count", None)
                            return original_call(*a, **kw)
                        trainer.call_dit = call
                        policy.zero_grad(set_to_none=True)
                        batch = dict(latents=latents, rejected_latents=rejected, krea2_vl_embed=embeds)
                        torch.manual_seed(203)
                        with patch.object(trainer, "_prepare_dit_inputs", wraps=trainer._prepare_dit_inputs) as prep:
                            loss, metrics = trainer.process_batch(args, accelerator, model, policy, batch, latents,
                                                                  noise, None, dtype, torch.float32, None, 0)
                            loss.backward()
                        grads = torch.cat([p.grad.flatten() for p in policy.parameters() if p.grad is not None])
                        rows.append((loss.detach(), grads.clone(), prep.call_count, metrics))
                    before, after = rows
                    self.assertEqual((before[2], after[2]), (2, 1))
                    self.assertEqual(set(before[3]), set(after[3]))
                    torch.testing.assert_close(after[0], before[0], rtol=.01, atol=1e-6)
                    if dtype == torch.float32:
                        torch.testing.assert_close(after[1], before[1], rtol=1e-5, atol=1e-6)
                    else:
                        relative = (after[1] - before[1]).norm() / before[1].norm()
                        cosine = torch.nn.functional.cosine_similarity(before[1].double(), after[1].double(), dim=0)
                        self.assertLess(relative.item(), .015)
                        self.assertGreater(cosine.item(), .9999)
                    self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
                    accelerator.trackers = ()
                    _, metrics = trainer.process_batch(args, accelerator, model, policy, batch, latents,
                                                        noise, None, dtype, torch.float32, None, 0)
                    self.assertEqual(metrics, {})


class ProfilingTests(unittest.TestCase):
    def test_invalid_config_profile_rejected_before_model_loading(self):
        import sys
        from krea2_trainer.krea2_train_network import main
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "invalid.toml"
            config.write_text("profile_steps = -1\n")
            with patch.object(sys, "argv", ["krea2-train-lora", "--config_file", str(config)]), \
                 patch("krea2_trainer.krea2_train_network.Krea2NetworkTrainer") as trainer:
                with self.assertRaisesRegex(ValueError, "nonnegative"):
                    main()
                trainer.assert_not_called()

    def test_cpu_bounded_window_and_epoch_continuation(self):
        with tempfile.TemporaryDirectory() as temp, patch("torch.cuda.Event", side_effect=AssertionError("unexpected CUDA")):
            path = Path(temp) / "profile.json"
            profiler = TrainingStepProfiler(2, 1, "cpu", path)
            seen = 0
            for epoch in range(2):
                for batch in profiler.iter_batches([torch.ones(2), torch.ones(2)]):
                    profiler.begin_step(seen, units=len(batch))
                    with profiler.phase("forward"):
                        batch.square()
                    profiler.end_step(optimizer_updated=seen % 2 == 1)
                    seen += 1
            report = profiler.finish()
            self.assertEqual(seen, 4)  # profiling never truncates training
            self.assertEqual(report["collected_microsteps"], 2)
            self.assertTrue(report["complete"])
            self.assertEqual([x["microstep"] for x in report["steps"]], [1, 2])
            self.assertTrue(all(x["cuda_stream_ms"] is None for x in report["steps"]))
            self.assertTrue(all(x["loader_and_placement_wall_ms"] >= 0 for x in report["steps"]))
            self.assertGreater(report["summary"]["units_per_second"], 0)
            self.assertEqual(json.loads(path.read_text()), report)

    def test_partial_window_and_no_samples_are_honest(self):
        with tempfile.TemporaryDirectory() as temp:
            profiler = TrainingStepProfiler(5, 0, "cpu", Path(temp) / "profile.json")
            report = profiler.finish()
            self.assertFalse(report["complete"])
            self.assertIsNone(report["summary"])
            profiler.begin_step(0)
            with self.assertRaises(RuntimeError):
                profiler.finish()
            profiler.end_step(optimizer_updated=True, auxiliary_work=True)
            report = profiler.finish()
            self.assertFalse(report["complete"])
            self.assertTrue(report["steps"][0]["sampling_or_save"])

    def test_disabled_and_rank_paths(self):
        self.assertIsNone(create_step_profiler(SimpleNamespace(), object()))
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(profile_steps=1, profile_warmup_steps=0, output_dir=temp,
                                   post_training="flow_cpo", mixed_precision="bf16", max_train_steps=2,
                                   gradient_accumulation_steps=2, gradient_checkpointing=True)
            accel = SimpleNamespace(device="cpu", num_processes=2, process_index=1)
            profiler = create_step_profiler(args, accel)
            self.assertEqual(profiler.output.name, "training-profile.rank1.json")
            self.assertEqual(profiler.metadata["unit"], "pairs")
            self.assertFalse(profiler.output.exists())
        for steps, warmup in ((-1, 0), (0, -1), (True, 0), (1, 1.5)):
            with self.assertRaises(ValueError):
                create_step_profiler(SimpleNamespace(profile_steps=steps, profile_warmup_steps=warmup), object())


if __name__ == "__main__":
    unittest.main()
