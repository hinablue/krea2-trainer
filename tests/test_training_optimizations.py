"""Compare optimized execution with the original math, including backward."""

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import os
import math
import tempfile
import unittest

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from safetensors.torch import save_file

from krea2_trainer.dataset.tensor_cache import TensorFileCache
from krea2_trainer.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT, temb
from krea2_trainer.krea2_train_network import Krea2NetworkTrainer
from krea2_trainer.modules.attention import AttentionParams, attention
from krea2_trainer.networks.lora_krea2 import create_arch_network
from krea2_trainer.training.flow_cpo import AdapterEMA, flow_cpo_loss, old_adapter_context
from krea2_trainer.training.timesteps import TQDScoreCache, sample_structure_detail_tqd, normalized_tqd_quality_weights
from krea2_trainer.training.trainer_base import DiTOutput, NetworkTrainer
from krea2_trainer.utils.tensor_checks import all_finite


class ScalarReads(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            self.count += 1
        return func(*args, **(kwargs or {}))


def legacy_forward(model, img, context, t, pos, mask):
    """Pre-optimization DiT: duplicated conditioning, pad=256, full last head."""
    img = model.first(img)
    t = model.tmlp(temb(t, model.config.tdim, device=img.device, dtype=img.dtype))
    tvec = model.tproj(t)
    imglen = img.shape[1]
    txtmask = mask[:, imglen:]
    nomask = AttentionParams.create_attention_params_from_mask(model.attn_mode, model.split_attn, 0, None)
    txtparams = AttentionParams.create_attention_params_from_mask(model.attn_mode, model.split_attn, 0, txtmask)
    context = model.txtmlp(model.txtfusion(context, nomask, txtparams))
    combined = torch.cat((img, context), dim=1)
    padlen = (-combined.shape[1]) % 256
    if padlen:
        combined = F.pad(combined, (0, 0, 0, padlen))
        pos = F.pad(pos, (0, 0, 0, padlen))
        txtmask = F.pad(txtmask, (0, padlen), value=False)
    params = AttentionParams.create_attention_params_from_mask(model.attn_mode, model.split_attn, imglen, txtmask)
    freqs = model.posemb(pos)
    for block in model.blocks:
        if model.gradient_checkpointing and model.training:
            combined = torch.utils.checkpoint.checkpoint(block, combined, tvec, freqs, params, use_reentrant=False)
        else:
            combined = block(combined, tvec, freqs, params)
    return model.last(combined, t)[:, :imglen]


def tiny_model(dtype=torch.float32, device="cpu", *, cpo=False):
    torch.manual_seed(76)
    config = SingleMMDiTConfig(
        features=32,
        tdim=8,
        txtdim=16,
        heads=2,
        kvheads=1,
        multiplier=1,
        layers=2,
        patch=2,
        channels=2,
        txtlayers=2,
        txtheads=2,
        txtkvheads=2,
    )
    model = SingleStreamDiT(config).to(device=device, dtype=dtype).requires_grad_(False)

    def attach(multiplier):
        network = create_arch_network(multiplier, 2, 2, None, None, model)
        network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
        return network.to(device=device, dtype=torch.float32)

    stage = attach(1.0).eval().requires_grad_(False) if cpo else None
    policy = attach(1.0)
    with torch.no_grad():
        for network in [stage, policy] if cpo else [policy]:
            for module in network.unet_loras:
                module.lora_up.weight.normal_(std=0.025)
    ema = AdapterEMA(policy, attach(0.0), 0.99) if cpo else None
    model.enable_gradient_checkpointing()
    model.train()
    return model, policy, ema


def inputs(device="cpu", lengths=(3, 5), paired=False):
    torch.manual_seed(88)
    count = len(lengths)
    context = torch.randn(count, max(lengths), 2, 16, device=device)
    textmask = torch.arange(max(lengths), device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
    context = context * textmask[:, :, None, None]
    times = torch.linspace(0.2, 0.8, count, device=device)
    if paired:
        context = torch.cat((context, context))
        times = torch.cat((times, times))
        textmask = torch.cat((textmask, textmask))
    batch = context.shape[0]
    img = torch.randn(batch, 6, 8, device=device)
    pos = torch.zeros(batch, 6 + max(lengths), 3, device=device)
    pos[:, :6, 1] = torch.arange(2, device=device).repeat_interleave(3)
    pos[:, :6, 2] = torch.arange(3, device=device).repeat(2)
    mask = torch.cat((torch.ones(batch, 6, device=device, dtype=torch.bool), textmask), dim=1)
    return img, context, times, pos, mask


def legacy_tqd_sample(structure, detail):
    mu = 0.5 + 0.5 * (structure - detail)
    kappa = 2.0 + 6.0 * (structure - detail).abs()
    u = torch.distributions.Beta((mu * kappa).clamp_min(1e-4), ((1 - mu) * kappa).clamp_min(1e-4)).sample()
    eps = torch.finfo(torch.float32).eps
    return torch.sigmoid(math.sqrt(2) * torch.erfinv(2 * u.clamp(eps, 1 - eps) - 1))


class ModelOptimizationTests(unittest.TestCase):
    def compare_model(self, *, cpo, dtype, lengths, device="cpu", steps=1):
        model, policy, ema = tiny_model(dtype, device, cpo=cpo)
        img, context, times, pos, mask = inputs(device, lengths, paired=cpo)
        metadata = model.prepare_training_metadata(2, 3, lengths, device, paired=cpo, pad_to_multiple=1)
        target = torch.randn(img.shape[0], 6, 8, device=device)

        def run(optimized):
            def forward():
                with torch.autocast(device, dtype=dtype, enabled=dtype == torch.bfloat16):
                    if optimized:
                        return model(img, context[: len(lengths)] if cpo else context, times, None, training_metadata=metadata)
                    return legacy_forward(model, img.clone().requires_grad_(), context.clone().requires_grad_(), times, pos, mask)

            if cpo:
                with old_adapter_context(policy, ema.ema_network, model):
                    old = forward()
                prediction = forward()
                n = len(lengths)
                loss, _ = flow_cpo_loss(prediction[:n], prediction[n:], old[:n], old[n:], target[:n], target[n:], 0.5, 1.0)
            else:
                prediction = forward()
                loss = (prediction.float() - target).square().mean()
                old = prediction.detach()
            grads = torch.autograd.grad(loss, tuple(policy.parameters()))
            return old.detach(), prediction.detach(), loss.detach(), grads

        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
        for _ in range(steps):
            expected = run(False)
            actual = run(True)
            if device == "cuda" and dtype == torch.bfloat16:
                # Changing GEMM batch/token dimensions changes BF16 tiling and
                # rounding. Relative error of an individual near-zero element
                # is unstable; check normalized error and gradient direction.
                for a, b in zip(actual[:3], expected[:3]):
                    a, b = a.float(), b.float()
                    self.assertLess(((a - b).norm() / b.norm().clamp_min(1e-8)).item(), 0.01)
                ga = torch.cat([g.flatten() for g in actual[3]])
                gb = torch.cat([g.flatten() for g in expected[3]])
                self.assertLess(((ga - gb).norm() / gb.norm()).item(), 0.015)
                self.assertGreater(F.cosine_similarity(ga, gb, dim=0).item(), 0.9999)
            else:
                tolerance = dict(rtol=0.05, atol=5e-4) if dtype == torch.bfloat16 else dict(rtol=2e-5, atol=2e-6)
                for actual_value, expected_value in zip(actual[:3], expected[:3]):
                    torch.testing.assert_close(actual_value, expected_value, **tolerance)
                for actual_grad, expected_grad in zip(actual[3], expected[3]):
                    torch.testing.assert_close(actual_grad, expected_grad, **tolerance)
            if steps > 1:
                for parameter, gradient in zip(policy.parameters(), actual[3]):
                    parameter.grad = gradient
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(policy)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        if cpo:
            self.assertTrue(all(parameter.grad is None for parameter in ema.ema_network.parameters()))

    def test_tqd_and_cpo_forward_backward_match_original(self):
        for cpo in (False, True):
            for dtype in (torch.float32, torch.bfloat16):
                for lengths in ((3,), (3, 5)):
                    with self.subTest(cpo=cpo, dtype=dtype, lengths=lengths):
                        self.compare_model(cpo=cpo, dtype=dtype, lengths=lengths)

    def test_conditioning_is_deduplicated_only_for_explicit_pairs(self):
        model, _, _ = tiny_model()
        img, context, times, _, _ = inputs(paired=True)
        metadata = model.prepare_training_metadata(2, 3, (3, 5), "cpu", paired=True, pad_to_multiple=1)
        seen = []
        handle = model.txtfusion.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape[0]))
        try:
            model(img, context[:2], times, None, training_metadata=metadata)
            ordinary = model.prepare_training_metadata(2, 3, (3, 5, 3, 5), "cpu", pad_to_multiple=1)
            model(img, context, times, None, training_metadata=ordinary)
        finally:
            handle.remove()
        self.assertEqual(seen, [2, 4])

    def test_metadata_cache_reuse_bound_invalidation_and_checkpoint_keys(self):
        model, _, _ = tiny_model()
        keys = set(model.state_dict())
        with patch.object(model.posemb, "forward", wraps=model.posemb.forward) as rope:
            first = model.prepare_training_metadata(2, 3, (3,), "cpu", pad_to_multiple=1)
            self.assertIs(first, model.prepare_training_metadata(2, 3, (3,), "cpu", pad_to_multiple=1))
            self.assertEqual(rope.call_count, 1)
        padded = model.prepare_training_metadata(2, 3, (3,), "cpu", pad_to_multiple=256)
        self.assertEqual((first.padlen, padded.padlen), (0, 247))
        for width in range(1, 12):
            model.prepare_training_metadata(2, width, (3,), "cpu", pad_to_multiple=1)
        self.assertEqual(len(model._training_metadata_cache), 8)
        model.float()
        self.assertFalse(model._training_metadata_cache)
        self.assertEqual(set(model.state_dict()), keys)

    def test_attention_no_scalar_reads_and_unchanged_gqa(self):
        for lengths in ((3, 3), (2, 3)):
            mask = torch.arange(5)[None, :] < torch.tensor(lengths)[:, None]
            params = AttentionParams.create_attention_params_from_mask("torch", False, 4, mask, text_lengths=lengths)
            q = torch.randn(2, 9, 4, 8, requires_grad=True)
            k = torch.randn(2, 9, 1, 8, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            with ScalarReads() as reads:
                actual = attention([q, k, v], attn_params=params)
                grads = torch.autograd.grad(actual.square().mean(), (q, k, v))
            self.assertEqual(reads.count, 0)
            expected = attention([q, k, v], attn_params=replace(params, lengths_known=False))
            expected_grads = torch.autograd.grad(expected.square().mean(), (q, k, v))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for a, b in zip(grads, expected_grads):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_compiled_tqd_block_accepts_prepared_metadata_without_graph_breaks(self):
        model, policy, _ = tiny_model()
        img, context, times, _, _ = inputs()
        metadata = model.prepare_training_metadata(2, 3, (3, 5), "cpu", pad_to_multiple=256)
        model.blocks[0] = torch.compile(model.blocks[0], backend="eager", fullgraph=True)
        output = model(img, context, times, None, training_metadata=metadata)
        torch.autograd.grad(output.square().mean(), tuple(policy.parameters()))
        self.assertEqual(metadata.padlen, 245)

    def test_tqd_dropout_preserves_original_shapes_draws_and_gradients(self):
        model, policy, _ = tiny_model()
        for module in policy.unet_loras:
            module.dropout = 0.2
        img, context, times, pos, mask = inputs()
        trainer = Krea2NetworkTrainer()
        accelerator = SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda model: model, autocast=nullcontext)
        # Six image tokens map to latent H=4,W=6 at patch=2.
        batch = {"latents": torch.zeros(2, 2, 1, 4, 6), "krea2_vl_embed": [context[0, :3], context[1]]}
        prepared = trainer._prepare_dit_inputs(
            SimpleNamespace(network_dropout=0.2, network_args=None, compile=False),
            accelerator,
            model,
            batch,
            torch.zeros_like(batch["latents"]),
            torch.zeros_like(batch["latents"]),
            times * 1000,
            torch.float32,
        )[0]["training_metadata"]
        self.assertEqual(prepared.padlen, 245)
        self.assertFalse(prepared.image_only_head)
        torch.manual_seed(447)
        expected = legacy_forward(model, img.clone().requires_grad_(), context.clone().requires_grad_(), times, pos, mask)
        expected_grads = torch.autograd.grad(expected.square().mean(), tuple(policy.parameters()))
        rng = torch.get_rng_state()
        torch.manual_seed(447)
        actual = model(img, context, times, None, training_metadata=prepared)
        grads = torch.autograd.grad(actual.square().mean(), tuple(policy.parameters()))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for a, b in zip(grads, expected_grads):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))


class TQDOptimizationTests(unittest.TestCase):
    def test_trainer_identity_weights_and_optional_metrics(self):
        from tests.test_post_training import TinyDiT, arguments, toy_accelerator

        trainer = Krea2NetworkTrainer()
        model = TinyDiT()
        args = arguments()
        args.timestep_sampling = "tqd_krea2_shift"
        args.tqd_quality_weighting = True
        accelerator = toy_accelerator()
        latents = torch.zeros(1, 2, 1, 2, 2)
        batch = dict(
            latents=latents,
            timesteps=None,
            krea2_vl_embed=[torch.ones(3, 2, 4)],
            tqd_structure_score=torch.tensor([0.8]),
            tqd_detail_score=torch.tensor([0.2]),
            tqd_score_values=((0.8, 0.2),),
        )
        for trackers in ((), (object(),)):
            accelerator.trackers = trackers
            with patch.object(trainer, "compute_loss", wraps=trainer.compute_loss) as loss:
                _, metrics = trainer.process_batch(
                    args,
                    accelerator,
                    model,
                    None,
                    batch,
                    latents,
                    torch.ones_like(latents),
                    None,
                    torch.bfloat16,
                    torch.float32,
                    None,
                    0,
                )
            self.assertIsNone(loss.call_args.kwargs["sample_weights"])
            if trackers:
                self.assertEqual(set(metrics), {"tqd/structure_mean", "tqd/detail_mean", "tqd/timestep_mean", "tqd/timestep_std"})
                self.assertAlmostEqual(metrics["tqd/structure_mean"], 0.8)
                self.assertEqual(metrics["tqd/timestep_std"], 0.0)
                self.assertTrue(all(isinstance(value, float) for value in metrics.values()))
            else:
                self.assertEqual(metrics, {})
        batch.update(tqd_structure_score=torch.zeros(1), tqd_detail_score=torch.zeros(1), tqd_score_values=((0.0, 0.0),))
        with self.assertRaisesRegex(ValueError, "non-zero"):
            trainer.process_batch(
                args,
                accelerator,
                model,
                None,
                batch,
                latents,
                torch.ones_like(latents),
                None,
                torch.bfloat16,
                torch.float32,
                None,
                0,
            )

    def test_cached_distribution_preserves_draws_rng_and_quality_weights(self):
        cache = TQDScoreCache(max_entries=2)
        values = ((0.8, 0.2), (0.3, 0.9))
        s, d = torch.tensor(values).unbind(1)
        prepared = cache.get(values, "cpu", 2, 8)
        torch.manual_seed(211)
        expected = legacy_tqd_sample(s, d)
        state = torch.get_rng_state()
        torch.manual_seed(211)
        with ScalarReads() as reads:
            actual = sample_structure_detail_tqd(s, d, kappa_base=2, kappa_max=8, sigmoid_scale=1, prepared=prepared)
        self.assertEqual(reads.count, 0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        torch.testing.assert_close(prepared.weights, normalized_tqd_quality_weights(s, d), rtol=0, atol=0)
        self.assertIs(cache.get(values, "cpu", 2, 8), prepared)
        self.assertIsNot(cache.get(values, "cpu", 3, 8), prepared)
        cache.get(((0.5, 0.5),), "cpu", 2, 8)
        self.assertEqual(len(cache._entries), 2)

    def test_cached_scores_reject_invalid_and_preserve_zero_quality_condition(self):
        cache = TQDScoreCache()
        for values in ((), ((float("nan"), 0.5),), ((1.1, 0.5),)):
            with self.assertRaises(ValueError):
                cache.get(values, "cpu", 2, 8)
        self.assertIsNone(cache.get(((0.0, 0.0),), "cpu", 2, 8).weights)
        for values in (((0.9, 0.2),), ((0.1, 0.05),)):
            torch.testing.assert_close(cache.get(values, "cpu", 2, 8).weights, torch.ones(1), rtol=0, atol=0)

    def test_weighted_reduction_matches_original_loss_and_gradients(self):
        trainer = NetworkTrainer()
        for count in (1, 3):
            pred = torch.randn(count, 2, 1, 4, 4, requires_grad=True)
            target = torch.randn_like(pred)
            weights = torch.linspace(0.5, 1.5, count)
            timestep_weights = torch.linspace(0.7, 1.2, count).view(-1, 1, 1, 1, 1)
            expected = (F.mse_loss(pred, target, reduction="none") * timestep_weights * weights.view(-1, 1, 1, 1, 1)).mean()
            with patch("krea2_trainer.training.trainer_base.compute_loss_weighting_for_sd3", return_value=timestep_weights):
                actual, _ = trainer.compute_loss(
                    SimpleNamespace(weighting_scheme="none"),
                    DiTOutput(pred, target),
                    torch.ones(count),
                    None,
                    torch.float32,
                    torch.float32,
                    0,
                    weights,
                )
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(torch.autograd.grad(actual, pred)[0], torch.autograd.grad(expected, pred)[0])

    def test_spatial_weights_retain_original_broadcasting(self):
        trainer = NetworkTrainer()
        pred = torch.randn(2, 1, 1, 2, 2, requires_grad=True)
        target = torch.randn_like(pred)
        # numel equals batch size, but these weights act along width, not batch.
        weights = torch.tensor([0.5, 1.5])
        expected = (F.mse_loss(pred, target, reduction="none") * weights).mean()
        with patch("krea2_trainer.training.trainer_base.compute_loss_weighting_for_sd3", return_value=weights):
            actual, _ = trainer.compute_loss(
                SimpleNamespace(weighting_scheme="none"),
                DiTOutput(pred, target),
                torch.ones(2),
                None,
                torch.float32,
                torch.float32,
                0,
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(actual, pred)[0], torch.autograd.grad(expected, pred)[0], rtol=0, atol=0)


class CacheAndEMATests(unittest.TestCase):
    def test_tensor_cache_refresh_isolation_and_memory_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.safetensors"
            save_file({"text": torch.ones(8)}, str(path))
            cache = TensorFileCache(max_bytes=64)
            from safetensors.torch import load_file

            with patch("krea2_trainer.dataset.tensor_cache.load_file", wraps=load_file) as load:
                cache.load(path)["text"].zero_()
                cached = cache.load(path)
                torch.testing.assert_close(cached["text"], torch.ones(8))
                self.assertEqual(load.call_count, 1)
                cached["text"].zero_()
                torch.testing.assert_close(cache.load(path)["text"], torch.ones(8))
                save_file({"text": torch.full((8,), 2.0)}, str(path))
                torch.testing.assert_close(cache.load(path)["text"], torch.full((8,), 2.0))
                self.assertEqual(load.call_count, 2)
            for i in range(4):
                other = Path(directory) / f"{i}.safetensors"
                save_file({"text": torch.ones(8)}, str(other))
                cache.load(other)
                self.assertLessEqual(cache.size_bytes, 64)

    def test_ema_checks_do_not_read_one_scalar_per_matrix(self):
        model, policy, ema = tiny_model(cpo=True)
        expected = {
            key: value * 0.99 + dict(policy.named_parameters())[key].detach() * 0.01
            for key, value in ema.ema_network.named_parameters()
        }
        with ScalarReads() as reads:
            ema.update(policy)
        self.assertLessEqual(reads.count, 8)
        for key, value in ema.ema_network.named_parameters():
            # Same order/scalars as production (1 - decay, not decimal 0.01).
            torch.testing.assert_close(value, expected[key], rtol=1e-6, atol=1e-7)
        before = {key: value.clone() for key, value in ema.ema_network.named_parameters()}
        with torch.no_grad():
            list(policy.parameters())[-1].fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            ema.update(policy)
        self.assertEqual(ema.updates, 1)
        for key, value in ema.ema_network.named_parameters():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_batched_finite_check_handles_nan_inf_and_large_values(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            limit = torch.finfo(dtype).max
            self.assertTrue(all_finite([torch.tensor([limit, -limit], dtype=dtype)]))
            for bad in (float("nan"), float("inf"), -float("inf")):
                self.assertFalse(all_finite([torch.ones(2, dtype=dtype), torch.tensor([bad], dtype=dtype)]))


@unittest.skipUnless(os.environ.get("RUN_GPU_OPTIMIZATION_TESTS") == "1", "Opt-in small CUDA verification")
class GPUOptimizationTests(unittest.TestCase):
    def test_bf16_cpo_and_tqd_through_three_updates(self):
        self.assertTrue(torch.cuda.is_available())
        checks = ModelOptimizationTests()
        for cpo in (False, True):
            for lengths in ((3,), (3, 5)):
                with self.subTest(cpo=cpo, lengths=lengths):
                    checks.compare_model(cpo=cpo, dtype=torch.bfloat16, lengths=lengths, device="cuda", steps=3)

    def test_cuda_ema_recurrence_and_tqd_rng_are_exact(self):
        _, policy, ema = tiny_model(torch.bfloat16, "cuda", cpo=True)
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.add_(0.0125)
        expected = {
            key: value * ema.decay + dict(policy.named_parameters())[key].detach() * (1 - ema.decay)
            for key, value in ema.ema_network.named_parameters()
        }
        ema.update(policy)
        for key, value in ema.ema_network.named_parameters():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        values = ((0.8, 0.2), (0.3, 0.9))
        s, d = torch.tensor(values, device="cuda").unbind(1)
        prepared = TQDScoreCache().get(values, "cuda", 2, 8)
        torch.cuda.manual_seed(197)
        original = legacy_tqd_sample(s, d)
        state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(197)
        optimized = sample_structure_detail_tqd(s, d, kappa_base=2, kappa_max=8, sigmoid_scale=1, prepared=prepared)
        torch.testing.assert_close(original, optimized, rtol=0, atol=0)
        self.assertTrue(torch.equal(state, torch.cuda.get_rng_state()))
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            self.assertTrue(all_finite([torch.tensor([torch.finfo(dtype).max], dtype=dtype, device="cuda")]))
            self.assertFalse(all_finite([torch.tensor([float("nan")], dtype=dtype, device="cuda")]))


if __name__ == "__main__":
    unittest.main()
