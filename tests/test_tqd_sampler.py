import argparse
import math
import unittest

import torch

from krea2_trainer.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from krea2_trainer.training.parser_common import _add_timestep_args
from krea2_trainer.training.timesteps import sample_structure_detail_tqd
from krea2_trainer.training.trainer_base import NetworkTrainer


class StructureDetailTQDSamplerTests(unittest.TestCase):
    def test_equal_scores_reproduce_native_pre_shift_logit_normal_for_fixed_cdf(self):
        scores = torch.full((3,), 0.5)
        cdf_samples = torch.tensor([0.1, 0.5, 0.9])

        actual = sample_structure_detail_tqd(
            scores,
            scores,
            kappa_base=2.0,
            kappa_max=8.0,
            sigmoid_scale=1.0,
            cdf_samples=cdf_samples,
        )
        expected = torch.sigmoid(math.sqrt(2.0) * torch.erfinv(2.0 * cdf_samples - 1.0))

        torch.testing.assert_close(actual, expected)

    def test_structure_dominant_samples_receive_higher_timesteps_than_detail_dominant_samples(self):
        count = 20_000
        torch.manual_seed(1234)
        structure_dominant = sample_structure_detail_tqd(
            torch.full((count,), 0.95),
            torch.full((count,), 0.05),
            kappa_base=2.0,
            kappa_max=8.0,
            sigmoid_scale=1.0,
        )
        torch.manual_seed(1234)
        detail_dominant = sample_structure_detail_tqd(
            torch.full((count,), 0.05),
            torch.full((count,), 0.95),
            kappa_base=2.0,
            kappa_max=8.0,
            sigmoid_scale=1.0,
        )

        self.assertGreater(structure_dominant.mean().item(), detail_dominant.mean().item())

    def test_rejects_invalid_inputs(self):
        scores = torch.tensor([0.5])
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            sample_structure_detail_tqd(scores, torch.tensor([0.5, 0.5]), kappa_base=2.0, kappa_max=8.0, sigmoid_scale=1.0)
        with self.assertRaisesRegex(ValueError, "within \\[0, 1\\]"):
            sample_structure_detail_tqd(torch.tensor([1.1]), scores, kappa_base=2.0, kappa_max=8.0, sigmoid_scale=1.0)
        with self.assertRaisesRegex(ValueError, "kappa_base"):
            sample_structure_detail_tqd(scores, scores, kappa_base=0.0, kappa_max=8.0, sigmoid_scale=1.0)

    def test_trainer_routes_structure_dominant_samples_to_higher_noise(self):
        trainer = NetworkTrainer()
        scheduler = FlowMatchDiscreteScheduler(shift=1.0, reverse=True, solver="euler")
        args = argparse.Namespace(
            timestep_sampling="tqd_krea2_shift",
            tqd_kappa_base=2.0,
            tqd_kappa_max=8.0,
            sigmoid_scale=1.0,
            min_timestep=None,
            max_timestep=None,
            preserve_distribution_shape=False,
        )
        latents = torch.zeros(4_000, 1, 2, 2)
        noise = torch.ones_like(latents)

        torch.manual_seed(42)
        _, structure_timesteps = trainer.get_noisy_model_input_and_timesteps(
            args,
            noise,
            latents,
            None,
            scheduler,
            torch.device("cpu"),
            torch.float32,
            tqd_structure_scores=torch.full((4_000,), 0.95),
            tqd_detail_scores=torch.full((4_000,), 0.05),
        )
        torch.manual_seed(42)
        _, detail_timesteps = trainer.get_noisy_model_input_and_timesteps(
            args,
            noise,
            latents,
            None,
            scheduler,
            torch.device("cpu"),
            torch.float32,
            tqd_structure_scores=torch.full((4_000,), 0.05),
            tqd_detail_scores=torch.full((4_000,), 0.95),
        )

        self.assertGreater(
            torch.as_tensor(structure_timesteps, dtype=torch.float32).mean().item(),
            torch.as_tensor(detail_timesteps, dtype=torch.float32).mean().item(),
        )

    def test_parser_accepts_tqd_mode_and_parameters(self):
        parser = argparse.ArgumentParser()
        _add_timestep_args(parser)
        args = parser.parse_args(["--timestep_sampling", "tqd_krea2_shift", "--tqd_kappa_base", "3", "--tqd_kappa_max", "9"])

        self.assertEqual(args.timestep_sampling, "tqd_krea2_shift")
        self.assertEqual(args.tqd_kappa_base, 3.0)
        self.assertEqual(args.tqd_kappa_max, 9.0)


if __name__ == "__main__":
    unittest.main()
