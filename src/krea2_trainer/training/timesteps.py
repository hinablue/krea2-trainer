"""Timestep sampling density and loss weighting utilities (SD3-style)."""

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass

import torch


logger = logging.getLogger(__name__)


def _checked_tqd_scores(structure_scores, detail_scores):
    if structure_scores.shape != detail_scores.shape:
        raise ValueError("TQD structure and detail score tensors must have matching shapes")
    structure = structure_scores.to(dtype=torch.float32)
    detail = detail_scores.to(device=structure.device, dtype=torch.float32)
    invalid = ~torch.isfinite(structure) | ~torch.isfinite(detail) | (structure < 0) | (structure > 1) | (detail < 0) | (detail > 1)
    if torch.any(invalid):
        raise ValueError("TQD structure and detail scores must be finite and within [0, 1]")
    return structure, detail


def _tqd_distribution(structure, detail, kappa_base, kappa_max):
    if not math.isfinite(kappa_base) or not math.isfinite(kappa_max) or kappa_base <= 0 or kappa_max < kappa_base:
        raise ValueError("TQD requires finite kappa_base > 0 and kappa_max >= kappa_base")
    if kappa_max > torch.finfo(torch.float32).max:
        raise ValueError("TQD kappa_max must be representable in float32")
    mu = 0.5 + 0.5 * (structure - detail)
    kappa = kappa_base + (kappa_max - kappa_base) * (structure - detail).abs()
    alpha = (mu * kappa).clamp_min(1e-4)
    beta = ((1.0 - mu) * kappa).clamp_min(1e-4)
    # Finite bounded scores and practical finite kappas make both positive.
    # Reject float32 overflow before bypassing distribution-internal checks.
    return torch.distributions.Beta(alpha, beta, validate_args=False)


@dataclass(frozen=True)
class PreparedTQDScores:
    structure: torch.Tensor
    detail: torch.Tensor
    distribution: torch.distributions.Beta
    weights: torch.Tensor | None


class TQDScoreCache:
    """Bounded, process-local cache keyed by actual immutable score values.

    Cache distribution parameters, never random draws. Sampling remains on the
    training device at the original point in the RNG stream.
    """

    def __init__(self, max_entries=256):
        self.max_entries = max_entries
        self._entries = OrderedDict()

    @torch.no_grad()
    def get(self, score_values, device, kappa_base, kappa_max):
        values = tuple(tuple(pair) for pair in score_values)
        key = (values, torch.device(device), kappa_base, kappa_max)
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        # The loader supplies Python score metadata, so validation is CPU-only.
        if not values or any(len(pair) != 2 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in pair) for pair in values):
            raise ValueError("TQD structure and detail scores must be finite and within [0, 1]")
        scores = torch.tensor(values, device=device, dtype=torch.float32)
        structure, detail = scores[:, 0], scores[:, 1]
        distribution = _tqd_distribution(structure, detail, kappa_base, kappa_max)
        quality = torch.maximum(structure, detail)
        mean_quality = quality.mean()
        weights = None if mean_quality <= torch.finfo(torch.float32).eps else quality / mean_quality
        result = PreparedTQDScores(structure, detail, distribution, weights)
        self._entries[key] = result
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return result


def compute_ideogram4_shift_timestep(
    uniform_samples: torch.Tensor,
    token_grid_height: int,
    token_grid_width: int,
    *,
    image_patch_size: int = 16,
    base_mean: float = 0.0,
    std: float = 1.5,
) -> torch.Tensor:
    """Map uniform samples to Ideogram 4's resolution-aware logit-normal t."""
    eps = 1e-7
    u = torch.clamp(uniform_samples.to(torch.float64), eps, 1.0 - eps)
    image_pixels = token_grid_height * image_patch_size * token_grid_width * image_patch_size
    mean = base_mean + 0.5 * math.log(image_pixels / (512 * 512))
    z = torch.special.ndtri(u)
    # musubi convention: t=1 is pure noise, t=0 is clean. Higher resolution -> larger
    # ``mean`` -> ``t`` skewed toward 1 (more noise). The trainer feeds the model
    # ``model_t = 1 - t``, so this reproduces the inference schedule's
    # ``model_t = 1 - sigmoid(mean + std * z)`` instead of mirroring it.
    t = torch.special.expit(mean + std * z)
    t_min = 1.0 / (1 + math.exp(0.5 * 18.0))
    t_max = 1.0 / (1 + math.exp(0.5 * -15.0))
    return t.clamp(1.0 - t_max, 1.0 - t_min).to(dtype=uniform_samples.dtype)


def normalized_tqd_quality_weights(structure_scores: torch.Tensor, detail_scores: torch.Tensor) -> torch.Tensor:
    """Return mean-one deterministic weights approximating TQD sample retention."""
    structure, detail = _checked_tqd_scores(structure_scores, detail_scores)

    quality = torch.maximum(structure, detail)
    mean_quality = quality.mean()
    if mean_quality <= torch.finfo(quality.dtype).eps:
        raise ValueError("TQD quality weights require at least one non-zero structure or detail score per batch")
    return quality / mean_quality


def sample_structure_detail_tqd(
    structure_scores: torch.Tensor,
    detail_scores: torch.Tensor,
    *,
    kappa_base: float,
    kappa_max: float,
    sigmoid_scale: float,
    cdf_samples: torch.Tensor | None = None,
    prepared: PreparedTQDScores | None = None,
) -> torch.Tensor:
    """Sample Krea2's pre-shift timestep from per-sample structure/detail scores.

    The Beta draw happens in CDF space. When scores are equal and
    ``kappa_base == 2``, it is uniform; applying the inverse normal CDF then
    reproduces Krea2's native logit-normal sample before resolution shifting.
    """
    if prepared is None:
        structure, detail = _checked_tqd_scores(structure_scores, detail_scores)
        distribution = _tqd_distribution(structure, detail, kappa_base, kappa_max)
    else:
        structure, distribution = prepared.structure, prepared.distribution

    if cdf_samples is None:
        cdf_samples = distribution.sample()
    else:
        if cdf_samples.shape != structure.shape:
            raise ValueError("TQD CDF samples must match the score tensor shape")

    assert cdf_samples is not None
    eps = torch.finfo(structure.dtype).eps
    u = cdf_samples.to(device=structure.device, dtype=structure.dtype).clamp(eps, 1.0 - eps)
    z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
    return torch.sigmoid(sigmoid_scale * z)


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)

    # if sum([(schedule_timesteps == t) for t in timesteps]) < len(timesteps):
    if any([(schedule_timesteps == t).sum() == 0 for t in timesteps]):
        # raise ValueError("Some timesteps are not in the schedule / 一部のtimestepsがスケジュールに含まれていません")
        # round to nearest timestep
        logger.warning("Some timesteps are not in the schedule / 一部のtimestepsがスケジュールに含まれていません")
        step_indices = [torch.argmin(torch.abs(schedule_timesteps - t)).item() for t in timesteps]
    else:
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def compute_loss_weighting_for_sd3(weighting_scheme: str, noise_scheduler, timesteps, device, dtype):
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt" or weighting_scheme == "cosmap":
        sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=5, dtype=dtype)
        if weighting_scheme == "sigma_sqrt":
            weighting = (sigmas**-2.0).float()
        else:
            bot = 1 - 2 * sigmas + 2 * sigmas**2
            weighting = 2 / (math.pi * bot)
    else:
        weighting = None  # torch.ones_like(sigmas)
    return weighting
