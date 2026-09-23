"""FlowCPO velocity objective and a separately attached, adapter-only EMA.

The old branch is the immutable base/stage-one hook chain plus EMA matrices,
not a deepcopy of the DiT and not an in-place swap of policy tensors. Attach
native networks in base -> stage-one -> policy -> EMA order before using these
helpers. The trainer owns their construction, device placement and attachment.
"""

from contextlib import contextmanager
import math
from numbers import Real
import os
from pathlib import Path
import re
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from krea2_trainer.training.metrics import materialize_metrics

from krea2_trainer.utils.tensor_checks import all_finite


FLOW_CPO_EMA_STATE = "krea2_flow_cpo_ema.safetensors"
_EMA_VERSION = "1"
_PARAMETER_KEY = re.compile(r"lora_(?:unet|te[0-9]*)_.+\.lora_(?:down|up)(?:\.[0-9]+)?\.weight")


def _finite_scalar(value, name, *, minimum, maximum=None, inclusive_minimum=True):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar, not a bool or coercible value")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if (
        not math.isfinite(result)
        or (result < minimum if inclusive_minimum else result <= minimum)
        or (maximum is not None and result >= maximum)
    ):
        interval = f"{'[' if inclusive_minimum else '('}{minimum}, {maximum if maximum is not None else 'inf'})"
        raise ValueError(f"{name} must be finite and in {interval}")
    return result


def flow_cpo_loss(
    policy_chosen, policy_rejected, old_chosen, old_rejected, target_chosen, target_rejected, beta, loss_lambda,
    *, collect_metrics=True, metrics_as_tensors=False,
):
    """Return ``(loss, metrics)`` from six native velocity tensors [pairs, ...].

    All six shapes must be identical, nonempty and at least two-dimensional.
    Convert inputs to FP32 *before* mixing and squaring, then mean over all
    non-batch dimensions and finally over pairs:

        mu_w = (1 - beta) * stopgrad(old_chosen) + beta * policy_chosen
        nu_l = (1 + beta) * stopgrad(old_rejected) - beta * policy_rejected
        loss = mean_pairs(MSE(mu_w, target_chosen)
                          + loss_lambda * MSE(nu_l, target_rejected))

    No logsigmoid, beta/2, timestep/SNR or additional preference weights apply.
    Metrics are detached Python floats: ``flow_cpo/loss``, ``chosen_mse`` and
    ``rejected_mse`` (the latter two refer to the mixed mu/nu branches).
    """
    beta = _finite_scalar(beta, "FlowCPO beta", minimum=0, inclusive_minimum=False)
    loss_lambda = _finite_scalar(loss_lambda, "FlowCPO loss_lambda", minimum=0)
    values = (policy_chosen, policy_rejected, old_chosen, old_rejected, target_chosen, target_rejected)
    if any(not isinstance(value, torch.Tensor) or not value.is_floating_point() for value in values):
        raise ValueError("FlowCPO requires six floating-point velocity tensors")
    shape, device = policy_chosen.shape, policy_chosen.device
    if any(value.shape != shape or value.ndim < 2 or not value.numel() for value in values):
        raise ValueError("FlowCPO velocity tensors must have identical nonempty [pairs, ...] shapes")
    if any(value.device != device or value.layout != torch.strided or value.is_meta for value in values):
        raise ValueError("FlowCPO velocity tensors must be dense, materialized and on the same device")
    if not all_finite(values):
        raise FloatingPointError("All six FlowCPO velocity branches must be finite")

    mu = (1 - beta) * old_chosen.detach().float() + beta * policy_chosen.float()
    nu = (1 + beta) * old_rejected.detach().float() - beta * policy_rejected.float()
    chosen = (mu - target_chosen.float()).square().flatten(1).mean(1)
    rejected = (nu - target_rejected.float()).square().flatten(1).mean(1)
    loss = (chosen + loss_lambda * rejected).mean()
    chosen_mean, rejected_mean = chosen.detach().mean(), rejected.detach().mean()
    if not all_finite((chosen, rejected, loss, chosen_mean, rejected_mean)):
        raise FloatingPointError("Non-finite FP32 FlowCPO objective or branch MSE")
    if not collect_metrics:
        return loss, {}
    metrics = {
        "flow_cpo/loss": loss.detach(),
        "flow_cpo/chosen_mse": chosen_mean,
        "flow_cpo/rejected_mse": rejected_mean,
    }
    return loss, metrics if metrics_as_tensors else materialize_metrics(metrics)


def _native_modules(network):
    if not isinstance(network, torch.nn.Module) or not getattr(network, "unet_loras", None):
        raise TypeError("FlowCPO requires an attached native Krea2 LoRA network")
    modules = list(network.modules())
    module_set = set(modules)
    if any(module not in module_set or not hasattr(module, "org_forward") for module in network.unet_loras):
        raise ValueError("Attach the native LoRA network before constructing or selecting FlowCPO EMA")
    multipliers = [(module, module.multiplier) for module in modules if hasattr(module, "multiplier")]
    if not multipliers:
        raise TypeError("FlowCPO requires native LoRA multipliers")
    return modules, multipliers


def _adapter_state(network, label):
    _native_modules(network)
    parameters = dict(network.named_parameters())
    if not parameters or any(_PARAMETER_KEY.fullmatch(key) is None for key in parameters):
        raise ValueError(f"{label} must contain only native LoRA parameter matrix keys")
    for key, value in parameters.items():
        if value.dtype != torch.float32 or value.ndim != 2 or not value.numel() or value.layout != torch.strided or value.is_meta:
            raise ValueError(f"{label} {key} must be a nonempty materialized FP32 parameter matrix")
    if not all_finite(parameters.values()):
        raise FloatingPointError(f"{label} matrices contain non-finite values")
    alphas = dict(network.named_buffers())
    expected_alphas = {key.rsplit(".lora_", 1)[0] + ".alpha" for key in parameters}
    if set(alphas) != expected_alphas:
        raise ValueError(f"{label} must contain exactly the corresponding LoRA alpha buffers")
    for key, value in alphas.items():
        if value.ndim != 0 or value.is_meta or value.is_complex() or value.dtype == torch.bool:
            raise ValueError(f"{label} {key} must be a finite scalar alpha buffer")
    if not all_finite(alphas.values()):
        raise ValueError(f"{label} must contain finite scalar alpha buffers")
    return parameters, alphas


def _matching_parameters(parameters, expected):
    if set(parameters) != set(expected):
        raise ValueError("FlowCPO EMA parameter key mismatch")
    if any(parameters[key].shape != expected[key] for key in expected):
        raise ValueError("FlowCPO EMA parameter shape mismatch")


def _matching_alphas(alphas, expected):
    if set(alphas) != set(expected):
        raise ValueError("FlowCPO EMA alpha buffers must remain exactly equal to the initial policy alphas")
    # Pack before copying: one transfer per device, not one per alpha scalar.
    devices = {value.device for value in alphas.values()} | {value.device for value in expected.values()}
    if len(devices) == 1:
        same = torch.equal(torch.stack([alphas[key] for key in expected]), torch.stack(list(expected.values())))
    else:

        def packed_cpu(values):
            groups = {}
            for key, value in values.items():
                groups.setdefault(value.device, []).append(key)
            result = {}
            for keys in groups.values():
                packed = torch.stack([values[key].detach() for key in keys]).cpu()
                result.update(zip(keys, packed.unbind()))
            return torch.stack([result[key] for key in expected])

        same = torch.equal(packed_cpu(alphas), packed_cpu(expected))
    if not same:
        raise ValueError("FlowCPO EMA alpha buffers must remain exactly equal to the initial policy alphas")


def _independent_parameters(policy, old):
    policy_storage = {(parameter.device, parameter.untyped_storage().data_ptr()) for parameter in policy.values()}
    if any((parameter.device, parameter.untyped_storage().data_ptr()) in policy_storage for parameter in old.values()):
        raise ValueError("FlowCPO policy and EMA must have independent parameter storage")


class AdapterEMA:
    """FP32 EMA of two already constructed and attached native LoRA networks.

    Both networks must contain FP32 matrices with identical keys/shapes and
    equal alpha buffers. Construction copies the policy matrices exactly,
    freezes/evaluates the EMA and sets its multiplier to zero. Alpha is checked,
    never copied, averaged or written to the EMA checkpoint.

    ``updates`` counts explicit successful ``update`` calls; the trainer must
    call once per completed (not skipped) optimizer step, not per microbatch.
    """

    def __init__(self, policy_network, ema_network, decay):
        decay = _finite_scalar(decay, "FlowCPO EMA decay", minimum=0, maximum=1)
        if policy_network is ema_network:
            raise ValueError("FlowCPO policy and EMA must be separately constructed networks")
        policy, policy_alphas = _adapter_state(policy_network, "Policy")
        old, old_alphas = _adapter_state(ema_network, "EMA")
        shapes = {key: value.shape for key, value in policy.items()}
        _matching_parameters(old, shapes)
        _matching_alphas(old_alphas, policy_alphas)
        _independent_parameters(policy, old)
        # Validate every branch before the first destination write.
        initial = {key: value.detach().to(device=old[key].device, dtype=torch.float32).clone() for key, value in policy.items()}
        with torch.no_grad():
            for key, value in old.items():
                value.copy_(initial[key])
        ema_network.requires_grad_(False)
        ema_network.eval()
        for module, _ in _native_modules(ema_network)[1]:
            module.multiplier = 0.0
        self.ema_network = ema_network
        self.decay = decay
        self.updates = 0
        self._shapes = shapes
        self._alphas = {key: value.detach().cpu().clone() for key, value in policy_alphas.items()}

    def _current_parameters(self):
        old, alphas = _adapter_state(self.ema_network, "EMA")
        _matching_parameters(old, self._shapes)
        _matching_alphas(alphas, self._alphas)
        return old

    @torch.no_grad()
    def update(self, policy_network):
        """Apply ``ema = decay * ema + (1 - decay) * policy`` in FP32.

        All source/destination matrices, alphas and candidate results are
        validated before mutation, so an invalid update changes neither the
        matrices nor ``updates``. Policy tensors are never modified.
        """
        old = self._current_parameters()
        policy, alphas = _adapter_state(policy_network, "Policy")
        _matching_parameters(policy, self._shapes)
        _matching_alphas(alphas, self._alphas)
        _independent_parameters(policy, old)
        self._validate_counters()
        destinations = list(old.values())
        sources = [policy[key].detach().to(device=value.device, dtype=torch.float32) for key, value in old.items()]
        # Keep two multiplies followed by an add, matching the original FP32
        # rounding. lerp/add(alpha=...) can change the recurrence via fusion.
        updated = torch._foreach_add(torch._foreach_mul(destinations, self.decay), torch._foreach_mul(sources, 1 - self.decay))
        if not all_finite(updated):
            raise FloatingPointError("Non-finite FP32 FlowCPO EMA update")
        torch._foreach_copy_(destinations, updated)
        self.updates += 1

    def _validate_counters(self):
        _finite_scalar(self.decay, "FlowCPO EMA decay", minimum=0, maximum=1)
        if type(self.updates) is not int or self.updates < 0:
            raise ValueError("FlowCPO EMA updates must be a nonnegative integer")

    def save(self, directory):
        """Atomically write the single EMA safetensors state; return its Path."""
        parameters = self._current_parameters()
        self._validate_counters()
        tensors = {
            key: value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone() for key, value in parameters.items()
        }
        metadata = {"version": _EMA_VERSION, "decay": str(self.decay), "updates": str(self.updates)}
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / FLOW_CPO_EMA_STATE
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=directory, prefix=f".{FLOW_CPO_EMA_STATE}.", suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
            save_file(tensors, str(temporary), metadata=metadata)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    def validate(self, directory):
        """Read once and preflight without mutation; return a validated payload.

        The dict contains ``version`` (str), ``decay`` (float), ``updates``
        (int), and ``parameters`` (dict of CPU FP32 tensors). Metadata keys
        must be exactly version/decay/updates. Decay must match exactly, not
        approximately, and updates must be canonical ASCII decimal notation.
        """
        self._current_parameters()
        self._validate_counters()
        path = Path(directory) / FLOW_CPO_EMA_STATE
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if set(metadata) != {"version", "decay", "updates"} or metadata.get("version") != _EMA_VERSION:
                raise ValueError("Invalid FlowCPO EMA metadata keys or version")
            try:
                decay = float(metadata["decay"])
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("Invalid FlowCPO EMA decay metadata") from error
            if not math.isfinite(decay) or decay != self.decay:
                raise ValueError("FlowCPO EMA checkpoint decay does not exactly match configured decay")
            if re.fullmatch(r"0|[1-9][0-9]*", metadata["updates"]) is None:
                raise ValueError("FlowCPO EMA updates metadata must be a canonical nonnegative integer")
            updates = int(metadata["updates"])
            if set(handle.keys()) != set(self._shapes):
                raise ValueError("FlowCPO EMA checkpoint parameter key mismatch")
            parameters = {key: handle.get_tensor(key) for key in handle.keys()}
        _matching_parameters(parameters, self._shapes)
        for key, value in parameters.items():
            if value.dtype != torch.float32:
                raise ValueError(f"FlowCPO EMA checkpoint {key} must be FP32")
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"FlowCPO EMA checkpoint {key} contains non-finite values")
        return {"version": _EMA_VERSION, "decay": decay, "updates": updates, "parameters": parameters}

    @torch.no_grad()
    def load(self, directory):
        """Read/validate the full checkpoint before applying matrices or count."""
        payload = self.validate(directory)
        old = self._current_parameters()
        # Resolve all device transfers before starting to overwrite the EMA.
        prepared = {key: value.to(device=old[key].device) for key, value in payload["parameters"].items()}
        for key, value in old.items():
            value.copy_(prepared[key])
        self.updates = payload["updates"]


@contextmanager
def old_adapter_context(policy_network, ema_network, transformer):
    """Select only the EMA delta over the unchanged base/stage-one hook chain.

    Temporarily set policy multipliers to zero and EMA multipliers to one;
    evaluate the transformer and both adapters under no-grad. Restore every
    original module training flag and adapter multiplier, including exceptions
    during setup/forward and nested use. No tensor is mutated. Call before the
    policy autograd graph is built, never between checkpointed forward/backward.
    """
    if policy_network is ema_network:
        raise ValueError("FlowCPO policy and EMA must be distinct networks")
    policy_modules, policy_multipliers = _native_modules(policy_network)
    ema_modules, ema_multipliers = _native_modules(ema_network)
    if set(policy_modules).intersection(ema_modules):
        raise ValueError("FlowCPO policy and EMA may not share adapter modules")
    modules = list(dict.fromkeys([*policy_modules, *ema_modules, *transformer.modules()]))
    training = [(module, module.training) for module in modules]
    try:
        for module, _ in policy_multipliers:
            module.multiplier = 0.0
        for module, _ in ema_multipliers:
            module.multiplier = 1.0
        policy_network.eval()
        ema_network.eval()
        transformer.eval()
        with torch.no_grad():
            yield
    finally:
        for module, multiplier in (*policy_multipliers, *ema_multipliers):
            module.multiplier = multiplier
        for module, was_training in training:
            module.training = was_training
