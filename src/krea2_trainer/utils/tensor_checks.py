"""Batched value checks without a host synchronization for every tensor."""

from collections import defaultdict

import torch


@torch.no_grad()
def all_finite(values) -> bool:
    groups = defaultdict(list)
    for value in values:
        groups[(value.device, value.dtype)].append(value.detach())
    for (device, dtype), tensors in groups.items():
        if dtype.is_floating_point and all(tensor.numel() for tensor in tensors):
            # Infinity norms cannot overflow on finite values, unlike L2 norms.
            with torch.autocast(device_type=device.type, enabled=False):
                maxima = torch.stack(torch._foreach_norm(tensors, float("inf")))
            valid = torch.isfinite(maxima).all()
        else:
            valid = torch.stack([torch.isfinite(tensor).all() for tensor in tensors]).all()
        if not bool(valid):
            return False
    return True
