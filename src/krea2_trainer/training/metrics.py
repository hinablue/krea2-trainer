"""Materialize detached scalar metrics with one device-to-host transfer."""

import torch


def materialize_metrics(metrics):
    if not metrics:
        return {}
    names = tuple(metrics)
    values = torch.stack([metrics[name].detach() for name in names]).cpu().tolist()
    return dict(zip(names, values))
