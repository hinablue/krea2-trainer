"""Lightweight media-grid helpers kept for the shared trainer.

The original upstream video helper is Hunyuan/video-oriented and pulls in
large non-Krea2 dependencies at import time. Krea2 training only needs save_images_grid for
sample images; save_videos_grid is provided for API compatibility.
"""
import os
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image


def save_images_grid(imgs: torch.Tensor, path: str, rescale=False, n_rows=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imgs = imgs.detach().cpu()
    if imgs.ndim == 5:  # B,C,F,H,W -> use first/only frame
        imgs = imgs[:, :, 0]
    grid = torchvision.utils.make_grid(imgs, nrow=n_rows)
    if rescale:
        grid = (grid + 1.0) / 2.0
    grid = torch.clamp(grid, 0, 1)
    grid = grid.transpose(0, 1).transpose(1, 2).numpy()
    Image.fromarray((grid * 255).astype(np.uint8)).save(path)


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    # Krea2 standalone is image-only. For compatibility, write the first frame as an image
    # if a video helper is accidentally called with an image extension.
    if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        if videos.ndim == 5:
            videos = videos[:, :, 0]
        save_images_grid(videos, path, rescale=rescale, n_rows=n_rows)
        return
    raise NotImplementedError("Krea2 standalone trainer is image-only; video grid export is not supported.")
