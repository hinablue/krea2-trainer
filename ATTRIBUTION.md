# Attribution and license notes

This standalone trainer was extracted from `kohya-ss/musubi-tuner` for Krea2 LoRA training.

Upstream README license summary at extraction time:

> Code under the `hunyuan_model` directory is modified from HunyuanVideo and follows their license.
> Code under the `hunyuan_video_1_5` directory is modified from HunyuanVideo 1.5 and follows their license.
> Code under the `wan` directory is modified from Wan2.1. The license is under the Apache License 2.0.
> Code under the `frame_pack` directory is modified from FramePack. The license is under the Apache License 2.0.
> Other code is under the Apache License 2.0. Some code is copied and modified from Diffusers.

This repository keeps a small `hunyuan_model` compatibility subset because upstream shared cache helpers import it at module load time. The public CLI surface is Krea2-only.
