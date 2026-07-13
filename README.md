# Krea2 Trainer

Krea2 Trainer 是從 [`kohya-ss/musubi-tuner`](https://github.com/kohya-ss/musubi-tuner) 抽離出來的 **Krea 2 專用 LoRA 訓練器**。

這不是把 `krea2_train_network.py` 硬改成單檔訓練器，而是保留 musubi-tuner 裡 Krea2 需要的共用訓練骨架，裁掉其他模型架構的公開入口，整理成可獨立安裝與執行的 package。

目標是做成一個乾淨的 **Krea2-only pruned standalone trainer**：

- 專注 Krea 2 LoRA training。
- 保留原 musubi-tuner 的 shared trainer / dataset / network 邏輯。
- 支援 latent cache、text encoder cache、RAW DiT LoRA training、Turbo sampling。
- 支援 Advanced Optimizers。
- 不支援非 Krea2 架構，不把訓練邏輯重寫成另一套。

---

## 專案狀態

目前版本：`0.1.0`

這份 trainer 適合第一版 standalone 工作流：

1. 建立資料集設定檔 `dataset.toml`
2. 快取 Qwen-Image VAE latents
3. 快取 Qwen3-VL text encoder outputs
4. 使用 Krea2 RAW DiT 訓練 LoRA
5. 可選擇在 sample 階段使用 Krea2 Turbo DiT 產圖

---

## 目錄結構

```text
krea2-trainer/
  pyproject.toml
  README.md
  ATTRIBUTION.md

  src/krea2_trainer/
    scripts/
      train_lora.py
      cache_latents.py
      cache_text_encoder.py

    krea2/
      krea2_mmdit.py
      krea2_encoder.py
      krea2_sampling.py
      krea2_utils.py

    qwen_image/
      qwen_image_autoencoder_kl.py
      qwen_image_utils.py
      qwen_image_model.py
      qwen_image_modules.py

    training/
      trainer_base.py
      parser_common.py
      accelerator_setup.py
      sampling_prompts.py
      timesteps.py

    dataset/
      architectures.py
      bucket.py
      cache_io.py
      config_utils.py
      datasources.py
      image_video_dataset.py
      media_utils.py

    networks/
      lora.py
      lora_krea2.py
      loha.py
      lokr.py
      network_arch.py

    modules/
      attention.py
      custom_offloading_utils.py
      fp8_optimization_utils.py
      lr_schedulers.py
      scheduling_flow_match_discrete.py

    utils/
      model_utils.py
      train_utils.py
      safetensors_utils.py
      lora_utils.py
      sai_model_spec.py
      huggingface_utils.py
      image_utils.py
      device_utils.py
```

`hunyuan_model/` 目前保留一小部分 compatibility subset，主要是因為 upstream shared cache helper 在 module import 階段仍會引用它。公開 CLI 仍然只針對 Krea2。

---

## 安裝

建議使用 Python 3.10～3.12。

### 使用 uv 安裝

CUDA 版本擇一安裝。

```bash
cd ~/Workspace/krea2-trainer
uv sync --extra cu128
```

或 CUDA 12.4：

```bash
uv sync --extra cu124
```

或 CUDA 13.0：

```bash
uv sync --extra cu130
```

### 使用 pip editable 安裝

```bash
cd ~/Workspace/krea2-trainer
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

如果要手動安裝 PyTorch，請依照你的 CUDA 環境安裝對應 wheel。

---

## 依賴

主要依賴包含：

- `torch`
- `torchvision`
- `accelerate==1.6.0`
- `bitsandbytes`
- `diffusers==0.32.1`
- `transformers==4.57.6`
- `safetensors==0.4.5`
- `adv_optm>=2.5.12`

---

## CLI 指令

安裝後會提供三個 console scripts：

```bash
krea2-cache-latents
krea2-cache-text
krea2-train-lora
```

也可以用 module 方式執行：

```bash
python -m krea2_trainer.scripts.cache_latents
python -m krea2_trainer.scripts.cache_text_encoder
python -m krea2_trainer.scripts.train_lora
```

訓練時通常搭配 `accelerate launch`：

```bash
accelerate launch -m krea2_trainer.scripts.train_lora ...
```

---

## 需要準備的模型檔

### 1. Krea2 RAW DiT

用於 LoRA 訓練。

參數：

```bash
--dit models/krea2/raw.safetensors
```

standalone 版也提供 alias：

```bash
--raw_dit models/krea2/raw.safetensors
```

`--raw_dit` 等同於 `--dit`。

### 2. Qwen-Image VAE

Krea2 使用 Qwen-Image VAE 做 latent encode/decode。

參數：

```bash
--vae models/qwen_image_vae.safetensors
```

### 3. Qwen3-VL text encoder

用於快取 Krea2 需要的 text hidden-state stack。

參數：

```bash
--text_encoder models/qwen3_vl_4b.safetensors
```

### 4. Krea2 Turbo DiT（選用）

推薦 workflow 是：

- RAW DiT：訓練 LoRA
- Turbo DiT：訓練中 sample / inference

參數：

```bash
--turbo_dit models/krea2/turbo.safetensors
```

可以加上：

```bash
--turbo_dit_cache
```

這會把 Turbo weights 留在 CPU RAM 中，sample 時比較快，但會增加 CPU 記憶體使用量。

---

## 資料集設定

Krea2 Trainer 沿用 musubi-tuner 的 TOML dataset config 流程。

基本概念：

- 圖片資料集由 `dataset.toml` 指定。
- 圖片 caption 可以來自 caption 檔或資料集設定。
- latent cache 與 text encoder cache 會依照 dataset config 產生對應 cache 檔。
- Krea2 是 image-only 工作流，不支援 video dataset。

簡化範例：

```toml
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "/path/to/images"
cache_directory = "/path/to/cache"
```

實際可用欄位仍以 upstream musubi-tuner dataset parser 支援的格式為準。

### Structure–Detail TQD（選用）

Krea2 是 image-only trainer，因此 TQD 使用 `structure_score` 取代影片 motion quality：它描述全局主體、姿勢與構圖是否可靠；`detail_score` 描述臉部、材質與局部紋理品質。分數必須在 `[0, 1]`，並以 JSONL manifest 提供給單一 dataset。

```toml
[[datasets]]
image_directory = "/path/to/images"
cache_directory = "/path/to/cache"
tqd_score_file = "/path/to/rosie_tqd_scores.jsonl"
```

`rosie_tqd_scores.jsonl` 每行一筆，`cache_file` 必須是 cache 目錄內的檔名，不可寫絕對路徑：

```json
{"cache_file":"rosie_0001_1024x1536_krea2.safetensors","structure_score":0.91,"detail_score":0.84}
{"cache_file":"rosie_0002_1024x1536_krea2.safetensors","structure_score":0.88,"detail_score":0.42}
```

啟動時：

```bash
accelerate launch -m krea2_trainer.scripts.train_lora \
  --raw_dit models/krea2/raw.safetensors \
  --vae models/qwen_image_vae.safetensors \
  --dataset_config dataset.toml \
  --output_dir output \
  --output_name my_krea2_tqd_lora \
  --timestep_sampling tqd_krea2_shift \
  --tqd_kappa_base 2 \
  --tqd_kappa_max 8
```

- structure 高於 detail 的圖會偏向高噪聲 timestep，強化構圖與語義結構。
- detail 高於 structure 的圖會偏向低噪聲 timestep，強化臉部、材質與局部細節。
- `structure_score == detail_score` 且 `--tqd_kappa_base 2` 時，會退化回 Krea2 原本的 pre-shift logit-normal sampling。
- 不能和 `--num_timestep_buckets` 同時使用，因為後者會預先指定與樣本無關的 timestep。
- `--tqd_quality_weighting` 會啟用 mean-one 的 `max(structure_score, detail_score)` loss weighting，近似論文中的 sample retention，但不改變每 epoch step 數。

---

## 標準訓練流程

### Step 1：快取 latents

```bash
krea2-cache-latents \
  --dataset_config dataset.toml \
  --vae models/qwen_image_vae.safetensors \
  --device cuda
```

這一步會：

- 讀取資料集圖片。
- 將圖片轉成 Qwen-Image VAE latents。
- 儲存 Krea2 訓練需要的 latent cache。

Krea2 latent 格式會走 Qwen-Image VAE normalization，圖片會轉成：

```text
(B, C, 1, H, W)
```

並 normalize 到 `[-1, 1]` 後進 VAE。

---

### Step 2：快取 text encoder outputs

```bash
krea2-cache-text \
  --dataset_config dataset.toml \
  --text_encoder models/qwen3_vl_4b.safetensors \
  --device cuda
```

這一步會：

- 讀取每張圖片的 caption。
- 使用 Qwen3-VL text encoder 編碼 prompt。
- 儲存 Krea2 DiT 需要的 selected-layer hidden-state stack。

Krea2 的 text cache 不是一般 CLIP/T5 embedding。它會保存 Qwen3-VL 多層 hidden states，形狀概念為：

```text
(B, seq, 12, 2560)
```

並且每筆資料只保存 non-padding tokens。

預設 selected layers：

```text
2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35
```

這部分不能簡化成一般 pooled embedding，否則 Krea2 DiT 的 text-fusion transformer 會吃不到正確輸入。

---

### Step 2.5：每個 epoch 重新快取 text encoder outputs（選用）

如果你希望 caption tag shuffle / dropout 在使用 text encoder cache 時仍然生效，可以在訓練時讓 trainer 於每個 epoch 開始前重建 Krea2 text encoder cache：

```bash
accelerate launch -m krea2_trainer.scripts.train_lora \
  --raw_dit models/krea2/raw.safetensors \
  --vae models/qwen_image_vae.safetensors \
  --text_encoder models/qwen3_vl_4b.safetensors \
  --dataset_config dataset.toml \
  --preset lora-default \
  --cache_te_every_epoch \
  --cache_te_shuffle_caption \
  --cache_te_caption_dropout_rate 0.1 \
  --cache_te_keep_tokens 1
```

行為：

- 在每個 epoch 的 dataloader 開始讀取前執行。
- 只有 main process 會重建 cache；DDP / multi-GPU 會在 cache 完成後同步等待。
- 每張圖的 caption 會依照 `seed + epoch + item_key` 做 deterministic augmentation。
- `--cache_te_keep_tokens` 會保留 caption 前 N 個 tag，不參與 shuffle/dropout。
- 預設用逗號 `,` 切 tag；可用 `--cache_te_caption_separator` 調整。
- 重建時會覆寫同一路徑的 `*_krea2_te.safetensors`。

常用參數：

```text
--cache_te_every_epoch
--cache_te_shuffle_caption
--cache_te_caption_dropout_rate 0.1
--cache_te_keep_tokens 1
--cache_te_caption_separator ","
--cache_te_batch_size 1
--cache_te_num_workers 8
--cache_te_device cuda
--cache_te_dtype bfloat16
```

注意：這會在每個 epoch 額外載入 Qwen3-VL text encoder 並重新 encode 全資料集；可以保留 caption 隨機性，但訓練時間會明顯增加。若 GPU VRAM 壓力太大，可用 `--cache_te_device cpu`，但速度會慢很多。

---

### Step 3：訓練 LoRA

最短推薦用法：

```bash
accelerate launch -m krea2_trainer.scripts.train_lora \
  --raw_dit models/krea2/raw.safetensors \
  --vae models/qwen_image_vae.safetensors \
  --dataset_config dataset.toml \
  --output_dir output \
  --output_name my_krea2_lora \
  --preset lora-default
```

`--preset lora-default` 會展開以下訓練預設：

```text
--sdpa
--mixed_precision bf16
--timestep_sampling shift
--weighting_scheme none
--optimizer_type adamw8bit
--learning_rate 1e-4
--gradient_checkpointing
--network_module krea2_trainer.networks.lora_krea2
--network_dim 32
--network_alpha 32
```

---

## 完整訓練範例

```bash
accelerate launch --num_cpu_threads_per_process 1 \
  -m krea2_trainer.scripts.train_lora \
  --raw_dit models/krea2/raw.safetensors \
  --vae models/qwen_image_vae.safetensors \
  --dataset_config dataset.toml \
  --preset lora-default \
  --max_train_epochs 16 \
  --save_every_n_epochs 1 \
  --output_dir output \
  --output_name my_krea2_lora
```

等同於較完整的展開寫法：

```bash
accelerate launch --num_cpu_threads_per_process 1 \
  -m krea2_trainer.scripts.train_lora \
  --dit models/krea2/raw.safetensors \
  --vae models/qwen_image_vae.safetensors \
  --dataset_config dataset.toml \
  --sdpa \
  --mixed_precision bf16 \
  --timestep_sampling shift \
  --weighting_scheme none \
  --discrete_flow_shift 2.5 \
  --optimizer_type adamw8bit \
  --learning_rate 1e-4 \
  --gradient_checkpointing \
  --network_module krea2_trainer.networks.lora_krea2 \
  --network_dim 32 \
  --network_alpha 32 \
  --max_train_epochs 16 \
  --save_every_n_epochs 1 \
  --output_dir output \
  --output_name my_krea2_lora
```

---

## W&B logging

Krea2 Trainer 支援透過 Hugging Face Accelerate tracker 將訓練 metrics、訓練設定、sample media 與 raw Python log file 送到 Weights & Biases。

### 安裝 W&B

如果使用 `uv sync`，加上 `tracking` extra：

```bash
uv sync --extra cu128 --extra tracking
```

或在現有環境中安裝：

```bash
uv add wandb
```

第一次使用前先登入：

```bash
wandb login
```

也可以在訓練時用 `--wandb_api_key` 指定 API key，或使用環境變數 `WANDB_API_KEY`。

### 啟用 W&B

```bash
accelerate launch --num_cpu_threads_per_process 1 \
  -m krea2_trainer.scripts.train_lora \
  --raw_dit models/krea2/raw.safetensors \
  --vae models/qwen_image_vae.safetensors \
  --dataset_config dataset.toml \
  --preset lora-default \
  --max_train_epochs 16 \
  --save_every_n_epochs 1 \
  --output_dir output \
  --output_name my_krea2_lora \
  --log_with wandb \
  --log_config \
  --log_tracker_name krea2-trainer \
  --wandb_run_name my_krea2_lora
```

會上傳的內容：

- step / epoch metrics，例如 loss、learning rate、gradient norm 等 tracker logs。
- `--log_config` 啟用後會上傳 sanitized training config；敏感 token 與本機路徑會被過濾。
- 若有設定 `--sample_prompts` 與 sample interval，sample image / video 會送到 W&B media panel。
- raw Python training log 會寫到 `train.log`，並在訓練結束時以 W&B artifact 上傳，artifact type 為 `training-log`。

### tracker config

可以用 TOML 指定 W&B project/entity/tags 等 `wandb.init()` 參數：

```toml
# wandb_tracker.toml
[wandb]
project = "krea2-trainer"
entity = "your-wandb-entity"
name = "my_krea2_lora"
tags = ["krea2", "lora"]
```

訓練時：

```bash
accelerate launch -m krea2_trainer.scripts.train_lora \
  ... \
  --log_with wandb \
  --log_config \
  --log_tracker_config wandb_tracker.toml
```

### log 檔位置

如果有指定 `--logging_dir`，raw log 會與 tracker run directory 放在同一個 timestamp 子目錄：

```bash
--logging_dir logs --log_prefix krea2_
# logs/krea2_YYYYMMDDHHMMSS/train.log
```

如果只使用 `--log_with wandb`、沒有指定 `--logging_dir`，raw log 會寫到：

```text
<output_dir>/logs/YYYYMMDDHHMMSS/train.log
```

若需要同時保留 TensorBoard 與 W&B：

```bash
--logging_dir logs --log_with all --log_config
```

---

## Turbo sampling

訓練 LoRA 時，可以用 RAW DiT 訓練，但在 sample 階段暫時換成 Turbo DiT 產圖。

```bash
accelerate launch -m krea2_trainer.scripts.train_lora \
  --raw_dit models/krea2/raw.safetensors \
  --turbo_dit models/krea2/turbo.safetensors \
  --vae models/qwen_image_vae.safetensors \
  --dataset_config dataset.toml \
  --sample_prompts sample_prompts.txt \
  --preset lora-default \
  --output_dir output \
  --output_name my_krea2_lora
```

可選擇啟用 Turbo cache：

```bash
--turbo_dit_cache
```

注意：

```text
--turbo_dit 不可與 --blocks_to_swap 同時使用
```

原因是 block-swap offloader 有自己的 CPU master weights。外部 RAW/Turbo swap 可能導致 RAW/Turbo 權重混用，所以 trainer 會直接阻止這種組合。

---

## fp8 設定

Krea2 支援 dynamic scaled fp8，但不支援單獨 plain fp8。

正確用法：

```bash
--fp8_base --fp8_scaled
```

錯誤用法：

```bash
--fp8_base
```

如果只設定 `--fp8_base`，trainer 會丟出錯誤。

原因是 Krea2 plain fp8 會把包含 norm 在內的部分轉成 fp8，容易導致模型壞掉。這個 standalone trainer 保留了 upstream 的保護條件。

---

## LoRA target layers

Krea2 預設 LoRA target 是 DiT 裡所有 `Linear` layers。

這包含：

- attention projections
- MLP layers
- text-fusion transformer
- time / text projection MLPs

預設 rank / alpha：

```bash
--network_dim 32
--network_alpha 32
```

這是 Krea2 model authors 推薦的預設方向。

### 自訂 LoRA target

可以透過 `--network_args` 使用 include / exclude patterns。

範例：排除特定模組：

```bash
--network_args "exclude_patterns=['.*\\.mlp\\..*']"
```

範例：先排除全部，再指定 include：

```bash
--network_args "exclude_patterns=['.*']" "include_patterns=['.*attn.*']"
```

---

## Advanced Optimizers 支援

本專案支援 [`adv_optm`](https://pypi.org/project/adv-optm/) 提供的 Advanced Optimizers。

可用 optimizer aliases：

```text
AdamW_adv
Prodigy_adv
Adopt_adv
Lion_adv
Muon_adv
AdaMuon_adv
SignSGD_adv
SinkSGD_adv
```

### Prodigy_adv 範例

```bash
accelerate launch -m krea2_trainer.scripts.train_lora \
  --raw_dit models/krea2/raw.safetensors \
  --vae models/qwen_image_vae.safetensors \
  --dataset_config dataset.toml \
  --preset lora-default \
  --optimizer_type Prodigy_adv \
  --optimizer_args weight_decay=0.01 state_precision="'bf16_sr'" \
  --output_dir output \
  --output_name my_krea2_lora
```

### AdamW_adv 範例

```bash
--optimizer_type AdamW_adv \
--optimizer_args weight_decay=0.01 betas="(0.9, 0.999)" state_precision="'bf16_sr'"
```

### Lion_adv 範例

```bash
--optimizer_type Lion_adv \
--optimizer_args weight_decay=0.01 betas="(0.9, 0.99)"
```

### Muon_adv 範例

```bash
--optimizer_type Muon_adv \
--optimizer_args weight_decay=0.01
```

`--optimizer_args` 會用 Python `ast.literal_eval` 解析，所以字串值需要額外保留引號。

例如：

```bash
state_precision="'bf16_sr'"
orthogonal_gradient="'flattened'"
```

數值與 tuple 可以直接寫：

```bash
weight_decay=0.01 betas="(0.9, 0.999)"
```

---

## 自訂 optimizer class

除了內建 alias，也可以指定完整 import path。

例如：

```bash
--optimizer_type torch.optim.AdamW
```

或：

```bash
--optimizer_type adv_optm.optim.Prodigy_adv
```

如果 `--optimizer_type` 不包含 `.`，且不是內建 alias，trainer 會嘗試從 `torch.optim` 尋找同名 optimizer。

---

## sample prompts

如果要訓練中定期 sample，需要提供 sample prompt 檔案：

```bash
--sample_prompts sample_prompts.txt
```

Krea2 sample prompt 會在訓練前用 Qwen3-VL text encoder 預先編碼，然後釋放 text encoder，避免訓練期間長時間佔用 VRAM。

如果使用 `--turbo_dit`，sample 階段會：

1. 暫時把 RAW DiT base weights 換成 Turbo DiT。
2. 保留 LoRA hook 套在目前 DiT 上。
3. 用 Turbo sampling schedule 產圖。
4. sample 結束後恢復 RAW DiT weights。

---

## 常見指令組合

### 使用預設 AdamW8bit

```bash
accelerate launch -m krea2_trainer.scripts.train_lora \
  --raw_dit models/krea2/raw.safetensors \
  --vae models/qwen_image_vae.safetensors \
  --dataset_config dataset.toml \
  --preset lora-default \
  --output_dir output \
  --output_name krea2_lora_adamw8bit
```

### 使用 Prodigy_adv

```bash
accelerate launch -m krea2_trainer.scripts.train_lora \
  --raw_dit models/krea2/raw.safetensors \
  --vae models/qwen_image_vae.safetensors \
  --dataset_config dataset.toml \
  --preset lora-default \
  --optimizer_type Prodigy_adv \
  --learning_rate 1.0 \
  --optimizer_args weight_decay=0.01 d_coef=1.0 state_precision="'bf16_sr'" \
  --output_dir output \
  --output_name krea2_lora_prodigy_adv
```

### 使用 fp8 scaled

```bash
accelerate launch -m krea2_trainer.scripts.train_lora \
  --raw_dit models/krea2/raw.safetensors \
  --vae models/qwen_image_vae.safetensors \
  --dataset_config dataset.toml \
  --preset lora-default \
  --fp8_base \
  --fp8_scaled \
  --output_dir output \
  --output_name krea2_lora_fp8_scaled
```

### 使用 Turbo sampling

```bash
accelerate launch -m krea2_trainer.scripts.train_lora \
  --raw_dit models/krea2/raw.safetensors \
  --turbo_dit models/krea2/turbo.safetensors \
  --vae models/qwen_image_vae.safetensors \
  --dataset_config dataset.toml \
  --sample_prompts sample_prompts.txt \
  --preset lora-default \
  --output_dir output \
  --output_name krea2_lora_turbo_sample
```

---

## 不支援範圍

第一版 standalone trainer 不支援：

- 非 Krea2 架構
- video dataset
- control / edit dataset
- full fine-tune
- GUI
- DreamBooth-style 自動資料集 wizard
- 多概念資料集編輯器

這些功能不是不能做，而是刻意不放進第一版，避免 standalone trainer 變回完整 musubi-tuner。

---

## 設計保留點

這份 trainer 保留幾個 Krea2 重要行為：

- RAW DiT training + optional Turbo sampling。
- `--turbo_dit` 與 `--blocks_to_swap` 互斥。
- `--fp8_base` 必須搭配 `--fp8_scaled`。
- Qwen3-VL selected-layer text cache。
- Qwen-Image VAE latent cache。
- Krea2 LoRA 預設 target 全部 DiT `Linear` layers。
- LoRA output 使用 safetensors。
- shared trainer 仍負責 optimizer、scheduler、accelerate、checkpoint、metadata、save/load。

---

## 疑難排解

### `No module named adv_optm`

請安裝：

```bash
pip install adv_optm
```

程式內 import path 是：

```python
adv_optm.optim
```

### `--fp8_base` 報錯

Krea2 不允許只開 plain fp8。請改用：

```bash
--fp8_base --fp8_scaled
```

### `--turbo_dit` 與 `--blocks_to_swap` 衝突

這是預期行為。Turbo sampling 需要外部替換 base weights；block swap offloader 會管理自己的 CPU master weights。兩者同時使用可能造成 RAW/Turbo 權重混用。

### 找不到 cache

請確認已依序執行：

```bash
krea2-cache-latents ...
krea2-cache-text ...
```

並確認 `dataset.toml` 的 `cache_directory` 與訓練時使用的是同一份設定。

### optimizer 字串參數解析失敗

`--optimizer_args` 使用 `ast.literal_eval`，字串需要保留 Python 字串格式。

正確：

```bash
--optimizer_args state_precision="'bf16_sr'"
```

錯誤：

```bash
--optimizer_args state_precision=bf16_sr
```

---

## 授權與來源

本專案從 musubi-tuner 抽離而來，請保留 upstream attribution 與授權資訊。

upstream README 摘要：

- `hunyuan_model` 目錄修改自 HunyuanVideo，遵循其授權。
- `hunyuan_video_1_5` 目錄修改自 HunyuanVideo 1.5，遵循其授權。
- `wan` 目錄修改自 Wan2.1，Apache License 2.0。
- `frame_pack` 目錄修改自 FramePack，Apache License 2.0。
- 其他程式碼為 Apache License 2.0，部分程式碼 copied and modified from Diffusers。

本 standalone repo 保留 `ATTRIBUTION.md` 說明來源與授權注意事項。
