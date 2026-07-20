# Krea 2：LoRA 與基礎模型融合整理

> 更新日期：2026-07-20
> 適用專案：`krea2-trainer`
> 主題：Krea 2 RAW／Turbo、LoRA 權重格式、永久烘焙、INT8／NVFP4／GGUF 量化、Diffusers／Musubi／ComfyUI 相容性

## 結論先講

1. **Krea 官方建議是「在 RAW 訓練 LoRA，在 Turbo 推論」**。RAW 是未蒸餾、適合微調的 base；Turbo 是 8-step 蒸餾版。兩者架構相同，RAW 訓練出的 LoRA 預期可直接套到 Turbo。
2. 一般使用時，優先保留 **Turbo base + 獨立 LoRA**，或在模型載入時把 LoRA 合入記憶體。這樣可調權重、可替換 LoRA，也不會製造約 26 GB 的新完整模型。
3. 只有部署端不支援 LoRA、要降低執行時 adapter 開銷、或要再量化成單一模型檔時，才值得做**永久烘焙**。
4. 真正烘焙到基礎模型的公式是：

   ```text
   W_fused = W_base + multiplier × (alpha / rank) × (W_up @ W_down)
   ```

   多個 LoRA 則逐一累加：

   ```text
   W_fused = W_base + Σ_i multiplier_i × (alpha_i / rank_i) × (W_up_i @ W_down_i)
   ```

5. 不要把 `lora_down` 與 `lora_up` 各自直接相加，誤當成精確的 LoRA 融合；這會產生交叉項。若目標是完整模型，直接將每個 LoRA 的 delta 加到 base 最乾淨。
6. 本專案的訓練 CLI 與轉換工具分開維護；先前完成並實際使用過的 Krea 2 融合、Diffusers→ComfyUI、NVFP4 與 GGUF repack Python 工具已收進 `tools/`。這些是部署工具，不納入 trainer 的固定 `diffusers==0.32.1` 執行環境。

---

## 1. Krea 2 模型家族

### 1.1 RAW

Krea 2 RAW 是未蒸餾的基礎 checkpoint，官方定位為：

- LoRA training
- fine-tuning／post-training
- 研究與需要較高可塑性的工作

官方單檔名稱為 `raw.safetensors`。本專案使用的對應檔案是：

```text
models/krea2-raw.safetensors
```

本機實際檢查結果：

- 檔案大小：`26,283,332,608 bytes`
- tensor 數：`430`
- native Krea／ComfyUI 命名，例如：
  - `blocks.0.attn.qknorm.qnorm.scale`
  - `blocks.0.mod.lin`
  - `blocks.0.prenorm.scale`
- normalization／modulation 等部分 tensor 是 F32；不能把整個模型無差別強制轉成 BF16／FP8。

### 1.2 Turbo

Krea 2 Turbo 是由 RAW 後訓練與蒸餾而來的 few-step checkpoint：

- 推薦 8 steps
- 官方 pipeline 使用 `guidance_scale=0.0`
- 官方 reference CLI 使用固定 `mu=1.15`
- 官方宣稱可在約 1K～2K 解析度生成

最重要的是：**RAW 與 Turbo 被設計成一組工作流，LoRA 在 RAW 訓練，於 Turbo 使用。**

### 1.3 架構重點

Krea 2 是 flow-matching text-to-image model，主要元件為：

- 約 12B／13B 級 dense single-stream MMDiT
- Grouped-Query Attention
- Qwen3-VL text encoder
- 從 12 個 decoder layer 取每個 token 的 hidden states，再由 DiT 內的 text-fusion stage 整合
- Qwen-Image VAE

因此 Krea 2 LoRA 主要作用在 DiT／transformer，不應把它當成 SDXL UNet LoRA，用一般 SDXL key conversion 硬套。

---

## 2. 本專案產出的 LoRA 格式

`krea2-trainer` 從 Musubi 的 Krea 2 trainer 抽離，預設 target 是 DiT 裡所有 `Linear`：

- 28 個主 blocks 的 attention
- MLP
- text-fusion transformer
- first／last projection
- time／text projection MLP

總計 264 個 target modules。modulation 與 RMSNorm 是 raw parameter，不是 `Linear`，所以不會被 LoRA 包覆。

本機抽查：

```text
output/hina_krea2_tqd_lora_v2.safetensors
```

結果：

- 檔案大小：`234,706,184 bytes`
- 共 `792` 個 tensors
- 全部為 BF16
- `264 × lora_down`
- `264 × lora_up`
- `264 × alpha`
- rank：`32`
- alpha tensor：全部為 `16`

也就是這個特定 LoRA 在 multiplier 為 1 時，每層有效 scale 是：

```text
alpha / rank = 16 / 32 = 0.5
```

範例 keys：

```text
lora_unet_blocks_0_attn_wq.lora_down.weight
lora_unet_blocks_0_attn_wq.lora_up.weight
lora_unet_blocks_0_attn_wq.alpha

lora_unet_blocks_0_mlp_down.lora_down.weight
lora_unet_blocks_0_mlp_down.lora_up.weight
lora_unet_blocks_0_mlp_down.alpha
```

注意：

- `ss_network_alpha` metadata 只是資訊來源之一；真正融合前仍應檢查每個 `.alpha` tensor。
- 不可假定所有 LoRA 都是 rank 32／alpha 32。
- ai-toolkit 可能使用 `diffusion_model.*.lora_A/lora_B.weight`，並把 alpha 放在 metadata；Musubi／Kohya 則常見 `lora_unet_*.lora_down/lora_up + .alpha`。
- 載入器必須辨識格式，並保證 `alpha/rank` **只套一次**。

---

## 3. 「融合」其實有四種不同意思

### 3.1 執行時掛載 LoRA

base 與 LoRA 保持為兩個檔案，forward 時動態計算：

```text
y = W_base x + multiplier × (alpha/rank) × W_up(W_down x)
```

優點：

- 可隨時切換／停用 LoRA
- 可調 multiplier
- 可同時組合多個 LoRA
- 不需要再存一份完整 26 GB 模型

缺點：

- loader 必須理解該 LoRA 格式
- 有少量執行時 adapter 開銷
- 某些量化／FP8 路徑不適合保留動態 hook

這是 ComfyUI 一般 `Load LoRA` 工作流的概念，也是多數日常推論最合理的方式。

### 3.2 載入時合入記憶體

模型與 LoRA 檔案仍分開，但初始化時把 delta 加進目前記憶體中的 base weights：

```text
load base → merge LoRA delta → inference
```

Musubi 的 `krea2_generate_image.py` 就採這條路，並明確指出在 FP8 路徑中，LoRA 應在載入時合入 base，再進行後續量化／推論。

優點：

- 不改動磁碟上的原始 base
- 推論期沒有 adapter hook
- 適合 FP8／block-swap loader 的模型建立流程

缺點：

- 每次啟動仍要重做融合
- 換 multiplier 或 LoRA 通常要重新載入模型

### 3.3 永久烘焙成完整 checkpoint

把已融合的 transformer state dict 寫成新的 `.safetensors` 或 Diffusers transformer directory：

```text
Turbo + 0.8 × LoRA → Krea 2 Turbo fused checkpoint
```

優點：

- 部署端只需一個 transformer
- 適合後續 INT8／FP8／NVFP4／GGUF 量化
- 不需依賴 LoRA loader

缺點：

- 檔案很大
- 不可調強度
- 來源與係數若沒記錄，之後難追溯
- BF16 寫回有捨入誤差，通常不能再「無損拆回」原始 base

### 3.4 LoRA 與 LoRA 合成另一個 LoRA

這和「LoRA 烘焙到 base」不同。

若直接做：

```text
down = wa × down_A + wb × down_B
up   = wa × up_A   + wb × up_B
```

則：

```text
up @ down
```

會包含 A/B 交叉項，不等於：

```text
wa × (up_A @ down_A) + wb × (up_B @ down_B)
```

要得到精確 delta 和，需做 rank concatenation：

```text
D_out = concat(D_A, D_B, dim=0)
U_out = concat(
  U_A × wa × alpha_A/rank_A,
  U_B × wb × alpha_B/rank_B,
  dim=1
)
alpha_out = rank_out
```

代價是輸出 rank 變成 `rank_A + rank_B`。若最後目標本來就是完整 checkpoint，通常直接把每個 delta 加到 base 比先做 LoRA-to-LoRA 更簡單也更精確。

---

## 4. 建議工作流

### 4.1 日常生成：保留 Turbo + LoRA

推薦：

```text
Krea 2 Turbo
  + LoRA(safetensors)
  + multiplier（常見從 0.6～1.0 試起）
  → 8-step inference
```

優先理由：

- 保留調整空間
- 可對不同 epoch checkpoint 做比較
- 可與其他 LoRA 組合
- 不增加大型完整模型檔

ComfyUI 官方文件也採 base model + style LoRA 的組法。

### 4.2 FP8 或單程序服務：載入時融合

Musubi 形式的概念命令：

```bash
python src/musubi_tuner/krea2_generate_image.py \
  "A portrait of the trained subject" \
  --dit /path/to/turbo.safetensors \
  --vae /path/to/qwen_image_vae.safetensors \
  --text_encoder /path/to/qwen3vl_4b.safetensors \
  --steps 8 \
  --guidance_scale 1 \
  --mu 1.15 \
  --lora_weight /path/to/lora.safetensors \
  --lora_multiplier 0.8 \
  --save_path /path/to/output
```

注意 Musubi 的 `guidance_scale` 定義與官方 Krea CLI 不同：

- 官方 Krea／Diffusers Turbo：`guidance_scale=0.0`
- Musubi 標準 CFG 表示法：`guidance_scale <= 1` 關閉 CFG，因此 Turbo 範例用 `1`
- RAW 的官方 guidance 4.5 對 Musubi 約等於 CFG scale 5.5

不要只抄數字而忽略 loader 的 CFG 定義。

### 4.3 要量化或部署單檔：先烘焙 BF16，再量化

推薦順序：

```text
原始 Turbo BF16
  → 以 FP32 計算 LoRA delta
  → 加入 base
  → 依目標格式寫回 BF16
  → 完整回讀驗證
  → 再做 INT8／FP8／NVFP4／GGUF
```

不要先把 base 量化，再嘗試用一般矩陣加法烘焙 LoRA；量化權重通常需要特定 dequantize／requantize 流程，而且誤差更難控制。

---

## 5. 實作永久融合時的核心演算法

對每個 LoRA module：

```python
rank = down.shape[0]
scale = multiplier * alpha / rank
delta = up.float() @ down.float()
base_weight = base_weight.float() + scale * delta
output_weight = base_weight.to(output_dtype)
```

多 LoRA：

```python
fused = base_weight.float()
for lora, multiplier in loras:
    rank = lora.down.shape[0]
    scale = multiplier * lora.alpha / rank
    fused.addmm_(lora.up.float(), lora.down.float(), beta=1.0, alpha=scale)
output = fused.to(output_dtype)
```

### 必須遵守的細節

1. 用 FP32 做 `up @ down` 與累加。
2. 每個 module 檢查：
   - down／up 是否成對
   - rank dimension 是否相符
   - base target shape 是否相符
   - alpha 是否存在；不存在時的預設規則要明確
3. 輸出 dtype 要依 base tensor 分類保留：
   - 大型 linear weights 可依目標使用 BF16
   - Krea 2 必須保留的 norm／scale／modulation F32 不可全域降精度
4. 不可就地覆蓋原始 RAW／Turbo。
5. 先寫到暫存檔，驗證成功後再 atomic rename。
6. metadata／旁車 JSON 至少記錄：
   - base 路徑與 SHA-256
   - LoRA 路徑與 SHA-256
   - 每個 multiplier
   - rank／alpha／格式
   - 融合公式
   - 輸出 dtype
   - 工具版本與日期

---

## 6. Key namespace 與格式相容性

### 6.1 本專案／Musubi LoRA

常見格式：

```text
lora_unet_blocks_0_attn_wq.lora_down.weight
lora_unet_blocks_0_attn_wq.lora_up.weight
lora_unet_blocks_0_attn_wq.alpha
```

### 6.2 Krea native／ComfyUI base

常見 base keys：

```text
blocks.0.attn.wq.weight
blocks.0.attn.wk.weight
blocks.0.attn.wv.weight
blocks.0.attn.wo.weight
blocks.0.mlp.gate.weight
blocks.0.mlp.up.weight
blocks.0.mlp.down.weight
```

另有：

```text
first.weight
last.linear.weight
tmlp.*
txtmlp.*
tproj.*
txtfusion.*
```

融合工具應透過與訓練器相同的 model/module walker 建立映射，而不是用全域字串取代猜 key。`lora_unet_` 中的底線既可能是 module separator，也可能是原 module 名的一部分，盲目 `replace("_", ".")` 不可靠。

### 6.3 ai-toolkit／PEFT

可能看到：

```text
diffusion_model.blocks.0.attn.wq.lora_A.weight
diffusion_model.blocks.0.attn.wq.lora_B.weight
```

或：

```text
base_model.model.blocks.0.attn.wq.lora_A.weight
base_model.model.blocks.0.attn.wq.lora_B.weight
```

alpha 可能只有 safetensors metadata `ss_network_alpha`，沒有每層 `.alpha` tensor。

### 6.4 Diffusers

Diffusers 0.39 的 Krea 2 loader 已包含 non-Diffusers Krea 2 LoRA conversion，能映射：

- `blocks.N.attn.wq/wk/wv/wo/gate`
- `blocks.N.mlp.gate/up/down`
- `first`
- `last.linear`
- `tmlp`
- `txtmlp`
- `tproj`
- `txtfusion`

到 `Krea2Transformer2DModel` 的 Diffusers namespace。

若使用 Diffusers 融合：

1. 只載入 `Krea2Transformer2DModel` 即可；若只是烘焙 transformer，沒必要同時載入 VAE 與 Qwen3-VL。
2. 載入 base 時用 `torch_dtype=torch.bfloat16`，不要再對整個 transformer 呼叫無差別的 `.to(dtype=bf16)`，以免應保留 FP32 的 norm 被改掉。
3. 載入 LoRA，確認 adapter 確實出現在 PEFT config。
4. `fuse_lora(..., safe_fusing=True)`。
5. unload adapter wrapper。
6. `save_pretrained()` 或另行轉成 Krea native／ComfyUI single-file。

概念範例：

```python
import torch
from diffusers import Krea2Pipeline

pipe = Krea2Pipeline.from_pretrained(
    "krea/Krea-2-Turbo",
    torch_dtype=torch.bfloat16,
)
pipe.load_lora_weights("/path/to/lora.safetensors", adapter_name="subject")
pipe.set_adapters(["subject"], adapter_weights=[0.8])
pipe.fuse_lora(adapter_names=["subject"], safe_fusing=True)
pipe.unload_lora_weights()
pipe.save_pretrained("/path/to/fused-krea2-turbo")
```

這是 API 方向示例，不代表本專案目前鎖定的 `diffusers==0.32.1` 已支援 Krea 2。Krea 2 pipeline／loader 需使用支援 Krea 2 的新版本（目前文件對應 Diffusers 0.39／main）。不要為了融合直接升級訓練環境而破壞 trainer；較安全做法是另建 conversion venv／container。

---

## 7. 多 LoRA 融合策略

### 7.1 最穩定：逐一加到 base

```text
Turbo + 0.8 × Character + 0.35 × Style
```

數學上直接累加兩個 delta：

```text
W' = W + 0.8ΔW_character + 0.35ΔW_style
```

不同 rank／alpha 也沒關係，只要各自正確換算 `alpha/rank` 並能映射到同一 base module。

### 7.2 多 adapter 動態組合

Diffusers／PEFT 可透過 `set_adapters()` 指定多個 adapter 與權重。這保留彈性，適合先做視覺 sweep，再決定是否烘焙。

### 7.3 選 multiplier 的實驗方式

建議固定：

- prompt
- seed
- resolution
- sampler／steps／CFG／mu

做矩陣測試：

```text
0.4, 0.6, 0.8, 1.0, 1.2
```

觀察：

- 身分／風格是否足夠
- prompt adherence 是否下降
- 五官、手部、材質是否開始崩壞
- Turbo 蒸餾風格是否被過度覆蓋

選定係數後才烘焙。烘焙不是找最佳係數的工具，而是部署決策的最後一步。

---

## 8. INT8、NVFP4 與 GGUF 量化

### 8.1 共通原則：先融合，再量化

三種格式都遵循同一條主線：

```text
Krea 2 Turbo／已融合 Diffusers base
  → 在 BF16／FP32 計算域烘焙 LoRA
  → 轉成單一 ComfyUI-native mixed BF16/F32 safetensors
  → 驗證 430 個邏輯 tensors、key、dtype、finite 與抽樣數值
  → 由同一份 BF16 來源分支轉 INT8／NVFP4／GGUF
  → 用實際 ComfyUI loader 驗證
```

不要直接把 LoRA 烘焙進已量化檔案。INT8、NVFP4 與 GGUF 都不是普通浮點 state dict；若先量化再融合，就必須逐層 dequantize、加入 delta、重新估 scale 並 requantize。這不但不會保留原量化誤差特性，也容易漏掉 companion tensors 或破壞格式 metadata。

Krea 2 單檔的原生 ComfyUI signature 應包含：

```text
txtfusion.projector.weight
first.weight
blocks.0.attn.wq.weight
blocks.27.mlp.down.weight
last.linear.weight
```

標準 transformer 目前是 `430` 個邏輯 tensors。這個數字可作為目前架構的強相容性檢查，但若上游模型架構更新，不應盲目硬套。

### 8.2 INT8：ComfyUI ConvRot

本機已驗證的路徑使用 Comfy-Org `comfy-model-tools` 的 `quant_int8_convrot.py`。來源必須是**單一 ComfyUI-native Krea 2 BF16 safetensors**；若來源仍是 Diffusers shards，應先用本文件的 conversion 工具轉成 native layout。

先 dry-run：

```bash
python /path/to/comfy-model-tools/quant_int8_convrot.py \
  output/Krea-2-fused-ComfyUI-BF16.safetensors \
  --dry-run
```

再量化並寫逐層報告：

```bash
python /path/to/comfy-model-tools/quant_int8_convrot.py \
  output/Krea-2-fused-ComfyUI-BF16.safetensors \
  output/Krea-2-fused-ComfyUI-INT8-ConvRot.safetensors \
  --verify-report output/Krea-2-fused-ComfyUI-INT8-ConvRot.verify.tsv
```

每個被量化的 linear 會形成一組：

```text
<base>.weight          I8
<base>.weight_scale    F32
<base>.comfy_quant     U8 JSON payload
```

驗證重點：

1. I8 weight、F32 scale、U8 config 數量相等。
2. `.comfy_quant` 可解碼，格式為 `int8_tensorwise`，ConvRot 啟用且 groupsize 符合 dry-run 計畫。
3. 將 `.weight_scale` 與 `.comfy_quant` 正規化回 `.weight` 後，邏輯 key set 要與 BF16 來源完全一致。
4. 解析 TSV，報告 layer count、平均／最大 relative error、最低 cosine 與超出 warning threshold 的層數。
5. 用 ComfyUI `nodes.UNETLoader().load_unet()` 實載，類別鏈應為：

   ```text
   ModelPatcher → Krea2 → SingleStreamDiT
   ```

標準 Krea 2 的代表性 dry-run 會選中 `240` 個 indexed-block linears、groupsize `256`，輸出約 `910` 個實體 tensors；這是本機目前工具版本的相容性預期，不是永久固定規格。

一般 INT8 可在 RTX 30／40／50 系列使用。第三方 model card 指出原生 row-wise INT8 與 ConvRot INT8 的 loader 支援狀態可能不同，所以實際部署必須以當前 ComfyUI build 與確切 loader 測試為準，不能只看副檔名。

### 8.3 NVFP4：Blackwell 原生路徑

本專案收錄的工具：

```text
tools/quant_krea2_nvfp4.py
```

它以 `comfy-kitchen` 的 `TensorCoreNVFP4Layout` 逐層量化，並保留 Krea 2 敏感路徑。原生 NVFP4 Tensor Core 加速需要 Blackwell，工具預設要求 CUDA capability `SM >= 10.0`；`--allow-non-blackwell` 只適合「在舊卡建立檔案、拿到 Blackwell 機器使用」，不代表舊卡能得到原生 NVFP4 加速。

先 dry-run：

```bash
python tools/quant_krea2_nvfp4.py \
  output/Krea-2-fused-ComfyUI-BF16.safetensors \
  output/Krea-2-fused-ComfyUI-NVFP4.safetensors \
  --profile krea2-safe \
  --device cuda \
  --dry-run
```

再執行並驗證：

```bash
python tools/quant_krea2_nvfp4.py \
  output/Krea-2-fused-ComfyUI-BF16.safetensors \
  output/Krea-2-fused-ComfyUI-NVFP4.safetensors \
  --profile krea2-safe \
  --device cuda \
  --verify \
  --verify-report output/Krea-2-fused-ComfyUI-NVFP4.verify.tsv \
  --sha256
```

每個 NVFP4 logical weight 會形成：

```text
<base>.weight
<base>.weight_scale
<base>.weight_scale_2
<base>.comfy_quant
```

比較邏輯 key 時，要把三種 companion suffix 都重建為 `<base>.weight`：

```python
re.sub(r"\.(weight_scale_2|weight_scale|comfy_quant)$", ".weight", key)
```

不要連續使用 `removesuffix()`；那會把 companion key 錯誤壓成 module base，產生假的 key mismatch。

`krea2-safe` profile 只量化符合條件的 indexed transformer linears，並保留 top-level、final、text projector／refiner、norm、modulation 等敏感路徑。實際驗證仍須包含：完整 logical key set、四件式 companion 數量、TSV 重建誤差，以及 ComfyUI `UNETLoader` 實載。

### 8.4 GGUF：Q4_K 與 ComfyUI-GGUF

GGUF 是推論封裝與量化格式，不是 Krea 2 的訓練 checkpoint。不要拿 GGUF 當 LoRA 訓練 base，也不要假設它能無損轉回原始 BF16。

本機驗證過的 Q4_K 流程分兩階段：

1. 用 `stable-diffusion.cpp` 產生 raw Q4_K payload。
2. 用 `tools/repack_krea2_q4k_gguf.py` 改成 ComfyUI-GGUF 可辨識的 Krea 2 artifact。

第一階段：

```bash
sd-cli --mode convert \
  --diffusion-model output/Krea-2-fused-ComfyUI-BF16.safetensors \
  --output output/Krea-2-fused-Q4_K-raw.gguf \
  --type q4_K
```

第二階段：

```bash
python tools/repack_krea2_q4k_gguf.py \
  output/Krea-2-fused-Q4_K-raw.gguf \
  output/Krea-2-fused-ComfyUI-Q4_K.gguf \
  output/Krea-2-fused-ComfyUI-BF16.safetensors
```

Repack 不是只加一個 metadata 字串。它會：

- 移除 `model.diffusion_model.` prefix；
- 寫入 `general.architecture=krea2`；
- 從 BF16 source 還原每個 logical shape；
- 將所有 1D tensor、`*.mod.lin`、`last.modulation.lin`、`tmlp.*`、`tproj.*` 從 BF16 source 取回並保存為 F32；
- 正確傳入量化 payload 的 byte shape；
- 寫暫存檔後 atomic rename。

直接由 Diffusers index 轉出的 GGUF 可能沒有 `general.architecture`，並保留 `transformer_blocks.*`／`text_fusion.*` namespace。這種檔案即使 payload 合法，ComfyUI-GGUF 仍可能報 `Unknown model architecture`；只補 architecture 字串也不夠，key layout 與 logical shapes 必須一起正規化。

最終必須用 ComfyUI-GGUF 的 `UnetLoaderGGUF` 實載，預期類別鏈：

```text
GGUFModelPatcher → Krea2 → SingleStreamDiT
```

### 8.5 三種格式的選擇

| 格式 | 主要硬體／loader | 優點 | 限制 |
|---|---|---|---|
| INT8 ConvRot | 一般 NVIDIA RTX；ComfyUI native quant loader | 畫質穩定、相容硬體廣、約半個 BF16 大小 | 仍需確認當前 ComfyUI 是否支援該 ConvRot metadata |
| NVFP4 | Blackwell SM 10.0+；ComfyUI + comfy-kitchen | 約 4-bit payload、Blackwell 原生 Tensor Core 路徑 | 硬體與 runtime 限制最強；相對誤差通常高於 INT8 |
| GGUF Q4_K | ComfyUI-GGUF／stable-diffusion.cpp 生態 | 可變量化格式、低 VRAM、部署彈性 | 不是訓練格式；Krea 2 需要正確 architecture、native keys、shape 與高精度 tensor policy |

### 8.6 本機實測參考

以同一份已融合的 Krea 2 Turbo ComfyUI BF16 source 為例：

| 產物／報告 | 實測結果 |
|---|---|
| BF16 source | `25,640,953,288 bytes`（`23.880 GiB`） |
| INT8 ConvRot verify report | 240 layers；平均 relative error `0.91065%`；最大 `1.4766%`；最低 cosine `0.999891` |
| NVFP4 | `7,921,260,448 bytes`（`7.377 GiB`，BF16 的 `30.89%`） |
| NVFP4 verify report | 240 layers；平均 relative error `9.41023%`；最大 `9.53457%`；最低 cosine `0.99545407` |
| GGUF Q4_K | `8,314,582,016 bytes`（`7.744 GiB`，BF16 的 `32.43%`） |

這些數字只代表目前的 Krea 2 checkpoint、保留層策略與工具版本，不是所有模型的保證。Relative error 也不能單獨等同於最終影像品質；仍須固定 prompt／seed 做影像回歸，並以確切 loader 實載。

不論選哪個格式，都要保留未量化的 BF16 fused source、量化報告、來源與係數 metadata。量化檔案是部署分支，不是新的融合母版。

---

## 9. 驗證清單

### 9.1 融合前

- [ ] base 是 Krea 2 RAW 或 Turbo，不是其他架構
- [ ] 確認 LoRA 的來源 base
- [ ] 列出 tensor 數、dtype、rank、alpha
- [ ] down／up／alpha key 完整
- [ ] 所有 target 都能映射到 base
- [ ] shape 完全相容
- [ ] 確認 alpha 不會重複套用
- [ ] 記錄 multiplier

### 9.2 融合後檔案

- [ ] safetensors header 可解析
- [ ] 官方／目標 loader 可重新打開
- [ ] key set 與預期 base 完整一致
- [ ] 沒有殘留 `lora_*` keys
- [ ] tensor 全部 finite，沒有 NaN／Inf
- [ ] 必要的 F32 norm／scale／modulation 仍為 F32
- [ ] 至少抽查一個 target tensor 確實改變
- [ ] 抽查公式誤差在輸出 dtype 合理範圍內
- [ ] 未被 LoRA target 的 tensor 與 base 完全一致
- [ ] 輸出不是原始 base 路徑
- [ ] provenance metadata／JSON 已保存

### 9.3 影像回歸

最少比較：

1. 原始 Turbo baseline
2. Turbo + 動態 LoRA
3. 永久 fused Turbo

固定相同 prompt／seed／解析度／steps／CFG／mu。理想結果是 2 與 3 視覺一致；若要求數值比對，應直接比 transformer output／latent，而不是只看 PNG。

BF16 可能因 matmul、scale、add 的運算順序產生一個 BF16 unit 的差異。未完全重現 loader 的 dtype 與 operation order 時，不應宣稱 bit-exact，只應報最大誤差與容許範圍。

---

## 10. 授權注意事項

Krea 2 weights 使用 **Krea 2 Community License**，不是 Apache-2.0 模型權重授權；官方 GitHub 程式碼授權與模型 weights 授權要分開看。

永久融合後的完整 checkpoint 明確屬於 license 定義中的 derivative／merged model。依 2026-07-20 讀到的條款，散布 derivative 時至少涉及：

- 附上 Krea 2 Community License
- 要求接收者受條款約束
- 模型名稱以 `Krea` 開頭
- 保留指定 attribution notice
- 說明模型已被修改
- 不得宣稱是 Krea 官方或獲官方背書
- 商業使用受公司年營收門檻與 enterprise license 條件限制
- 部署需實作合理的 content filtering／review safeguards

LoRA 本身是否構成 derivative、散布時應附哪些條款，也應依最新 Krea license 與實際發佈方式判斷。公開發布前應重新讀最新條款；本節是工程整理，不是法律意見。

---

## 11. 本專案已收錄的 Python 工具

先前在 `/home/hina/Workspace/Krea2` 完成並實際跑過的工具已複製到本專案 `tools/`。原始工作區檔案沒有移除或覆寫。

### 11.1 工具索引

| 檔案 | 用途 | 主要輸入 | 主要輸出 |
|---|---|---|---|
| `tools/merge_krea2_lora.py` | 兩個 LoRA 的 key 檢查、Krea 2 namespace 雙向映射與 tensor-wise blend | 兩個 LoRA safetensors | 新 LoRA safetensors |
| `tools/merge_krea2_diffusion_model.py` | 將一個或多個 LoRA 依序永久烘焙到 Krea 2 transformer | Krea 2 Diffusers base + LoRA | Diffusers transformer 目錄或單一 ComfyUI BF16 safetensors |
| `tools/convert_fused_diffusers_to_comfy_krea2.py` | 將 sharded Diffusers Krea 2 transformer 映射成單一 ComfyUI-native checkpoint | Diffusers transformer shards/index | 430-tensor ComfyUI safetensors |
| `tools/quant_krea2_nvfp4.py` | Krea 2-aware NVFP4 dry-run、轉換與逐層驗證 | ComfyUI-native BF16/FP16 safetensors | NVFP4 safetensors + TSV report |
| `tools/repack_krea2_q4k_gguf.py` | 將 stable-diffusion.cpp raw Q4_K GGUF 重包成 ComfyUI-GGUF Krea 2 artifact | raw GGUF + BF16 shape/high-precision source | `general.architecture=krea2` 的 GGUF |

`quant_int8_convrot.py` 是 Comfy-Org `comfy-model-tools` 的上游 GPL-3.0 工具，不在這裡複製一份；文件固定連到上游，避免混淆來源、授權與版本。

### 11.2 LoRA 與 LoRA blend

`merge_krea2_lora.py` 能將 Musubi `lora_unet_*` 與 ai-toolkit `diffusion_model.*` key 正規化後比較，支援 missing-key policy、block filter、BF16 raw merge 與 dry-run：

```bash
python tools/merge_krea2_lora.py \
  --a output/subject.safetensors --wa 0.7 \
  --b output/style.safetensors --wb 0.3 \
  --key-map krea2-diffusion \
  --missing error \
  --out-dtype bf16 \
  --out output/subject-style.safetensors \
  --dry-run
```

移除 `--dry-run` 才會寫檔。這支工具的預設公式是 tensor-wise `A*wa + B*wb`；它會保留 rank，但對 down/up 各自相加會產生 cross terms，**不是精確的 delta addition**。需要精確多 LoRA 完整模型時，應直接用下一節將每個 LoRA delta 依序烘焙到 base。

### 11.3 將 LoRA 永久烘焙到 Krea 2 base

`merge_krea2_diffusion_model.py` 支援：

- Musubi／Kohya `lora_unet_*.lora_down/lora_up + .alpha`；
- ai-toolkit `diffusion_model.*.lora_A/lora_B`；
- safetensors metadata `ss_network_alpha`；
- `alpha/rank` 只 bake 一次；
- 多個 `--lora PATH SCALE` 依序融合；
- 單一 ComfyUI-native mixed BF16/F32 輸出；
- temp file + atomic rename；
- 430 tensor 與必要 Krea 2 signatures 檢查。

```bash
python tools/merge_krea2_diffusion_model.py \
  --base /path/to/Krea-2-Turbo-diffusers \
  --lora output/character.safetensors 0.8 \
  --lora output/style.safetensors 0.35 \
  --output output/Krea-2-character-style-ComfyUI-BF16.safetensors
```

若 `--output` 不是 `.safetensors`，工具會寫 sharded Diffusers transformer directory 與 `merge_info.json`。

這支工具需要支援 Krea 2 的 Diffusers；目前實測基準是 `diffusers==0.39.0`。本 trainer 的 `pyproject.toml` 仍鎖定 `diffusers==0.32.1`，不要直接升級訓練 venv。請使用獨立 conversion venv／container，並依 PEFT 檢查結果準備相容的 `torchao>=0.16`。

### 11.4 Diffusers shards 轉單一 ComfyUI Krea 2

若已有融合後的 Diffusers transformer shards：

```bash
python tools/convert_fused_diffusers_to_comfy_krea2.py \
  --src-dir /path/to/fused/transformer \
  --dst output/Krea-2-fused-ComfyUI-BF16.safetensors
```

`--src-dir` 應包含 `diffusion_pytorch_model.safetensors.index.json`。可加 `--oracle /path/to/known-good-krea2.safetensors`，比較正規化後的完整 logical key set。

### 11.5 NVFP4 與 Q4_K GGUF

NVFP4 使用方式見 8.3；工具預設不覆蓋既有輸出、拒絕已帶 `.comfy_quant`／`.weight_scale_2` 的來源，並支援 `--dry-run`、`--verify-report`、`--sha256`。

Q4_K 必須先由 `sd-cli` 產生 raw GGUF，再用 `repack_krea2_q4k_gguf.py RAW DST BF16_SOURCE`；第三個參數不是可省略的參考檔，它提供原始 logical shapes 與需要保留為 F32 的敏感 tensors。

### 11.6 相依套件與執行邊界

| 工具 | 額外需求 |
|---|---|
| `merge_krea2_lora.py` | `numpy`；Torch backend 另需 `torch` + `safetensors` |
| `merge_krea2_diffusion_model.py` | `torch`、`safetensors`、`peft`、Krea 2-capable `diffusers`；部分環境需相容 `torchao` |
| `convert_fused_diffusers_to_comfy_krea2.py` | `torch`、`safetensors` |
| `quant_krea2_nvfp4.py` | `torch`、`safetensors`、`comfy-kitchen`；原生加速需 Blackwell |
| `repack_krea2_q4k_gguf.py` | `gguf`、`torch`、`safetensors`，以及先行產生 raw GGUF 的 `sd-cli` |

所有大型轉換都應先檢查輸入 header、可用磁碟與輸出路徑，禁止覆蓋 BF16 母版。產物在結構驗證與 exact loader 實載成功前，不應複製到 ComfyUI 正式模型目錄。

---

## 12. 來源

### 官方／第一方

1. Krea 2 官方 GitHub
   <https://github.com/krea-ai/krea-2>
2. Krea 2 RAW model card
   <https://huggingface.co/krea/Krea-2-Raw>
3. Krea 2 Turbo model card
   <https://huggingface.co/krea/Krea-2-Turbo>
4. Krea 2 Open Source 頁面
   <https://www.krea.ai/krea-2-open-source>
5. Krea 2 Community License
   <https://www.krea.ai/krea-2-licensing>

### 實作文件／程式碼

6. Hugging Face Diffusers：Krea 2 pipeline
   <https://huggingface.co/docs/diffusers/main/en/api/pipelines/krea2>
7. Hugging Face Diffusers：Merge LoRAs
   <https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras>
8. Diffusers Krea 2 LoRA conversion source
   <https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/lora_conversion_utils.py>
9. Musubi Tuner Krea 2 文件
   <https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md>
10. Musubi Krea 2 inference／load-time merge
    <https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/krea2_generate_image.py>
11. ComfyUI Krea 2 workflow 文件
    <https://docs.comfy.org/tutorials/image/krea/krea-2>
12. Comfy-Org comfy-model-tools（INT8 ConvRot）
    <https://github.com/Comfy-Org/comfy-model-tools>
13. ComfyUI-GGUF
    <https://github.com/city96/ComfyUI-GGUF>
14. stable-diffusion.cpp
    <https://github.com/leejet/stable-diffusion.cpp>
15. 社群 Krea 2 INT8／NVFP4 model card（第三方實作與硬體提示）
    <https://huggingface.co/chfm/Krea-2-Base-Turbo-NVFP4-FP8-INT8>

### 本專案實際檢查

16. `README.md`
17. `src/krea2_trainer/networks/lora.py`
18. `src/krea2_trainer/networks/lora_krea2.py`
19. `models/krea2-raw.safetensors` header
20. `output/hina_krea2_tqd_lora_v2.safetensors` header／rank／alpha
21. `tools/merge_krea2_lora.py`
22. `tools/merge_krea2_diffusion_model.py`
23. `tools/convert_fused_diffusers_to_comfy_krea2.py`
24. `tools/quant_krea2_nvfp4.py`
25. `tools/repack_krea2_q4k_gguf.py`

> 所有網路資料於 2026-07-20 讀取。Krea 2 與 Diffusers 支援仍在快速更新，執行前應重新確認上游版本、model card 與授權條款。
