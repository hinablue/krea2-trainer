# Krea2 離線後訓練：RFT / FlowDPO

此功能在既有 Krea2 RAW DiT 訓練器加入離線偏好後訓練，不包含生成候選、線上 reward model 或 RL rollout。建議保留滿意的第一階段 TQD LoRA，用同一批人工偏好資料分別訓練兩個第二階段 adapter，比較未見 prompts 的好圖率、特徵與多樣性。實作與測試通過不等於已證明真實圖片品質改善。

## 模型與輸出合約

推薦使用 `--reference_lora /path/to/stage1.safetensors`：

- RAW DiT 與第一階段 adapter 凍結。
- 初始化一個獨立、zero-output 的可訓練第二階段 Krea2 LoRA。
- RFT 只在 winner 上做原本的 flow-matching 訓練。
- FlowDPO 的 reference 是 RAW DiT **加第一階段 LoRA**；只在 reference forward 關閉第二階段 adapter，不是把全部 LoRA 都關掉。
- 輸出是**增量第二階段 LoRA**，推論必須使用相同 RAW DiT + stage1 + stage2，各 adapter multiplier 預設為 1。不能把 stage2 當完整 stage1 替代品。

也可用非 FP8 的 `--base_weights` 先合併第一階段權重，或提供已合併的 `--dit`。若沒有提供第一階段權重，reference 就是 `--dit` 本身，訓練器不會猜測你的上一個 checkpoint。`--reference_lora` 與 `--base_weights` 不可同時使用；FP8 路徑不要先 `--base_weights` 合併，改用獨立 reference adapter。

不要用 `--network_weights` 載入第一階段後直接繼續更新它；本介面刻意拒絕這種會模糊 reference／輸出語意的用法。

`--reference_lora` 與 `--base_weights` 必須是相容的原生 Krea2 Linear LoRA：每個目標都要有完整的 `alpha`／`lora_down.weight`／`lora_up.weight`，且形狀、rank、有限數值及模型目標必須吻合。允許只訓練部分 Linear 的原生 adapter，但不允許提供無法對應的額外目標；不會靜默略過錯誤權重。多個 base-weight 來源會全部驗證後才開始合併。

## 偏好 manifest

JSONL，一行一組。名稱是原始圖片 basename（含副檔名），**不是 latent cache 檔名，也不是完整路徑**：

```jsonl
{"pair_id":"p001-ab","prompt":"a woman reading beside a window","chosen":"p001_a.png","rejected":"p001_b.png","split":"train"}
{"pair_id":"p001-ac","prompt":"a woman reading beside a window","chosen":"p001_a.png","rejected":"p001_c.png","split":"train"}
{"pair_id":"p002-ab","prompt":"a woman walking through a rainy city","chosen":"p002_a.png","rejected":"p002_b.png","split":"validation"}
```

- `pair_id` 唯一；`prompt` 必須與 text cache 的 `caption1` metadata **逐字相同**。
- `split` 省略時為 `train`；可用 `train`、`validation`、`test`，只有 train 進入 optimizer。held-out 記錄目前僅排除訓練，不自動產圖／評分。
- 同一 prompt 不可跨 split。概念近似但文字不同的 prompt 分組仍須由資料準備者負責，程式不能替你判斷概念洩漏。
- 拒絕重複 pair、自我配對、相反勝負及不同圖片共用 stem 的歧義。
- FlowDPO 要求 pair 兩張圖片的 latent shape／bucket 與文字 conditioning 一致；同 prompt 的兩份 cache embedding 必須相同。
- RFT 對重複 `(chosen, prompt)` 去重，不因同一 winner 配了多個 loser 而過度重複訓練。RFT 不需要讀取 loser cache，但 manifest 仍須合法。
- source dataset 的 `num_repeats` 不會變成偏好樣本的隱性加權；batch 由 `--preference_batch_size` 控制。
- 可以保留額外的人工原因標籤，這些欄位不參與 loss。

候選生成可以使用不同 seed；**訓練** FlowDPO pair 的重新加噪會共用同一份 Gaussian noise 及 timestep，這是不同層次的規則。

## 準備 caches

沿用一般的圖片 dataset TOML 與 cache 命令。每張候選的 caption 放同一個生成 prompt，不要把「好手指／壞手指」寫進 conditioning；那是偏好標籤，不是兩個不同 prompt。

```toml
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "/absolute/path/to/preference/images"
cache_directory = "/absolute/path/to/preference/cache"
num_repeats = 1
```

```bash
.venv/bin/krea2-cache-latents \
  --dataset_config /absolute/path/to/preference/dataset.toml \
  --vae /absolute/path/to/qwen_image_vae.safetensors

.venv/bin/krea2-cache-text \
  --dataset_config /absolute/path/to/preference/dataset.toml \
  --text_encoder /absolute/path/to/qwen3_vl_text_encoder.safetensors
```

快取後不要更改 prompt 或圖片內容而沿用舊 cache；caption metadata 一致不能證明 cache 對應未被換過的圖片。

## 開始兩個獨立實驗

以下 LR、steps、rank、beta 是**可執行的起點示例，不是已验证的 Krea2 最佳值**。替換路徑後執行，勿直接接既有 TQD launcher（它可能帶入互斥參數）。兩個分支都從同一 stage1 開始，而不是 FlowDPO 接在這次 RFT 之後。

### RFT

```bash
.venv/bin/krea2-train-lora \
  --post_training rft \
  --dataset_config /absolute/path/to/preference/dataset.toml \
  --preference_manifest /absolute/path/to/preference/pairs.jsonl \
  --preference_batch_size 2 \
  --dit /absolute/path/to/krea2_raw.safetensors \
  --reference_lora /absolute/path/to/stage1.safetensors \
  --network_module krea2_trainer.networks.lora_krea2 \
  --network_dim 32 --network_alpha 32 \
  --timestep_sampling uniform --weighting_scheme none \
  --mixed_precision bf16 --sdpa --gradient_checkpointing \
  --optimizer_type AdamW --learning_rate 1e-5 \
  --max_train_steps 200 --seed 42 \
  --output_dir /absolute/path/to/output/rft --output_name stage2_rft \
  --save_state
```

RFT 的 `preference_batch_size=2` 代表每 device microbatch 兩張不同 winner，不是兩組 pair 的四張圖。除了禁止 TQD，RFT 可使用原本支援的一般 FM sampler；上例與 FlowDPO 同樣使用 uniform，便於控制變因。

### FlowDPO

```bash
.venv/bin/krea2-train-lora \
  --post_training flow_dpo \
  --dataset_config /absolute/path/to/preference/dataset.toml \
  --preference_manifest /absolute/path/to/preference/pairs.jsonl \
  --preference_batch_size 1 --flow_dpo_beta 100 \
  --dit /absolute/path/to/krea2_raw.safetensors \
  --reference_lora /absolute/path/to/stage1.safetensors \
  --network_module krea2_trainer.networks.lora_krea2 \
  --network_dim 32 --network_alpha 32 \
  --timestep_sampling uniform --weighting_scheme none \
  --mixed_precision bf16 --sdpa --gradient_checkpointing \
  --optimizer_type AdamW --learning_rate 1e-5 \
  --max_train_steps 200 --seed 42 \
  --output_dir /absolute/path/to/output/flow_dpo --output_name stage2_flow_dpo \
  --save_state
```

FlowDPO batch 1 是一組 pair（winner + loser），仍有偏好學習訊號；和 TQD batch 內 quality normalization 抵消不是同一件事。FlowDPO 需要 policy／reference 的前向計算，成本不能當作普通 batch 1。grad accumulation 是累積 pair 梯度，不改變各 pair 的 shared-noise 規則。

## FlowDPO 公式與來源

採原論文 Appendix C 的 uniform `t ∼ U[0,1)`，配合作者 T2I code 的 FP32 per-image **mean** velocity MSE reduction：

```text
x_t = (1-t) * x_0 + t * noise
velocity_target = noise - x_0
Krea2_model_timestep = 1000*t + 1
error = mean_image_latent_elements((velocity_prediction - target)^2)
delta_chosen = policy_error_chosen - stopgrad(reference_error_chosen)
delta_rejected = policy_error_rejected - stopgrad(reference_error_rejected)
logit = (beta / 2) * (delta_rejected - delta_chosen)
loss = mean_pairs(-logsigmoid(logit))
```

- 每 pair 先算 logit／logsigmoid，再平均 batch。
- constant beta，沒有 `(1-t)^2`、SNR 或其他額外時間權重；不是 FlowCPO，也不是 signed regression。
- beta 的數字與 sum／mean reduction 及 latent 尺度密切相關，不能跨實作直接搬值。
- policy=reference 時 loss 接近 `log(2)` 是正常初始化，**不代表 gradient 為零**。
- loss 下降／內部 win-rate 上升不代表人眼好圖率已改善；必須使用未見 prompts 做盲測。

來源：
- [Improving Video Generation with Human Feedback](https://arxiv.org/html/2501.13918)，§4.2、Appendix C。
- [VideoAlign 官方 repository](https://github.com/KlingAIResearch/VideoAlign)。
- [作者 T2I implementation，固定 revision](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/scripts/train_sd3_dpo.py#L918-L931)。公開 T2I script 的 scheduler-index logit-normal 與本實作 uniform 不同；只採其 loss reduction，不移植線上生成／reference refresh loop。

## 相容性與安全限制

- 後訓練必須在 CLI 或 config 明確指定 `--mixed_precision bf16` 或 `fp16`；省略或使用 `no` 會在初始化前被拒絕，避免環境繼承的精度變更逃過續訓 reference 檢查。建議沿用上例的 `bf16`。
- 預設 `--post_training none` 保留普通訓練；指定後訓練專用選項卻未啟用 mode 會報錯，避免 flag 被忽略。
- 後訓練不支援 `--preset`、TQD routing／quality weighting、`--num_timestep_buckets`、TE 每 epoch 重建、Turbo sample model、HF resume、既有 `--network_weights` warm-start。
- FlowDPO 明確要求 `--timestep_sampling uniform --weighting_scheme none`，不會偷偷覆蓋原 sampler。拒絕時間截斷／bucket、dropout 及其他不會被 uniform objective 消費的變更選項。
- FlowDPO 或獨立 `--reference_lora` 暫不支援 `--compile`／block swap／gradient-checkpoint CPU offload；也要求 `--dynamo_backend NO` 與 `ACCELERATE_DYNAMO_BACKEND=NO`（或未設定），避免 Accelerate 暗中啟用編譯。普通 gradient checkpointing 可用。這是保守的 reference／裝置一致性界線，不代表永久不可能支援。
- 保存 metadata 是必要條件。恢復完整 optimizer 訓練請使用 `--resume /path/to/state` 並保留相同 reference 與 objective；不將恢復時 policy 當成新的 reference。沿用現有 trainer 的計步方式，這次 run 的 `--max_train_steps`／輸出 steps 重新計算，不是自動從先前總步數扣除剩餘預算。
- state sidecar 保存 manifest、dataset TOML、reference adapter 的 SHA256 與檔案資訊；大型 DiT 只保存 path／size／mtime，**不是內容雜湊保證**。不要原地改寫底模或 caches 後恢復；cache 本身未做逐檔內容鎖定。

## 驗證與評估

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/krea2-train-lora --help
```

測試使用合成 latent／text safetensors 與小模型，驗證資料、loss、梯度及 reference 合約，不修改實際資料，也不啟動完整 Krea2 訓練。正式驗收需另跑自己的短程 smoke 與 held-out 圖像比較；不要用訓練 loss 取代這一步。
