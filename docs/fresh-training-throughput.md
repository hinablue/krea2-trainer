# 全新 LoRA／TQD 訓練：throughput 模式與 RAW 模型實測

## 範圍與結論

本輪只處理**從 RAW 底模建立新 LoRA 的一般／TQD 訓練**，不再優化 RFT、FlowDPO、FlowCPO。本輪開始時已保存 `src/` 基準，既有 Python trainer/model source 與後訓練檔案保持不變。

真正有收益的改動是：將目前預設關閉的 **主幹 `torch.compile` 路徑**整理成可直接使用、可退出的 fresh-training throughput 配方。沒有宣稱重寫了模型 kernel，也沒有修改 loss、learning rate、rank/alpha、batch size 或 sampler 配方。

**GB10、真實 Krea2 RAW 權重、1024×1024／batch 1 的新 TQD 訓練短測：**

- Eager：median **8.523 秒／step**，p95 **8.533 秒**。
- Compiled：median **6.143 秒／step**，p95 **6.188 秒**。
- 此固定 workload 的 step time 減少 **27.92%**。
- PyTorch peak allocated：**28.796 → 28.769 GiB**，沒有明顯增加。

兩次正式對照都**關閉分階段 profiler**，各 4 steps 暖機＋8 steps 量測，全部 12 次 optimizer updates 均完成並輸出有效 native LoRA。輸入是隔離目錄中的合成 latent／文字 cache，不是你的正式資料集。這證明完整模型的計算路徑收益，**不證明真實資料的長程收斂或畫質相同**。

## 使用方法

### 沿用原有訓練設定

在原本的環境／模型／資料路徑設定外，加上：

```bash
TRAIN_PERFORMANCE=throughput ./run_training_tqd.sh
# 一般新 LoRA：
TRAIN_PERFORMANCE=throughput ./run_training.sh
```

若既有 shell 或 `ENV_FILE` 已設定 `ENABLE_COMPILE=0`，它會優先於 profile 預設；請改成 `1` 或移除該 override。可明確指定：

```bash
TRAIN_PERFORMANCE=throughput ENABLE_COMPILE=1 \
COMPILE_MODE=max-autotune-no-cudagraphs ./run_training_tqd.sh
```

這是正常訓練入口，不是自動停止的 benchmark。請沿用正確的資料與模型設定，並使用新的 output 名稱／目錄，避免覆蓋既有 LoRA。

本機目前找到的模型位於 `workspace/models/` 子目錄，不是兩個 local wrapper 的歷史平面預設路徑。沒有既有 `ENV_FILE` 時，需另外明確傳入：

```bash
RAW_DIT="$PWD/workspace/models/diffusion_models/krea2_raw_bf16.safetensors"
VAE="$PWD/workspace/models/vae/qwen_image_vae.safetensors"
TEXT_ENCODER="$PWD/workspace/models/text_encoders/Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors"
```

上述路徑需以環境變數傳給 launcher（例如 `export` 或同一行命令前綴）；本輪沒有擅自改你的模型、資料、輸出路徑。

### Profile 契約

- `TRAIN_PERFORMANCE=balanced`：預設，保留先前 compile-off 行為。
- `TRAIN_PERFORMANCE=throughput`：預設啟用既有的主幹 compile。
- 預設 compile mode：`max-autotune-no-cudagraphs`，**不啟用 CUDA Graphs**。
- 預設 dynamic：`auto`，cache size limit：32。
- `TORCHINDUCTOR_COMPILE_THREADS` 未設定時使用 2，限制 cold compile 的 CPU 並行度。
- BF16、gradient checkpointing、rank 32/alpha 16、原 optimizer/loss 配方維持。
- FP8 預設仍關閉；不把儲存精度當作原生 FP8 GEMM 加速。
- 手動環境／CLI override 仍保留；啟動訊息明確寫的是 **profile defaults**，實際編譯設定另由 trainer 記錄。
- `ENABLE_COMPILE=0` 可退出編譯，不需還原程式碼。
- 只有共用 fresh launcher 的 `standard`／`tqd` 模式使用此設定；它不會修改後訓練限制。

之所以保留 opt-in：compiled BF16 不保證逐位元相同，且首次編譯、更多 bucket、不同 PyTorch／GPU 會改變成本。沒有以一個合成 workload 的短跑結果強制替換所有既有成功訓練。

## 實測設計

### 完整 RAW 模型對照

- GPU：NVIDIA GB10；PyTorch 2.12.1+cu130。
- 真實 checkpoint：`workspace/models/diffusion_models/krea2_raw_bf16.safetensors`，檔案 26,283,332,608 bytes。
- 新 LoRA，沒有載入 reference LoRA、resume 或 post-training mode。
- 固定 1024×1024、batch 1、rank 32／alpha 16、BF16、SDPA、gradient checkpointing。
- Optimizer：`Adopt_adv`，兩側相同 optimizer args；learning rate 5e-5。
- 相同 seed、4 筆合成 cache、96 text tokens、TQD scores；使用短 fixture 的 2-step LR warmup，並非修改一般 launcher 的 warmup。
- Eager／compiled 唯一訓練選項差異是既有 `--compile` 路徑及其既有 padding 行為。
- compiled 使用 `max-autotune-no-cudagraphs`／dynamic auto。
- 正式計時先跑 compiled 再跑 eager；另有先 eager 再 compiled 的同步 profiler 診斷作交叉檢查，**兩種量測不混算**。
- 主機原有 GPU services 未停止，所以不是 GPU 獨占的實驗。

### Timing 的意義

`scripts/benchmark_fresh_training.py` 在原生 `LossRecorder.add` 完成後記錄時間。trainer 在前一行已有 `loss.item()` 同步，因此連續 timestamp 包含 native step 的資料載入、forward、backward、optimizer 及相鄰 logging 邊界，不新增逐步 CUDA synchronize。

- 第一個量測 interval 從最後一個 warmup step 完成算起。
- median/p95 不含模型載入、cold compile、warmup 與最終 checkpoint save。
- JSON 另列 `total_wall_seconds_including_load_compile_save`，不把編譯成本藏在 steady-state 數字中。
- `--profile` 是另外的同步診斷選項；有開啟時 JSON 明確標記，不應和 unprofiled A/B 比較。
- 此次主對照僅每側 8 個量測 steps。27.92% 是這組固定 cache／shape 的結果，不是所有 buckets、batch sizes 或長跑的保證。

### 實際驗證

完整模型兩側均驗證：

- 明確維持 `post_training=none` 與指定 steps。
- 底模參數 frozen、未被 optimizer 擁有，完成後 parameter version 與 gradient state 不變。
- RAW 檔案 size／mtime／ctime 未變。
- 每步 loss 有限。
- 匯出的 native LoRA 全部 tensor 有限，原始 zero-initialized up projections 確實更新。
- checkpoint `ss_steps` 等於指定 update 數。
- compiled run 無記錄到 graph break。

另外完成：

- 一般 fresh LoRA（`krea2_shift`，非 TQD）完整 RAW compiled 4-step smoke，通過同一組驗證；沒有把它當成 standard 模式的 A/B 加速百分比。
- 縮小模型的 eager／主幹 compiled 對照：輸出相對 L2 約 0.6405%、gradient 相對 L2 約 0.3427%、cosine 約 0.99999417，RNG state 相同。符合先前數值門檻，但**不是完整 RAW gradient parity 證明**。
- 小模型在三種 latent geometry／文字長度間切換，dynamic auto 完成 9 次新 TQD updates，記錄 3 個 graphs、無 graph break；並未窮盡所有 dataset buckets。
- fresh launcher、TQD dataset／sampler 相關 CPU tests：40 項通過。
- 新腳本與測試 Ruff、shell syntax、`git diff --check` 通過。
- fresh-only 獨立 code review 通過，沒有 Important；另修正 `--profile` 報表的窗口 peak aggregation，並用「前一步 100、最後一步 50」的 CPU regression 防止低報峰值。
- 附件保留原始實測 runner 及其 SHA；最終 runner 只多了診斷模式的 peak 彙整。已比對 AST，unprofiled 主流程與實測版本相同，沒有改寫原始 benchmark 的 SHA 或數字。

## 可重跑的受限實驗

輸出目錄必須不存在；script 自建獨立合成 cache，不讀寫正式 dataset：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TORCHINDUCTOR_COMPILE_THREADS=2 PYTHONPATH="$PWD/src:$PWD" \
.venv/bin/python scripts/benchmark_fresh_training.py \
  --dit workspace/models/diffusion_models/krea2_raw_bf16.safetensors \
  --output /tmp/krea2-fresh-compiled-new \
  --variant compiled --training-mode tqd --resolution 1024 --warmup 4 --steps 8
```

另一側用新的 output 路徑及 `--variant eager`。一般新 LoRA 改為 `--training-mode standard`。

此工具只允許 CUDA、fresh modes、512/1024 resolution、有限 warmup/steps；限制 CUDA allocator 預算為 80 GiB 或裝置總量的 70%（取較小者）。生成的 `synthetic_benchmark.safetensors` **不是品質模型，不應拿去正式推論或合併**。

## 沒有採用的候選

- 擴大編譯到 text-fusion blocks 或整個 fusion module：小模型沒有比只編譯主幹更快，沒有加進 production。
- 原生 GQA：只有 attention primitive 探索，未改 attention backend，也未以其微測代替全模型速度。
- 關閉 checkpointing：不做；保留使用者已驗證的記憶體邊界。
- 後訓練優化：不做；本輪開始後既有 `src/` 檔案逐檔比對保持相同。
