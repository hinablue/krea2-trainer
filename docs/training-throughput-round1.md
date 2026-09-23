# Throughput 第一輪：步數控制、DPO 去重與分階段量測

本輪保留原 loss、rank/alpha、optimizer 配方與 gradient checkpointing；沒有更換 attention backend、打開 FP8/compile、重建正式 cache 或修改正式權重。這裡的 GPU 實驗使用縮小模型與合成輸入，**不是正式 RAW DiT 的速度／畫質保證**。

## 已實作

### 1. 共享 launcher 的明確 step cap

`scripts/train_from_env.sh`（包含 `run_training.sh`、`run_training_tqd.sh` 的共用入口）：

- 設定 `MAX_TRAIN_STEPS` 或 CLI `--max_train_steps` 時，不再注入預設 `--max_train_epochs`。
- 未設定 step cap 時，仍使用原本 epoch-based 訓練。
- 明確在 launcher CLI 混用 epoch 與 step 限制會提早拒絕，避免假裝只跑短測。
- `MAX_TRAIN_STEPS` 必須是正整數。
- Launcher 限制旗標必須完整拼寫；`--max_train_step`、`--max_train_s`、`--max_train_epoch` 等縮寫在 cache／模型作業前拒絕。直接 Python CLI 的縮寫支援不變。
- **直接 Python CLI 的既有語意不變**：明確傳 `--max_train_epochs` 仍覆蓋 steps。另有 config file 時，應檢查啟動列出的最終 steps；不要在 config 裡再混入衝突的 epoch 配方。

回歸測試不只檢查 argv，還通過真實 DataLoader／step-resolution 方法確認最終步數。

### 2. DPO 使用 paired conditioning

FlowDPO 接上原本 FlowCPO 使用的 `_krea2_pair_count` metadata：

- reference 與 policy 共用相同 raw inputs／target／幾何 preparation。
- 每個分支只算每對一份 text/time conditioning，再以保留 autograd 的方式複製至 chosen/rejected。
- **不跨 reference/policy 共用可訓練中間輸出**，也不在 policy forward 與 checkpointed backward 中途切換 adapter。
- 對沒有 metadata API 的自訂小型 DiT 保留相容路徑。

### 3. 按 tracker 需求計算 metrics

- DPO/CPO trainer 沒有 tracker 時，不產生不會被使用的統計；仍保留訓練主迴圈的 loss 顯示。
- 有 tracker 時，將 detached tensor metrics 與 timestep 合併搬回 CPU。
- DPO input finite checks 改為按 device/dtype 批次 reduction。
- loss helper 仍預設回傳 Python float metrics，保留現有呼叫介面；內部可選 `collect_metrics=False` 或 `metrics_as_tensors=True`。
- 非有限值／overflow 檢查、EMA 更新次數與原子性維持；沒有將防護換成抽查。

## 分階段 profiler

新增：

- `--profile_steps N`：收集 N 個 **microsteps**；預設 0（完全不開啟診斷同步）。這不是訓練停止條件。
- `--profile_warmup_steps N`：略過前 N 個 microsteps；預設 5。
- `--profile_output PATH`：JSON 檔；預設 `<output_dir>/training-profile.json`。多程序時附加 rank，避免相互覆寫。

階段包括 loader/device placement wall time、input preparation、process batch、DPO reference/CPO old forward、policy forward、backward、gradient sync/clip、optimizer、post-optimizer hook 與 logging。

JSON 包含採樣窗口、實際收集數、是否完整、每步單位／optimizer update 訊號、wall time、CUDA stream elapsed time、CUDA allocator peak 及 median/p95。CPU 沒有 CUDA 測量時填 `null`，不足窗口時 `complete=false`，不捏造零延遲或完整性。

### 量測界線

- 這是 **opt-in、同步化的診斷窗口**。正式吞吐 A/B 要關閉 profiler，另行量測。
- CUDA event 記錄的是 current-stream 經過時間，包含排程空隙，**不是純 kernel 執行時間**；不能推論 GPU utilization。
- 父子 phase 是 inclusive，不能相加當總時間。
- loader 時間包含 Accelerate device placement／prefetch，不能直接叫磁碟 IO。
- 採樣中遇到 sampling/save 會包含在總 wall time 並標記；epoch setup、iterator construction、最後一次 save 不在此窗口。
- 單位為每裝置 image/pair，不是 global throughput 或 optimizer updates。GPU allocator peak 不代表所有程序或整台主機的記憶體。
- profiling 不改訓練停止條件、sampling、save/resume，也不保存 prompt／latent／credentials。

## 可重跑的小型訓練

```bash
cd /home/hina/Workspace/krea2-trainer
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python scripts/benchmark_tiny_training.py \
  --mode flow_dpo --steps 30 --warmup 5 \
  --output /tmp/krea2-tiny-dpo-new-run
```

`--output` 必須是尚未存在的目錄，避免覆寫實驗。`--mode` 支援 `tqd`、`flow_dpo`、`flow_cpo`。`--collect-metrics` 只啟用 metrics 計算，不連接 W&B 等服務；`--profile` 則另外生成 `phases.json`，勿將該次 timing 混入無 profiler 的 A/B。

此 runner：

- 使用真實 reduced SingleStreamDiT、native LoRA、BF16 autocast、checkpointed backward、AdamW 與 CPO EMA。
- 固定 rank 32/alpha 16，合成 latent／varlen conditioning；不載正式權重或資料。
- CUDA allocator 上限為 4 GiB 或裝置總量的 10%（取較小者）；不停止其他服務。
- 限制每次 steps、warmup、batch，避免意外長跑。
- 驗證 policy 確實更新且有限、底模/reference 不變、EMA update count。
- 輸出 JSON timing、比較用 tensors 與標示 synthetic-only 的 tiny policy；不應用於正式推論。

比較 revision 時，使用同一 runner，以不同 `PYTHONPATH` 在**不同程序**選擇 baseline／candidate source；JSON 會記錄載入的 `trainer_module.__file__`。不能只改工作目錄卻仍測到 editable install 的同一份程式。

## 本輪 GPU 實驗

環境：GB10、PyTorch 2.12.1+cu130。baseline 為 `34874016c261cdc67f9cd7820f1fa9234494277e`；candidate 為此輪工作樹。

主測試每個模式各做三組 baseline/candidate，交錯順序；每次暖機 5 steps、量測 30 steps，batch=2 images/pairs、checkpointing 開啟、無 external tracker、profiler 關閉。共 18 次、630 個 optimizer updates。另測三模式 GPU profiler 與既有 opt-in CUDA regression。

以各 run median 的中位數彙總：

- **DPO**：47.902 → 47.044 ms/step，step time 降 1.79%；PyTorch peak allocated 223.78 → 199.86 MiB，降 10.69%。
- **CPO**：56.204 → 56.928 ms/step，約慢 1.29%；peak 未變，單次 run 存在可見波動。
- **TQD**：29.921 → 29.710 ms/step，約快 0.71%；peak 未變。

**判讀：DPO 的記憶體減量明確；各模式速度差都未達本輪採用的保守 5% 顯著收益門檻，不能宣稱整體訓練已明顯加速。** 主機仍有既有 GPU services，未獨占 GPU；短測的 timing 只能當開發線索，不能外推完整模型或長跑。

### 數值驗收

採用既有 GPU tolerance：prediction/loss 相對誤差 <1%、完整 policy gradient 相對 L2 <1.5%、gradient cosine >0.9999。

- DPO first-step gradient relative L2 約 0.3562%，cosine 約 0.99999366；prediction 相對 L2 <0.24%；35 updates 後 policy 相對 L2 約 0.0338%。通過。
- CPO/TQD first-step prediction/loss/gradient 對照一致；CPO 的最終 policy 最大相對差約 0.0148%，TQD 的最終 policy 一致。
- 同組 initial policy 與 first-step CPU/CUDA RNG state 一致。
- 所有主測試底模／reference 保持不變，policy 真實更新且有限，CPO EMA 次數正確。
- CPU integration 覆蓋 RFT/DPO/CPO/TQD、dense/varlen cache、save/resume 與梯度累積；新 profiler 也在該迴圈中驗證。

這些是小型模型的工程驗收，不是長程收斂或畫質驗收。後續應拿正式 workload 的 profile 決定 attention／compile／EMA 管理的下一個優化點，而不是因微小 timing 差就切換高風險 backend。

## 最終回歸與審查

- 完整 unittest suite：190 項，CPU 執行 188 項通過、2 項 opt-in GPU 測試跳過；該 2 項另於 GB10 執行通過。
- 新增 profiler 的 CLI help、修改範圍 Ruff、launcher `bash -n`、`git diff --check` 通過。
- 獨立 code review 找到一項 Important：argparse 的合法 limit 縮寫可繞過 shell step-cap guard。先重現四個紅燈案例，再加入 prefix 拒絕；最終 suite 通過。直接 Python CLI 的既有縮寫仍有測試保護。
- Review 的 config/epoch 優先觀察保留為已知契約，未宣稱 launcher 能強制覆蓋所有 TOML 配方。
- 修改維持本地工作樹，未 commit／push；既有 Compose／Modal 等使用者修改不在本輪變更範圍。
