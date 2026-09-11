# FlowCPO：Krea2 離線偏好後訓練

本版已由隔離開發副本合併回 `/home/hina/Workspace/krea2-trainer`，原 CLI 可選擇 `--post_training flow_cpo`。它是依 [FlowCPO 論文](https://arxiv.org/html/2609.09905) Eq. 13、Algorithm 1 移植的實驗實作，不是作者官方程式；論文這版未附程式碼。本次整合未啟動正式訓練；下方訓練命令由使用者執行後會開始訓練。

## 方法與本版選擇

配對共用 uniform timestep、Gaussian noise、文字 conditioning：

```text
x_t = (1-t)*x0 + t*noise
u = noise-x0
mu_w = (1-beta)*stopgrad(v_old_w) + beta*v_policy_w
nu_l = (1+beta)*stopgrad(v_old_l) - beta*v_policy_l
loss = mean_pairs(mean_nonbatch((mu_w-u_w)^2)
                  + lambda*mean_nonbatch((nu_l-u_l)^2))
```

- 先以 FP32 混合 velocity，再平方與平均。採每張圖 mean MSE 是這個 Krea2 移植的尺度選擇，不是論文公式 sum 的逐字照搬。
- `v_old` 是固定底模＋固定第一階段 LoRA＋第二階段 LoRA 的 EMA；不是 FlowDPO 固定不動的 reference。
- EMA 是第二階段 LoRA down/up **參數矩陣**的 FP32 移動平均，不是每個輸出 velocity 的平均，也不等於合併後 dense delta 的精確平均。
- EMA decay `eta`：`old = eta*old + (1-eta)*policy`。只在真正成功的 optimizer step 後更新；梯度累積 microstep 或 overflow 跳過的 step 不更新。
- EMA forward 在 policy forward 前完成，只有 no-grad 與 multiplier 切換，不交換 policy 權重，不複製大型 DiT。EMA／第一階段不進 optimizer。
- 初始化時 policy 增量為零，EMA 與準備完成的初始 policy 相同。續訓載入後不再重設 EMA。
- 輸出的是 **policy 增量 LoRA**，不是 EMA。推論仍需要完全相同的底模＋第一階段。

## 參數

- `--post_training flow_cpo`：一般 CLI 模式選擇。下方專用 launcher 已代入，無需重複。
- `--flow_cpo_beta 0.5`：正向／鏡像 velocity 分支的混合係數，必須有限且 >0。
- `--flow_cpo_lambda 1`：rejected 分支權重，必須有限且 >=0。`beta=1, lambda=0` 可作 chosen-only loss 消融，但仍沿用配對資料頻率，不等於 RFT 去重資料模式。
- `--flow_cpo_ema_decay 0.99`：EMA decay，範圍 `[0,1)`。
- **不要傳入 `--flow_dpo_beta`**，兩者不是同一種 beta；即使傳 DPO 預設值也會拒絕。
- beta／lambda／EMA decay 都會進入續訓相容性紀錄，不能在同一份 state 上任意改變。

## 從原訓練目錄執行

保留既有 interpreter，不做 pip／uv install，launcher 用 `PYTHONPATH` 明確選擇本 checkout 的 source。可用 `KREA2_PYTHON` 指向另一個已備妥的相容 interpreter。

```bash
cd /home/hina/Workspace/krea2-trainer
bash scripts/train_flowcpo.sh --help
```

以下是全新 CPO run 的保守起始範例，**不要與 DPO 同時在同一 GPU 啟動**。讀取原有 manifest／cache／底模／stage-one，輸出放在獨立的 `flow_cpo` 子目錄，不覆蓋 DPO checkpoint。正式資料仍須已符合相同 prompt、conditioning、latent shape 與 bucket 的配對要求。

```bash
cd /home/hina/Workspace/krea2-trainer
bash scripts/train_flowcpo.sh \
  --dataset_config /home/hina/Workspace/krea2-trainer/configs/asian_flowdpo.toml \
  --preference_manifest /home/hina/Workspace/krea2-trainer/configs/pairs.jsonl \
  --preference_batch_size 1 \
  --flow_cpo_beta 0.5 --flow_cpo_lambda 1 --flow_cpo_ema_decay 0.99 \
  --dit /home/hina/Workspace/krea2-trainer/workspace/models/diffusion_models/krea2_raw_bf16.safetensors \
  --reference_lora /home/hina/Workspace/krea2-trainer/workspace/output/checkpoints/krea2_asianMix_lora-000005.safetensors \
  --network_module krea2_trainer.networks.lora_krea2 \
  --network_dim 32 --network_alpha 16 \
  --timestep_sampling uniform --weighting_scheme none \
  --mixed_precision bf16 --sdpa --gradient_checkpointing \
  --optimizer_type AdamW --optimizer_args weight_decay=0.01 --learning_rate 1e-5 \
  --max_train_steps 500 --seed 17415 \
  --save_state --save_every_n_steps 100 \
  --log_with wandb --log_config \
  --log_tracker_name krea2-trainer-tqd --wandb_run_name stage2_flow_cpo \
  --output_dir /home/hina/Workspace/krea2-trainer/workspace/output/checkpoints/flow_cpo \
  --output_name stage2_flow_cpo
```

這是可啟動的試驗設定，不是已針對 Hina 資料調好的最佳超參數；論文 SD3.5-M 的品質數據不可直接當成 Krea2 收益。

## Optimizer 與其他限制

本版先明確允許 AdamW、Adam、SGD、AdamW8bit 與程式列出的 `torch.optim` 名稱；CPU 整合實測使用 AdamW，AdamW8bit GPU 路徑未驗證。**目前不接受原本 DPO 使用的 `Adopt_adv` 或其 `--optimizer_args` 配方**，不要直接整段複製 DPO 命令。帶 train/eval 權重交換的 schedule-free optimizer 會拒絕，避免 EMA 混到錯的參數表示。

- 保留 `--gradient_checkpointing`；Hina 已實測此完整尺寸 workload 不開會 OOM。
- 需要明確 `bf16` 或 `fp16` autocast，policy／EMA 矩陣保持 FP32；拒絕 full reduced-precision policy。
- 不允許 compile／Accelerator Dynamo、block swap／H2D-only、checkpoint CPU offload。
- 不允許 nonzero `scale_weight_norms`、adapter dropout、TQD、timestep buckets、TE refresh、Turbo swap 等尚未驗證組合。
- 無 online rollout、reward model 或額外圖像生成。現有 `pairs.jsonl` 與 dense／varlen caches 可沿用。
- CPO 仍需要 EMA 與 policy 計算；不能保證比目前 FlowDPO 快，也不能由 CPU 測試推論 GPU 峰值記憶體或畫質。

## 保存、續訓與推論

啟用 `--save_state` 才能完整恢復 optimizer／EMA；`--save_every_n_steps` 決定中途保存頻率。state 目錄包含原生 Accelerator state，加上：

- `krea2_post_training_state.json`：原始底模／stage-one／資料／objective 相容性紀錄。
- `krea2_flow_cpo_ema.safetensors`：FP32 EMA down/up 矩陣，metadata 含 `version`、`decay`、`updates`。

`--resume /absolute/path/to/the-CPO-state-directory` 必須指向相容 **CPO** state，不接受 DPO state。缺失、損毀、非有限值、keys／shape／dtype／decay 不符時拒絕，不會默默把 EMA 換成目前 policy。底模沿用檔案資訊 fingerprint，非全內容 hash；cache 內容未逐檔鎖定。

繼承原 trainer 的步數語意：續訓時這次指定的 `--max_train_steps` 是這次 run 的步數上限，不會自動扣除歷史步數；EMA update count 則延續保存的累計值。

推論使用 `stage2_flow_cpo.safetensors` 搭配同一底模＋stage-one。不要把 EMA sidecar 當一般 LoRA 載入。

## 驗證與部署邊界

- `tests/test_flow_cpo.py`：loss、梯度、配置拒絕、EMA recurrence、跳過／累積、狀態損壞、reference 安全性等。
- `tests/test_post_training_integration.py`：真實縮小 SingleStreamDiT CLI、dense／varlen caches、bf16＋gradient checkpointing、兩步訓練＋save＋續訓一步。CPO 使用 gradient accumulation=2，核對每次 EMA 更新與保存／恢復後的精確張量、update count。
- 已先比對 baseline hashes、備份再合併回原 repo；沒有覆蓋使用者的其他 dirty files。原始檔備份與異動清單位於 `/home/hina/.hermes/profiles/rosie-agent/cache/krea2-flowcpo-merge/20260911-232154-687689`。
- 不載入正式模型、不生成圖片、不修改正式資料、不啟動 GPU 訓練。完整尺寸／畫質驗收需另做。
