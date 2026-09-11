# CPO／TQD 計算優化

2026-09-12 實作。沿用現有訓練命令，沒有調整 learning rate、rank、loss 係數、EMA decay 或 batch size。GQA 仍使用原本 K/V `repeat_interleave`＋SDPA，沒有啟用原生 GQA 或改動 SDPA 呼叫。

## CPO 1～5

1. eval、multiplier=0 且矩陣已凍結的 EMA adapter 直接跳過 down/up projection；仍執行底層 base／stage-one／policy。可訓練的零 multiplier adapter 保留零梯度語意。
2. finite 檢查按 device／dtype 批次 reduction；alpha 先合併再比較／搬移。EMA 使用 foreach 的兩次乘法、一次加法，保留原 FP32 recurrence 的運算順序。全部候選值驗證成功後才寫入；overflow、累積步數、save/resume 的更新計數不變。
3. SDPA 的等長判斷與裁切長度在 attention metadata 建立時處理。Krea2 訓練直接提供 CPU 文字長度，block 與 checkpoint 重算不用逐層讀回 GPU 純量。
4. CPO 每對共用 raw conditioning；old、policy 各自在自己的 adapter 狀態下，將 text fusion／text MLP／time MLP／time projection 算一次，再用保留 autograd 的複製供 chosen/rejected 使用。兩個模型分支之間只共用原始輸入、target 與幾何資訊，不共用可訓練的中間輸出。
5. 無 dropout 的 eager SDPA 訓練省略 256 倍數 padding，輸出頭只處理 image token。compiled TQD 保留原 padding。非零 adapter dropout 的 TQD 同時保留原 padding 與完整輸出頭形狀，避免改變 dropout RNG 的消耗。

## TQD 與共用路徑

- masks、位置與 RoPE 使用模型內最多 8 種幾何配置的快取。device 轉換時清除，不加入 state_dict。
- 移除 non-reentrant checkpoint 不需要的 raw image/context `requires_grad`；LoRA 與其上游必需的梯度仍保留。
- 依實際 score 值、device、kappa 快取最多 256 組 Beta 分布參數與品質權重。每步仍在原來的位置、原來的 device 抽樣，不快取隨機 timestep。
- 分數先驗證，再省掉 Beta／Dirichlet 的重複驗證。batch=1 通過品質檢查後省略恆等權重；全零品質仍按原規則拒絕。
- loss 優先做每張圖的 MSE reduction，再乘每圖固定權重；空間權重保留原路徑。無權重時直接 mean MSE。loss 保留既有 network_dtype（目前 trainer 為 FP32）。
- TQD 四個 metrics 一次搬回 CPU；沒有 tracker 時不計算這些統計。
- TQD latent／文字檔案快取在每個 loader process 共用 64 MiB／128 檔案上限。以 inode、size、mtime_ns、ctime_ns 判斷失效；每 epoch 重建文字 cache 後會重新讀取。回傳張量不與快取共用可修改的 storage。
- CPO／DPO rejected 只讀 latent；原有初始化階段的配對 conditioning 一致性驗證保留。

## 驗證與數值界線

完整 unittest suite 共 180 項：CPU 執行 178 項通過，2 項 opt-in CUDA 測試另外在 GB10 執行並通過。修改檔案的 Ruff 檢查與 `git diff --check` 亦通過。

`tests/test_training_optimizations.py` 對照優化前的 DiT 計算流程，涵蓋 CPO／普通 TQD 路徑、batch=1、多筆 varlen conditioning、FP32／BF16、checkpoint backward、dropout RNG、cache 失效、EMA 原子更新與 compiled block 無 graph break。

`tests/test_post_training_integration.py` 涵蓋 RFT／DPO／CPO／TQD × dense／varlen cache，使用實際縮小 SingleStreamDiT、CPU BF16、CLI、optimizer、save 與 resume。CPO 額外檢查 gradient accumulation 與 EMA recurrence。

GB10／PyTorch 2.12.1+cu130 的縮小模型測試涵蓋 CPO／TQD 各 3 次 optimizer 更新。CUDA EMA recurrence、TQD 抽樣結果與 RNG state 逐值一致。metadata-only 對照的輸出／梯度亦一致。

裁切 token 或配對去重會改變 GEMM 的矩陣形狀，因此 BF16 不保證逐位元一致。GPU 對照採輸出／loss 相對 L2 誤差 <1%、完整 policy gradient 相對 L2 誤差 <1.5%、梯度 cosine >0.9999；CPU 另做逐張量對照。這些是縮小模型的驗證界線，不是正式模型畫質或長程收斂保證。

GPU 驗證可獨立執行（只建立小模型）：

```bash
RUN_GPU_OPTIMIZATION_TESTS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m unittest tests.test_training_optimizations.GPUOptimizationTests -v
```

正式模型吞吐量與畫質尚未量測；既有訓練程序不會因磁碟上的 Python 程式更新而自動切換，新啟動的程序才套用這些優化。checkpoint 格式未改，不需要重新產生 latent／文字 cache。
