# RFT / FlowDPO 實作驗證紀錄

使用說明、參數與原始論文來源：[後訓練指南](post-training.md)。

## 驗證結果

在專案既有 `.venv` 中實際執行：

```text
.venv/bin/python -m unittest discover -s tests -q
Ran 116 tests in 11.439s
OK
```

- Ruff：本次涉及的七個 Python 檔案全部通過。
- `compileall`：`src/krea2_trainer` 與 `tests` 通過。
- `git diff --check`：通過。
- CLI help：訓練、latent cache、text cache 入口通過。
- 文件中的兩個訓練命令及兩個 cache 命令完成 parser 檢查；範例 dataset TOML 完成 blueprint 驗證。
- 資料與訓練部分完成獨立規格、程式品質與整合審查；發現的阻擋問題已加入回歸測試並通過複審。

## 真正執行過的整合範圍

`tests/test_post_training_integration.py` 使用縮小尺寸、真實的 `SingleStreamDiT`，只替換大型 checkpoint 的載入，其他路徑執行正式程式：

- 真正 CLI main → dataset TOML → safetensors latent／text caches。
- RFT／FlowDPO × 固定長度／變動長度文字 conditioning。
- CPU bf16 autocast、原生 stage-one／stage-two LoRA、gradient checkpointing。
- 兩步 AdamW 訓練，驗證 stage-two 不再是 zero-output 初始化。
- 儲存增量 checkpoint 與 reference state sidecar。
- 重新建立 base／stage-one、恢復 optimizer，再執行一步更新。
- 驗證原始底模與第一階段權重未被 optimizer 改動，續訓 reference 身分不變。
- 明確 bf16 CLI 與相反的 fp16 環境設定並存時，實際保存的精度仍為 bf16。

其他測試涵蓋 FlowDPO loss 符號／reduction、配對共用 noise／timestep、checkpointed gradients、reference 凍結、錯誤權重在修改底模／hooks 前拒絕、精度變更續訓拒絕，以及 persistent-worker 配對與 epoch shuffle。

## 審查後修正

- 禁止省略後訓練 precision，避免繼承環境後造成 reference contract 與實際計算不符。
- 嚴格驗證所有 `base_weights`／`reference_lora` 目標、triplets、形狀與有限數值，避免沒有匹配到 LoRA 卻靜默訓練裸底模。
- FlowDPO 同時接收 dense tensor 與 varlen list conditioning，並測試完整 dataset → forward／backward → save／resume 路徑。

## 未驗證與未執行

- 沒有載入正式 Krea2 checkpoint 進行完整尺寸 GPU 訓練。
- 沒有使用或修改正式圖片資料集。
- 沒有宣稱好圖率、特徵維持或多樣性已改善；這些需另外做 held-out prompts 的圖像評估。
- FP8／大型 GPU 記憶體表現不由上述 CPU fixture 證明。
- 大型 DiT 身分只用檔案資訊，不是全內容雜湊；cache 未逐檔鎖定內容。具體相容性與續訓限制見指南。
