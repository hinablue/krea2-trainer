# FlowCPO 驗證紀錄：隔離開發與原目錄整合

使用說明：[FlowCPO 指南](flow-cpo.md)。

## 原目錄整合驗證

- 已按使用者暫停訓練後的授權，先檢查來源 baseline、備份，再搬回 CPO 範圍的 source／tests／docs／launcher。
- 備份與逐檔雜湊清單：`/home/hina/.hermes/profiles/rosie-agent/cache/krea2-flowcpo-merge/20260911-232154-687689`。
- 未 commit／push，未啟動正式訓練；未修改模型或正式資料。
- 移除測試環境的 `PYTHONPATH`，實際 import 指向 `/home/hina/Workspace/krea2-trainer/src/krea2_trainer/krea2_flow_cpo.py`，CUDA=False、PyTorch threads=1。
- 原目錄全套 CPU 測試：`Ran 163 tests in 16.218s`、`OK`、exit 0；log：`/tmp/flowcpo-live-final-suite.log`。
- 原 CLI help、launcher bash syntax、Ruff 與 `git diff --check` 通過。
- 整理後 README 與 FlowCPO 指南中的命令通過真實 parser／post-training validation；以其 optimizer args 建立 tiny AdamW 成功。僅核對正式輸入路徑存在，不載入正式模型。
- 指南的相對文件連結與 code fences 檢查通過。

## 來源與隔離（整合前歷史）

- 原始目錄 `/home/hina/Workspace/krea2-trainer`。
- 實作／測試目錄 `/home/hina/Workspace/krea2-trainer-flowcpo-dev`。
- `FLOWCPO_BASELINE.json` 記錄原始 SHA256 baseline；parent 驗證 live baseline changed 為空清單。
- 共用原有 `.venv/bin/python`，沒有安裝或改動套件。
- `PYTHONPATH` 明確選擇 dev source；實測 package `__file__` 指向 dev，CUDA unavailable，PyTorch threads=1。
- 未啟動正式訓練、未讀取正式模型／圖片、未改動資料／checkpoint、未合併到 live source。

## Parent 實際執行

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHONPATH=/home/hina/Workspace/krea2-trainer-flowcpo-dev/src:/home/hina/Workspace/krea2-trainer-flowcpo-dev
/home/hina/Workspace/krea2-trainer/.venv/bin/python -m unittest discover -s tests -q
Ran 163 tests in 34.087s
OK
```

完整 log：`/tmp/flowcpo-full-parent.log`。

- Ruff：8 個涉及 Python 檔案通過。
- `compileall -q src tests` 通過。
- `bash -n scripts/train_flowcpo.sh` 通過。
- 專用 launcher 的 CLI `--help` 通過，包含 CPO 選項。
- 指南的完整訓練命令通過真實 parser 與 post-training validation；只解析，不啟動訓練。

## 整合測試

`tests/test_post_training_integration.py` 實際執行：

- RFT／FlowDPO／FlowCPO × dense／varlen conditioning。
- 真實 miniature `SingleStreamDiT`、native LoRA、safetensors cache、CLI main、AdamW、CPU bf16、gradient checkpointing。
- CPO 兩組 pairs、gradient accumulation=2。
- 初次两步 optimizer update，EMA 更新事件 `[false,true,false,true]`；逐 microstep 比對精確 recurrence。
- policy-only native checkpoint 與 EMA／provenance state 保存。
- fresh trainer／base／stage-one 重建、恢復 optimizer，於 `on_train_start` 確認 EMA tensor/count 未重設。
- 再訓練一步，EMA 更新事件 `[false,true]`，累計 count=3；保存的 EMA 張量與記憶體中一致。
- 原底模與 stage-one 未被訓練更新。

## 獨立 SPEC review

結果：**PASS**，無 blocking spec gaps。

- 42 個 CPO focused tests 通過；既有 post-training／dataset regression 72 項通過。
- reviewer 額外用真實 CPU GradScaler overflow、Accelerator accumulation、scheduler／zero_grad 驗證 EMA counts `[0,1,1,1,1,2]`，跳過 overflow 與非同步 microstep。
- 實際 save/load 恢復 policy、EMA、count；錯誤 provenance 與 FP16 EMA state 在 policy／optimizer load 前拒絕。
- Reviewer 未修改任何檔案，probe fixtures 已清理。

## 最後品質／整合審查

結果：**APPROVED**，獨立 reviewer 未重現阻擋性的程式品質或整合問題（`deleg_ae25506a`）。

- Reviewer 在隔離副本執行完整 CPU suite：163 tests passed、31.713s、exit 0；八個相關 Python 檔案 Ruff 通過。
- 檢查真實縮小模型 CLI train/save/resume、dense／varlen、checkpointed backward、EMA 精確 recurrence 與恢復、policy-only export，以及既有 RFT／DPO 行為隔離。
- 隔離 launcher 的 shell syntax、從 `/tmp` 執行 help、拒絕 mode override、無參數 fail-closed，以及文件指令 parser／validation 通過。
- Reviewer 未修改檔案。此 approval 只涵蓋受審的隔離版本，不把 parent 後續 launcher／文件調整算成 reviewer 審查成果。
- Parent 已另行完成原目錄部署驗證（見本文件首節），並確認搬回的 source／tests 與受審副本逐檔相同；baseline 範圍內沒有預期清單之外的變更。

## 未驗證邊界

- 沒有完整尺寸 GPU 效能、VRAM、画質結果。
- 沒有 GPU fp16 overflow、AdamW8bit 或 multi-process runtime 測試。
- `Adopt_adv` 是明確拒絕的初版限制，不得宣稱相容。
- 論文未附官方程式；本版採 LoRA parameter EMA 與 mean MSE 尺度，是有明示選擇的 Krea2 移植。
