# 在 RunPod 上訓練 Krea 2 LoRA

這份操作指南說明如何建置映像、準備 Network Volume、設定 Pod，並以可重跑的 launcher 執行模型下載、cache 與訓練。容器啟動後只會待命，不會自動占用 GPU 開始訓練。

## 選擇 GPU 與儲存空間

Krea2 Trainer 使用 CUDA 13.0 與 PyTorch 2.9.1。建議從下列設定開始：

| Profile | GPU | 起始設定 |
|---|---|---|
| Preferred | RTX PRO 6000 96 GB | batch size 1、FP8 base/scaled、gradient checkpointing、`ENABLE_COMPILE=1` |
| Economy | A6000、A40 或 L40S 48 GB | batch size 1、FP8 base/scaled、gradient checkpointing、`ENABLE_COMPILE=0` |

Container disk 建議設為 50 GB。Network Volume 至少需要 200 GB，建議配置 500 GB，並與 Pod 放在同一個 data center。SSH 需要 `22/tcp`，Jupyter 則另加 `8888/http`。

## 建置並推送映像

`runpod/Dockerfile` 使用 `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404`。建置時只複製專案程式碼與文件，不會把本機 models、datasets、output 或 secrets 寫進映像。

```bash
docker build \
  -f runpod/Dockerfile \
  -t registry.example.com/krea2-trainer:cu130 \
  .
docker push registry.example.com/krea2-trainer:cu130
```

映像透過 `uv sync --frozen --extra cu130` 安裝 `uv.lock` 內的依賴。若依賴有變更，先在 repository 更新 lockfile，再重新建置映像。

## 準備 Network Volume

把 Network Volume 掛載到 `/workspace`。launcher 依賴固定的持久化目錄契約：

```text
/workspace/
  models/
    diffusion_models/
    text_encoders/
    vae/
  datasets/my_dataset/
    images/
    cache/
    scores.tqd.jsonl
  configs/
    dataset.toml
    train.env
  output/
    checkpoints/
    logs/
  .cache/huggingface/
```

將圖片與同名 `.txt` caption 放進 `images/`。複製下列範例作為設定起點：

```bash
cp /opt/krea2-trainer/runpod/dataset.example.toml \
  /workspace/configs/dataset.toml
cp /opt/krea2-trainer/runpod/train.env.example \
  /workspace/configs/train.env
chmod 600 /workspace/configs/train.env
```

`dataset.toml` 內的路徑必須使用容器路徑，例如 `/workspace/datasets/my_dataset/images`，不可填入本機工作站路徑。

## 設定訓練環境

launcher 依照下列優先序解析設定，越左側優先權越高：

```text
訓練 CLI 尾端參數 > Pod process environment > ENV_FILE > launcher defaults
```

Pod 建議設定這些非機密環境變數：

```dotenv
ENV_FILE=/workspace/configs/train.env
DATASET_CONFIG=/workspace/configs/dataset.toml
MODEL_DIR=/workspace/models
OUTPUT_DIR=/workspace/output/checkpoints
LOGGING_DIR=/workspace/output/logs
HF_HOME=/workspace/.cache/huggingface
```

`ENV_FILE` 只接受 `KEY=VALUE`，不執行 shell 語法。常用訓練值可放進 `/workspace/configs/train.env`：

```dotenv
TRAIN_MODE=standard
MODEL_FETCH=if_missing
HF_MODEL_REPO=Comfy-Org/Krea-2
HF_MODEL_REVISION=main
CACHE_MODE=all
CACHE_SKIP_EXISTING=1
FORCE_REBUILD_CACHE=0
ENABLE_COMPILE=1
OUTPUT_NAME=my_krea2_lora
MAX_TRAIN_EPOCHS=5
```

模型下載模式如下：

- `if_missing`: 重用完整檔案，只下載缺少的模型
- `never`: 不連線下載，模型缺少時立即失敗
- `force`: 重新驗證並下載全部模型

`Comfy-Org/Krea-2` 的公開檔案不需要 token。需要 `HF_TOKEN` 或 `WANDB_API_KEY` 時，請使用 RunPod Secrets 或 Pod process environment。不要把 secret 寫入 dotenv、image、repository 或命令列。

## 建立 Pod

### 使用 RunPod UI

1. 選擇由 `runpod/Dockerfile` 建置的 image
2. 將 Container disk 設為 50 GB
3. 把 200 至 500 GB 的 Network Volume 掛載到 `/workspace`
4. 加入 `22/tcp`，需要 Jupyter 時再加入 `8888/http`
5. 設定非機密環境變數，機密值只放進 Secrets
6. 建立 Pod 並等待容器進入 ready 狀態

容器預設執行 `sleep infinity`。這讓你能先檢查 volume、資料集與設定，再明確啟動訓練。

### 使用 API 或 RunPod MCP

建立 Pod 或 Template 時，使用相同的 image、volume mount、ports 與 environment。Environment object 可採用下列結構：

```json
{
  "ENV_FILE": "/workspace/configs/train.env",
  "DATASET_CONFIG": "/workspace/configs/dataset.toml",
  "MODEL_DIR": "/workspace/models",
  "OUTPUT_DIR": "/workspace/output/checkpoints",
  "LOGGING_DIR": "/workspace/output/logs",
  "HF_TOKEN": "[REDACTED]",
  "WANDB_API_KEY": "[REDACTED]"
}
```

API 或 MCP 呼叫中的 GPU type、data center 與 Network Volume ID 會依帳號資源而異。先查詢可用 GPU 與 volume，再建立 Pod，不要在文件或腳本中寫死資源 ID。

## 啟動與監控訓練

先確認 GPU、設定檔與資料集掛載正常：

```bash
nvidia-smi
ls -lah /workspace/configs
ls -lah /workspace/datasets/my_dataset/images
```

再明確執行 launcher：

```bash
/opt/krea2-trainer/runpod/train.sh
```

launcher 會依序執行：

1. 驗證 CUDA 13.0、設定檔與持久化目錄
2. 下載或檢查 Krea 2、Qwen3-VL 與 VAE 模型
3. 建立或重用 latent cache
4. 建立或重用 text encoder cache
5. 透過 Accelerate 啟動訓練

即時查看輸出目錄：

```bash
ls -lah /workspace/output/checkpoints
ls -lah /workspace/output/logs
```

## 執行 TQD 訓練

將 `TRAIN_MODE` 改成 `tqd`，並在 dataset 設定加入 `tqd_score_file`。Timestep-aware Quality Decoupling (TQD) manifest 使用來源圖片檔名作為索引，因此可在 cache 建立前完成：

```json
{"image_file":"0001.png","structure_score":0.91,"detail_score":0.84}
{"image_file":"0002.webp","structure_score":0.88,"detail_score":0.42}
```

每張來源圖片必須剛好有一筆資料。檔名只能是 basename，不能包含路徑。`structure_score` 與 `detail_score` 必須是 `[0, 1]` 內的有限數值。同一個 stem 不可同時使用不同副檔名，例如 `0001.png` 與 `0001.webp`。

## 從中斷狀態恢復

預設的 `MODEL_FETCH=if_missing` 與 `CACHE_SKIP_EXISTING=1` 會重用完整模型與既有 cache。下載或 cache 中斷後，修正錯誤並重新執行同一支 launcher 即可。

不要刪除 Hugging Face cache 內的 partial 或 lock 檔案。Hub client 會處理續傳與原子完成。若要離線執行，先按照既定 layout 放好三個模型，再設定 `MODEL_FETCH=never`。

## 停止計費與保留成果

訓練完成後，先確認 checkpoint 已寫入 Network Volume，再停止或刪除 Pod。停止 Pod 會停止 GPU 計費，但 Network Volume 仍會產生儲存費用。

保留 Network Volume 可保存 models、cache、checkpoints 與 logs。確認成果已下載或同步到外部儲存後，再刪除不需要的 volume。

## 排除常見錯誤

- `Dataset TOML not found`: 確認 volume 掛載到 `/workspace`，且 `DATASET_CONFIG` 使用容器路徑
- `CUDA 13.0 is required`: 重新建置指定的 RunPod image，並確認 Pod 沒有覆寫 runtime
- `Insufficient disk space`: 模型全缺時需至少 40 GiB 可用空間，另加 dataset、cache 與 output 空間
- `CACHE_SKIP_EXISTING and FORCE_REBUILD_CACHE are mutually exclusive`: 兩者只能啟用一個
- `Missing TQD score`: 確認 manifest 的 `image_file` stem 與來源圖片一致
- `Duplicate TQD image stem`: 移除重複資料，並避免同 stem 的跨副檔名碰撞
- W&B 未登入: 透過 RunPod Secrets 注入 `WANDB_API_KEY`，或將 `LOG_WITH` 設成空字串停用 tracking
