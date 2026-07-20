# 使用 Docker Compose 在本機訓練 Krea 2 LoRA

這份操作指南說明如何在本機 NVIDIA Docker 主機建置待命容器、準備持久化 workspace，並以和 RunPod 相同的 launcher 啟動訓練。`docker compose up` 不會下載模型、建立 cache 或開始訓練。

## 確認主機需求

本機映像以 `nvcr.io/nvidia/pytorch:25.11-py3` 為基底，支援 `linux/amd64` 與 `linux/arm64`。主機需要：

- Docker Engine
- Docker Compose v2
- 可用的 NVIDIA driver
- NVIDIA Container Toolkit
- 50 GB 以上 image 空間
- 40 GiB 以上模型空間，另加 dataset、cache 與 output 空間

先確認 Docker 能存取 GPU：

```bash
docker run --rm --gpus all \
  nvcr.io/nvidia/pytorch:25.11-py3 \
  nvidia-smi
```

若這個命令失敗，先修復 driver 或 NVIDIA Container Toolkit，再建置 trainer。

## 初始化持久化 workspace

Compose 預設把 repository 內的 `./workspace` bind mount 到容器的 `/workspace`。建立目錄並複製範例設定：

```bash
cd /path/to/krea2-trainer
mkdir -p workspace/{configs,models,.cache/huggingface}
mkdir -p workspace/datasets/my_dataset/{images,cache}
mkdir -p workspace/output/{checkpoints,logs}
cp runpod/dataset.example.toml workspace/configs/dataset.toml
cp runpod/train.env.example workspace/configs/train.env
```

把圖片與同名 `.txt` caption 放進 `workspace/datasets/my_dataset/images/`。`dataset.toml` 必須使用容器路徑：

```toml
[[datasets]]
image_directory = "/workspace/datasets/my_dataset/images"
cache_directory = "/workspace/datasets/my_dataset/cache"
```

需要改用其他資料盤時，設定 `KREA2_WORKSPACE`：

```bash
export KREA2_WORKSPACE=/mnt/training/krea2
```

該目錄會完整掛載到 `/workspace`。Compose 不會建立 named volume，也不會在 `docker compose down` 時刪除這些資料。

## 設定訓練參數

編輯下列兩個檔案：

- `workspace/configs/dataset.toml`: dataset、caption、resolution 與 cache 路徑
- `workspace/configs/train.env`: cache、training mode、epochs 與 output name

`train.env` 只接受 `KEY=VALUE`，不支援 shell 展開或 command substitution。Compose 已設定：

```dotenv
ENV_FILE=/workspace/configs/train.env
DATASET_CONFIG=/workspace/configs/dataset.toml
MODEL_DIR=/workspace/models
OUTPUT_DIR=/workspace/output/checkpoints
LOGGING_DIR=/workspace/output/logs
HF_HOME=/workspace/.cache/huggingface
```

設定優先序如下：

```text
訓練 CLI 尾端參數 > Compose process environment > ENV_FILE > launcher defaults
```

## 建置並啟動待命容器

執行以下命令：

```bash
docker compose build trainer
docker compose up -d trainer
docker compose ps
```

等待 healthcheck 通過，再檢查 PyTorch 與 GPU：

```bash
docker compose exec trainer python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
```

預設 image 名稱是 `krea2-trainer:local-cu130`。可透過 `KREA2_IMAGE` 改名，也能透過 `KREA2_BASE_IMAGE` 覆寫 base image：

```bash
export KREA2_IMAGE=registry.example.com/krea2-trainer:local
export KREA2_BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
```

替代 base image 必須提供 Linux、Python 3.10 至 3.12，以及 CUDA 13.0 相容環境。

## 傳入 secrets

不要把 Hugging Face 或 Weights & Biases (W&B) token 寫入 repository 或 `train.env`。在啟動 Compose 前，從目前 shell 注入：

```bash
export HF_TOKEN="[REDACTED]"
export WANDB_API_KEY="[REDACTED]"
docker compose up -d --force-recreate trainer
```

容器只在建立時取得 environment。修改 token 後必須重新建立容器，單純 restart 不會更新值。公開的 `Comfy-Org/Krea-2` 模型不需要 `HF_TOKEN`。

## 明確啟動訓練

確認設定後，執行和 RunPod 相同的 launcher：

```bash
docker compose exec trainer \
  /opt/krea2-trainer/runpod/train.sh
```

launcher 固定執行下列流程：

1. 驗證 CUDA 13.0、dataset config 與可寫目錄
2. 下載或檢查模型
3. 建立或重用 latent cache
4. 建立或重用 text encoder cache
5. 啟動 Accelerate training

`MODEL_FETCH=if_missing` 與 `CACHE_SKIP_EXISTING=1` 讓流程能在中斷後重跑。若模型已手動放進 `workspace/models`，可設定 `MODEL_FETCH=never` 進行離線訓練。

額外的 trainer CLI 參數可附加在 launcher 後方，且優先於環境設定：

```bash
docker compose exec trainer \
  /opt/krea2-trainer/runpod/train.sh \
  --max_train_steps 100
```

## 執行 TQD 訓練

將 `TRAIN_MODE=tqd` 寫入 `workspace/configs/train.env`，並在 dataset 設定加入：

```toml
tqd_score_file = "/workspace/datasets/my_dataset/scores.tqd.jsonl"
```

Timestep-aware Quality Decoupling (TQD) manifest 使用來源圖片檔名，不依賴 cache 檔名：

```json
{"image_file":"0001.png","structure_score":0.91,"detail_score":0.84}
```

每張圖片必須剛好一筆。兩個 score 都必須是 `[0, 1]` 內的有限數值。

## 查看狀態與停止容器

常用操作如下：

```bash
# 查看容器狀態與啟動 log
docker compose ps
docker compose logs -f trainer

# 進入 shell
docker compose exec trainer bash

# 停止容器並保留 workspace
docker compose down
```

Checkpoint 與 log 會保存在宿主機：

- `workspace/output/checkpoints/`
- `workspace/output/logs/`

原始碼、lockfile 或依賴變更後，重建並重新建立容器：

```bash
docker compose build trainer
docker compose up -d --force-recreate trainer
```

## 調整 Compose 資源

預設 shared memory 是 16 GB，可透過 `KREA2_SHM_SIZE` 調整：

```bash
export KREA2_SHM_SIZE=32gb
docker compose up -d --force-recreate trainer
```

Compose 設定 `gpus: all`，不會限制特定 GPU。多 GPU 主機若需要精確選卡，請在啟動前設定 Docker 支援的 GPU visibility，並在容器內以 `nvidia-smi` 驗證。

## 排除常見錯誤

- `could not select device driver ... gpu`: 安裝或修復 NVIDIA Container Toolkit
- healthcheck 顯示 unhealthy: 執行 GPU 檢查命令，確認 `torch.cuda.is_available()` 與 CUDA 13.0
- `Dataset TOML not found`: 確認 `KREA2_WORKSPACE` 指向正確目錄，且 `workspace/configs/dataset.toml` 存在
- `Insufficient disk space`: 釋放 workspace 所在磁碟空間，或改用容量較大的 `KREA2_WORKSPACE`
- cache flags incompatible: `CACHE_SKIP_EXISTING=1` 與 `FORCE_REBUILD_CACHE=1` 不可同時使用
- 容器看不到新 token: 使用 `docker compose up -d --force-recreate trainer` 重新建立容器
- 修改程式碼後容器仍使用舊版本: 執行 `docker compose build trainer`，再重新建立容器
