#!/bin/bash
# 在有外网的【本地电脑】运行，把生成物 SFTP 上传到服务器 data1/ml-2-3/
set -euo pipefail

DEST="${1:-./offline_bundle}"
mkdir -p "$DEST/wheels" "$DEST/pretrained" "$DEST/code"

echo "==> 1) 下载 Python 依赖 wheel（需与服务器 Python 版本接近，建议 3.10）"
pip download -r requirements.txt -d "$DEST/wheels"

echo "==> 2) 下载预训练权重"
pip install -q huggingface_hub ultralytics
python3 << 'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
from ultralytics import YOLO

dest = Path("offline_bundle/pretrained")
dest.mkdir(parents=True, exist_ok=True)

# YOLOv8n
YOLO("yolov8n.pt")
import shutil
shutil.copy2("yolov8n.pt", dest / "yolov8n.pt")

# TrOCR
snapshot_download(
    repo_id="microsoft/trocr-base-handwritten",
    local_dir=str(dest / "trocr-base-handwritten"),
)
print("pretrained OK ->", dest)
PY

echo "==> 3) 复制训练脚本"
cp train_scheme_a.py env.sh requirements.txt "$DEST/code/"

echo ""
echo "完成。请用 SFTP 上传整个 offline_bundle/ 到服务器，例如："
echo "  服务器: /data1/ml-2-3/"
echo "  - wheels/          -> /data1/ml-2-3/wheels/"
echo "  - pretrained/      -> /data1/ml-2-3/pretrained/"
echo "  - code/            -> /data1/ml-2-3/code/handwritten-to-data/"
echo ""
echo "数据集请单独下载 RUKOPYS (~17GB) 到 /data1/ml-2-3/data/rukopys/"
echo "  huggingface-cli download UkrainianCatholicUniversity/rukopys --repo-type dataset --local-dir ./rukopys"
