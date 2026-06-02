# %% [markdown]
# # Handwritten to Data - Hugging Face Data + YOLO + TrOCR
#
# 这个文件是给 Kaggle Notebook 用的。
#
# 比赛页面说明完整数据在 Hugging Face：
#
# ```python
# from datasets import load_dataset
# ds = load_dataset("UkrainianCatholicUniversity/rukopys")
# ```
#
# 所以这版代码不再依赖 Kaggle Input 里的 `train.csv/test.csv`。
# 它会直接从 Hugging Face 下载 `train/silver/test`，然后：
#
# 1. 用 YOLOv8 训练 `bbox + type` 检测器
# 2. 用 TrOCR 训练区域文字识别器
# 3. 对 test 图片生成 `submission.csv`
#
# 最终提交：
#
# ```text
# /kaggle/working/submission.csv
# ```

# %% [markdown]
# ## 1. 安装依赖
# Kaggle 右侧 Settings 里需要打开 Internet。

# %%
import importlib.util
import subprocess
import sys

# Kaggle already includes many GPU/CUDA packages. Avoid aggressive upgrades because
# they can create scary but unrelated CUDA/numba dependency conflict messages.
PACKAGE_IMPORTS = {
    "datasets": "datasets",
    "huggingface_hub": "huggingface_hub",
    "ultralytics": "ultralytics",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "jiwer": "jiwer",
    "sentencepiece": "sentencepiece",
}
missing = [pkg for pkg, import_name in PACKAGE_IMPORTS.items() if importlib.util.find_spec(import_name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
print("Package check complete. Missing installed:", missing)

# %% [markdown]
# ## 2. 导入库和参数

# %%
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
from ultralytics import YOLO


DATASET_ID = "UkrainianCatholicUniversity/rukopys"
WORK_DIR = Path("/kaggle/working")
IMAGE_DIR = WORK_DIR / "rukopys_images"
YOLO_DIR = WORK_DIR / "yolo_regions"
OCR_MODEL_DIR = WORK_DIR / "trocr_region_ocr"
SUBMISSION_PATH = WORK_DIR / "submission.csv"

SEED = 42
N_SPLITS = 5
FOLD = 0

# 第一次建议 False，先跑通。想冲更高分再改 True，用 silver 自训练数据。
USE_SILVER = False

# 如果只是测试流程，可以把这两个改小，比如 200、80。正式训练设为 None。
MAX_TRAIN_IMAGES = None
MAX_TEST_IMAGES = None

YOLO_MODEL = "yolov8s.pt"  # 显存不够改 yolov8n.pt；冲分可试 yolov8m.pt
YOLO_EPOCHS = 30
YOLO_IMGSZ = 1280
YOLO_BATCH = 4
YOLO_CONF = 0.15
YOLO_IOU = 0.50

OCR_MODEL = "microsoft/trocr-base-handwritten"
OCR_EPOCHS = 2
OCR_BATCH = 4
OCR_GRAD_ACCUM = 4
OCR_LR = 5e-5
MAX_TARGET_LENGTH = 128
MAX_NEW_TOKENS = 128
CROP_MARGIN = 8

np.random.seed(SEED)
torch.manual_seed(SEED)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## 3. 从 Hugging Face 加载数据

# %%
ds = load_dataset(DATASET_ID)
print(ds)
print("Splits:", list(ds.keys()))

train_ds = ds["train"]
if USE_SILVER and "silver" in ds:
    train_ds = concatenate_datasets([train_ds, ds["silver"]])
test_ds = ds["test"]

if MAX_TRAIN_IMAGES is not None:
    train_ds = train_ds.select(range(min(MAX_TRAIN_IMAGES, len(train_ds))))
if MAX_TEST_IMAGES is not None:
    test_ds = test_ds.select(range(min(MAX_TEST_IMAGES, len(test_ds))))

print("train images:", len(train_ds))
print("test images:", len(test_ds))
print("example keys:", train_ds[0].keys())
print("example file_name:", train_ds[0].get("file_name"))
print("example regions:", train_ds[0].get("regions")[:2])

# %% [markdown]
# ## 4. 工具函数

# %%
@dataclass(frozen=True)
class Region:
    bbox: tuple[int, int, int, int]
    region_type: str
    text: str


def image_name(example):
    file_name = example.get("file_name")
    if file_name:
        return Path(file_name).name
    image = example["image"]
    if getattr(image, "filename", None):
        return Path(image.filename).name
    return f"{abs(hash(str(example))) % (10**12)}.jpg"


def pil_image(example):
    image = example["image"]
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def parse_regions(value):
    if value is None:
        return []
    out = []
    for item in value:
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(
            Region(
                bbox=(x1, y1, x2, y2),
                region_type=str(item.get("type", "handwritten")),
                text=str(item.get("text", "")),
            )
        )
    return out


def serialize_regions(regions):
    payload = [
        {
            "bbox": [int(v) for v in r.bbox],
            "type": str(r.region_type),
            "text": str(r.text),
        }
        for r in regions
    ]
    return json.dumps(payload, ensure_ascii=False)


def clamp_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_to_yolo(bbox, width, height):
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return cx, cy, w, h


def add_margin(bbox, width, height, margin=CROP_MARGIN):
    x1, y1, x2, y2 = bbox
    return clamp_bbox((x1 - margin, y1 - margin, x2 + margin, y2 + margin), width, height)


def reading_order(regions):
    # Dataset README says evaluation sorts by center_y bucket then center_x.
    return sorted(regions, key=lambda r: (((r.bbox[1] + r.bbox[3]) / 2) // 15, (r.bbox[0] + r.bbox[2]) / 2))


def save_example_image(example, split):
    out_dir = IMAGE_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    name = image_name(example)
    path = out_dir / name
    if not path.exists():
        pil_image(example).save(path, quality=95)
    return path


def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

# %% [markdown]
# ## 5. 建训练记录和类别

# %%
records = []
for idx, example in enumerate(tqdm(train_ds, desc="prepare train records")):
    path = save_example_image(example, "train")
    regions = parse_regions(example.get("regions"))
    records.append(
        {
            "idx": idx,
            "image_name": path.name,
            "image_path": str(path),
            "regions": regions,
            "source": example.get("source"),
            "annotation_source": example.get("annotation_source"),
        }
    )

class_names = sorted({r.region_type for rec in records for r in rec["regions"]})
if not class_names:
    raise ValueError("没有读到任何 regions，请把本格输出发给我。")
class_to_id = {name: idx for idx, name in enumerate(class_names)}
id_to_class = {idx: name for name, idx in class_to_id.items()}
print("class_to_id:", class_to_id)
print("records:", len(records))
print("regions:", sum(len(r["regions"]) for r in records))

kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
for rec in records:
    rec["fold"] = -1
for fold, (_, val_idx) in enumerate(kfold.split(records)):
    for i in val_idx:
        records[i]["fold"] = fold

# %% [markdown]
# ## 6. 准备 YOLO 数据集

# %%
def prepare_yolo_dataset():
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)

    for split_name, split_records in [
        ("train", [r for r in records if r["fold"] != FOLD]),
        ("val", [r for r in records if r["fold"] == FOLD]),
    ]:
        image_out = YOLO_DIR / split_name / "images"
        label_out = YOLO_DIR / split_name / "labels"
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for rec in tqdm(split_records, desc=f"prepare yolo {split_name}"):
            src = Path(rec["image_path"])
            dst = image_out / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

            with Image.open(src) as image:
                width, height = image.size

            lines = []
            for region in rec["regions"]:
                bbox = clamp_bbox(region.bbox, width, height)
                if bbox is None:
                    continue
                cx, cy, w, h = bbox_to_yolo(bbox, width, height)
                lines.append(f"{class_to_id[region.region_type]} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}")
            (label_out / f"{src.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    data_yaml = {
        "path": str(YOLO_DIR),
        "train": "train/images",
        "val": "val/images",
        "names": {idx: name for idx, name in id_to_class.items()},
    }
    yaml_path = YOLO_DIR / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return yaml_path


yolo_yaml = prepare_yolo_dataset()
print(yolo_yaml.read_text())

# %% [markdown]
# ## 7. 训练 YOLO

# %%
detector = YOLO(YOLO_MODEL)
detector.train(
    data=str(yolo_yaml),
    epochs=YOLO_EPOCHS,
    imgsz=YOLO_IMGSZ,
    batch=YOLO_BATCH,
    seed=SEED,
    project=str(WORK_DIR / "yolo_runs"),
    name="regions",
    exist_ok=True,
)

best_yolo = WORK_DIR / "yolo_runs" / "regions" / "weights" / "best.pt"
detector = YOLO(str(best_yolo))
print("best_yolo:", best_yolo)

# %% [markdown]
# ## 8. 准备 OCR 数据

# %%
ocr_rows = []
for rec in records:
    for region in rec["regions"]:
        if normalize_text(region.text) == "":
            continue
        ocr_rows.append(
            {
                "image_path": rec["image_path"],
                "bbox": region.bbox,
                "type": region.region_type,
                "text": region.text,
                "fold": rec["fold"],
            }
        )

ocr_df = pd.DataFrame(ocr_rows)
print("ocr_df:", ocr_df.shape)
print(ocr_df["type"].value_counts())
ocr_train = ocr_df[ocr_df["fold"] != FOLD].reset_index(drop=True)
ocr_val = ocr_df[ocr_df["fold"] == FOLD].reset_index(drop=True)

# %% [markdown]
# ## 9. 训练 TrOCR

# %%
class RegionOCRDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        width, height = image.size
        bbox = add_margin(tuple(row["bbox"]), width, height)
        if bbox is not None:
            image = image.crop(bbox)

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            normalize_text(row["text"]),
            padding="max_length",
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
        ).input_ids
        labels = [x if x != self.processor.tokenizer.pad_token_id else -100 for x in labels]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels, dtype=torch.long)}


def collate_fn(features):
    return {
        "pixel_values": torch.stack([f["pixel_values"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
    }


processor = TrOCRProcessor.from_pretrained(OCR_MODEL)
ocr_model = VisionEncoderDecoderModel.from_pretrained(OCR_MODEL)
ocr_model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
ocr_model.config.pad_token_id = processor.tokenizer.pad_token_id
ocr_model.config.vocab_size = ocr_model.config.decoder.vocab_size
ocr_model.config.eos_token_id = processor.tokenizer.sep_token_id
ocr_model.config.max_length = MAX_NEW_TOKENS
ocr_model.config.early_stopping = True
ocr_model.config.no_repeat_ngram_size = 3
ocr_model.config.length_penalty = 2.0
ocr_model.config.num_beams = 4


def compute_metrics(eval_pred):
    pred_ids = eval_pred.predictions
    label_ids = eval_pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = [normalize_text(x) for x in processor.batch_decode(pred_ids, skip_special_tokens=True)]
    label_str = [normalize_text(x) for x in processor.batch_decode(label_ids, skip_special_tokens=True)]
    return {"cer": jiwer.cer(label_str, pred_str), "wer": jiwer.wer(label_str, pred_str)}


training_args = Seq2SeqTrainingArguments(
    output_dir=str(OCR_MODEL_DIR),
    num_train_epochs=OCR_EPOCHS,
    per_device_train_batch_size=OCR_BATCH,
    per_device_eval_batch_size=OCR_BATCH,
    gradient_accumulation_steps=OCR_GRAD_ACCUM,
    learning_rate=OCR_LR,
    weight_decay=0.01,
    warmup_ratio=0.05,
    predict_with_generate=True,
    generation_max_length=MAX_NEW_TOKENS,
    fp16=torch.cuda.is_available(),
    logging_steps=25,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=ocr_model,
    args=training_args,
    train_dataset=RegionOCRDataset(ocr_train, processor),
    eval_dataset=RegionOCRDataset(ocr_val, processor),
    data_collator=collate_fn,
    processing_class=processor,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model(OCR_MODEL_DIR)
processor.save_pretrained(OCR_MODEL_DIR)

# %% [markdown]
# ## 10. 保存 test 图片并预测

# %%
test_records = []
for example in tqdm(test_ds, desc="save test images"):
    path = save_example_image(example, "test")
    test_records.append({"image_name": path.name, "image_path": str(path)})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ocr_model.to(device)
ocr_model.eval()


def ocr_crops(image, regions, batch_size=OCR_BATCH):
    crops = []
    width, height = image.size
    for region in regions:
        bbox = add_margin(region.bbox, width, height)
        crops.append(image.crop(bbox) if bbox else image)

    texts = []
    for start in range(0, len(crops), batch_size):
        batch = crops[start : start + batch_size]
        pixel_values = processor(images=batch, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated = ocr_model.generate(pixel_values, max_new_tokens=MAX_NEW_TOKENS, num_beams=4)
        texts.extend([normalize_text(x) for x in processor.batch_decode(generated, skip_special_tokens=True)])
    return texts


def predict_image_regions(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    result = detector.predict(
        source=str(image_path),
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        verbose=False,
    )[0]

    detected = []
    if result.boxes is not None:
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        for box, class_id in zip(xyxy, cls):
            bbox = tuple(int(round(float(v))) for v in box.tolist())
            bbox = clamp_bbox(bbox, width, height)
            if bbox is None:
                continue
            detected.append(Region(bbox=bbox, region_type=id_to_class.get(int(class_id), "handwritten"), text=""))

    detected = reading_order(detected)
    if detected:
        texts = ocr_crops(image, detected)
        detected = [Region(bbox=r.bbox, region_type=r.region_type, text=t) for r, t in zip(detected, texts)]
    return detected


pred_map = {}
for rec in tqdm(test_records, desc="predict test"):
    regions = predict_image_regions(Path(rec["image_path"]))
    pred_map[rec["image_name"]] = serialize_regions(regions)

# %% [markdown]
# ## 11. 生成 submission.csv

# %%
sample_paths = [
    Path("/kaggle/input/competitions/handwritten-to-data/sample_submission.csv"),
    Path("/kaggle/input/handwritten-to-data/sample_submission.csv"),
]
sample_path = next((p for p in sample_paths if p.exists()), None)
if sample_path is None:
    sample_path = Path(hf_hub_download(repo_id=DATASET_ID, repo_type="dataset", filename="sample_submission.csv"))

submission = pd.read_csv(sample_path)
image_col = submission.columns[0]
regions_col = submission.columns[1]
submission[regions_col] = submission[image_col].map(pred_map).fillna("[]")
submission.to_csv(SUBMISSION_PATH, index=False, quoting=csv.QUOTE_MINIMAL)

print("Saved:", SUBMISSION_PATH)
print(submission.head().to_string(index=False))

# %% [markdown]
# ## 12. 提交什么？
#
# 提交这个文件：
#
# ```text
# /kaggle/working/submission.csv
# ```
#
# 在 Kaggle Notebook 右侧 Output 里找到 `submission.csv`，点 Submit。
