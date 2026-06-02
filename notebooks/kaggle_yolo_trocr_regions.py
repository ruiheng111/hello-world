# %% [markdown]
# # Handwritten to Data - YOLO + TrOCR Regions Baseline
#
# 你上传的 `sample_submission.csv` 格式是：
#
# ```text
# image,regions
# ```
#
# `regions` 是 JSON list，每个元素包含：
#
# ```json
# {"bbox": [x1, y1, x2, y2], "type": "handwritten", "text": "..."}
# ```
#
# 所以更适合的方案是两阶段：
#
# 1. YOLOv8 检测每个 region 的 `bbox` 和 `type`
# 2. TrOCR 对每个裁剪出来的 region 识别 `text`
#
# 先跑通这个 baseline，再根据 public LB 和错误样例继续提分。

# %% [markdown]
# ## 1. 安装依赖
# 如果 Kaggle Notebook 没开 Internet，这一步会失败。先在右侧 Settings 打开 Internet。

# %%
import subprocess
import sys

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "-U",
    "ultralytics",
    "transformers",
    "accelerate",
    "jiwer",
    "sentencepiece",
])

# %% [markdown]
# ## 2. 导入库和参数

# %%
import ast
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import torch
import yaml
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


SEED = 42
N_SPLITS = 5
FOLD = 0

DATA_DIR = Path("/kaggle/input/handwritten-to-data")
WORK_DIR = Path("/kaggle/working")
YOLO_DIR = WORK_DIR / "yolo_regions"
OCR_MODEL_DIR = WORK_DIR / "trocr_region_ocr"
SUBMISSION_PATH = WORK_DIR / "submission.csv"

# 第一次先跑小一点，确认能成功提交；之后再调大。
YOLO_MODEL = "yolov8s.pt"  # 显存不够可改成 yolov8n.pt；想冲分可试 yolov8m.pt
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
# ## 3. 检查数据
# 如果后面报错，把这一段的输出发给我。

# %%
print("DATA_DIR:", DATA_DIR)
print("Top-level files:")
for p in sorted(DATA_DIR.iterdir()):
    print("-", p.name)

csv_paths = sorted(DATA_DIR.rglob("*.csv"))
print("\nCSV files:")
for p in csv_paths:
    df_tmp = pd.read_csv(p)
    print("\n-", p.relative_to(DATA_DIR))
    print("  shape:", df_tmp.shape)
    print("  columns:", list(df_tmp.columns))
    print(df_tmp.head(3).to_string(index=False))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
image_paths_all = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
print("\nImage count:", len(image_paths_all))
for p in image_paths_all[:20]:
    print("-", p.relative_to(DATA_DIR))

# %% [markdown]
# ## 4. 数据读取和 regions 解析工具

# %%
ID_CANDIDATES = ["image", "id", "ID", "Id", "image_id", "file_id", "filename", "file_name"]
IMAGE_CANDIDATES = ["image", "image_path", "path", "filepath", "file_path", "filename", "file_name"]
REGIONS_CANDIDATES = ["regions", "region", "annotations", "labels"]


@dataclass(frozen=True)
class Region:
    bbox: tuple[int, int, int, int]
    region_type: str
    text: str


def find_csv(candidates):
    for name in candidates:
        direct = DATA_DIR / name
        if direct.exists():
            return direct
        matches = list(DATA_DIR.rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Cannot find any of: {candidates}")


def first_existing_column(columns, candidates):
    column_set = set(columns)
    for c in candidates:
        if c in column_set:
            return c
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        found = lower_map.get(c.lower())
        if found is not None:
            return found
    return None


def detect_required_columns(df, require_regions):
    id_col = first_existing_column(df.columns, ID_CANDIDATES)
    image_col = first_existing_column(df.columns, IMAGE_CANDIDATES)
    regions_col = first_existing_column(df.columns, REGIONS_CANDIDATES)
    if id_col is None:
        raise ValueError(f"Cannot detect image/id column. Columns: {list(df.columns)}")
    if require_regions and regions_col is None:
        raise ValueError(f"Cannot detect regions column. Columns: {list(df.columns)}")
    return id_col, image_col, regions_col


def parse_regions(value):
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, list):
        raw_regions = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            raw_regions = json.loads(text)
        except json.JSONDecodeError:
            raw_regions = ast.literal_eval(text)

    regions = []
    for item in raw_regions:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            continue
        regions.append(
            Region(
                bbox=(x1, y1, x2, y2),
                region_type=str(item.get("type", "handwritten")),
                text=str(item.get("text", "")),
            )
        )
    return regions


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


def candidate_image_names(value):
    raw = str(value)
    path = Path(raw)
    names = [raw]
    if path.name != raw:
        names.append(path.name)
    if path.suffix.lower() in IMAGE_EXTS:
        return list(dict.fromkeys(names))
    for ext in IMAGE_EXTS:
        names.append(raw + ext)
        if path.name != raw:
            names.append(path.name + ext)
    return list(dict.fromkeys(names))


def resolve_image_path(value):
    search_dirs = [
        DATA_DIR,
        DATA_DIR / "train",
        DATA_DIR / "test",
        DATA_DIR / "images",
        DATA_DIR / "train_images",
        DATA_DIR / "test_images",
        DATA_DIR / "images" / "train",
        DATA_DIR / "images" / "test",
    ]
    for name in candidate_image_names(value):
        for d in search_dirs:
            candidate = d / name
            if candidate.exists():
                return candidate
    # 最后用全局图片索引兜底。
    basename = Path(str(value)).name
    for p in image_paths_all:
        if p.name == basename or p.stem == Path(basename).stem:
            return p
    raise FileNotFoundError(f"Cannot resolve image path for {value!r}")


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
    return sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0], r.bbox[3], r.bbox[2]))

# %% [markdown]
# ## 5. 读取 train/test/sample_submission

# %%
train_csv = find_csv(["train.csv", "annotations.csv", "train_annotations.csv"])
test_csv = find_csv(["test.csv"])
sample_csv = find_csv(["sample_submission.csv"])

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
sample_submission = pd.read_csv(sample_csv)

train_id_col, train_image_col, regions_col = detect_required_columns(train_df, require_regions=True)
test_id_col, test_image_col, _ = detect_required_columns(test_df, require_regions=False)

print("train_csv:", train_csv)
print("test_csv:", test_csv)
print("sample_csv:", sample_csv)
print("train columns:", list(train_df.columns))
print("test columns:", list(test_df.columns))
print("sample columns:", list(sample_submission.columns))
print("detected:", train_id_col, train_image_col, regions_col, test_id_col, test_image_col)

train_df["image_path"] = [
    str(resolve_image_path(row[train_image_col or train_id_col]))
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="resolve train")
]
test_df["image_path"] = [
    str(resolve_image_path(row[test_image_col or test_id_col]))
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="resolve test")
]

train_df["parsed_regions"] = train_df[regions_col].apply(parse_regions)
class_names = sorted({r.region_type for regions in train_df["parsed_regions"] for r in regions})
if not class_names:
    raise ValueError("No training regions found. Please send train.csv columns and first rows to me.")
class_to_id = {name: idx for idx, name in enumerate(class_names)}
id_to_class = {idx: name for name, idx in class_to_id.items()}
print("class_to_id:", class_to_id)
print("regions per image:")
print(train_df["parsed_regions"].apply(len).describe())

# %% [markdown]
# ## 6. 准备 YOLO 检测数据集

# %%
def prepare_yolo_dataset():
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)

    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(kfold.split(train_df)):
        train_df.loc[val_idx, "fold"] = fold

    for split_name, split_df in [
        ("train", train_df[train_df["fold"] != FOLD]),
        ("val", train_df[train_df["fold"] == FOLD]),
    ]:
        image_out = YOLO_DIR / split_name / "images"
        label_out = YOLO_DIR / split_name / "labels"
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"prepare yolo {split_name}"):
            src = Path(row["image_path"])
            dst = image_out / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

            with Image.open(src) as image:
                width, height = image.size

            label_lines = []
            for region in row["parsed_regions"]:
                bbox = clamp_bbox(region.bbox, width, height)
                if bbox is None:
                    continue
                cx, cy, w, h = bbox_to_yolo(bbox, width, height)
                label_lines.append(
                    f"{class_to_id[region.region_type]} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}"
                )
            (label_out / f"{src.stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")

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
# ## 7. 训练 YOLO 检测 bbox/type
#
# 如果训练太慢：把 `YOLO_MODEL` 改为 `yolov8n.pt`，或者把 `YOLO_EPOCHS` 改小。
# 如果想冲分：训练成功后可尝试 `yolov8m.pt`、更大 `YOLO_EPOCHS`、更大 `YOLO_IMGSZ`。

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
# ## 8. 准备 OCR 训练数据
#
# OCR 只训练有文字的 region。`formula` 也先一起训练，后面如果公式识别差，再单独处理。

# %%
ocr_rows = []
for _, row in train_df.iterrows():
    image_path = row["image_path"]
    for region in row["parsed_regions"]:
        if str(region.text).strip() == "":
            continue
        ocr_rows.append(
            {
                "image_path": image_path,
                "bbox": region.bbox,
                "type": region.region_type,
                "text": region.text,
                "fold": row["fold"],
            }
        )
ocr_df = pd.DataFrame(ocr_rows)
print("ocr_df:", ocr_df.shape)
print(ocr_df["type"].value_counts())
print(ocr_df.head())

ocr_train = ocr_df[ocr_df["fold"] != FOLD].reset_index(drop=True)
ocr_val = ocr_df[ocr_df["fold"] == FOLD].reset_index(drop=True)

# %% [markdown]
# ## 9. 训练 TrOCR 识别裁剪区域文字

# %%
def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


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
        text = normalize_text(row["text"])
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
        ).input_ids
        labels = [x if x != self.processor.tokenizer.pad_token_id else -100 for x in labels]
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


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
    return {
        "cer": jiwer.cer(label_str, pred_str),
        "wer": jiwer.wer(label_str, pred_str),
    }


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
# ## 10. 对 test 图片检测 + OCR

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ocr_model.to(device)
ocr_model.eval()


def ocr_crops(image, regions, batch_size=OCR_BATCH):
    texts = []
    crops = []
    width, height = image.size
    for region in regions:
        bbox = add_margin(region.bbox, width, height)
        if bbox is None:
            crops.append(image)
        else:
            crops.append(image.crop(bbox))

    for start in range(0, len(crops), batch_size):
        batch = crops[start : start + batch_size]
        pixel_values = processor(images=batch, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated = ocr_model.generate(
                pixel_values,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,
            )
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
            region_type = id_to_class.get(int(class_id), "handwritten")
            detected.append(Region(bbox=bbox, region_type=region_type, text=""))

    detected = reading_order(detected)
    if detected:
        texts = ocr_crops(image, detected)
        detected = [
            Region(bbox=r.bbox, region_type=r.region_type, text=t)
            for r, t in zip(detected, texts)
        ]
    return detected


pred_map = {}
for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="predict test"):
    image_name = row[test_id_col]
    regions = predict_image_regions(Path(row["image_path"]))
    pred_map[image_name] = serialize_regions(regions)

# %% [markdown]
# ## 11. 生成 submission.csv
#
# 你上传的 sample_submission 是 `image,regions`，这里会覆盖所有 `regions`。

# %%
submission = sample_submission.copy()
sub_image_col = sample_submission.columns[0]
sub_regions_col = sample_submission.columns[1]

if sub_image_col in test_df.columns:
    test_key_col = sub_image_col
else:
    test_key_col = test_id_col

if test_key_col != test_id_col:
    pred_map = {
        row[test_key_col]: pred_map[row[test_id_col]]
        for _, row in test_df.iterrows()
    }

submission[sub_regions_col] = submission[sub_image_col].map(pred_map).fillna("[]")
submission.to_csv(SUBMISSION_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
print("Saved:", SUBMISSION_PATH)
print(submission.head().to_string(index=False))

# %% [markdown]
# ## 12. 提交
#
# Kaggle Notebook 右侧 Output 会出现 `submission.csv`，可以直接点 Submit。
# 也可以取消下面一行注释运行。

# %%
# subprocess.check_call([
#     "kaggle",
#     "competitions",
#     "submit",
#     "-c",
#     "handwritten-to-data",
#     "-f",
#     str(SUBMISSION_PATH),
#     "-m",
#     "yolo + trocr regions baseline",
# ])

# %% [markdown]
# ## 13. 下一轮提分方向
#
# 跑出第一版分数后，把这些发给我：
#
# - public LB 分数
# - 第 10 名分数
# - 训练日志里 YOLO 的 mAP
# - TrOCR 验证集 CER/WER
# - 几个错误样例截图：漏框、错框、文字识别错、type 错
#
# 然后可以继续做：
#
# - YOLOv8m / YOLOv8l、更高分辨率、更久训练
# - 按 `handwritten` 和 `formula` 分开训练 OCR
# - 用 PaddleOCR / LaTeX OCR / formula 专用模型处理公式
# - 图像增强、旋转矫正、裁剪边距优化
# - pseudo-labeling 和模型 ensemble
