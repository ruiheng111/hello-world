# %% [markdown]
# # Handwritten to Data - Kaggle Notebook TrOCR Baseline
#
# 使用方法：
# 1. 在 Kaggle 比赛页面点击 Code -> New Notebook
# 2. 右侧 Settings 打开 GPU
# 3. Add Input 添加 `handwritten-to-data` 比赛数据
# 4. 把本文件代码复制到 Kaggle Notebook 里运行
#
# 先跑通 baseline，再根据分数和错误样例继续优化。

# %% [markdown]
# ## 1. 安装依赖
# Kaggle 环境通常已经有 torch/pandas/sklearn。这里升级/安装 OCR 需要的包。

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
    "transformers",
    "evaluate",
    "jiwer",
    "accelerate",
    "sentencepiece",
])

# %% [markdown]
# ## 2. 导入库和参数

# %%
import os
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import torch
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

SEED = 42
N_SPLITS = 5
FOLD = 0

MODEL_NAME = "microsoft/trocr-base-handwritten"
MAX_TARGET_LENGTH = 128
MAX_NEW_TOKENS = 128

# 第一次建议小一点，确认能跑通后再调大。
EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUM = 8
LEARNING_RATE = 5e-5

DATA_DIR = Path("/kaggle/input/handwritten-to-data")
WORK_DIR = Path("/kaggle/working")
MODEL_DIR = WORK_DIR / "trocr-baseline"
SUBMISSION_PATH = WORK_DIR / "submission.csv"

torch.manual_seed(SEED)
np.random.seed(SEED)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## 3. 检查比赛数据
#
# 先运行这一段，把输出发给我，我可以帮你确认列名是否识别正确。

# %%
print("DATA_DIR:", DATA_DIR)
print("Top-level files:")
for p in sorted(DATA_DIR.iterdir()):
    print("-", p.name)

csv_files = sorted(DATA_DIR.rglob("*.csv"))
print("\nCSV files:")
for p in csv_files:
    print("-", p.relative_to(DATA_DIR))
    df_tmp = pd.read_csv(p)
    print("  shape:", df_tmp.shape)
    print("  columns:", list(df_tmp.columns))
    print(df_tmp.head(3).to_string(index=False))

image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
image_files = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in image_exts]
print("\nImage count:", len(image_files))
print("First images:")
for p in image_files[:20]:
    print("-", p.relative_to(DATA_DIR))

# %% [markdown]
# ## 4. 自动识别列名和图片路径
#
# 如果这里报错，说明比赛列名比较特殊。把报错和上一步输出发给我。

# %%
ID_CANDIDATES = ["id", "ID", "Id", "image_id", "file_id", "filename", "file_name"]
IMAGE_CANDIDATES = ["image", "image_path", "path", "filepath", "file_path", "filename", "file_name"]
TARGET_CANDIDATES = ["text", "label", "target", "transcription", "ground_truth", "answer", "value"]
TRAIN_IMAGE_DIRS = ["train", "train_images", "images/train", "images"]
TEST_IMAGE_DIRS = ["test", "test_images", "images/test", "images"]


def find_csv(name: str) -> Path:
    direct = DATA_DIR / name
    if direct.exists():
        return direct
    matches = list(DATA_DIR.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Cannot find {name} under {DATA_DIR}")
    return matches[0]


def first_existing_column(columns, candidates):
    column_set = set(columns)
    for c in candidates:
        if c in column_set:
            return c
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def detect_columns(df: pd.DataFrame, require_target: bool):
    id_col = first_existing_column(df.columns, ID_CANDIDATES)
    image_col = first_existing_column(df.columns, IMAGE_CANDIDATES)
    target_col = first_existing_column(df.columns, TARGET_CANDIDATES)

    if id_col is None:
        raise ValueError(f"Cannot detect id column. Available columns: {list(df.columns)}")
    if require_target and target_col is None:
        raise ValueError(f"Cannot detect target column. Available columns: {list(df.columns)}")
    return id_col, image_col, target_col


def candidate_image_names(value):
    raw = str(value)
    path = Path(raw)
    names = [raw]
    if path.name != raw:
        names.append(path.name)
    if path.suffix.lower() in image_exts:
        return list(dict.fromkeys(names))
    for ext in image_exts:
        names.append(raw + ext)
        if path.name != raw:
            names.append(path.name + ext)
    return list(dict.fromkeys(names))


def resolve_image_path(row, id_col, image_col, image_dirs):
    value = row[image_col] if image_col else row[id_col]
    for name in candidate_image_names(value):
        direct = DATA_DIR / name
        if direct.exists():
            return direct
        for d in image_dirs:
            candidate = DATA_DIR / d / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Cannot resolve image for value={value!r}")


train_csv = find_csv("train.csv")
test_csv = find_csv("test.csv")
sample_submission_csv = find_csv("sample_submission.csv")

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
sample_submission = pd.read_csv(sample_submission_csv)

id_col, image_col, target_col = detect_columns(train_df, require_target=True)
test_id_col, test_image_col, _ = detect_columns(test_df, require_target=False)

print("Detected train columns:")
print("id_col:", id_col)
print("image_col:", image_col)
print("target_col:", target_col)
print("\nDetected test columns:")
print("test_id_col:", test_id_col)
print("test_image_col:", test_image_col)
print("\nSample submission columns:", list(sample_submission.columns))

train_df["image_path"] = [
    str(resolve_image_path(row, id_col, image_col, TRAIN_IMAGE_DIRS))
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="resolve train images")
]
test_df["image_path"] = [
    str(resolve_image_path(row, test_id_col, test_image_col, TEST_IMAGE_DIRS))
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="resolve test images")
]

print(train_df[[id_col, "image_path", target_col]].head())
print(test_df[[test_id_col, "image_path"]].head())

# %% [markdown]
# ## 5. 建本地验证集

# %%
kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
train_df["fold"] = -1
for fold, (_, val_idx) in enumerate(kfold.split(train_df)):
    train_df.loc[val_idx, "fold"] = fold

trn_df = train_df[train_df["fold"] != FOLD].reset_index(drop=True)
val_df = train_df[train_df["fold"] == FOLD].reset_index(drop=True)

print("train:", trn_df.shape, "val:", val_df.shape)
print(train_df["fold"].value_counts().sort_index())

# %% [markdown]
# ## 6. Dataset 和指标

# %%
def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


class OCRDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        text = normalize_text(row[target_col])
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


def data_collator(features):
    return {
        "pixel_values": torch.stack([f["pixel_values"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
    }


def compute_metrics(eval_pred):
    pred_ids = eval_pred.predictions
    label_ids = eval_pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    pred_str = [normalize_text(x) for x in pred_str]
    label_str = [normalize_text(x) for x in label_str]
    return {
        "cer": jiwer.cer(label_str, pred_str),
        "wer": jiwer.wer(label_str, pred_str),
    }

# %% [markdown]
# ## 7. 训练 TrOCR
#
# 第一次 `EPOCHS=1` 只是确认流程跑通。跑通后可以改成 3、5 或更多。

# %%
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = MAX_NEW_TOKENS
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

train_dataset = OCRDataset(trn_df, processor)
val_dataset = OCRDataset(val_df, processor)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(MODEL_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
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
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=processor,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(MODEL_DIR)
processor.save_pretrained(MODEL_DIR)

# %% [markdown]
# ## 8. 预测 test

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

predictions = []
for start in tqdm(range(0, len(test_df), BATCH_SIZE), desc="predict test"):
    batch = test_df.iloc[start : start + BATCH_SIZE]
    images = [Image.open(p).convert("RGB") for p in batch["image_path"]]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
        )
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    predictions.extend([normalize_text(x) for x in text])

test_df["prediction"] = predictions
print(test_df[[test_id_col, "prediction"]].head())

# %% [markdown]
# ## 9. 生成 submission.csv
#
# 如果这里列名不对，把 `sample_submission.head()` 和报错发给我。

# %%
submission = sample_submission.copy()
sub_id_col = sample_submission.columns[0]
prediction_cols = [c for c in sample_submission.columns if c != sub_id_col]
if not prediction_cols:
    raise ValueError("sample_submission.csv 没有预测列")
sub_pred_col = prediction_cols[0]

if sub_id_col in test_df.columns:
    pred_map = test_df.set_index(sub_id_col)["prediction"].to_dict()
else:
    pred_map = test_df.set_index(test_id_col)["prediction"].to_dict()

submission[sub_pred_col] = submission[sub_id_col].map(pred_map).fillna("")
submission.to_csv(SUBMISSION_PATH, index=False)

print("Saved:", SUBMISSION_PATH)
print(submission.head())

# %% [markdown]
# ## 10. 提交
#
# Kaggle Notebook 右侧 Output 里会出现 `submission.csv`。
# 可以直接点 Submit，也可以运行：

# %%
# !kaggle competitions submit -c handwritten-to-data -f /kaggle/working/submission.csv -m "trocr baseline"
