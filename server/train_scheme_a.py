#!/usr/bin/env python3
"""
Handwritten to Data — 方案 A（YOLO 检测 + TrOCR 识别）
在离线 4090 服务器上运行。先 source env.sh

用法:
  python train_scheme_a.py --quick          # 快速试跑
  python train_scheme_a.py                  # 完整训练
  python train_scheme_a.py --infer-only     # 仅推理（需已有权重）
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 路径：全部在数据盘
# ---------------------------------------------------------------------------
ML_ROOT = Path(os.environ.get("ML_ROOT", "/data1/ml-2-3"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", ML_ROOT / "data" / "rukopys"))
WORK = Path(os.environ.get("WORK_DIR", ML_ROOT / "output" / "scheme_a"))
YOLO_DIR = WORK / "yolo_rukopys"
REC_DIR = WORK / "rec_crops"
DET_WEIGHTS = WORK / "det_best.pt"
REC_MODEL_DIR = WORK / "trocr_uk"
SUBMISSION_PATH = WORK / "submission.csv"

PRETRAINED_YOLO = Path(os.environ.get("PRETRAINED_YOLO", ML_ROOT / "pretrained" / "yolov8n.pt"))
PRETRAINED_TROCR = Path(os.environ.get("PRETRAINED_TROCR", ML_ROOT / "pretrained" / "trocr-base-handwritten"))

TRAIN_IMG_DIR = DATA_ROOT / "train" / "images"
TEST_IMG_DIR = DATA_ROOT / "test" / "images"
TRAIN_META = DATA_ROOT / "train" / "metadata.jsonl"
TEST_META = DATA_ROOT / "test" / "metadata.jsonl"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"

REGION_TYPES = ["handwritten", "printed", "formula", "table", "annotation", "image", "graph"]
TYPE2ID = {t: i for i, t in enumerate(REGION_TYPES)}
ID2TYPE = {i: t for t, i in TYPE2ID.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="少量数据快速试跑")
    p.add_argument("--infer-only", action="store_true", help="跳过训练，仅生成 submission")
    p.add_argument("--det-epochs", type=int, default=None)
    p.add_argument("--rec-epochs", type=int, default=None)
    return p.parse_args()


def load_jsonl(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def image_id_from_row(row):
    return Path(row["file_name"]).name


def yolo_label_line(bbox, cls_id, w, h):
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"


def build_yolo_dataset(rows, split_name, train_img_dir):
    img_out = YOLO_DIR / "images" / split_name
    lbl_out = YOLO_DIR / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for row in tqdm(rows, desc=f"YOLO {split_name}"):
        img_name = image_id_from_row(row)
        src = train_img_dir / img_name
        if not src.exists():
            continue
        dst = img_out / img_name
        if not dst.exists():
            shutil.copy2(src, dst)
        w, h = row["image_width"], row["image_height"]
        lines = []
        for reg in row.get("regions") or []:
            t = reg.get("type", "handwritten")
            if t not in TYPE2ID:
                continue
            line = yolo_label_line(reg["bbox"], TYPE2ID[t], w, h)
            if line:
                lines.append(line)
        with open(lbl_out / (Path(img_name).stem + ".txt"), "w") as f:
            f.writelines(lines)


def crop_region(img, bbox, pad=4):
    x1, y1, x2, y2 = bbox
    w, h = img.size
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img.crop((x1, y1, x2, y2))


def train_detector(device, det_epochs, img_size, batch_det):
    from ultralytics import YOLO

    weights = str(PRETRAINED_YOLO) if PRETRAINED_YOLO.exists() else "yolov8n.pt"
    det_model = YOLO(weights)
    yaml_path = YOLO_DIR / "data.yaml"
    det_results = det_model.train(
        data=str(yaml_path),
        epochs=det_epochs,
        imgsz=img_size,
        batch=batch_det,
        device=0 if device == "cuda" else "cpu",
        project=str(WORK),
        name="yolo_train",
        exist_ok=True,
        patience=10,
        verbose=True,
    )
    best_pt = Path(det_results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best_pt, DET_WEIGHTS)
    return YOLO(str(DET_WEIGHTS))


def build_rec_crops(train_rows, train_img_dir, va_rows, max_samples):
    crop_train = REC_DIR / "train"
    crop_val = REC_DIR / "val"
    crop_train.mkdir(parents=True, exist_ok=True)
    crop_val.mkdir(parents=True, exist_ok=True)
    val_ids = {image_id_from_row(r) for r in va_rows}
    samples = []
    counter = 0
    for row in tqdm(train_rows, desc="Build REC crops"):
        img_name = image_id_from_row(row)
        img_path = train_img_dir / img_name
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        split_dir = crop_val if img_name in val_ids else crop_train
        for j, reg in enumerate(row.get("regions") or []):
            t = reg.get("type", "handwritten")
            if t in ("image", "graph"):
                continue
            text = (reg.get("text") or "").strip()
            if not text:
                continue
            crop = crop_region(img, reg["bbox"])
            if crop is None or crop.width < 8 or crop.height < 8:
                continue
            out_path = split_dir / f"{Path(img_name).stem}_{j}.jpg"
            crop.save(out_path, quality=90)
            samples.append({"path": str(out_path), "text": text, "type": t})
            counter += 1
            if max_samples and counter >= max_samples:
                break
        if max_samples and counter >= max_samples:
            break
    df = pd.DataFrame(samples)
    train_df = df[df["path"].str.contains("/train/")].reset_index(drop=True)
    val_df = df[df["path"].str.contains("/val/")].reset_index(drop=True)
    return train_df, val_df


def train_recognizer(device, rec_epochs, batch_rec, train_df, val_df):
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        default_data_collator,
    )

    trocr_path = str(PRETRAINED_TROCR)
    if not PRETRAINED_TROCR.exists():
        trocr_path = "microsoft/trocr-base-handwritten"

    processor = TrOCRProcessor.from_pretrained(trocr_path, local_files_only=PRETRAINED_TROCR.exists())
    ocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_path, local_files_only=PRETRAINED_TROCR.exists())

    chars = set()
    for txt in pd.concat([train_df["text"], val_df["text"]], ignore_index=True):
        chars.update(list(str(txt)))
    extra = sorted(chars - set(processor.tokenizer.get_vocab().keys()))
    if extra:
        processor.tokenizer.add_tokens(extra)
        ocr_model.decoder.resize_token_embeddings(len(processor.tokenizer))
        print(f"Added {len(extra)} tokens")

    ocr_model.config.eos_token_id = processor.tokenizer.eos_token_id
    ocr_model.config.pad_token_id = processor.tokenizer.pad_token_id
    ocr_model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    ocr_model.config.max_length = 128

    max_len = 128
    target_h = 64

    class RecDataset(Dataset):
        def __init__(self, frame):
            self.frame = frame.reset_index(drop=True)

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, idx):
            row = self.frame.iloc[idx]
            img = Image.open(row["path"]).convert("RGB")
            w, h = img.size
            new_w = max(1, int(w * (target_h / h)))
            img = img.resize((new_w, target_h), Image.BICUBIC)
            pixel_values = processor(img, return_tensors="pt").pixel_values.squeeze(0)
            labels = processor.tokenizer(
                row["text"],
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            labels[labels == processor.tokenizer.pad_token_id] = -100
            return {"pixel_values": pixel_values, "labels": labels}

    train_ds = RecDataset(train_df)
    val_ds = RecDataset(val_df if len(val_df) else train_df.head(50))

    eval_kw = "evaluation_strategy"  # 兼容旧版 transformers
    import inspect
    if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters:
        eval_kw = "eval_strategy"

    args_kwargs = dict(
        output_dir=str(REC_MODEL_DIR),
        per_device_train_batch_size=batch_rec,
        per_device_eval_batch_size=batch_rec,
        num_train_epochs=rec_epochs,
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=4e-5,
        predict_with_generate=True,
        generation_max_length=max_len,
        fp16=device == "cuda",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )
    args_kwargs[eval_kw] = "epoch"

    args = Seq2SeqTrainingArguments(**args_kwargs)
    trainer = Seq2SeqTrainer(
        model=ocr_model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(str(REC_MODEL_DIR))
    processor.save_pretrained(str(REC_MODEL_DIR))
    ocr_model = VisionEncoderDecoderModel.from_pretrained(str(REC_MODEL_DIR)).to(device)
    ocr_model.eval()
    return ocr_model, processor, max_len, target_h


@torch.inference_mode()
def run_inference(det_model, ocr_model, processor, max_len, target_h, device, conf_det, iou_det):
    from ultralytics import YOLO
    if isinstance(det_model, str):
        det_model = YOLO(det_model)

    def recognize_crop(pil_img):
        w, h = pil_img.size
        new_w = max(1, int(w * (target_h / max(h, 1))))
        img = pil_img.resize((new_w, target_h), Image.BICUBIC)
        pv = processor(img, return_tensors="pt").pixel_values.to(device)
        gen = ocr_model.generate(pv, max_length=max_len)
        return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def detect_regions(pil_img):
        res = det_model.predict(source=np.array(pil_img), conf=conf_det, iou=iou_det, verbose=False)[0]
        regions = []
        if res.boxes is None or len(res.boxes) == 0:
            return regions
        for box in res.boxes:
            cls_id = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            regions.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "type": ID2TYPE.get(cls_id, "handwritten"),
            })
        return regions

    def sort_regions(regions):
        def key(r):
            x1, y1, x2, y2 = r["bbox"]
            return (int((y1 + y2) / 2) // 15, (x1 + x2) / 2)
        return sorted(regions, key=key)

    def predict_page(img_path):
        img = Image.open(img_path).convert("RGB")
        dets = sort_regions(detect_regions(img))
        out = []
        for reg in dets:
            t = reg["type"]
            if t in ("image", "graph"):
                out.append({"bbox": reg["bbox"], "type": t, "text": ""})
                continue
            crop = crop_region(img, reg["bbox"])
            if crop is None:
                continue
            text = recognize_crop(crop)
            out.append({"bbox": reg["bbox"], "type": t, "text": text})
        return out

    if SAMPLE_SUB.exists():
        sub_df = pd.read_csv(SAMPLE_SUB)
        test_images = sub_df["image"].tolist()
    else:
        test_images = sorted(p.name for p in TEST_IMG_DIR.glob("*.jpg"))
        sub_df = pd.DataFrame({"image": test_images})

    preds = []
    for img_name in tqdm(test_images, desc="Predict test"):
        img_path = TEST_IMG_DIR / img_name
        if not img_path.exists():
            preds.append("[]")
            continue
        preds.append(json.dumps(predict_page(img_path), ensure_ascii=False))

    sub_df["regions"] = preds
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(SUBMISSION_PATH, index=False)
    print("Saved:", SUBMISSION_PATH, sub_df.shape)


def main():
    args = parse_args()
    quick = args.quick
    det_epochs = args.det_epochs or (3 if quick else 30)
    rec_epochs = args.rec_epochs or (2 if quick else 8)
    max_pages = 80 if quick else None
    max_rec = 2000 if quick else None
    img_size = 640 if quick else 960
    batch_det = 8
    batch_rec = 16 if not quick else 8  # 4090 可适当加大

    for p in [TRAIN_META, TRAIN_IMG_DIR, TEST_IMG_DIR]:
        if not p.exists():
            raise FileNotFoundError(f"缺少数据: {p}\n请把 RUKOPYS 放到 {DATA_ROOT}")

    WORK.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    train_rows = load_jsonl(TRAIN_META)
    if max_pages:
        train_rows = train_rows[:max_pages]

    idx = list(range(len(train_rows)))
    random.shuffle(idx)
    n_val = max(1, int(0.1 * len(idx)))
    va_rows = [train_rows[i] for i in idx[:n_val]]
    tr_rows = [train_rows[i] for i in idx[n_val:]]

    if not args.infer_only:
        for p in [YOLO_DIR / "images" / "train", YOLO_DIR / "labels" / "train",
                  YOLO_DIR / "images" / "val", YOLO_DIR / "labels" / "val"]:
            if p.exists():
                shutil.rmtree(p)
        build_yolo_dataset(tr_rows, "train", TRAIN_IMG_DIR)
        build_yolo_dataset(va_rows, "val", TRAIN_IMG_DIR)
        yaml_path = YOLO_DIR / "data.yaml"
        yaml_path.write_text(
            f"path: {YOLO_DIR}\ntrain: images/train\nval: images/val\n"
            f"nc: {len(REGION_TYPES)}\nnames: {REGION_TYPES}\n",
            encoding="utf-8",
        )
        det_model = train_detector(device, det_epochs, img_size, batch_det)

        train_df, val_df = build_rec_crops(train_rows, TRAIN_IMG_DIR, va_rows, max_rec)
        print("REC train:", len(train_df), "val:", len(val_df))
        ocr_model, processor, max_len, target_h = train_recognizer(
            device, rec_epochs, batch_rec, train_df, val_df
        )
    else:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from ultralytics import YOLO
        if not DET_WEIGHTS.exists() or not REC_MODEL_DIR.exists():
            raise FileNotFoundError("infer-only 需要已有 det_best.pt 和 trocr_uk/")
        det_model = YOLO(str(DET_WEIGHTS))
        processor = TrOCRProcessor.from_pretrained(str(REC_MODEL_DIR))
        ocr_model = VisionEncoderDecoderModel.from_pretrained(str(REC_MODEL_DIR)).to(device)
        ocr_model.eval()
        max_len, target_h = 128, 64

    run_inference(det_model, ocr_model, processor, max_len, target_h, device, 0.15, 0.45)


if __name__ == "__main__":
    main()
