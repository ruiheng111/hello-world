from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from src.htd.config import ensure_dir, load_config
from src.htd.data import detect_columns
from src.htd.metrics import char_error_rate, normalize_text, word_error_rate


class OCRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        processor: TrOCRProcessor,
        image_col: str,
        target_col: str,
        max_target_length: int,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.image_col = image_col
        self.target_col = target_col
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image = Image.open(row[self.image_col]).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        text = normalize_text(row[self.target_col])
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids
        labels = [token if token != self.processor.tokenizer.pad_token_id else -100 for token in labels]
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class DataCollator:
    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.stack([f["pixel_values"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/baseline.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    return parser.parse_args()


def build_compute_metrics(processor: TrOCRProcessor):
    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = [normalize_text(x) for x in pred_str]
        label_str = [normalize_text(x) for x in label_str]
        return {
            "cer": char_error_rate(label_str, pred_str),
            "wer": word_error_rate(label_str, pred_str),
        }

    return compute_metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    validation = cfg["validation"]

    model_dir = ensure_dir(paths["model_dir"])
    folds_path = Path(paths["processed_dir"]) / "train_folds.csv"
    if not folds_path.exists():
        raise FileNotFoundError("Run scripts/01_make_folds.py before training.")

    df = pd.read_csv(folds_path)
    spec = detect_columns(df, cfg.get("columns", {}), require_target=True)
    image_col = "resolved_image_path"
    target_col = spec.target_col
    if target_col is None:
        raise ValueError("Target column is required for training.")

    fold = int(validation["fold"])
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    if args.max_train_samples:
        train_df = train_df.head(args.max_train_samples)
    if args.max_val_samples:
        val_df = val_df.head(args.max_val_samples)

    model_name = model_cfg["name"]
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = int(model_cfg["max_new_tokens"])
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    train_ds = OCRDataset(
        train_df,
        processor,
        image_col=image_col,
        target_col=target_col,
        max_target_length=int(model_cfg["max_target_length"]),
    )
    val_ds = OCRDataset(
        val_df,
        processor,
        image_col=image_col,
        target_col=target_col,
        max_target_length=int(model_cfg["max_target_length"]),
    )

    epochs = args.epochs or int(train_cfg["epochs"])
    batch_size = args.batch_size or int(train_cfg["batch_size"])
    grad_accum = args.grad_accum or int(train_cfg["grad_accum"])
    learning_rate = args.learning_rate or float(train_cfg["learning_rate"])

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(model_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        predict_with_generate=True,
        generation_max_length=int(model_cfg["max_new_tokens"]),
        fp16=bool(train_cfg["fp16"]) and torch.cuda.is_available(),
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        dataloader_num_workers=int(train_cfg["num_workers"]),
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollator(),
        processing_class=processor,
        compute_metrics=build_compute_metrics(processor),
    )
    trainer.train()
    trainer.save_model(model_dir)
    processor.save_pretrained(model_dir)
    print(f"Saved model and processor to {model_dir}")


if __name__ == "__main__":
    main()
