from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.htd.config import ensure_dir, load_config
from src.htd.data import add_image_paths, detect_columns, read_csv
from src.htd.metrics import normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/baseline.yaml")
    parser.add_argument("--model-dir")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def load_images(paths: list[str], processor: TrOCRProcessor, device: torch.device) -> torch.Tensor:
    images = [Image.open(path).convert("RGB") for path in paths]
    return processor(images=images, return_tensors="pt").pixel_values.to(device)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]
    model_cfg = cfg["model"]

    data_dir = Path(paths["data_dir"])
    test = read_csv(data_dir / paths["test_csv"])
    spec = detect_columns(test, cfg.get("columns", {}), require_target=False)
    test = add_image_paths(test, spec, data_dir, paths["test_image_dirs"])
    if args.max_samples:
        test = test.head(args.max_samples).copy()

    model_dir = Path(args.model_dir or paths["model_dir"])
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions: list[str] = []
    batch_size = args.batch_size
    image_paths = test["resolved_image_path"].tolist()
    for start in tqdm(range(0, len(image_paths), batch_size), desc="predict"):
        batch_paths = image_paths[start : start + batch_size]
        pixel_values = load_images(batch_paths, processor, device)
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=int(model_cfg["max_new_tokens"]),
                num_beams=4,
            )
        batch_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(normalize_text(x) for x in batch_text)

    out_dir = ensure_dir(paths["submission_dir"])
    pred_path = out_dir / "predictions.csv"
    out = test.copy()
    out["prediction"] = predictions
    out.to_csv(pred_path, index=False)
    print(f"Wrote {pred_path} with shape={out.shape}")


if __name__ == "__main__":
    main()
