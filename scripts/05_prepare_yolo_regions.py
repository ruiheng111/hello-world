from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import KFold

from src.htd.config import ensure_dir, load_config
from src.htd.data import add_image_paths, detect_columns, read_csv
from src.htd.regions import bbox_to_yolo, clamp_bbox, parse_regions


def find_regions_column(df: pd.DataFrame) -> str:
    for name in ("regions", "region", "annotations", "labels"):
        if name in df.columns:
            return name
    raise ValueError(f"Could not find regions column. Available columns: {list(df.columns)}")


def write_split(
    df: pd.DataFrame,
    split_dir: Path,
    image_col: str,
    regions_col: str,
    class_to_id: dict[str, int],
) -> None:
    image_out_dir = ensure_dir(split_dir / "images")
    label_out_dir = ensure_dir(split_dir / "labels")

    for _, row in df.iterrows():
        src = Path(row[image_col])
        dst = image_out_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

        with Image.open(src) as image:
            width, height = image.size

        label_lines: list[str] = []
        for region in parse_regions(row[regions_col]):
            bbox = clamp_bbox(region.bbox, width, height)
            if bbox is None:
                continue
            cx, cy, w, h = bbox_to_yolo(bbox, width, height)
            class_id = class_to_id[region.region_type]
            label_lines.append(f"{class_id} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}")

        (label_out_dir / f"{src.stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/baseline.yaml")
    parser.add_argument("--output-dir", default="data/processed/yolo_regions")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    validation = cfg["validation"]

    data_dir = Path(paths["data_dir"])
    train = read_csv(data_dir / paths["train_csv"])
    spec = detect_columns(train, cfg.get("columns", {}), require_target=False)
    train = add_image_paths(train, spec, data_dir, paths["train_image_dirs"])
    regions_col = find_regions_column(train)

    all_types = sorted(
        {
            region.region_type
            for value in train[regions_col]
            for region in parse_regions(value)
        }
    )
    if not all_types:
        raise ValueError("No regions found in training data.")
    class_to_id = {name: idx for idx, name in enumerate(all_types)}

    kfold = KFold(
        n_splits=int(validation["n_splits"]),
        shuffle=True,
        random_state=int(validation["seed"]),
    )
    train["fold"] = -1
    for fold, (_, val_idx) in enumerate(kfold.split(train)):
        train.loc[val_idx, "fold"] = fold

    fold = int(validation["fold"])
    out_dir = ensure_dir(args.output_dir)
    write_split(train[train["fold"] != fold], out_dir / "train", "resolved_image_path", regions_col, class_to_id)
    write_split(train[train["fold"] == fold], out_dir / "val", "resolved_image_path", regions_col, class_to_id)

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {idx: name for name, idx in class_to_id.items()},
    }
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote YOLO dataset to {out_dir}")
    print(f"Wrote {yaml_path}")
    print(f"Classes: {class_to_id}")


if __name__ == "__main__":
    main()
