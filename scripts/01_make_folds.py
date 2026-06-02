from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold

from src.htd.config import ensure_dir, load_config
from src.htd.data import add_image_paths, detect_columns, read_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/baseline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    validation = cfg["validation"]

    data_dir = Path(paths["data_dir"])
    processed_dir = ensure_dir(paths["processed_dir"])
    train_csv = data_dir / paths["train_csv"]

    train = read_csv(train_csv)
    spec = detect_columns(train, cfg.get("columns", {}), require_target=True)
    train = add_image_paths(train, spec, data_dir, paths["train_image_dirs"])

    n_splits = int(validation["n_splits"])
    seed = int(validation["seed"])
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    train["fold"] = -1
    for fold, (_, val_idx) in enumerate(kfold.split(train)):
        train.loc[val_idx, "fold"] = fold

    out_path = processed_dir / "train_folds.csv"
    train.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape={train.shape}")
    print(f"Detected id_col={spec.id_col}, image_col={spec.image_col}, target_col={spec.target_col}")
    print(train["fold"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
