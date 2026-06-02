from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def print_csv_summary(path: Path) -> None:
    print(f"\n[CSV] {path}")
    df = pd.read_csv(path)
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print("head:")
    print(df.head(5).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    print(f"Data directory: {data_dir.resolve()}")

    csv_files = sorted(data_dir.rglob("*.csv"))
    if not csv_files:
        print("No CSV files found.")
    for csv_path in csv_files:
        print_csv_summary(csv_path)

    image_files = [p for p in data_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    print(f"\nImage files found: {len(image_files)}")
    for sample in image_files[:20]:
        print(f"- {sample.relative_to(data_dir)}")

    top_level = sorted(p.name for p in data_dir.iterdir())
    print("\nTop-level data_dir entries:")
    for name in top_level:
        print(f"- {name}")


if __name__ == "__main__":
    main()
