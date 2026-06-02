from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.htd.config import ensure_dir, load_config
from src.htd.data import detect_columns, infer_submission_columns, read_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/baseline.yaml")
    parser.add_argument("--predictions", default="submissions/predictions.csv")
    parser.add_argument("--output", default="submissions/submission.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    data_dir = Path(paths["data_dir"])

    sample = read_csv(data_dir / paths["sample_submission_csv"])
    preds = pd.read_csv(args.predictions)

    test_spec = detect_columns(preds, cfg.get("columns", {}), require_target=False)
    submission_id_col, submission_prediction_col = infer_submission_columns(
        sample,
        cfg.get("submission", {}),
    )

    out = sample.copy()
    if submission_id_col in preds.columns:
        pred_map = preds.set_index(submission_id_col)["prediction"].to_dict()
    else:
        pred_map = preds.set_index(test_spec.id_col)["prediction"].to_dict()

    out[submission_prediction_col] = out[submission_id_col].map(pred_map).fillna("")
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    out.to_csv(output_path, index=False)
    print(f"Wrote {output_path} with shape={out.shape}")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
