from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ID_CANDIDATES = ("id", "ID", "Id", "image_id", "file_id", "filename", "file_name")
IMAGE_CANDIDATES = (
    "image",
    "image_path",
    "path",
    "filepath",
    "file_path",
    "filename",
    "file_name",
)
TARGET_CANDIDATES = (
    "text",
    "label",
    "target",
    "transcription",
    "ground_truth",
    "answer",
    "value",
)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class ColumnSpec:
    id_col: str
    image_col: str | None
    target_col: str | None


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def first_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate

    lower_to_original = {c.lower(): c for c in columns}
    for candidate in candidates:
        found = lower_to_original.get(candidate.lower())
        if found is not None:
            return found
    return None


def detect_columns(
    df: pd.DataFrame,
    configured: dict,
    require_target: bool,
) -> ColumnSpec:
    id_col = configured.get("id") or first_existing_column(df.columns, ID_CANDIDATES)
    image_col = configured.get("image") or first_existing_column(df.columns, IMAGE_CANDIDATES)
    target_col = configured.get("target") or first_existing_column(df.columns, TARGET_CANDIDATES)

    if id_col is None:
        raise ValueError(
            "Could not detect id column. Set columns.id in config/baseline.yaml. "
            f"Available columns: {list(df.columns)}"
        )
    if require_target and target_col is None:
        raise ValueError(
            "Could not detect target column. Set columns.target in config/baseline.yaml. "
            f"Available columns: {list(df.columns)}"
        )
    return ColumnSpec(id_col=id_col, image_col=image_col, target_col=target_col)


def candidate_image_names(value: object) -> list[str]:
    raw = str(value)
    path = Path(raw)
    names = [raw]
    if path.name != raw:
        names.append(path.name)

    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return list(dict.fromkeys(names))

    for ext in IMAGE_EXTENSIONS:
        names.append(raw + ext)
        if path.name != raw:
            names.append(path.name + ext)
    return list(dict.fromkeys(names))


def resolve_image_path(row: pd.Series, spec: ColumnSpec, data_dir: Path, image_dirs: list[str]) -> Path:
    key_value = row[spec.image_col] if spec.image_col else row[spec.id_col]
    names = candidate_image_names(key_value)

    for name in names:
        direct = data_dir / name
        if direct.exists():
            return direct
        for image_dir in image_dirs:
            candidate = data_dir / image_dir / name
            if candidate.exists():
                return candidate

    searched = [str(data_dir / d) for d in image_dirs]
    raise FileNotFoundError(
        f"Could not resolve image for value={key_value!r}. "
        f"Looked in data_dir and image_dirs={searched}."
    )


def add_image_paths(
    df: pd.DataFrame,
    spec: ColumnSpec,
    data_dir: Path,
    image_dirs: list[str],
    path_col: str = "resolved_image_path",
) -> pd.DataFrame:
    out = df.copy()
    out[path_col] = [
        str(resolve_image_path(row, spec, data_dir, image_dirs))
        for _, row in out.iterrows()
    ]
    return out


def infer_submission_columns(sample_submission: pd.DataFrame, configured: dict) -> tuple[str, str]:
    id_col = configured.get("id") or first_existing_column(sample_submission.columns, ID_CANDIDATES)
    if id_col is None:
        id_col = sample_submission.columns[0]

    prediction_col = configured.get("prediction")
    if prediction_col is None:
        non_id = [c for c in sample_submission.columns if c != id_col]
        if not non_id:
            raise ValueError("sample_submission.csv must have at least one prediction column.")
        prediction_col = non_id[0]

    return id_col, prediction_col
