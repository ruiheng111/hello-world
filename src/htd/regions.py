from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Region:
    bbox: tuple[int, int, int, int]
    region_type: str
    text: str


def parse_regions(value: Any) -> list[Region]:
    if value is None:
        return []
    if isinstance(value, float) and value != value:
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

    regions: list[Region] = []
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


def serialize_regions(regions: list[Region]) -> str:
    payload = [
        {
            "bbox": [int(v) for v in region.bbox],
            "type": region.region_type,
            "text": region.text,
        }
        for region in regions
    ]
    return json.dumps(payload, ensure_ascii=False)


def clamp_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_to_yolo(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return cx, cy, w, h


def sort_regions_reading_order(regions: list[Region]) -> list[Region]:
    return sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0], r.bbox[3], r.bbox[2]))
