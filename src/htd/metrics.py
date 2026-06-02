from __future__ import annotations

import jiwer


def char_error_rate(references: list[str], predictions: list[str]) -> float:
    return jiwer.cer(references, predictions)


def word_error_rate(references: list[str], predictions: list[str]) -> float:
    return jiwer.wer(references, predictions)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()
