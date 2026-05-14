from __future__ import annotations

import shutil
from typing import Sequence

import cv2
import numpy as np


def preprocess_ocr_crop(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if max(gray.shape[:2]) < 1400:
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def ocr_confidence_for_bbox(
    image_bgr: np.ndarray,
    bbox_xywh: Sequence[int],
    psm: int,
) -> tuple[float, int]:
    if shutil.which("tesseract") is None:
        return -1.0, 0

    try:
        import pytesseract
        from pytesseract import Output
    except ImportError:
        return -1.0, 0

    x, y, width, height = bbox_xywh
    pad = max(4, int(min(width, height) * 0.02))
    x0 = max(0, x + pad)
    y0 = max(0, y + pad)
    x1 = min(image_bgr.shape[1], x + width - pad)
    y1 = min(image_bgr.shape[0], y + height - pad)
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0

    crop = image_bgr[y0:y1, x0:x1]
    prepared = preprocess_ocr_crop(crop)
    try:
        data = pytesseract.image_to_data(
            prepared,
            output_type=Output.DICT,
            config=f"--psm {psm}",
        )
    except pytesseract.pytesseract.TesseractNotFoundError:
        return -1.0, 0
    except RuntimeError:
        return 0.0, 0

    confidences: list[float] = []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        if not str(text).strip():
            continue
        try:
            value = float(conf)
        except ValueError:
            continue
        if value >= 0:
            confidences.append(value)

    if not confidences:
        return 0.0, 0
    return float(np.mean(confidences)), len(confidences)
