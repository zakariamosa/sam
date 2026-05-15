from __future__ import annotations

import json
import math
import sys
import threading
from dataclasses import asdict, dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_BACKEND_SAM = "sam"
MODEL_BACKEND_MOBILESAM = "mobilesam"
MODEL_BACKENDS = {MODEL_BACKEND_SAM, MODEL_BACKEND_MOBILESAM}
DEFAULT_CHECKPOINTS = {
    MODEL_BACKEND_SAM: Path("models/sam_vit_b_01ec64.pth"),
    MODEL_BACKEND_MOBILESAM: Path("models/mobile_sam.pt"),
}
DEFAULT_MODEL_TYPES = {
    MODEL_BACKEND_SAM: "vit_b",
    MODEL_BACKEND_MOBILESAM: "vit_t",
}


@dataclass(frozen=True)
class DetectorConfig:
    max_dim: int = 1024
    points_per_side: int = 16
    points_per_batch: int = 8
    pred_iou_thresh: float = 0.82
    stability_score_thresh: float = 0.88
    box_nms_thresh: float = 0.72
    crop_n_layers: int = 0
    min_mask_region_area: int = 150
    mask_iou_threshold: float = 0.45
    bbox_iou_threshold: float = 0.42
    containment_threshold: float = 0.76
    min_area_ratio: float = 0.003
    final_min_area_ratio: float = 0.008
    max_area_ratio: float = 0.45
    min_width_ratio: float = 0.08
    min_height_ratio: float = 0.12
    min_aspect: float = 1.05
    max_aspect: float = 8.0
    max_top_ratio: float = 0.68
    min_light_ratio: float = 0.45
    min_dark_ratio: float = 0.002
    same_width_tolerance: float = 0.0
    ocr_min_confidence: float = 0.0
    ocr_min_words: int = 2
    ocr_psm: int = 6
    polygon_epsilon: float = 0.006


@dataclass
class TicketMask:
    polygon: list[list[int]]
    bbox_xywh: list[int]
    area: int
    score: float
    predicted_iou: float
    stability_score: float
    light_ratio: float
    dark_ratio: float
    aspect_ratio: float
    source: str
    mask: np.ndarray
    width_deviation: float = 0.0
    ocr_confidence: float | None = None
    ocr_word_count: int = 0


def config_from_mapping(values: dict[str, Any] | None = None, base: DetectorConfig | None = None) -> DetectorConfig:
    current = base or DetectorConfig()
    if not values:
        return current

    fields = set(asdict(current).keys())
    cleaned = {key: value for key, value in values.items() if key in fields and value is not None}
    if not cleaned:
        return current
    return replace(current, **cleaned)


def decode_image_bytes(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("Empty image payload.")

    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Use a standard image format such as JPG or PNG.")
    return image


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        yield path
        return

    for candidate in sorted(path.iterdir()):
        if candidate.suffix.lower() in IMAGE_EXTENSIONS:
            yield candidate


def resize_for_sam(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_dim / max(height, width))
    if scale == 1.0:
        return image.copy(), 1.0
    size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def sam_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU.", file=sys.stderr)
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but not available; using CPU.", file=sys.stderr)
        return "cpu"
    return requested


def normalize_model_backend(model_backend: str) -> str:
    normalized = model_backend.strip().lower().replace("_", "").replace("-", "")
    if normalized == "mobile":
        normalized = MODEL_BACKEND_MOBILESAM
    if normalized not in MODEL_BACKENDS:
        valid = ", ".join(sorted(MODEL_BACKENDS))
        raise ValueError(f"Unsupported MODEL_BACKEND '{model_backend}'. Expected one of: {valid}.")
    return normalized


def default_checkpoint_for_backend(model_backend: str) -> Path:
    return DEFAULT_CHECKPOINTS[normalize_model_backend(model_backend)]


def default_model_type_for_backend(model_backend: str) -> str:
    return DEFAULT_MODEL_TYPES[normalize_model_backend(model_backend)]


def import_sam_backend():
    try:
        module = import_module("segment_anything")
    except ImportError as exc:
        raise ImportError(
            "SAM backend requires segment-anything. Install it with "
            "'pip install git+https://github.com/facebookresearch/segment-anything.git'."
        ) from exc
    return module


def import_mobilesam_backend():
    try:
        module = import_module("mobile_sam")
    except ImportError as exc:
        raise ImportError(
            "MobileSAM backend requires MobileSAM. Install it with "
            "'pip install git+https://github.com/ChaoningZhang/MobileSAM.git'."
        ) from exc
    return module


def build_registered_model(module, checkpoint: Path, model_type: str, device: str, backend_label: str):
    registry = module.sam_model_registry
    try:
        model_builder = registry[model_type]
    except KeyError as exc:
        valid = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown {backend_label} model type '{model_type}'. Available: {valid}.") from exc

    model = model_builder(checkpoint=str(checkpoint))
    model.to(device=device)
    model.eval()
    return model


def load_sam(checkpoint: Path, model_type: str, device: str):
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {checkpoint}\n"
            "Download Meta's ViT-B checkpoint, sam_vit_b_01ec64.pth, into the models folder."
        )

    module = import_sam_backend()
    return build_registered_model(module, checkpoint, model_type, device, "SAM")


def load_mobilesam(checkpoint: Path, model_type: str, device: str):
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"MobileSAM checkpoint not found: {checkpoint}\n"
            "Download MobileSAM's mobile_sam.pt checkpoint into the models folder."
        )

    module = import_mobilesam_backend()
    return build_registered_model(module, checkpoint, model_type, device, "MobileSAM")


def backend_module(model_backend: str):
    backend = normalize_model_backend(model_backend)
    if backend == MODEL_BACKEND_SAM:
        return import_sam_backend()
    if backend == MODEL_BACKEND_MOBILESAM:
        return import_mobilesam_backend()
    raise AssertionError(f"Unhandled model backend: {backend}")


def load_model(model_backend: str, checkpoint: Path, model_type: str, device: str):
    backend = normalize_model_backend(model_backend)
    if backend == MODEL_BACKEND_SAM:
        return load_sam(checkpoint, model_type, device)
    if backend == MODEL_BACKEND_MOBILESAM:
        return load_mobilesam(checkpoint, model_type, device)
    raise AssertionError(f"Unhandled model backend: {backend}")


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def order_polygon(points: Sequence[Sequence[float]]) -> list[list[int]]:
    pts = np.asarray(points, dtype=np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = np.lexsort((pts[:, 0], pts[:, 1]))[0]
    pts = np.roll(pts, -start, axis=0)
    return [[int(round(x)), int(round(y))] for x, y in pts]


def mask_to_polygon(mask: np.ndarray, scale: float, epsilon_ratio: float) -> list[list[int]] | None:
    contour = largest_contour(mask)
    if contour is None or cv2.contourArea(contour) < 4:
        return None

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
    if len(approx) < 3:
        rect = cv2.minAreaRect(contour)
        points = cv2.boxPoints(rect)
    else:
        points = approx.reshape(-1, 2)

    points = points / scale
    return order_polygon(points)


def bbox_from_polygon(polygon: Sequence[Sequence[int]]) -> list[int]:
    pts = np.asarray(polygon, dtype=np.int32)
    x, y, width, height = cv2.boundingRect(pts)
    return [int(x), int(y), int(width), int(height)]


def bbox_iou_xywh(a: Sequence[int], b: Sequence[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1)


def bbox_intersection_xywh(a: Sequence[int], b: Sequence[int]) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def bbox_containment_xywh(a: Sequence[int], b: Sequence[int]) -> float:
    intersection = bbox_intersection_xywh(a, b)
    if intersection == 0:
        return 0.0
    smaller = min(a[2] * a[3], b[2] * b[3])
    return intersection / max(smaller, 1)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(intersection / max(union, 1))


def mask_color_stats(image_bgr: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    pixels = mask.astype(bool)
    if pixels.sum() == 0:
        return 0.0, 0.0

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    light = ((value > 115) & (saturation < 120) & pixels).sum() / pixels.sum()
    dark = ((gray < 145) & pixels).sum() / pixels.sum()
    return float(light), float(dark)


def candidate_from_mask(
    mask: np.ndarray,
    image_bgr: np.ndarray,
    scale: float,
    *,
    source: str,
    predicted_iou: float,
    stability_score: float,
    config: DetectorConfig,
    strict: bool,
) -> TicketMask | None:
    height, width = mask.shape[:2]
    area = int(mask.sum())
    image_area = height * width
    area_ratio = area / max(image_area, 1)
    if area_ratio < config.min_area_ratio or area_ratio > config.max_area_ratio:
        return None

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    box_w = max(1, x1 - x0 + 1)
    box_h = max(1, y1 - y0 + 1)
    aspect_ratio = box_h / box_w

    aspect_floor = config.min_aspect if strict else min(0.65, config.min_aspect)
    min_width = width * (config.min_width_ratio if strict else min(0.015, config.min_width_ratio))
    min_height = height * (config.min_height_ratio if strict else min(0.07, config.min_height_ratio))
    max_aspect_ceiling = config.max_aspect if strict else max(config.max_aspect, 12.0)
    if box_w < min_width or box_h < min_height:
        return None
    if aspect_ratio < aspect_floor:
        return None
    if aspect_ratio > max_aspect_ceiling:
        return None
    if strict and box_w > width * 0.62:
        return None
    if strict and y0 > height * config.max_top_ratio:
        return None
    if strict and box_h < box_w * 1.05:
        return None

    fill_ratio = area / max(box_w * box_h, 1)
    if fill_ratio < (0.22 if strict else 0.12):
        return None

    light_ratio, dark_ratio = mask_color_stats(image_bgr, mask)
    if light_ratio < (config.min_light_ratio if strict else min(0.25, config.min_light_ratio)):
        return None
    if strict and dark_ratio < config.min_dark_ratio:
        return None

    contour = largest_contour(mask)
    if contour is None:
        return None
    contour_area = cv2.contourArea(contour)
    if contour_area < area * 0.30:
        return None

    polygon = mask_to_polygon(mask, scale, config.polygon_epsilon)
    if polygon is None:
        return None
    bbox = bbox_from_polygon(polygon)

    quality = (predicted_iou + stability_score) / 2.0
    paper_score = min(light_ratio, 1.0) * 0.20 + min(dark_ratio * 8.0, 1.0) * 0.10
    shape_score = min(aspect_ratio / 4.0, 1.0) * 0.15 + min(fill_ratio, 1.0) * 0.10
    score = float(min(1.0, quality * 0.55 + paper_score + shape_score))

    return TicketMask(
        polygon=polygon,
        bbox_xywh=bbox,
        area=int(round(area / (scale * scale))),
        score=score,
        predicted_iou=float(predicted_iou),
        stability_score=float(stability_score),
        light_ratio=light_ratio,
        dark_ratio=dark_ratio,
        aspect_ratio=float(aspect_ratio),
        source=source,
        mask=mask.astype(bool),
    )


def dedupe_tickets(candidates: Sequence[TicketMask], config: DetectorConfig) -> list[TicketMask]:
    kept: list[TicketMask] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        duplicate = False
        for existing in kept:
            if bbox_iou_xywh(candidate.bbox_xywh, existing.bbox_xywh) > config.bbox_iou_threshold:
                duplicate = True
                break
            if bbox_containment_xywh(candidate.bbox_xywh, existing.bbox_xywh) > config.containment_threshold:
                duplicate = True
                break
            if mask_iou(candidate.mask, existing.mask) > config.mask_iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return sorted(kept, key=lambda item: item.bbox_xywh[0])


def passes_final_shape_constraints(
    ticket: TicketMask,
    image_shape: tuple[int, int],
    config: DetectorConfig,
    mode: str,
) -> bool:
    height, width = image_shape
    _, y, box_w, box_h = ticket.bbox_xywh
    area_ratio = ticket.area / max(width * height, 1)
    aspect_ratio = box_h / max(box_w, 1)

    if area_ratio < config.final_min_area_ratio:
        return False
    if box_w < width * config.min_width_ratio:
        return False
    if box_h < height * config.min_height_ratio:
        return False
    if aspect_ratio < config.min_aspect or aspect_ratio > config.max_aspect:
        return False
    if mode == "auto" and y > height * config.max_top_ratio:
        return False
    return True


def apply_same_width_normalization(tickets: Sequence[TicketMask], tolerance: float) -> list[TicketMask]:
    if tolerance <= 0 or len(tickets) < 3:
        for ticket in tickets:
            ticket.width_deviation = 0.0
        return list(tickets)

    widths = np.asarray([ticket.bbox_xywh[2] for ticket in tickets], dtype=np.float32)
    median_width = float(np.median(widths))
    if median_width <= 0:
        return list(tickets)

    kept: list[TicketMask] = []
    for ticket in tickets:
        width_deviation = abs(ticket.bbox_xywh[2] - median_width) / median_width
        ticket.width_deviation = float(width_deviation)
        if width_deviation <= tolerance:
            kept.append(ticket)
    return kept


def apply_ocr_confidence_check(
    image_bgr: np.ndarray,
    tickets: Sequence[TicketMask],
    config: DetectorConfig,
    enable_ocr: bool,
) -> list[TicketMask]:
    if not enable_ocr or config.ocr_min_confidence <= 0:
        return list(tickets)

    from ocr_support import ocr_confidence_for_bbox

    checked: list[TicketMask] = []
    warned_unavailable = False
    for ticket in tickets:
        confidence, word_count = ocr_confidence_for_bbox(image_bgr, ticket.bbox_xywh, config.ocr_psm)
        ticket.ocr_confidence = None if confidence < 0 else confidence
        ticket.ocr_word_count = word_count

        if confidence < 0:
            if not warned_unavailable:
                print("OCR check skipped: optional OCR dependencies are unavailable.", file=sys.stderr)
                warned_unavailable = True
            checked.append(ticket)
            continue

        if confidence >= config.ocr_min_confidence and word_count >= config.ocr_min_words:
            checked.append(ticket)

    return checked


def apply_logical_constraints(
    image_bgr: np.ndarray,
    tickets: Sequence[TicketMask],
    config: DetectorConfig,
    *,
    mode: str = "auto",
    enable_ocr: bool = False,
) -> list[TicketMask]:
    image_shape = image_bgr.shape[:2]
    shape_filtered = [
        ticket for ticket in tickets if passes_final_shape_constraints(ticket, image_shape, config, mode)
    ]
    overlap_filtered = dedupe_tickets(shape_filtered, config)
    ocr_filtered = apply_ocr_confidence_check(image_bgr, overlap_filtered, config, enable_ocr)
    width_filtered = apply_same_width_normalization(ocr_filtered, config.same_width_tolerance)
    final = dedupe_tickets(width_filtered, config)
    return sorted(final, key=lambda item: item.bbox_xywh[0])


def draw_tickets(image_bgr: np.ndarray, tickets: Sequence[TicketMask]) -> np.ndarray:
    annotated = image_bgr.copy()
    overlay = annotated.copy()

    for index, ticket in enumerate(tickets, start=1):
        pts = np.asarray(ticket.polygon, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 220, 255))
        cv2.polylines(annotated, [pts], True, (0, 180, 255), 5, cv2.LINE_AA)
        x, y, _, _ = cv2.boundingRect(pts)
        label = f"{index} {ticket.source} {ticket.score:.2f}"
        cv2.putText(
            annotated,
            label,
            (x, max(35, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            label,
            (x, max(35, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

    return cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0)


def ticket_to_debug_json(ticket: TicketMask, index: int, crop_path: Path | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": index,
        "source": ticket.source,
        "score": round(ticket.score, 6),
        "predicted_iou": round(ticket.predicted_iou, 6),
        "stability_score": round(ticket.stability_score, 6),
        "area": ticket.area,
        "bbox_xywh": ticket.bbox_xywh,
        "polygon": ticket.polygon,
        "light_ratio": round(ticket.light_ratio, 6),
        "dark_ratio": round(ticket.dark_ratio, 6),
        "aspect_ratio": round(ticket.aspect_ratio, 6),
        "width_deviation": round(ticket.width_deviation, 6),
        "ocr_confidence": None if ticket.ocr_confidence is None else round(ticket.ocr_confidence, 6),
        "ocr_word_count": ticket.ocr_word_count,
    }
    if crop_path is not None:
        record["crop_path"] = str(crop_path)
    return record


def ticket_to_api_json(ticket: TicketMask, index: int, include_debug: bool = False) -> dict[str, Any]:
    x, y, width, height = ticket.bbox_xywh
    record: dict[str, Any] = {
        "id": index,
        "score": round(ticket.score, 6),
        "bbox": {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
        },
        "polygon": [{"x": int(point[0]), "y": int(point[1])} for point in ticket.polygon],
    }

    if include_debug:
        record["source"] = ticket.source
        record["predictedIou"] = round(ticket.predicted_iou, 6)
        record["stabilityScore"] = round(ticket.stability_score, 6)
        record["area"] = ticket.area
        record["lightRatio"] = round(ticket.light_ratio, 6)
        record["darkRatio"] = round(ticket.dark_ratio, 6)
        record["aspectRatio"] = round(ticket.aspect_ratio, 6)
        record["widthDeviation"] = round(ticket.width_deviation, 6)
        record["ocrConfidence"] = (
            None if ticket.ocr_confidence is None else round(ticket.ocr_confidence, 6)
        )
        record["ocrWordCount"] = ticket.ocr_word_count

    return record


def tickets_api_response(
    image_bgr: np.ndarray,
    tickets: Sequence[TicketMask],
    include_debug: bool = False,
) -> dict[str, Any]:
    return {
        "imageWidth": int(image_bgr.shape[1]),
        "imageHeight": int(image_bgr.shape[0]),
        "ticketCount": len(tickets),
        "tickets": [
            ticket_to_api_json(ticket, index, include_debug=include_debug)
            for index, ticket in enumerate(tickets, start=1)
        ],
    }


def save_outputs(
    image_path: Path,
    image_bgr: np.ndarray,
    tickets: Sequence[TicketMask],
    output_dir: Path,
    *,
    save_crops: bool,
    model_type: str,
    checkpoint: Path,
    device: str,
    model_backend: str = MODEL_BACKEND_SAM,
) -> tuple[Path, Path]:
    model_backend = normalize_model_backend(model_backend)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / f"{image_path.stem}_sam_tickets{image_path.suffix}"
    json_path = output_dir / f"{image_path.stem}_sam_tickets.json"
    crop_dir = output_dir / f"{image_path.stem}_crops"
    if save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    annotated = draw_tickets(image_bgr, tickets)
    cv2.imwrite(str(annotated_path), annotated)

    records = []
    for index, ticket in enumerate(tickets, start=1):
        crop_path = None
        if save_crops:
            x, y, width, height = ticket.bbox_xywh
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(image_bgr.shape[1], x + width), min(image_bgr.shape[0], y + height)
            crop = image_bgr[y0:y1, x0:x1]
            crop_path = crop_dir / f"ticket_{index:02d}.png"
            cv2.imwrite(str(crop_path), crop)
        records.append(ticket_to_debug_json(ticket, index, crop_path))

    payload = {
        "image": str(image_path),
        "width": int(image_bgr.shape[1]),
        "height": int(image_bgr.shape[0]),
        "model": {
            "name": (
                "mobile-sam"
                if model_backend == MODEL_BACKEND_MOBILESAM
                else "segment-anything"
            ),
            "backend": model_backend,
            "model_type": model_type,
            "checkpoint": str(checkpoint),
            "device": device,
        },
        "ticket_count": len(tickets),
        "tickets": records,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return annotated_path, json_path


def parse_points(points_text: str) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for item in points_text.split(";"):
        item = item.strip()
        if not item:
            continue
        x_text, y_text = item.split(",", maxsplit=1)
        x, y = int(float(x_text)), int(float(y_text))
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Invalid point '{item}'.")
        points.append((x, y))
    return points


class TicketDetector:
    def __init__(
        self,
        checkpoint: Path,
        model_type: str = "vit_b",
        device: str = "auto",
        config: DetectorConfig | None = None,
        model_backend: str = MODEL_BACKEND_SAM,
    ) -> None:
        self.config = config or DetectorConfig()
        self.model_backend = normalize_model_backend(model_backend)
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.device = sam_device(device)
        self.sam = load_model(self.model_backend, checkpoint, model_type, self.device)
        module = backend_module(self.model_backend)
        self._predictor_cls = module.SamPredictor
        self.generator = module.SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=self.config.points_per_side,
            points_per_batch=self.config.points_per_batch,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_score_thresh,
            box_nms_thresh=self.config.box_nms_thresh,
            crop_n_layers=self.config.crop_n_layers,
            min_mask_region_area=self.config.min_mask_region_area,
            output_mode="binary_mask",
        )
        self._lock = threading.Lock()

    def detect(
        self,
        image_bgr: np.ndarray,
        *,
        config: DetectorConfig | None = None,
        enable_ocr: bool = False,
        verbose: bool = False,
    ) -> list[TicketMask]:
        runtime_config = config or self.config
        sam_bgr, scale = resize_for_sam(image_bgr, runtime_config.max_dim)
        sam_rgb = cv2.cvtColor(sam_bgr, cv2.COLOR_BGR2RGB)

        with self._lock:
            with torch.inference_mode():
                raw_masks = self.generator.generate(sam_rgb)

        candidates: list[TicketMask] = []
        for ann in raw_masks:
            candidate = candidate_from_mask(
                ann["segmentation"].astype(bool),
                sam_bgr,
                scale,
                source=f"{self.model_backend}-auto",
                predicted_iou=float(ann.get("predicted_iou", 0.0)),
                stability_score=float(ann.get("stability_score", 0.0)),
                config=runtime_config,
                strict=True,
            )
            if candidate is not None:
                candidates.append(candidate)

        if verbose:
            print(f"SAM masks generated: {len(raw_masks)}")
            print(f"Ticket-like masks after filtering: {len(candidates)}")

        deduped = dedupe_tickets(candidates, runtime_config)
        return apply_logical_constraints(
            image_bgr,
            deduped,
            runtime_config,
            mode="auto",
            enable_ocr=enable_ocr,
        )

    def detect_from_bytes(
        self,
        data: bytes,
        *,
        config: DetectorConfig | None = None,
        enable_ocr: bool = False,
    ) -> tuple[np.ndarray, list[TicketMask]]:
        image_bgr = decode_image_bytes(data)
        tickets = self.detect(image_bgr, config=config, enable_ocr=enable_ocr)
        return image_bgr, tickets

    def prompt_masks(
        self,
        image_bgr: np.ndarray,
        points: Sequence[tuple[int, int]],
        *,
        config: DetectorConfig | None = None,
    ) -> list[TicketMask]:
        runtime_config = config or self.config
        sam_bgr, scale = resize_for_sam(image_bgr, runtime_config.max_dim)
        sam_rgb = cv2.cvtColor(sam_bgr, cv2.COLOR_BGR2RGB)
        scaled_points = [(int(round(x * scale)), int(round(y * scale))) for x, y in points]

        with self._lock:
            predictor = self._predictor_cls(self.sam)
            predictor.set_image(sam_rgb)

            candidates: list[TicketMask] = []
            for point in scaled_points:
                point_coords = np.asarray([point], dtype=np.float32)
                point_labels = np.asarray([1], dtype=np.int32)
                with torch.inference_mode():
                    masks, scores, _ = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True,
                    )

                prompt_candidates: list[TicketMask] = []
                for mask, score in zip(masks, scores):
                    candidate = candidate_from_mask(
                        mask.astype(bool),
                        sam_bgr,
                        scale,
                        source=f"{self.model_backend}-prompt",
                        predicted_iou=float(score),
                        stability_score=1.0,
                        config=runtime_config,
                        strict=False,
                    )
                    if candidate is not None:
                        prompt_candidates.append(candidate)

                if prompt_candidates:
                    candidates.append(max(prompt_candidates, key=lambda item: item.score))
                else:
                    print(f"No ticket-like mask accepted for point {point}.", file=sys.stderr)

            predictor.reset_image()

        deduped = dedupe_tickets(candidates, runtime_config)
        return apply_logical_constraints(
            image_bgr,
            deduped,
            runtime_config,
            mode="prompt",
            enable_ocr=False,
        )
