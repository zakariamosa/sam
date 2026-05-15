from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from ticket_detector import (
    DetectorConfig,
    TicketDetector,
    default_checkpoint_for_backend,
    default_model_type_for_backend,
    iter_images,
    load_bgr,
    normalize_model_backend,
    parse_points,
    resize_for_sam,
    save_outputs,
)


def config_from_args(args: argparse.Namespace) -> DetectorConfig:
    return DetectorConfig(
        max_dim=args.max_dim,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        crop_n_layers=args.crop_n_layers,
        min_mask_region_area=args.min_mask_region_area,
        mask_iou_threshold=args.mask_iou_threshold,
        bbox_iou_threshold=args.bbox_iou_threshold,
        containment_threshold=args.containment_threshold,
        min_area_ratio=args.min_area_ratio,
        final_min_area_ratio=args.final_min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        min_width_ratio=args.min_width_ratio,
        min_height_ratio=args.min_height_ratio,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
        max_top_ratio=args.max_top_ratio,
        min_light_ratio=args.min_light_ratio,
        min_dark_ratio=args.min_dark_ratio,
        same_width_tolerance=args.same_width_tolerance,
        ocr_min_confidence=args.ocr_min_confidence,
        ocr_min_words=args.ocr_min_words,
        ocr_psm=args.ocr_psm,
        polygon_epsilon=args.polygon_epsilon,
    )


def collect_points(image_bgr, display_max_dim: int) -> tuple[list[tuple[int, int]], float]:
    display, display_scale = resize_for_sam(image_bgr, display_max_dim)
    points: list[tuple[int, int]] = []
    window = "Click ticket interiors. Enter/Space: finish. Backspace/right-click: undo. Esc: cancel."

    def redraw() -> None:
        canvas = display.copy()
        for index, (x, y) in enumerate(points, start=1):
            cv2.circle(canvas, (x, y), 8, (0, 220, 255), -1)
            cv2.putText(
                canvas,
                str(index),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow(window, canvas)

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()
    print(window)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (13, 10, 32, ord("q")):
            break
        if key in (8, 127) and points:
            points.pop()
            redraw()
        if key == 27:
            points = []
            break
    cv2.destroyWindow(window)
    return points, display_scale


def print_tickets(image_path: Path, tickets, annotated_path: Path, json_path: Path) -> None:
    print(f"\nImage: {image_path}")
    print(f"Tickets detected: {len(tickets)}")
    for index, ticket in enumerate(tickets, start=1):
        print(f"Ticket {index} ({ticket.source}, score={ticket.score:.3f})")
        print(json.dumps(ticket.polygon))
    print(f"Annotated image: {annotated_path}")
    print(f"JSON output: {json_path}")


def show_image(path: Path) -> None:
    image = load_bgr(path)
    window = str(path)
    cv2.imshow(window, image)
    print("Press any key in the image window to close it...")
    cv2.waitKey(0)
    cv2.destroyWindow(window)


def process_image(
    image_path: Path,
    detector: TicketDetector,
    config: DetectorConfig,
    args: argparse.Namespace,
) -> int:
    original_bgr = load_bgr(image_path)

    if args.mode == "auto":
        tickets = detector.detect(
            original_bgr,
            config=config,
            enable_ocr=args.ocr_min_confidence > 0,
            verbose=True,
        )
    elif args.mode == "prompt":
        if args.points:
            points = parse_points(args.points)
        else:
            clicked, display_scale = collect_points(original_bgr, args.display_max_dim)
            points = [(int(round(x / display_scale)), int(round(y / display_scale))) for x, y in clicked]
        tickets = detector.prompt_masks(original_bgr, points, config=config)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    annotated_path, json_path = save_outputs(
        image_path,
        original_bgr,
        tickets,
        args.output_dir,
        save_crops=args.save_crops,
        model_backend=args.backend,
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        device=detector.device,
    )
    print_tickets(image_path, tickets, annotated_path, json_path)
    if args.show:
        show_image(annotated_path)
    return len(tickets)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segment kitchen tickets with SAM or MobileSAM.")
    parser.add_argument("input", type=Path, help="One image file or a directory of images.")
    parser.add_argument("--output-dir", type=Path, default=Path("sam_outputs"))
    parser.add_argument("--backend", default="sam", choices=("sam", "mobilesam"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--mode", default="auto", choices=("auto", "prompt"))
    parser.add_argument("--points", default="", help='Prompt points as original-image coordinates: "x,y;x,y".')
    parser.add_argument("--save-crops", action="store_true", help="Save one crop per accepted ticket.")
    parser.add_argument("--show", action="store_true", help="Display the annotated output image after saving.")
    parser.add_argument("--max-dim", type=int, default=1024, help="Resize image longest side before SAM inference.")
    parser.add_argument("--display-max-dim", type=int, default=1400, help="Interactive click window longest side.")
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--points-per-batch", type=int, default=8)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.82)
    parser.add_argument("--stability-score-thresh", type=float, default=0.88)
    parser.add_argument("--box-nms-thresh", type=float, default=0.72)
    parser.add_argument("--crop-n-layers", type=int, default=0)
    parser.add_argument("--min-mask-region-area", type=int, default=150)
    parser.add_argument("--mask-iou-threshold", type=float, default=0.45)
    parser.add_argument("--bbox-iou-threshold", type=float, default=0.42)
    parser.add_argument("--containment-threshold", type=float, default=0.76)
    parser.add_argument("--min-area-ratio", type=float, default=0.003)
    parser.add_argument("--final-min-area-ratio", type=float, default=0.008)
    parser.add_argument("--max-area-ratio", type=float, default=0.45)
    parser.add_argument("--min-width-ratio", type=float, default=0.08)
    parser.add_argument("--min-height-ratio", type=float, default=0.12)
    parser.add_argument("--min-aspect", type=float, default=1.05)
    parser.add_argument("--max-aspect", type=float, default=8.0)
    parser.add_argument("--max-top-ratio", type=float, default=0.68)
    parser.add_argument("--min-light-ratio", type=float, default=0.45)
    parser.add_argument("--min-dark-ratio", type=float, default=0.002)
    parser.add_argument("--same-width-tolerance", type=float, default=0.0)
    parser.add_argument("--ocr-min-confidence", type=float, default=0.0)
    parser.add_argument("--ocr-min-words", type=int, default=2)
    parser.add_argument("--ocr-psm", type=int, default=6)
    parser.add_argument("--polygon-epsilon", type=float, default=0.006)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.backend = normalize_model_backend(args.backend)
    if args.checkpoint is None:
        args.checkpoint = default_checkpoint_for_backend(args.backend)
    if args.model_type is None:
        args.model_type = default_model_type_for_backend(args.backend)

    images = list(iter_images(args.input))
    if not images:
        print(f"No images found: {args.input}", file=sys.stderr)
        return 1

    config = config_from_args(args)
    print(f"Loading {args.backend} {args.model_type} on {args.device}...")
    detector = TicketDetector(args.checkpoint, args.model_type, args.device, config, args.backend)
    print(f"Using device: {detector.device}")

    total = 0
    for image_path in images:
        total += process_image(image_path, detector, config, args)

    print(f"\nProcessed {len(images)} image(s); total tickets detected: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
