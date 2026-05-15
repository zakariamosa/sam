from __future__ import annotations

import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from ticket_detector import (
    DetectorConfig,
    TicketDetector,
    decode_image_bytes,
    default_checkpoint_for_backend,
    default_model_type_for_backend,
    normalize_model_backend,
    tickets_api_response,
)


logger = logging.getLogger("sam-ticket-service")


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def config_from_env() -> DetectorConfig:
    defaults = DetectorConfig()
    return DetectorConfig(
        max_dim=env_int("SAM_MAX_DIM", defaults.max_dim),
        points_per_side=env_int("SAM_POINTS_PER_SIDE", defaults.points_per_side),
        points_per_batch=env_int("SAM_POINTS_PER_BATCH", defaults.points_per_batch),
        pred_iou_thresh=env_float("SAM_PRED_IOU_THRESH", defaults.pred_iou_thresh),
        stability_score_thresh=env_float("SAM_STABILITY_SCORE_THRESH", defaults.stability_score_thresh),
        box_nms_thresh=env_float("SAM_BOX_NMS_THRESH", defaults.box_nms_thresh),
        crop_n_layers=env_int("SAM_CROP_N_LAYERS", defaults.crop_n_layers),
        min_mask_region_area=env_int("SAM_MIN_MASK_REGION_AREA", defaults.min_mask_region_area),
        mask_iou_threshold=env_float("SAM_MASK_IOU_THRESHOLD", defaults.mask_iou_threshold),
        bbox_iou_threshold=env_float("SAM_BBOX_IOU_THRESHOLD", defaults.bbox_iou_threshold),
        containment_threshold=env_float("SAM_CONTAINMENT_THRESHOLD", defaults.containment_threshold),
        min_area_ratio=env_float("SAM_MIN_AREA_RATIO", defaults.min_area_ratio),
        final_min_area_ratio=env_float("SAM_FINAL_MIN_AREA_RATIO", defaults.final_min_area_ratio),
        max_area_ratio=env_float("SAM_MAX_AREA_RATIO", defaults.max_area_ratio),
        min_width_ratio=env_float("SAM_MIN_WIDTH_RATIO", defaults.min_width_ratio),
        min_height_ratio=env_float("SAM_MIN_HEIGHT_RATIO", defaults.min_height_ratio),
        min_aspect=env_float("SAM_MIN_ASPECT", defaults.min_aspect),
        max_aspect=env_float("SAM_MAX_ASPECT", defaults.max_aspect),
        max_top_ratio=env_float("SAM_MAX_TOP_RATIO", defaults.max_top_ratio),
        min_light_ratio=env_float("SAM_MIN_LIGHT_RATIO", defaults.min_light_ratio),
        min_dark_ratio=env_float("SAM_MIN_DARK_RATIO", defaults.min_dark_ratio),
        same_width_tolerance=env_float("SAM_SAME_WIDTH_TOLERANCE", defaults.same_width_tolerance),
        polygon_epsilon=env_float("SAM_POLYGON_EPSILON", defaults.polygon_epsilon),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = config_from_env()
    model_backend = normalize_model_backend(os.getenv("MODEL_BACKEND", "sam"))
    checkpoint_env = (
        "MOBILESAM_CHECKPOINT" if model_backend == "mobilesam" else "SAM_CHECKPOINT"
    )
    model_type_env = (
        "MOBILESAM_MODEL_TYPE" if model_backend == "mobilesam" else "SAM_MODEL_TYPE"
    )
    checkpoint = Path(
        os.getenv(checkpoint_env, str(default_checkpoint_for_backend(model_backend)))
    )
    model_type = os.getenv(model_type_env, default_model_type_for_backend(model_backend))
    device = os.getenv("SAM_DEVICE", "auto")

    load_start = perf_counter()
    logger.info("Loading %s %s from %s on %s", model_backend, model_type, checkpoint, device)
    app.state.detector = TicketDetector(
        checkpoint=checkpoint,
        model_type=model_type,
        device=device,
        config=config,
        model_backend=model_backend,
    )
    load_seconds = perf_counter() - load_start
    logger.info(
        "Model loaded backend=%s model_type=%s device=%s load_time=%.3fs",
        app.state.detector.model_backend,
        app.state.detector.model_type,
        app.state.detector.device,
        load_seconds,
    )
    yield


app = FastAPI(
    title="SAM Ticket Detection Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    authorization: str | None = Header(default=None),
) -> None:
    expected = os.getenv("SAM_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SAM_API_KEY is not configured on the server.",
        )

    supplied_keys = []
    if x_api_key:
        supplied_keys.append(x_api_key)
    if authorization and authorization.lower().startswith("bearer "):
        supplied_keys.append(authorization[7:].strip())

    if not any(secrets.compare_digest(supplied, expected) for supplied in supplied_keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )


def detector_from_app(request: Request) -> TicketDetector:
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SAM model is not loaded.",
        )
    return detector


def validate_size(data: bytes) -> None:
    max_bytes = env_int("SAM_MAX_IMAGE_BYTES", 25 * 1024 * 1024)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image payload is too large. Limit is {max_bytes} bytes.",
        )


async def image_bytes_from_multipart(request: Request) -> bytes:
    form = await request.form()
    upload = form.get("image") or form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Multipart request must include an image file field named 'image' or 'file'.",
        )

    data = await upload.read()
    validate_size(data)
    return data


async def image_bytes_from_url(image_url: str) -> bytes:
    parsed = urlparse(image_url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="imageUrl must use http or https.",
        )

    if not env_bool("SAM_ALLOW_IMAGE_URL", True):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="imageUrl input is disabled on this server.",
        )

    timeout = env_float("SAM_IMAGE_URL_TIMEOUT_SECONDS", 15.0)
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            response = await client.get(image_url)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"imageUrl returned HTTP {exc.response.status_code}.",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not fetch imageUrl: {exc}",
        ) from exc

    data = response.content
    validate_size(data)
    return data


async def image_bytes_from_request(request: Request) -> bytes:
    content_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0].lower()

    if content_type == "multipart/form-data":
        return await image_bytes_from_multipart(request)

    if content_type == "application/json":
        try:
            payload: dict[str, Any] = await request.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON body.",
            ) from exc

        image_url = payload.get("imageUrl")
        if not isinstance(image_url, str) or not image_url.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="JSON body must include a non-empty imageUrl string.",
            )
        return await image_bytes_from_url(image_url.strip())

    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail="Use multipart/form-data with an image file or application/json with imageUrl.",
    )


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    detector = getattr(request.app.state, "detector", None)
    return {
        "status": "ok" if detector is not None else "starting",
        "modelLoaded": detector is not None,
        "modelBackend": None if detector is None else detector.model_backend,
        "modelType": None if detector is None else detector.model_type,
        "device": None if detector is None else detector.device,
    }


@app.post("/detect-tickets", dependencies=[Depends(require_api_key)])
async def detect_tickets(request: Request) -> dict[str, Any]:
    total_start = perf_counter()
    detector = detector_from_app(request)
    data = await image_bytes_from_request(request)

    try:
        image_bgr = decode_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        inference_start = perf_counter()
        tickets = detector.detect(
            image_bgr,
            enable_ocr=False,
            verbose=False,
        )
        inference_seconds = perf_counter() - inference_start
    except Exception as exc:
        logger.exception("Ticket detection failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ticket detection failed: {exc}",
        ) from exc

    total_seconds = perf_counter() - total_start
    logger.info(
        (
            "detect-tickets backend=%s model_type=%s device=%s image=%dx%d "
            "tickets=%d inference_time=%.3fs total_request_time=%.3fs"
        ),
        detector.model_backend,
        detector.model_type,
        detector.device,
        image_bgr.shape[1],
        image_bgr.shape[0],
        len(tickets),
        inference_seconds,
        total_seconds,
    )
    return tickets_api_response(image_bgr, tickets)
