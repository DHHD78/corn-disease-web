import cv2
import numpy as np
import re
import zipfile
from datetime import datetime
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pathlib import Path

from . import config
from .inference import detect_image_bytes, save_history
from .model_manager import get_model

router = APIRouter(prefix="/api/detect", tags=["detect"])
downloads_router = APIRouter(tags=["downloads"])


def _validate_image(file: UploadFile) -> bytes:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.ALLOWED_IMAGE_EXT:
        raise HTTPException(400, f"不支持的图片格式: {ext or '(无扩展名)'}")
    data = file.file.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"文件过大，上限 {config.MAX_UPLOAD_MB}MB")
    return data


@router.post("/image")
def detect_image(
    file: UploadFile = File(...),
    model: str = Form(""),
    conf: float = Form(config.DEFAULT_CONF),
    iou: float = Form(config.DEFAULT_IOU),
    img_size: int = Form(config.DEFAULT_IMG_SIZE),
):
    data = _validate_image(file)
    try:
        yolo = get_model(model)
    except ValueError as e:
        raise HTTPException(404, str(e))

    try:
        annotated, detections, stats = detect_image_bytes(yolo, data, conf, iou, img_size)
    except ValueError as e:
        raise HTTPException(400, str(e))

    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    h, w = frame.shape[:2]
    hist_id = save_history(
        source="image",
        params={"model": model, "conf": conf, "iou": iou, "img_size": img_size},
        detections=detections,
        stats=stats,
        original=data,
        annotated=annotated,
        img_w=w,
        img_h=h,
    )
    return {
        "history_id": hist_id,
        "detections": detections,
        "stats": stats,
        "annotated_url": f"/api/history/{hist_id}/annotated",
    }


@router.post("/batch")
def detect_batch(
    files: list[UploadFile] = File(...),
    model: str = Form(""),
    conf: float = Form(config.DEFAULT_CONF),
    iou: float = Form(config.DEFAULT_IOU),
    img_size: int = Form(config.DEFAULT_IMG_SIZE),
):
    try:
        yolo = get_model(model)
    except ValueError as e:
        raise HTTPException(404, str(e))

    results = []
    for f in files:
        data = _validate_image(f)
        annotated, detections, stats = detect_image_bytes(yolo, data, conf, iou, img_size)

        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        h, w = frame.shape[:2]
        hist_id = save_history(
            source="batch",
            params={"model": model, "conf": conf, "iou": iou, "img_size": img_size},
            detections=detections,
            stats=stats,
            original=data,
            annotated=annotated,
            img_w=w,
            img_h=h,
        )
        results.append(
            {
                "filename": f.filename,
                "history_id": hist_id,
                "detections": detections,
                "stats": stats,
            }
        )

    config.DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    zip_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = config.DOWNLOADS_DIR / zip_name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            ann = config.HISTORY_DIR / r["history_id"] / "annotated.jpg"
            zf.write(ann, arcname=f"{r['filename']}_annotated.jpg")

    return {
        "total": len(results),
        "results": results,
        "zip_url": f"/api/downloads/{zip_name}",
    }


_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@downloads_router.get("/api/downloads/{filename}")
def download(filename: str):
    if not _SAFE_FILENAME.match(filename):
        raise HTTPException(400, "非法文件名")
    path = config.DOWNLOADS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "文件不存在")
    return FileResponse(path, media_type="application/zip", filename=filename)
