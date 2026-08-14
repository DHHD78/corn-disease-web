import json
from collections import Counter
from datetime import datetime

import cv2
import numpy as np

from . import config
from utils.detector import draw_detections, extract_detections


def detect_image_bytes(model, img_bytes: bytes, conf: float, iou: float, img_size: int):
    """对图片字节做推理，返回 (标注JPEG字节, detections, stats)"""
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("无法解码图片")

    results = model.predict(source=frame, conf=conf, iou=iou, imgsz=img_size, verbose=False)
    detections = extract_detections(results)
    annotated = draw_detections(frame, detections)

    ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("标注图编码失败")

    stats = {
        "total": len(detections),
        "classes": dict(Counter(d["class_name"] for d in detections)),
    }
    return buf.tobytes(), detections, stats


def save_history(
    source: str,
    params: dict,
    detections: list,
    stats: dict,
    original: bytes,
    annotated: bytes,
    img_w: int,
    img_h: int,
) -> str:
    """将一次图片/批量检测结果落盘为历史记录，返回记录 id"""
    now = datetime.now()
    hist_id = f"{now.strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{source}"
    d = config.HISTORY_DIR / hist_id
    d.mkdir(parents=True, exist_ok=True)

    (d / "original.jpg").write_bytes(original)
    (d / "annotated.jpg").write_bytes(annotated)

    payload = {
        "id": hist_id,
        "source": source,
        "params": params,
        "detections": detections,
        "stats": stats,
        "created_at": now.isoformat(timespec="seconds"),
    }
    (d / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # YOLO 格式标注：class_id cx cy w h（归一化坐标）
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        lines.append(
            f"{det['class_id']} {cx / img_w:.6f} {cy / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"
        )
    (d / "labels.txt").write_text("\n".join(lines), encoding="utf-8")

    return hist_id
