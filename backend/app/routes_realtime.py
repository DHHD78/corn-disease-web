import asyncio
import base64
import json
import time
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from . import config
from .inference import detect_image_bytes
from .model_manager import get_model

router = APIRouter(tags=["realtime"])


@router.post("/api/realtime/video")
def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.ALLOWED_VIDEO_EXT:
        raise HTTPException(400, f"不支持的视频格式: {ext or '(无扩展名)'}")
    data = file.file.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"文件过大，上限 {config.MAX_UPLOAD_MB}MB")

    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    video_id = uuid.uuid4().hex
    (config.UPLOAD_DIR / f"{video_id}{ext}").write_bytes(data)
    return {"video_id": video_id}


@router.websocket("/ws/detect")
async def ws_detect(ws: WebSocket):
    await ws.accept()
    model_name = ""
    conf = config.DEFAULT_CONF
    iou = config.DEFAULT_IOU
    img_size = config.DEFAULT_IMG_SIZE
    yolo = None
    cap = None

    try:
        while True:
            raw = await ws.receive()
            if raw["type"] == "websocket.disconnect":
                break

            if raw["type"] == "websocket.receive" and raw.get("bytes") is not None:
                data = raw["bytes"]
                if yolo is None:
                    await ws.send_json({"type": "error", "message": "请先发送 config 配置模型"})
                    continue
                t0 = time.time()
                try:
                    annotated, detections, stats = detect_image_bytes(yolo, data, conf, iou, img_size)
                except ValueError as e:
                    await ws.send_json({"type": "error", "message": str(e)})
                    continue
                fps = 1.0 / (time.time() - t0) if time.time() - t0 > 0 else 0.0
                await ws.send_json(
                    {
                        "type": "result",
                        "annotated": base64.b64encode(annotated).decode("ascii"),
                        "detections": detections,
                        "stats": stats,
                        "fps": round(fps, 1),
                    }
                )
                continue

            if raw["type"] != "websocket.receive" or raw.get("text") is None:
                continue

            try:
                msg = json.loads(raw.get("text", "{}"))
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "消息不是合法 JSON"})
                continue

            mtype = msg.get("type")
            if mtype == "config":
                model_name = msg.get("model", "")
                conf = float(msg.get("conf", config.DEFAULT_CONF))
                iou = float(msg.get("iou", config.DEFAULT_IOU))
                img_size = int(msg.get("img_size", config.DEFAULT_IMG_SIZE))
                try:
                    yolo = get_model(model_name)
                except ValueError as e:
                    yolo = None
                    await ws.send_json({"type": "error", "message": str(e)})
                    continue
                await ws.send_json({"type": "ready"})

            elif mtype == "video":
                if cap is not None:
                    cap.release()
                video_id = msg.get("video_id", "")
                candidates = list(config.UPLOAD_DIR.glob(f"{video_id}.*"))
                if not candidates:
                    await ws.send_json({"type": "error", "message": "视频不存在或已过期"})
                    continue
                cap = cv2.VideoCapture(str(candidates[0]))
                if not cap.isOpened():
                    await ws.send_json({"type": "error", "message": "无法打开视频"})
                    continue

                src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    ok, buf = cv2.imencode(".jpg", frame)
                    if not ok:
                        continue
                    t0 = time.time()
                    try:
                        annotated, detections, stats = detect_image_bytes(
                            yolo, buf.tobytes(), conf, iou, img_size
                        )
                    except ValueError as e:
                        await ws.send_json({"type": "error", "message": str(e)})
                        break
                    fps = 1.0 / (time.time() - t0) if time.time() - t0 > 0 else 0.0
                    await ws.send_json(
                        {
                            "type": "result",
                            "annotated": base64.b64encode(annotated).decode("ascii"),
                            "detections": detections,
                            "stats": stats,
                            "fps": round(fps, 1),
                        }
                    )
                    if src_fps > 0:
                        await asyncio.sleep(1.0 / src_fps)
                cap.release()
                cap = None

            elif mtype == "stop":
                if cap is not None:
                    cap.release()
                    cap = None
                await ws.send_json({"type": "ready", "stopped": True})

    except WebSocketDisconnect:
        pass
    finally:
        if cap is not None:
            cap.release()
