# Corn Disease Detection Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将「基于改进 YOLOv8 的玉米叶部病害实时检测系统」改造成完整的前后端 Web 应用：图片/批量检测、摄像头/视频实时检测（WebSocket）、参数调节、模型切换、历史记录。

**Architecture:** FastAPI 后端复用训练仓库 `corn_disease_detection/utils/detector.py` 的模型加载/结果提取/画框逻辑；图片与批量走 REST API，摄像头/视频实时走 WebSocket；图片与批量检测自动落盘 `runs/web/<id>/` 作为历史记录。前端 Vue 3 + Vite + Element Plus，四个页面：图片检测、批量检测、实时检测、历史记录。

**Tech Stack:** Python 3.10+ / FastAPI / uvicorn / ultralytics / torch / opencv-python；Vue 3 / Vite / Element Plus / axios。

## Global Constraints

- 后端 Python >= 3.10；前端 Node >= 18
- 训练仓库默认路径 `F:\CURSOR\corn_disease_detection`，可用环境变量 `CORN_PROJECT_ROOT` 覆盖；权重只读、不复制
- 推理统一复用 `utils/detector.py` 的 `load_model` / `extract_detections` / `draw_detections`（由 `app/config.py` 将训练仓库加入 `sys.path`）
- 历史目录：`corn-disease-web/runs/web/<id>/`，仅图片/批量检测落盘；实时检测不存历史
- 上传大小上限 20MB；图片扩展名 `{.jpg,.jpeg,.png,.bmp,.tif,.webp}`；视频扩展名 `{.mp4,.avi,.mov,.mkv}`
- 错误统一返回中文 `{"detail": "..."}`；WebSocket 错误发送 `{type: "error", message}`
- 每个任务结束必须运行该任务的测试并通过，然后提交一次
- 检测参数默认值：`conf=0.25`、`iou=0.45`、`img_size=640`

---

## File Structure

```
corn-disease-web/
├── .gitignore
├── backend/
│   ├── requirements.txt
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py          # 路径、上传限制、默认参数（Task 1）
│   │   ├── main.py            # FastAPI 入口、CORS、路由注册、启动预加载（Task 1/2/4）
│   │   ├── model_manager.py   # scan_models / get_model / active_model_name（Task 2）
│   │   ├── inference.py       # detect_image_bytes / save_history（Task 3）
│   │   ├── routes_detect.py   # 图片/批量 REST（Task 4/5）
│   │   ├── routes_history.py  # 历史列表/回看/下载（Task 6）
│   │   └── routes_realtime.py # 视频上传 + WebSocket（Task 7）
│   └── tests/
│       ├── conftest.py
│       ├── test_health.py
│       ├── test_model_manager.py
│       ├── test_inference.py
│       ├── test_detect.py
│       ├── test_batch.py
│       ├── test_history.py
│       └── test_realtime.py
└── frontend/
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.js
        ├── App.vue            # 顶部导航 + 参数侧栏
        ├── api.js             # axios 实例
        ├── store.js           # 全局参数（模型/阈值/尺寸）
        └── views/
            ├── ImageDetect.vue
            ├── BatchDetect.vue
            ├── Realtime.vue
            └── History.vue
```

---

### Task 1: 后端骨架（配置 + 健康检查）

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/.gitignore`
- Create: `backend/app/__init__.py`
- Create: `backend/app/config.py`
- Create: `backend/app/main.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_health.py`

**Interfaces:**
- Consumes: 无
- Produces: `app.config`（模块级常量 `CORN_PROJECT_ROOT`、`TRAIN_DIR`、`HISTORY_DIR`、`UPLOAD_DIR`、`DOWNLOADS_DIR`、`MAX_UPLOAD_BYTES`、`ALLOWED_IMAGE_EXT`、`ALLOWED_VIDEO_EXT`、`DEFAULT_CONF`、`DEFAULT_IOU`、`DEFAULT_IMG_SIZE`）；`app.main.app`（FastAPI 实例，`GET /api/health` 返回 `{"status": "ok"}`

- [ ] **Step 1: 编写失败测试**

`backend/tests/test_health.py`:

```python
from fastapi.testclient import TestClient

from app.main import app


def test_health():
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
```

- [ ] **Step 2: 运行测试确认失败**

运行: `cd F:\CURSOR\corn-disease-web\backend && python -m pytest tests/test_health.py -v`

预期: FAIL（`ModuleNotFoundError: No module named 'app'` 或 `app.main` 不存在）

- [ ] **Step 3: 实现最小代码**

`backend/requirements.txt`:

```
fastapi>=0.110
uvicorn>=0.29
python-multipart>=0.0.9
ultralytics>=8.1.0
torch>=2.0.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=9.0.0
pytest>=8.0
httpx>=0.27
```

`backend/.gitignore`:

```
__pycache__/
.venv/
.pytest_cache/
runs/
```

`backend/app/__init__.py`: 空文件

`backend/app/config.py`:

```python
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 训练仓库根目录：权重与 utils.detector 的唯一来源
CORN_PROJECT_ROOT = Path(
    os.environ.get("CORN_PROJECT_ROOT", r"F:\CURSOR\corn_disease_detection")
)
sys.path.insert(0, str(CORN_PROJECT_ROOT))  # 使 `from utils.detector import ...` 可用

TRAIN_DIR = CORN_PROJECT_ROOT / "runs" / "train"
HISTORY_DIR = PROJECT_ROOT / "runs" / "web"
UPLOAD_DIR = PROJECT_ROOT / "runs" / "uploads"
DOWNLOADS_DIR = HISTORY_DIR / "downloads"

MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMG_SIZE = 640
```

`backend/app/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="玉米叶部病害检测 Web 系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}
```

`backend/tests/conftest.py`:

```python
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """每个测试使用独立的临时目录，避免污染真实 runs/"""
    import app.config as config

    monkeypatch.setattr(config, "HISTORY_DIR", tmp_path / "history")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "DOWNLOADS_DIR", tmp_path / "history" / "downloads")
    for d in (config.HISTORY_DIR, config.UPLOAD_DIR, config.DOWNLOADS_DIR):
        d.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: 运行测试确认通过**

运行: `python -m pytest tests/test_health.py -v`

预期: PASS（1 passed）

- [ ] **Step 5: 提交**

```bash
git add backend
git commit -m "feat(backend): scaffold FastAPI app with health check"
```

---

### Task 2: 模型管理器与模型列表 API

**Files:**
- Create: `backend/app/model_manager.py`
- Modify: `backend/app/main.py`（注册 `/api/models`、扩展 `/api/health`）
- Create: `backend/tests/test_model_manager.py`
- Modify: `backend/tests/test_health.py`

**Interfaces:**
- Consumes: `app.config.TRAIN_DIR`
- Produces:
  - `model_manager.scan_models() -> list[dict]`，每项 `{name, path, mtime, size}`，按 `mtime` 倒序
  - `model_manager.get_model(name: str, loader=None) -> object`；`name` 不存在抛 `ValueError`；`loader(path)` 默认懒加载 `utils.detector.load_model`；同一时刻只缓存一个模型
  - `model_manager.active_model_name() -> str | None`
  - `GET /api/models` 返回模型列表；`GET /api/health` 增加 `active_model` 与 `models_count`

- [ ] **Step 1: 编写失败测试**

`backend/tests/test_model_manager.py`:

```python
import time

import pytest

import app.config as config
from app import model_manager


@pytest.fixture(autouse=True)
def _clear_active():
    """每个测试前清空模型缓存，避免用例间串扰"""
    model_manager._active.clear()
    yield
    model_manager._active.clear()


def _make_fake_runs(tmp_path):
    for exp in ("exp_old", "exp_new"):
        d = tmp_path / "runs" / "train" / exp / "weights"
        d.mkdir(parents=True)
        (d / "best.pt").write_bytes(b"weights")
    old = tmp_path / "runs" / "train" / "exp_old" / "weights" / "best.pt"
    old.touch()
    new = tmp_path / "runs" / "train" / "exp_new" / "weights" / "best.pt"
    time.sleep(0.01)
    new.touch()
    return tmp_path / "runs" / "train"


def test_scan_models_sorted_desc(tmp_path, monkeypatch):
    train_dir = _make_fake_runs(tmp_path)
    monkeypatch.setattr(config, "TRAIN_DIR", train_dir)
    models = model_manager.scan_models()
    assert [m["name"] for m in models] == ["exp_new", "exp_old"]
    assert models[0]["path"].endswith("best.pt")
    assert models[0]["size"] == 7


def test_scan_models_empty_when_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRAIN_DIR", tmp_path / "no_such_dir")
    assert model_manager.scan_models() == []


def test_get_model_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRAIN_DIR", tmp_path / "runs" / "train")
    with pytest.raises(ValueError):
        model_manager.get_model("no_such_model")


def test_get_model_caches_and_switches(tmp_path, monkeypatch):
    train_dir = _make_fake_runs(tmp_path)
    monkeypatch.setattr(config, "TRAIN_DIR", train_dir)
    calls = []

    def fake_loader(path):
        calls.append(path)
        return object()

    m1 = model_manager.get_model("exp_old", loader=fake_loader)
    m1_again = model_manager.get_model("exp_old", loader=fake_loader)
    assert m1 is m1_again
    assert len(calls) == 1

    model_manager.get_model("exp_new", loader=fake_loader)
    assert len(calls) == 2
    assert model_manager.active_model_name() == "exp_new"


def test_models_api_and_health(tmp_path, monkeypatch):
    train_dir = _make_fake_runs(tmp_path)
    monkeypatch.setattr(config, "TRAIN_DIR", train_dir)
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert [m["name"] for m in resp.json()] == ["exp_new", "exp_old"]

    health = client.get("/api/health").json()
    assert health["models_count"] == 2
    assert health["active_model"] is None
```

- [ ] **Step 2: 运行测试确认失败**

运行: `python -m pytest tests/test_model_manager.py tests/test_health.py -v`

预期: FAIL（`No module named 'app.model_manager'`）

- [ ] **Step 3: 实现**

`backend/app/model_manager.py`:

```python
import torch

import config

_active: dict[str, object] = {}


def scan_models() -> list[dict]:
    """扫描 runs/train/*/weights/best.pt，按实验目录修改时间倒序"""
    train_dir = config.TRAIN_DIR
    if not train_dir.exists():
        return []
    entries = []
    for best in train_dir.glob("*/weights/best.pt"):
        st = best.stat()
        entries.append(
            {
                "name": best.parent.parent.name,
                "path": str(best),
                "mtime": st.st_mtime,
                "size": st.st_size,
            }
        )
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return entries


def _default_loader(path: str):
    from utils.detector import load_model

    return load_model(path)


def get_model(name: str, loader=None):
    """按名称加载模型；同一时刻只保留一个模型，切换时释放旧的并清理显存"""
    if name in _active:
        return _active[name]

    entry = next((m for m in scan_models() if m["name"] == name), None)
    if entry is None:
        raise ValueError(f"模型不存在: {name}")

    if _active:
        old_key = next(iter(_active))
        del _active[old_key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    loader = loader or _default_loader
    _active[name] = loader(entry["path"])
    return _active[name]


def active_model_name() -> str | None:
    return next(iter(_active), None)
```

修改 `backend/app/main.py`：

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model_manager import active_model_name, scan_models

app = FastAPI(title="玉米叶部病害检测 Web 系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "active_model": active_model_name(),
        "models_count": len(scan_models()),
    }


@app.get("/api/models")
def list_models():
    return scan_models()
```

- [ ] **Step 4: 运行测试确认通过**

运行: `python -m pytest tests/ -v`

预期: PASS（health 1 + model_manager 5）

- [ ] **Step 5: 提交**

```bash
git add backend
git commit -m "feat(backend): add model manager and models API"
```

---

### Task 3: 推理封装与历史落盘

**Files:**
- Create: `backend/app/inference.py`
- Create: `backend/tests/test_inference.py`
- Create: `backend/tests/stubs.py`（Stub 模型，供本任务及后续任务复用）
- Create: `backend/tests/__init__.py`（空文件，保证 `from tests.stubs import ...` 可导入）

**Interfaces:**
- Consumes: `app.config.HISTORY_DIR`；`utils.detector.extract_detections` / `utils.detector.draw_detections`
- Produces:
  - `inference.detect_image_bytes(model, img_bytes: bytes, conf, iou, img_size) -> tuple[bytes, list[dict], dict]`，返回 `(annotated_jpeg, detections, stats)`；`stats = {"total": int, "classes": {类别名: 数量}}`；无法解码抛 `ValueError("无法解码图片")`
  - `inference.save_history(source: str, params: dict, detections: list, stats: dict, original: bytes, annotated: bytes, img_w: int, img_h: int) -> str`，在 `HISTORY_DIR/<id>/` 写 `original.jpg`、`annotated.jpg`、`result.json`、`labels.txt`，返回 `id`；`id = YYYYmmdd_HHMMSS_fff_<source>`

- [ ] **Step 1: 编写失败测试**

`backend/tests/stubs.py`:

```python
"""Stub 模型：模拟 ultralytics 预测结果，避免测试依赖真实权重/GPU"""

import numpy as np


class FakeArr:
    def __init__(self, a):
        self.a = a

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakeBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = FakeArr(xyxy)
        self.cls = FakeArr(cls)
        self.conf = FakeArr(conf)
        self._n = len(xyxy)

    def __len__(self):
        return self._n


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class StubModel:
    """predict 返回一个包含 1 个目标（class 0，置信度 0.9）的结果"""

    def __init__(self):
        self.boxes = FakeBoxes(
            np.array([[10.0, 10.0, 50.0, 50.0]]),
            np.array([0]),
            np.array([0.9]),
        )

    def predict(self, source, conf=0.25, iou=0.45, imgsz=640, verbose=False):
        return [FakeResult(self.boxes)]


def make_jpeg_bytes() -> bytes:
    import cv2

    img = np.full((100, 100, 3), 120, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()
```

`backend/tests/test_inference.py`:

```python
import json

import pytest

import app.config as config
from app import inference
from tests.stubs import StubModel, make_jpeg_bytes


def test_detect_image_bytes_returns_annotated_and_stats():
    img = make_jpeg_bytes()
    annotated, detections, stats = inference.detect_image_bytes(
        StubModel(), img, conf=0.25, iou=0.45, img_size=640
    )
    assert annotated[:2] == b"\xff\xd8"
    assert len(detections) == 1
    assert detections[0]["class_id"] == 0
    assert detections[0]["confidence"] == 0.9
    assert stats["total"] == 1
    assert sum(stats["classes"].values()) == 1


def test_detect_image_bytes_invalid_image_raises():
    with pytest.raises(ValueError):
        inference.detect_image_bytes(StubModel(), b"not an image", 0.25, 0.45, 640)


def test_save_history_writes_all_files():
    img = make_jpeg_bytes()
    hist_id = inference.save_history(
        source="image",
        params={"conf": 0.25, "iou": 0.45, "img_size": 640, "model": "exp"},
        detections=[{"class_id": 0, "confidence": 0.9, "bbox": [10, 10, 50, 50]}],
        stats={"total": 1, "classes": {"玉米锈病": 1}},
        original=img,
        annotated=img,
        img_w=100,
        img_h=100,
    )
    d = config.HISTORY_DIR / hist_id
    assert (d / "original.jpg").exists()
    assert (d / "annotated.jpg").exists()
    assert (d / "labels.txt").exists()
    payload = json.loads((d / "result.json").read_text(encoding="utf-8"))
    assert payload["source"] == "image"
    assert payload["stats"]["total"] == 1
```

- [ ] **Step 2: 运行测试确认失败**

运行: `python -m pytest tests/test_inference.py -v`

预期: FAIL（`No module named 'app.inference'`）

- [ ] **Step 3: 实现**

`backend/app/inference.py`:

```python
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config
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
```

- [ ] **Step 4: 运行测试确认通过**

运行: `python -m pytest tests/test_inference.py -v`

预期: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add backend
git commit -m "feat(backend): add inference wrapper and history persistence"
```

---

### Task 4: 图片检测 API

**Files:**
- Create: `backend/app/routes_detect.py`
- Modify: `backend/app/main.py`（注册路由）
- Create: `backend/tests/test_detect.py`

**Interfaces:**
- Consumes: `app.config`、`model_manager.get_model`、`inference.detect_image_bytes`、`inference.save_history`
- Produces:
  - `POST /api/detect/image`（multipart：`file`、`model: str`、`conf: float`、`iou: float`、`img_size: int`）→ `200 {history_id, detections, stats, annotated_url}`；`400 {"detail": 中文错误}`（非法扩展名/超大小/无法解码）；`404 {"detail": "模型不存在: <name>"}`
  - 路由模块内部 `routes_detect.get_model` 与 `routes_detect.save_history` 引用模块级名称，测试可 monkeypatch

- [ ] **Step 1: 编写失败测试**

`backend/tests/test_detect.py`:

```python
import app.config as config
from app import routes_detect
from tests.stubs import StubModel, make_jpeg_bytes


def _post_image(client, filename="leaf.jpg", data=b"", **form):
    form = {
        "model": "exp",
        "conf": 0.25,
        "iou": 0.45,
        "img_size": 640,
        **form,
    }
    files = {"file": (filename, data, "image/jpeg")}
    return client.post("/api/detect/image", data=form, files=files)


def test_detect_image_ok(client, monkeypatch):
    monkeypatch.setattr(routes_detect, "get_model", lambda name: StubModel())
    resp = _post_image(client, data=make_jpeg_bytes())
    assert resp.status_code == 200
    body = resp.json()
    assert body["stats"]["total"] == 1
    assert body["annotated_url"].startswith("/api/history/")
    assert (config.HISTORY_DIR / body["history_id"] / "annotated.jpg").exists()


def test_detect_image_bad_extension(client):
    resp = _post_image(client, filename="leaf.txt", data=b"abc")
    assert resp.status_code == 400
    assert "不支持" in resp.json()["detail"]


def test_detect_image_too_large(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 10)
    resp = _post_image(client, data=b"x" * 100)
    assert resp.status_code == 400
    assert "过大" in resp.json()["detail"]


def test_detect_image_model_missing(client, monkeypatch):
    def raise_missing(name):
        raise ValueError(f"模型不存在: {name}")

    monkeypatch.setattr(routes_detect, "get_model", raise_missing)
    resp = _post_image(client, data=make_jpeg_bytes())
    assert resp.status_code == 404


def test_detect_image_invalid_image(client, monkeypatch):
    monkeypatch.setattr(routes_detect, "get_model", lambda name: StubModel())
    resp = _post_image(client, data=b"not an image")
    assert resp.status_code == 400
```

- [ ] **Step 2: 运行测试确认失败**

运行: `python -m pytest tests/test_detect.py -v`

预期: FAIL（`No module named 'app.routes_detect'`）

- [ ] **Step 3: 实现**

`backend/app/routes_detect.py`:

```python
import cv2
import numpy as np
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import config
from inference import detect_image_bytes, save_history
from model_manager import get_model

router = APIRouter(prefix="/api/detect", tags=["detect"])


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
```

说明：`frame` 已由 `detect_image_bytes` 解码，这里重复解码只为取宽高。为避免重复，后续实现可将 `detect_image_bytes` 返回宽高；本计划保持现状（性能无影响）。

修改 `backend/app/main.py`：

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model_manager import active_model_name, scan_models
from routes_detect import router as detect_router

app = FastAPI(title="玉米叶部病害检测 Web 系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect_router)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "active_model": active_model_name(),
        "models_count": len(scan_models()),
    }


@app.get("/api/models")
def list_models():
    return scan_models()
```

`backend/tests/conftest.py` 增加 `client` fixture（追加到文件末尾）：

```python
@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)
```

- [ ] **Step 4: 运行测试确认通过**

运行: `python -m pytest tests/ -v`

预期: PASS（health 1 + model_manager 5 + inference 3 + detect 5）

- [ ] **Step 5: 提交**

```bash
git add backend
git commit -m "feat(backend): add image detection API"
```

---

### Task 5: 批量检测 API

**Files:**
- Modify: `backend/app/routes_detect.py`（增加 `/batch` 与 zip 下载路由）
- Create: `backend/tests/test_batch.py`

**Interfaces:**
- Consumes: `app.config.DOWNLOADS_DIR`、Task 4 的检测/历史函数
- Produces:
  - `POST /api/detect/batch`（multipart：`files[]`、`model`、`conf`、`iou`、`img_size`）→ `200 {total, results: [{filename, history_id, detections, stats}], zip_url}`
  - `GET /api/downloads/{filename}` → 下载 `DOWNLOADS_DIR` 下的文件（文件名做白名单校验，防目录穿越）

- [ ] **Step 1: 编写失败测试**

`backend/tests/test_batch.py`:

```python
from fastapi.testclient import TestClient

from app import routes_detect
from tests.stubs import StubModel, make_jpeg_bytes


def test_batch_detect_ok(client, monkeypatch):
    monkeypatch.setattr(routes_detect, "get_model", lambda name: StubModel())
    files = [
        ("files", ("a.jpg", make_jpeg_bytes(), "image/jpeg")),
        ("files", ("b.jpg", make_jpeg_bytes(), "image/jpeg")),
    ]
    form = {"model": "exp", "conf": 0.25, "iou": 0.45, "img_size": 640}
    resp = client.post("/api/detect/batch", data=form, files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert [r["filename"] for r in body["results"]] == ["a.jpg", "b.jpg"]
    assert all(r["stats"]["total"] == 1 for r in body["results"])
    assert body["zip_url"].startswith("/api/downloads/")

    zip_resp = client.get(body["zip_url"])
    assert zip_resp.status_code == 200
    assert zip_resp.headers["content-type"] == "application/zip"


def test_batch_detect_one_invalid(client, monkeypatch):
    monkeypatch.setattr(routes_detect, "get_model", lambda name: StubModel())
    files = [
        ("files", ("ok.jpg", make_jpeg_bytes(), "image/jpeg")),
        ("files", ("bad.txt", b"abc", "text/plain")),
    ]
    form = {"model": "exp", "conf": 0.25, "iou": 0.45, "img_size": 640}
    resp = client.post("/api/detect/batch", data=form, files=files)
    assert resp.status_code == 400


def test_downloads_path_traversal_blocked(client):
    resp = client.get("/api/downloads/..%2F..%2Fsecret.txt")
    assert resp.status_code in (400, 404)
```

- [ ] **Step 2: 运行测试确认失败**

运行: `python -m pytest tests/test_batch.py -v`

预期: FAIL（404：`/api/detect/batch` 不存在）

- [ ] **Step 3: 实现**

在 `backend/app/routes_detect.py` 中追加：

```python
import io
import re
import zipfile
from datetime import datetime

from fastapi.responses import FileResponse


downloads_router = APIRouter(tags=["downloads"])


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
        import cv2
        import numpy as np

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
            {"filename": f.filename, "history_id": hist_id, "detections": detections, "stats": stats}
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
```

注意：`_validate_image` 抛 `HTTPException(400, ...)`，批量中任一文件非法即整体 400（符合测试预期）。

修改 `backend/app/main.py`：将 import 改为 `from routes_detect import detect_router, downloads_router`，并追加 `app.include_router(downloads_router)`。

- [ ] **Step 4: 运行测试确认通过**

运行: `python -m pytest tests/test_batch.py -v`

预期: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add backend
git commit -m "feat(backend): add batch detection API with zip download"
```

---

### Task 6: 历史记录 API

**Files:**
- Create: `backend/app/routes_history.py`
- Modify: `backend/app/main.py`（注册路由）
- Create: `backend/tests/test_history.py`

**Interfaces:**
- Consumes: `app.config.HISTORY_DIR`
- Produces:
  - `GET /api/history` → `[{id, source, created_at, params, stats, annotated_url}]` 按时间倒序
  - `GET /api/history/{id}/annotated`、`/original`、`/json`、`/labels` → 对应文件
  - `GET /api/history/{id}/zip` → 打包整条记录的 zip
  - `id` 必须匹配 `^\d{8}_\d{6}_\d{3}_\w+$`，否则 404

- [ ] **Step 1: 编写失败测试**

`backend/tests/test_history.py`:

```python
import json

import app.config as config


def _seed_history(tmp_path):
    d = config.HISTORY_DIR / "20260814_120000_000_image"
    d.mkdir(parents=True)
    (d / "original.jpg").write_bytes(b"orig")
    (d / "annotated.jpg").write_bytes(b"ann")
    (d / "labels.txt").write_text("0 0.5 0.5 0.2 0.2")
    (d / "result.json").write_text(
        json.dumps(
            {
                "id": d.name,
                "source": "image",
                "created_at": "2026-08-14T12:00:00",
                "params": {"model": "exp"},
                "stats": {"total": 1},
                "detections": [],
            }
        ),
        encoding="utf-8",
    )
    return d.name


def test_history_list(client, tmp_path):
    hid = _seed_history(tmp_path)
    resp = client.get("/api/history")
    assert resp.status_code == 200
    assert [h["id"] for h in resp.json()] == [hid]
    assert resp.json()[0]["annotated_url"] == f"/api/history/{hid}/annotated"


def test_history_files(client, tmp_path):
    hid = _seed_history(tmp_path)
    assert client.get(f"/api/history/{hid}/annotated").status_code == 200
    assert client.get(f"/api/history/{hid}/original").status_code == 200
    assert client.get(f"/api/history/{hid}/json").json()["stats"]["total"] == 1
    zip_resp = client.get(f"/api/history/{hid}/zip")
    assert zip_resp.status_code == 200
    assert zip_resp.headers["content-type"] == "application/zip"


def test_history_invalid_id(client):
    assert client.get("/api/history/..%2F..%2Fetc/annotated").status_code in (400, 404)
    assert client.get("/api/history/no_such_id/annotated").status_code == 404
```

- [ ] **Step 2: 运行测试确认失败**

运行: `python -m pytest tests/test_history.py -v`

预期: FAIL（404：`/api/history` 不存在）

- [ ] **Step 3: 实现**

`backend/app/routes_history.py`:

```python
import io
import json
import re
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

import config

router = APIRouter(prefix="/api/history", tags=["history"])

_ID_PATTERN = re.compile(r"^\d{8}_\d{6}_\d{3}_\w+$")


def _record_dir(hist_id: str):
    if not _ID_PATTERN.match(hist_id):
        raise HTTPException(400, "非法记录 id")
    d = config.HISTORY_DIR / hist_id
    if not d.is_dir():
        raise HTTPException(404, "记录不存在")
    return d


@router.get("")
def list_history():
    records = []
    if config.HISTORY_DIR.exists():
        for d in config.HISTORY_DIR.iterdir():
            if not (d / "result.json").exists():
                continue
            try:
                payload = json.loads((d / "result.json").read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            records.append(
                {
                    "id": d.name,
                    "source": payload.get("source", ""),
                    "created_at": payload.get("created_at", ""),
                    "params": payload.get("params", {}),
                    "stats": payload.get("stats", {}),
                    "annotated_url": f"/api/history/{d.name}/annotated",
                }
            )
    records.sort(key=lambda r: r["created_at"], reverse=True)
    return records


@router.get("/{hist_id}/annotated")
def annotated(hist_id: str):
    d = _record_dir(hist_id)
    return FileResponse(d / "annotated.jpg", media_type="image/jpeg")


@router.get("/{hist_id}/original")
def original(hist_id: str):
    d = _record_dir(hist_id)
    return FileResponse(d / "original.jpg", media_type="image/jpeg")


@router.get("/{hist_id}/json")
def record_json(hist_id: str):
    d = _record_dir(hist_id)
    payload = json.loads((d / "result.json").read_text(encoding="utf-8"))
    return JSONResponse(payload)


@router.get("/{hist_id}/zip")
def record_zip(hist_id: str):
    d = _record_dir(hist_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ("original.jpg", "annotated.jpg", "result.json", "labels.txt"):
            f = d / name
            if f.exists():
                zf.write(f, arcname=f"{hist_id}_{name}")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{hist_id}.zip"'},
    )
```

修改 `backend/app/main.py`：在 `app.include_router(detect_router)` 后追加 `app.include_router(history_router)`，并 import `from routes_history import router as history_router`。

- [ ] **Step 4: 运行测试确认通过**

运行: `python -m pytest tests/test_history.py -v`

预期: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add backend
git commit -m "feat(backend): add history API"
```

---

### Task 7: 实时检测（视频上传 + WebSocket）

**Files:**
- Create: `backend/app/routes_realtime.py`
- Modify: `backend/app/main.py`（注册路由）
- Create: `backend/tests/test_realtime.py`

**Interfaces:**
- Consumes: `app.config.UPLOAD_DIR`、`model_manager.get_model`、`inference.detect_image_bytes`
- Produces:
  - `POST /api/realtime/video`（multipart：`file`）→ `200 {video_id}`；校验扩展名/大小
  - `WS /ws/detect`：
    - 客户端二进制消息 = 摄像头 JPEG 帧 → 回复 `{type:"result", annotated: base64, detections, stats, fps}`
    - 客户端 JSON `{"type":"config", model, conf, iou, img_size}` → 回复 `{type:"ready"}`
    - 客户端 JSON `{"type":"video", video_id}` → 服务端用 OpenCV 逐帧推理推流，帧间回复同 `result`
    - 客户端 JSON `{"type":"stop"}` → 停止并释放
    - 任何异常 → `{type:"error", message}`（不关闭连接）

- [ ] **Step 1: 编写失败测试**

`backend/tests/test_realtime.py`:

```python
from fastapi.testclient import TestClient

from app import routes_realtime
from tests.stubs import StubModel, make_jpeg_bytes


def test_upload_video_ok(client):
    resp = client.post(
        "/api/realtime/video",
        files={"file": ("demo.mp4", b"fake-mp4-bytes", "video/mp4")},
    )
    assert resp.status_code == 200
    assert resp.json()["video_id"]


def test_upload_video_bad_ext(client):
    resp = client.post(
        "/api/realtime/video",
        files={"file": ("demo.txt", b"abc", "text/plain")},
    )
    assert resp.status_code == 400


def test_ws_frame_and_error_paths(client, monkeypatch):
    monkeypatch.setattr(routes_realtime, "get_model", lambda name: StubModel())
    with client.websocket_connect("/ws/detect") as ws:
        ws.send_json({"type": "config", "model": "exp", "conf": 0.25, "iou": 0.45, "img_size": 640})
        assert ws.receive_json()["type"] == "ready"

        ws.send_bytes(make_jpeg_bytes())
        msg = ws.receive_json()
        assert msg["type"] == "result"
        assert msg["stats"]["total"] == 1
        assert msg["annotated"]

        ws.send_json({"type": "video", "video_id": "no_such_video"})
        assert ws.receive_json()["type"] == "error"


def test_ws_config_missing_model(client, monkeypatch):
    def raise_missing(name):
        raise ValueError(f"模型不存在: {name}")

    monkeypatch.setattr(routes_realtime, "get_model", raise_missing)
    with client.websocket_connect("/ws/detect") as ws:
        ws.send_json({"type": "config", "model": "nope"})
        assert ws.receive_json()["type"] == "error"
```

- [ ] **Step 2: 运行测试确认失败**

运行: `python -m pytest tests/test_realtime.py -v`

预期: FAIL（404：`/api/realtime/video` 不存在）

- [ ] **Step 3: 实现**

`backend/app/routes_realtime.py`:

```python
import base64
import time
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

import config
from inference import detect_image_bytes
from model_manager import get_model

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

            if raw["type"] == "websocket.receive_bytes":
                data = raw.get("bytes")
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

            if raw["type"] != "websocket.receive_text":
                continue

            import json

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
                video_path = config.UPLOAD_DIR / f"{msg.get('video_id', '')}"
                candidates = list(config.UPLOAD_DIR.glob(f"{msg.get('video_id', '')}.*"))
                if not candidates:
                    await ws.send_json({"type": "error", "message": "视频不存在或已过期"})
                    continue
                cap = cv2.VideoCapture(str(candidates[0]))
                if not cap.isOpened():
                    await ws.send_json({"type": "error", "message": "无法打开视频"})
                    continue
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
```

说明：WebSocket 循环按 FastAPI 实际消息格式处理 `websocket.receive_bytes` / `websocket.receive_text`。

修改 `backend/app/main.py`：import `from routes_realtime import router as realtime_router`，并 `app.include_router(realtime_router)`。

- [ ] **Step 4: 运行测试确认通过**

运行: `python -m pytest tests/ -v`

预期: PASS（新增 realtime 4 个用例）

- [ ] **Step 5: 提交**

```bash
git add backend
git commit -m "feat(backend): add realtime video upload and WebSocket detection"
```

---

### Task 8: 前端脚手架（Vite + Vue 3 + Element Plus）

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.js`
- Create: `frontend/index.html`
- Create: `frontend/src/main.js`
- Create: `frontend/src/api.js`
- Create: `frontend/src/store.js`
- Create: `frontend/src/App.vue`
- Create: `.gitignore`（项目根，补充 node_modules/dist）

**Interfaces:**
- Consumes: 后端 API（`/api/models`、`/api/health`）
- Produces:
  - `src/store.js`：导出 `params`（reactive：`model/conf/iou/imgSize/models`）、`loadModels()`、`persistParams()`
  - `src/api.js`：默认导出 axios 实例（baseURL `/api`，timeout 180s）
  - `src/App.vue`：顶部导航四页 + 侧栏参数，用 `<component :is>` 切换视图

- [ ] **Step 1: 创建脚手架文件**

`frontend/package.json`:

```json
{
  "name": "corn-disease-web-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "@element-plus/icons-vue": "^2.3.1",
    "axios": "^1.7.0",
    "element-plus": "^2.7.0",
    "vue": "^3.4.0"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.0.0",
    "vite": "^5.2.0"
  }
}
```

`frontend/vite.config.js`:

```js
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
```

`frontend/index.html`:

```html
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>玉米叶部病害智能检测系统</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.js"></script>
  </body>
</html>
```

`frontend/src/main.js`:

```js
import { createApp } from "vue";
import ElementPlus from "element-plus";
import "element-plus/dist/index.css";
import App from "./App.vue";

createApp(App).use(ElementPlus).mount("#app");
```

`frontend/src/api.js`:

```js
import axios from "axios";

const api = axios.create({ baseURL: "/api", timeout: 180000 });
export default api;
```

`frontend/src/store.js`:

```js
import { reactive } from "vue";
import api from "./api";

const KEY = "corn_disease_params";

function loadSaved() {
  try {
    return JSON.parse(localStorage.getItem(KEY)) || {};
  } catch {
    return {};
  }
}

export const params = reactive({
  model: "",
  conf: 0.25,
  iou: 0.45,
  imgSize: 640,
  models: [],
  ...loadSaved(),
});

export function persistParams() {
  localStorage.setItem(
    KEY,
    JSON.stringify({
      model: params.model,
      conf: params.conf,
      iou: params.iou,
      imgSize: params.imgSize,
    })
  );
}

export async function loadModels() {
  const { data } = await api.get("/models");
  params.models = data;
  if (!params.model && data.length) {
    params.model = data[0].name;
    persistParams();
  }
}
```

`frontend/src/App.vue`:

```vue
<template>
  <el-container class="app">
    <el-header class="header">
      <span class="title">🌽 玉米叶部病害智能检测系统</span>
      <el-menu mode="horizontal" :default-active="active" @select="active = $event" class="menu">
        <el-menu-item index="image">图片检测</el-menu-item>
        <el-menu-item index="batch">批量检测</el-menu-item>
        <el-menu-item index="realtime">实时检测</el-menu-item>
        <el-menu-item index="history">历史记录</el-menu-item>
      </el-menu>
    </el-header>
    <el-container>
      <el-aside width="260px" class="aside">
        <el-form label-width="90px">
          <el-form-item label="模型">
            <el-select v-model="params.model" style="width: 100%">
              <el-option v-for="m in params.models" :key="m.name" :label="m.name" :value="m.name" />
            </el-select>
          </el-form-item>
          <el-form-item label="置信度">
            <el-slider v-model="params.conf" :min="0.05" :max="0.95" :step="0.05" />
          </el-form-item>
          <el-form-item label="IoU">
            <el-slider v-model="params.iou" :min="0.1" :max="0.9" :step="0.05" />
          </el-form-item>
          <el-form-item label="图像尺寸">
            <el-select v-model="params.imgSize">
              <el-option v-for="s in [416, 512, 640, 768]" :key="s" :label="`${s}`" :value="s" />
            </el-select>
          </el-form-item>
        </el-form>
      </el-aside>
      <el-main>
        <component :is="currentView" />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { computed, onMounted, ref, watch } from "vue";
import { loadModels, params, persistParams } from "./store";
import ImageDetect from "./views/ImageDetect.vue";
import BatchDetect from "./views/BatchDetect.vue";
import Realtime from "./views/Realtime.vue";
import History from "./views/History.vue";

const active = ref("image");
const views = { image: ImageDetect, batch: BatchDetect, realtime: Realtime, history: History };
const currentView = computed(() => views[active.value]);

watch(params, persistParams, { deep: true });
onMounted(loadModels);
</script>

<style>
.header { display: flex; align-items: center; gap: 24px; border-bottom: 1px solid #eee; }
.title { font-size: 18px; font-weight: bold; color: #2c3e50; }
.menu { flex: 1; border-bottom: none; }
.aside { border-right: 1px solid #eee; padding: 16px 12px; }
</style>
```

项目根 `.gitignore`：

```
__pycache__/
.venv/
.pytest_cache/
runs/
node_modules/
dist/
```

- [ ] **Step 2: 安装依赖并构建**

运行: `cd F:\CURSOR\corn-disease-web\frontend && npm.cmd install && npm.cmd run build`

预期: 构建成功，生成 `dist/`

说明：`npm.cmd` 是 Windows PowerShell 下的 npm 入口；当前目录会创建 `node_modules`（已被 .gitignore 忽略）。

- [ ] **Step 3: 提交**

```bash
git add frontend .gitignore
git commit -m "feat(frontend): scaffold Vite + Vue3 + Element Plus app"
```

---

### Task 9: 图片检测页面

**Files:**
- Create: `frontend/src/views/ImageDetect.vue`

**Interfaces:**
- Consumes: `api`、`params`（Task 8）；`POST /api/detect/image`（Task 4）
- Produces: 图片上传 → 标注图 + 统计 + 检测框表格

- [ ] **Step 1: 创建组件**

`frontend/src/views/ImageDetect.vue`:

```vue
<template>
  <div>
    <el-upload
      drag
      :auto-upload="false"
      :limit="1"
      accept="image/*"
      :on-change="onFileChange"
      :on-remove="clear"
    >
      <div style="font-size: 16px">拖拽图片到这里，或点击选择</div>
      <div style="color: #909399; font-size: 13px">支持 jpg / png / bmp / tif / webp，≤20MB</div>
    </el-upload>

    <div v-if="file" style="margin-top: 12px">
      <el-button type="primary" :loading="loading" @click="run">开始检测</el-button>
      <el-button @click="clear">清除</el-button>
    </div>

    <el-alert v-if="error" type="error" :title="error" style="margin-top: 12px" show-icon />

    <div v-if="result" style="display: flex; gap: 20px; margin-top: 16px; flex-wrap: wrap">
      <img :src="result.annotatedUrl" style="max-width: 560px; border: 1px solid #eee; border-radius: 8px" />
      <div style="min-width: 300px; flex: 1">
        <el-descriptions title="检测结果" :column="1" border>
          <el-descriptions-item label="目标总数">{{ result.stats.total }}</el-descriptions-item>
          <el-descriptions-item v-for="(v, k) in result.stats.classes" :key="k" :label="k">
            {{ v }}
          </el-descriptions-item>
        </el-descriptions>
        <el-table :data="result.detections" size="small" max-height="360" style="margin-top: 12px">
          <el-table-column prop="class_name" label="类别" />
          <el-table-column prop="confidence" label="置信度">
            <template #default="{ row }">{{ (row.confidence * 100).toFixed(1) }}%</template>
          </el-table-column>
          <el-table-column label="坐标">
            <template #default="{ row }">{{ row.bbox.map((n) => Math.round(n)).join(", ") }}</template>
          </el-table-column>
        </el-table>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { ElMessage } from "element-plus";
import api from "../api";
import { params } from "../store";

const file = ref(null);
const loading = ref(false);
const error = ref("");
const result = ref(null);

function onFileChange(f) {
  file.value = f.raw;
  result.value = null;
  error.value = "";
}

function clear() {
  file.value = null;
  result.value = null;
  error.value = "";
}

async function run() {
  if (!file.value) return;
  loading.value = true;
  error.value = "";
  try {
    const form = new FormData();
    form.append("file", file.value);
    form.append("model", params.model);
    form.append("conf", params.conf);
    form.append("iou", params.iou);
    form.append("img_size", params.imgSize);
    const { data } = await api.post("/detect/image", form);
    result.value = { ...data, annotatedUrl: data.annotated_url };
  } catch (e) {
    error.value = e.response?.data?.detail || "检测失败";
    ElMessage.error(error.value);
  } finally {
    loading.value = false;
  }
}
</script>
```

- [ ] **Step 2: 构建验证**

运行: `cd F:\CURSOR\corn-disease-web\frontend && npm.cmd run build`

预期: 构建成功，无编译错误

- [ ] **Step 3: 提交**

```bash
git add frontend
git commit -m "feat(frontend): add image detection page"
```

---

### Task 10: 批量检测页面

**Files:**
- Create: `frontend/src/views/BatchDetect.vue`

**Interfaces:**
- Consumes: `api`、`params`；`POST /api/detect/batch`、`/api/downloads/{filename}`（Task 5）
- Produces: 多文件选择 → 汇总表格 + zip 下载

- [ ] **Step 1: 创建组件**

`frontend/src/views/BatchDetect.vue`:

```vue
<template>
  <div>
    <el-upload
      drag
      multiple
      :auto-upload="false"
      accept="image/*"
      :on-change="onFiles"
      :on-remove="onFiles"
      :file-list="fileList"
    >
      <div style="font-size: 16px">选择多张图片进行批量检测</div>
      <div style="color: #909399; font-size: 13px">每张 ≤20MB，逐张处理并自动保存历史</div>
    </el-upload>

    <div style="margin-top: 12px">
      <el-button type="primary" :loading="loading" :disabled="!files.length" @click="run">
        批量检测（{{ files.length }} 张）
      </el-button>
      <el-button v-if="zipUrl" type="success" @click="downloadZip">下载标注图 zip</el-button>
    </div>

    <el-table v-if="results.length" :data="results" style="margin-top: 16px">
      <el-table-column prop="filename" label="文件" />
      <el-table-column prop="stats.total" label="目标数" width="100" />
      <el-table-column label="操作" width="200">
        <template #default="{ row }">
          <el-link type="primary" :href="`/api/history/${row.history_id}/annotated`" target="_blank">
            查看标注图
          </el-link>
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { ElMessage } from "element-plus";
import api from "../api";
import { params } from "../store";

const files = ref([]);
const fileList = ref([]);
const loading = ref(false);
const results = ref([]);
const zipUrl = ref("");

function onFiles(_f, list) {
  fileList.value = list;
  files.value = list.map((i) => i.raw);
}

async function run() {
  loading.value = true;
  try {
    const form = new FormData();
    for (const f of files.value) form.append("files", f);
    form.append("model", params.model);
    form.append("conf", params.conf);
    form.append("iou", params.iou);
    form.append("img_size", params.imgSize);
    const { data } = await api.post("/detect/batch", form);
    results.value = data.results;
    zipUrl.value = data.zip_url;
    ElMessage.success(`处理完成，共 ${data.total} 张`);
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || "批量检测失败");
  } finally {
    loading.value = false;
  }
}

function downloadZip() {
  window.open(zipUrl.value, "_blank");
}
</script>
```

- [ ] **Step 2: 构建验证**

运行: `cd F:\CURSOR\corn-disease-web\frontend && npm.cmd run build`

预期: 构建成功

- [ ] **Step 3: 提交**

```bash
git add frontend
git commit -m "feat(frontend): add batch detection page"
```

---

### Task 11: 实时检测页面

**Files:**
- Create: `frontend/src/views/Realtime.vue`

**Interfaces:**
- Consumes: `api`、`params`；`POST /api/realtime/video`、`WS /ws/detect`（Task 7）
- Produces: 摄像头（getUserMedia 抽帧）与视频文件（上传后服务端推流）两种实时检测

- [ ] **Step 1: 创建组件**

`frontend/src/views/Realtime.vue`:

```vue
<template>
  <div>
    <el-tabs v-model="tab">
      <el-tab-pane label="摄像头" name="camera">
        <el-button type="primary" :disabled="running" @click="startCamera">开始摄像头检测</el-button>
        <el-button :disabled="!running" @click="stop">停止</el-button>
        <p style="color: #909399; font-size: 13px">浏览器将请求摄像头权限，画面以约 15fps 抽帧推送给后端</p>
      </el-tab-pane>
      <el-tab-pane label="视频文件" name="video">
        <el-upload :auto-upload="false" :limit="1" :on-change="onVideo" accept="video/*">
          <el-button>选择视频（mp4/avi/mov/mkv）</el-button>
        </el-upload>
        <div style="margin-top: 12px">
          <el-button type="primary" :disabled="!videoId || running" @click="startVideo">
            开始视频检测
          </el-button>
          <el-button :disabled="!running" @click="stop">停止</el-button>
        </div>
      </el-tab-pane>
    </el-tabs>

    <div style="display: flex; gap: 20px; margin-top: 16px; flex-wrap: wrap">
      <canvas ref="canvas" width="640" height="480" style="border: 1px solid #ddd; background: #000; border-radius: 8px" />
      <div style="min-width: 200px">
        <el-statistic title="FPS" :value="fps" />
        <el-statistic title="目标总数" :value="total" style="margin-top: 16px" />
        <div v-for="(v, k) in classes" :key="k" style="margin-top: 8px">{{ k }}: {{ v }}</div>
      </div>
    </div>

    <video ref="video" style="display: none" playsinline />
  </div>
</template>

<script setup>
import { onBeforeUnmount, ref } from "vue";
import { ElMessage } from "element-plus";
import api from "../api";
import { params } from "../store";

const tab = ref("camera");
const canvas = ref(null);
const video = ref(null);
const running = ref(false);
const fps = ref(0);
const total = ref(0);
const classes = ref({});
const videoId = ref("");

let ws = null;
let stream = null;
let rafId = null;

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws/detect`;
}

function openWs(onOpen) {
  ws = new WebSocket(wsUrl());
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    ws.send(
      JSON.stringify({
        type: "config",
        model: params.model,
        conf: params.conf,
        iou: params.iou,
        img_size: params.imgSize,
      })
    );
    onOpen && onOpen();
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "result") {
      const img = new Image();
      img.onload = () => {
        const ctx = canvas.value.getContext("2d");
        ctx.drawImage(img, 0, 0, canvas.value.width, canvas.value.height);
      };
      img.src = "data:image/jpeg;base64," + msg.annotated;
      fps.value = msg.fps;
      total.value = msg.stats.total;
      classes.value = msg.stats.classes;
    } else if (msg.type === "error") {
      ElMessage.error(msg.message);
    }
  };
  ws.onclose = () => {
    running.value = false;
    if (rafId) cancelAnimationFrame(rafId);
  };
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.value.srcObject = stream;
    await video.value.play();
    openWs(() => {
      running.value = true;
      const sendFrame = () => {
        if (!running.value || !ws || ws.readyState !== WebSocket.OPEN) return;
        const ctx = canvas.value.getContext("2d");
        ctx.drawImage(video.value, 0, 0, 640, 480);
        canvas.value.toBlob((blob) => {
          if (blob && ws.readyState === WebSocket.OPEN) ws.send(blob);
        }, "image/jpeg", 0.7);
        rafId = requestAnimationFrame(sendFrame);
      };
      rafId = requestAnimationFrame(sendFrame);
    });
  } catch (e) {
    ElMessage.error("无法访问摄像头: " + e.message);
  }
}

async function onVideo(f) {
  const form = new FormData();
  form.append("file", f.raw);
  try {
    const { data } = await api.post("/realtime/video", form);
    videoId.value = data.video_id;
    ElMessage.success("视频上传成功，可开始检测");
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || "视频上传失败");
  }
}

function startVideo() {
  openWs(() => {
    running.value = true;
    ws.send(JSON.stringify({ type: "video", video_id: videoId.value }));
  });
}

function stop() {
  running.value = false;
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "stop" }));
  if (ws) ws.close();
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  if (rafId) cancelAnimationFrame(rafId);
}

onBeforeUnmount(stop);
</script>
```

- [ ] **Step 2: 构建验证**

运行: `cd F:\CURSOR\corn-disease-web\frontend && npm.cmd run build`

预期: 构建成功

- [ ] **Step 3: 提交**

```bash
git add frontend
git commit -m "feat(frontend): add realtime detection page"
```

---

### Task 12: 历史记录页面

**Files:**
- Create: `frontend/src/views/History.vue`

**Interfaces:**
- Consumes: `api`；`GET /api/history`、`/api/history/{id}/...`（Task 6）
- Produces: 历史列表（缩略图）+ 查看/下载

- [ ] **Step 1: 创建组件**

`frontend/src/views/History.vue`:

```vue
<template>
  <div>
    <el-button style="margin-bottom: 12px" @click="load">刷新</el-button>
    <el-table :data="records" v-loading="loading">
      <el-table-column prop="created_at" label="时间" width="180" />
      <el-table-column prop="source" label="类型" width="100" />
      <el-table-column prop="params.model" label="模型" width="200" />
      <el-table-column prop="stats.total" label="目标数" width="90" />
      <el-table-column label="缩略图" width="130">
        <template #default="{ row }">
          <img :src="row.annotated_url" style="width: 90px; height: 64px; object-fit: cover; border-radius: 4px" />
        </template>
      </el-table-column>
      <el-table-column label="操作" width="280">
        <template #default="{ row }">
          <el-link type="primary" :href="row.annotated_url" target="_blank">标注图</el-link>
          <el-link type="primary" :href="`/api/history/${row.id}/original`" target="_blank" style="margin-left: 12px">
            原图
          </el-link>
          <el-link type="primary" :href="`/api/history/${row.id}/zip`" style="margin-left: 12px">
            下载 zip
          </el-link>
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>

<script setup>
import { onMounted, ref } from "vue";
import api from "../api";

const records = ref([]);
const loading = ref(false);

async function load() {
  loading.value = true;
  try {
    const { data } = await api.get("/history");
    records.value = data;
  } finally {
    loading.value = false;
  }
}

onMounted(load);
</script>
```

- [ ] **Step 2: 构建验证**

运行: `cd F:\CURSOR\corn-disease-web\frontend && npm.cmd run build`

预期: 构建成功

- [ ] **Step 3: 提交**

```bash
git add frontend
git commit -m "feat(frontend): add history page"
```

---

### Task 13: README 与端到端冒烟

**Files:**
- Create: `README.md`（corn-disease-web 根）
- 手动验证（非自动化）

**Interfaces:**
- Consumes: 全部任务产物
- Produces: 可运行说明 + 验证清单

- [ ] **Step 1: 编写 README**

`README.md`:

```markdown
# 玉米叶部病害检测 Web 系统

基于改进 YOLOv8 的田间玉米主要叶部病害检测 Web 应用：图片/批量检测、摄像头/视频实时检测、参数调节、模型切换、历史记录。

## 功能

- 图片检测：上传 → 标注图 + 检测框 + 分类统计，自动保存历史
- 批量检测：多文件 → 汇总表格 + 标注图 zip 下载
- 实时检测：摄像头（浏览器抽帧）与视频文件（服务端逐帧推理），WebSocket 推流
- 历史记录：图片/批量检测自动落盘 `runs/web/`，可回看与下载

## 环境要求

- Python 3.10+（后端）
- Node 18+（前端）
- 训练权重位于 `corn_disease_detection/runs/train/*/weights/best.pt`
- 有 NVIDIA GPU 时推理自动使用 GPU（无 GPU 也可运行，速度较慢）

## 启动后端

```powershell
cd F:\CURSOR\corn-disease-web\backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> 训练仓库默认路径 `F:\CURSOR\corn_disease_detection`，可通过环境变量 `CORN_PROJECT_ROOT` 覆盖。

## 启动前端

```powershell
cd F:\CURSOR\corn-disease-web\frontend
npm.cmd install
npm.cmd run dev
```

浏览器打开 `http://localhost:5173`

## 测试

```powershell
cd F:\CURSOR\corn-disease-web\backend
python -m pytest tests/ -v
```

## 目录结构

见 `docs/superpowers/specs/2026-08-14-corn-disease-web-design.md`。
```

- [ ] **Step 2: 端到端冒烟（手动）**

按 README 启动后端与前端，浏览器逐项验证：

1. `GET http://localhost:8000/api/health` 返回 `status=ok`；`/api/models` 列出权重
2. 图片检测：上传一张玉米叶图片，出现标注图、检测框表格与分类统计
3. 批量检测：选择 2 张以上图片，汇总表格正确，zip 可下载
4. 实时检测-摄像头：授权摄像头后画面持续刷新，FPS > 0
5. 实时检测-视频：上传 mp4 后开始推流，画面持续刷新
6. 历史记录：出现刚才的图片/批量记录，缩略图与 zip 可打开
7. 错误路径：上传 txt 文件提示"不支持的图片格式"

- [ ] **Step 3: 提交**

```bash
git add README.md
git commit -m "docs: add corn-disease-web run guide"
```
