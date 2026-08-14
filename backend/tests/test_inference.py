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
