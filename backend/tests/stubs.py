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
