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
