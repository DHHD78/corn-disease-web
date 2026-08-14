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
