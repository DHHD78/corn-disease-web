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
