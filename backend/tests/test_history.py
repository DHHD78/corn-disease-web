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
    assert client.get("/api/history/no_such_id/annotated").status_code == 400
    assert client.get("/api/history/20260814_120000_000_image/annotated").status_code == 404
