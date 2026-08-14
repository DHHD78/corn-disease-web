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
