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


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)
