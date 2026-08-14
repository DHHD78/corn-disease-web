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
