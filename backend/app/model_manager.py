from . import config

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
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass  # 测试环境可能未安装 torch，跳过显存清理

    loader = loader or _default_loader
    _active[name] = loader(entry["path"])
    return _active[name]


def active_model_name() -> str | None:
    return next(iter(_active), None)
