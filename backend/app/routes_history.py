import io
import json
import re
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import config

router = APIRouter(prefix="/api/history", tags=["history"])

_ID_PATTERN = re.compile(r"^\d{8}_\d{6}_\d{3}_\w+$")


def _record_dir(hist_id: str):
    if not _ID_PATTERN.match(hist_id):
        raise HTTPException(400, "非法记录 id")
    d = config.HISTORY_DIR / hist_id
    if not d.is_dir():
        raise HTTPException(404, "记录不存在")
    return d


@router.get("")
def list_history():
    records = []
    if config.HISTORY_DIR.exists():
        for d in config.HISTORY_DIR.iterdir():
            if not (d / "result.json").exists():
                continue
            try:
                payload = json.loads((d / "result.json").read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            records.append(
                {
                    "id": d.name,
                    "source": payload.get("source", ""),
                    "created_at": payload.get("created_at", ""),
                    "params": payload.get("params", {}),
                    "stats": payload.get("stats", {}),
                    "annotated_url": f"/api/history/{d.name}/annotated",
                }
            )
    records.sort(key=lambda r: r["created_at"], reverse=True)
    return records


@router.get("/{hist_id}/annotated")
def annotated(hist_id: str):
    d = _record_dir(hist_id)
    return FileResponse(d / "annotated.jpg", media_type="image/jpeg")


@router.get("/{hist_id}/original")
def original(hist_id: str):
    d = _record_dir(hist_id)
    return FileResponse(d / "original.jpg", media_type="image/jpeg")


@router.get("/{hist_id}/json")
def record_json(hist_id: str):
    d = _record_dir(hist_id)
    payload = json.loads((d / "result.json").read_text(encoding="utf-8"))
    return JSONResponse(payload)


@router.get("/{hist_id}/zip")
def record_zip(hist_id: str):
    d = _record_dir(hist_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ("original.jpg", "annotated.jpg", "result.json", "labels.txt"):
            f = d / name
            if f.exists():
                zf.write(f, arcname=f"{hist_id}_{name}")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{hist_id}.zip"'},
    )
