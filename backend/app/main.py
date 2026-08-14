from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .model_manager import active_model_name, scan_models
from .routes_detect import downloads_router, router as detect_router
from .routes_history import router as history_router
from .routes_realtime import router as realtime_router

app = FastAPI(title="玉米叶部病害检测 Web 系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect_router)
app.include_router(downloads_router)
app.include_router(history_router)
app.include_router(realtime_router)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "active_model": active_model_name(),
        "models_count": len(scan_models()),
    }


@app.get("/api/models")
def list_models():
    return scan_models()
