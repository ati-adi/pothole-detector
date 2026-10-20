"""Веб-демо: загрузите фото или видео дороги — получите найденные ямы и точку на карте.

    uvicorn app:app --reload        # http://127.0.0.1:8000

На Vercel этот же файл запускается как Python-функция (FastAPI определяется автоматически).
"""

import base64
import os
import shutil
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pothole_ai import (IMAGE_EXT, VIDEO_EXT, detect_image, detect_video, draw, gps_from_exif,
                        load_model, read_image, summarize)
from storage import LocalStorage, storage

ROOT = Path(__file__).resolve().parent
ON_VERCEL = bool(os.environ.get("VERCEL"))
MAX_UPLOAD_MB = 4.5 if ON_VERCEL else None  # лимит тела запроса Vercel Functions
MAX_GALLERY = 30


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()  # прогрев, чтобы первый запрос не ждал загрузки модели
    yield


app = FastAPI(title="Детектор ям", lifespan=lifespan)
if isinstance(storage, LocalStorage):
    app.mount(storage.url_prefix, StaticFiles(directory=storage.dir), name="files")
_model_lock = threading.Lock()


@app.middleware("http")
async def vercel_request_context(request: Request, call_next):
    # Blob SDK берёт OIDC-токен Vercel из заголовков текущего запроса.
    try:
        from vercel.headers import set_headers
        set_headers(dict(request.headers))
    except ImportError:
        pass
    return await call_next(request)


def _jpeg(img: np.ndarray, max_side: int, quality: int = 85) -> bytes:
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        r = max_side / max(h, w)
        img = cv2.resize(img, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def _parse_coord(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


@app.get("/")
def index():
    # no-cache: после деплоя браузер сразу берёт новую версию страницы
    return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/config")
def config():
    return {"max_upload_mb": MAX_UPLOAD_MB, "storage": type(storage).__name__}


@app.get("/api/reports")
async def reports():
    return await storage.list_reports()


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...), lat: str | None = Form(None), lon: str | None = Form(None),
            conf: float = Form(0.35)):
    ext = Path(file.filename or "").suffix.lower()
    rid = uuid.uuid4().hex[:10]
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    browser_gps = (_parse_coord(lat), _parse_coord(lon))
    browser_gps = browser_gps if None not in browser_gps else None
    conf = min(max(conf, 0.05), 0.95)

    if ext in IMAGE_EXT or (file.content_type or "").startswith("image/"):
        try:
            frame, pil = read_image(file.file)
        except Exception:
            raise HTTPException(400, "Не удалось прочитать изображение")
        with _model_lock:
            potholes = detect_image(frame, conf=conf)
        gps = gps_from_exif(pil)
        thumb = draw(frame, potholes)
        report = {
            "id": rid, "type": "image", "filename": file.filename, "created_at": created,
            "gps": gps or browser_gps, "gps_source": "exif" if gps else ("browser" if browser_gps else None),
            "summary": summarize(potholes), "potholes": [p.to_dict() for p in potholes],
            "preview_url": _data_url(_jpeg(thumb, 1280)),
        }

    elif ext in VIDEO_EXT or (file.content_type or "").startswith("video/"):
        with tempfile.NamedTemporaryFile(suffix=ext or ".mp4", delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
        try:
            with _model_lock:
                res = detect_video(tmp.name, conf=conf)
        except ValueError:
            raise HTTPException(400, "Не удалось прочитать видео")
        finally:
            Path(tmp.name).unlink(missing_ok=True)
        items = [{**c["pothole"].to_dict(), "time_sec": c["time_sec"], "frames_seen": c["frames_seen"],
                  "image_url": _data_url(_jpeg(c["snapshot"], 640, quality=80))}
                 for c in res["potholes"][:MAX_GALLERY]]
        thumb = res["potholes"][0]["snapshot"] if res["potholes"] else None
        report = {
            "id": rid, "type": "video", "filename": file.filename, "created_at": created,
            "gps": browser_gps, "gps_source": "browser" if browser_gps else None,
            "duration_sec": res["duration_sec"], "summary": res["summary"], "potholes": items,
        }
    else:
        raise HTTPException(400, "Поддерживаются фото (jpg, png, webp, heic) и видео (mp4, mov, avi)")

    report["saved"] = bool(report["summary"]["count"] and report["gps"])
    if report["saved"]:
        # на карту — сводка и один снимок; галерея кадров и превью остаются только в ответе
        report["image_url"] = storage.save_image(f"{rid}.jpg", _jpeg(thumb, 800))
        storage.add_report({k: v for k, v in report.items() if k not in ("potholes", "preview_url")})
    return report
