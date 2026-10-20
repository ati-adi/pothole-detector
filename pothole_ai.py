"""Ядро: поиск ям на фото и видео, оценка опасности, GPS из EXIF.

Инференс идёт через ONNX Runtime: без PyTorch зависимости весят ~100 МБ вместо ~1 ГБ,
поэтому то же ядро работает и локально, и в serverless-функции на Vercel.
PyTorch/Ultralytics нужны только для обучения (train.py).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

try:  # фото с iPhone (HEIC)
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "pothole.onnx"
IMGSZ = 640
NMS_IOU = 0.7  # как в Ultralytics по умолчанию

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

# Доля кадра, которую занимает яма. Грубая эвристика без калибровки камеры:
# для реальных размеров в сантиметрах нужна известная высота и угол камеры.
SEVERITY_THRESHOLDS = [(0.015, "low"), (0.06, "medium")]
SEVERITY_RU = {"low": "низкая", "medium": "средняя", "high": "высокая"}
SEVERITY_COLOR = {"low": (0, 200, 255), "medium": (0, 128, 255), "high": (0, 0, 255)}  # BGR


@dataclass
class Pothole:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    area_ratio: float
    severity: str
    track_id: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity_ru"] = SEVERITY_RU[self.severity]
        return d

    @property
    def box(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


def severity_for(area_ratio: float) -> str:
    for limit, label in SEVERITY_THRESHOLDS:
        if area_ratio < limit:
            return label
    return "high"


@lru_cache(maxsize=1)
def load_model() -> ort.InferenceSession:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Нет модели {MODEL_PATH}. Сначала запустите: python train.py")
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def read_image(path_or_file) -> tuple[np.ndarray, Image.Image]:
    """Читает фото с учётом поворота из EXIF (снимки с телефона). Возвращает (BGR, PIL)."""
    pil = Image.open(path_or_file)
    rgb = ImageOps.exif_transpose(pil).convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR), pil


def gps_from_exif(pil: Image.Image) -> tuple[float, float] | None:
    """Координаты съёмки из EXIF, если телефон их записал."""
    try:
        gps = pil.getexif().get_ifd(0x8825)
    except Exception:
        return None
    if not gps or 2 not in gps or 4 not in gps:
        return None

    def to_deg(dms) -> float:
        d, m, s = (float(v) for v in dms)
        return d + m / 60 + s / 3600

    lat, lon = to_deg(gps[2]), to_deg(gps[4])
    if gps.get(1) == "S":
        lat = -lat
    if gps.get(3) == "W":
        lon = -lon
    return round(lat, 6), round(lon, 6)


# ---------- инференс ----------

def _letterbox(frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Вписывает кадр в 640×640 с серыми полями, как при обучении Ultralytics."""
    h, w = frame.shape[:2]
    r = min(IMGSZ / h, IMGSZ / w)
    nw, nh = round(w * r), round(h * r)
    px, py = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR→RGB, NCHW
    return np.ascontiguousarray(blob), r, (px, py)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU всех пар рамок a[N,4] × b[M,4] в формате xyxy."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    order, keep = np.argsort(-scores), []
    while order.size:
        i = order[0]
        keep.append(int(i))
        order = order[1:][_iou(boxes[i:i + 1], boxes[order[1:]])[0] < iou_thr]
    return keep


def predict(frame: np.ndarray, conf: float = 0.35) -> list[Pothole]:
    session = load_model()
    blob, r, (px, py) = _letterbox(frame)
    out = session.run(None, {session.get_inputs()[0].name: blob})[0][0].T  # [8400, 5]: cx, cy, w, h, score
    out = out[out[:, 4] >= conf]
    if not len(out):
        return []
    cx, cy, bw, bh, scores = out.T
    boxes = np.stack([cx - bw / 2 - px, cy - bh / 2 - py, cx + bw / 2 - px, cy + bh / 2 - py], 1) / r
    h, w = frame.shape[:2]
    boxes = np.clip(boxes, 0, [w, h, w, h])

    result = []
    for i in _nms(boxes, scores, NMS_IOU):
        x1, y1, x2, y2 = boxes[i]
        area = float((x2 - x1) * (y2 - y1) / (w * h))
        result.append(Pothole(int(x1), int(y1), int(x2), int(y2), round(float(scores[i]), 3),
                              round(area, 4), severity_for(area)))
    return _dedupe(result)


def _dedupe(potholes: list[Pothole], overlap: float = 0.6) -> list[Pothole]:
    """NMS пропускает вложенные рамки (у них низкий IoU), и на одну яму бывает две рамки.
    Убираем рамку, если она больше чем на `overlap` лежит внутри более уверенной."""
    kept: list[Pothole] = []
    for p in sorted(potholes, key=lambda p: -p.confidence):
        pa = (p.x2 - p.x1) * (p.y2 - p.y1)
        dup = False
        for k in kept:
            iw = min(p.x2, k.x2) - max(p.x1, k.x1)
            ih = min(p.y2, k.y2) - max(p.y1, k.y1)
            ka = (k.x2 - k.x1) * (k.y2 - k.y1)
            if iw > 0 and ih > 0 and iw * ih / max(min(pa, ka), 1) > overlap:
                dup = True
                break
        if not dup:
            kept.append(p)
    return kept


def detect_image(frame: np.ndarray, conf: float = 0.35) -> list[Pothole]:
    return predict(frame, conf)


def draw(frame: np.ndarray, potholes: list[Pothole]) -> np.ndarray:
    img = frame.copy()
    thick = max(2, round(min(img.shape[:2]) / 300))
    for p in potholes:
        color = SEVERITY_COLOR[p.severity]
        cv2.rectangle(img, (p.x1, p.y1), (p.x2, p.y2), color, thick)
        label = f"{'#' + str(p.track_id) + ' ' if p.track_id else ''}{p.severity} {p.confidence:.2f}"
        scale = thick / 3
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, thick // 2))
        y = max(p.y1, th + 6)
        cv2.rectangle(img, (p.x1, y - th - 6), (p.x1 + tw + 6, y), color, -1)
        cv2.putText(img, label, (p.x1 + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), max(1, thick // 2), cv2.LINE_AA)
    return img


def summarize(potholes: list[Pothole]) -> dict:
    counts = {k: sum(p.severity == k for p in potholes) for k in SEVERITY_RU}
    worst = next((k for k in ("high", "medium", "low") if counts[k]), None)
    return {"count": len(potholes), "by_severity": counts, "worst": worst,
            "worst_ru": SEVERITY_RU.get(worst, "нет ям")}


# ---------- видео ----------

class Tracker:
    """Упрощённый ByteTrack: сопоставление рамок по IoU между кадрами, без фильтра Калмана.

    Модель часто видит яму с уверенностью 0.3–0.6, поэтому слабые детекции (от `low`)
    не создают треков, но продлевают существующие — так яма не теряется на кадр-другой.
    Положение трека экстраполируется по скорости: камера едет вперёд, яма сдвигается вниз.
    """

    def __init__(self, high: float = 0.3, low: float = 0.1, match_iou: float = 0.2, max_lost: int = 30):
        self.high, self.low, self.match_iou, self.max_lost = high, low, match_iou, max_lost
        self.tracks: list[dict] = []
        self._next_id = 1

    def update(self, dets: list[Pothole]) -> list[Pothole]:
        for t in self.tracks:
            t["pred"] = t["box"] + t["vel"]
        unmatched = list(range(len(self.tracks)))
        out = []
        for stage in (self.high, self.low):
            pool = [d for d in dets if d.track_id is None and d.confidence >= stage]
            if not pool or not unmatched:
                continue
            iou = _iou(np.stack([self.tracks[i]["pred"] for i in unmatched]), np.stack([d.box for d in pool]))
            for ti, di in sorted(np.argwhere(iou >= self.match_iou), key=lambda x: -iou[x[0], x[1]]):
                track, det = self.tracks[unmatched[ti]], pool[di]
                if track.get("hit") or det.track_id is not None:
                    continue
                track["hit"] = True
                track["vel"] = 0.5 * track["vel"] + 0.5 * (det.box - track["box"])
                track["box"], track["lost"] = det.box, 0
                det.track_id = track["id"]
                out.append(det)
            unmatched = [i for i in unmatched if not self.tracks[i].pop("hit", False)]
            for t in self.tracks:
                t.pop("hit", None)

        for d in dets:  # новые ямы — только из уверенных детекций
            if d.track_id is None and d.confidence >= self.high:
                d.track_id = self._next_id
                self._next_id += 1
                self.tracks.append({"id": d.track_id, "box": d.box, "vel": np.zeros(4, np.float32), "lost": 0})
                out.append(d)
        for i in unmatched:
            self.tracks[i]["lost"] += 1
        self.tracks = [t for t in self.tracks if t["lost"] <= self.max_lost]
        return out


def detect_video(path: str | Path, out_path: str | Path | None = None, conf: float = 0.35,
                 min_frames: int = 3, frame_stride: int = 1) -> dict:
    """Ищет ямы на видео с трекингом, чтобы одна яма считалась один раз, а не в каждом кадре.

    Яма засчитывается, если её трек виден хотя бы в `min_frames` кадрах и хотя бы раз
    уверенность была >= `conf`. Для каждой ямы сохраняется кадр с максимальной уверенностью.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker, tracks, writer, idx = Tracker(), {}, None, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % frame_stride:
            continue
        potholes = tracker.update(predict(frame, conf=min(0.1, conf)))
        for p in potholes:
            t = tracks.setdefault(p.track_id, {"frames": 0, "best": None})
            t["frames"] += 1
            if t["best"] is None or p.confidence > t["best"].confidence:
                t["best"], t["best_frame"], t["best_idx"] = p, frame, idx

        if out_path is not None:
            if writer is None:
                h, w = frame.shape[:2]
                writer = _open_writer(out_path, fps / frame_stride, (w, h))
            writer.write(draw(frame, potholes))
    cap.release()
    if writer is not None:
        writer.release()

    confirmed = []
    for t in sorted(tracks.values(), key=lambda t: t["best_idx"]):
        if t["frames"] < min_frames or t["best"].confidence < conf:
            continue
        confirmed.append({
            "pothole": t["best"],
            "frames_seen": t["frames"],
            "time_sec": round(t["best_idx"] / fps, 2),
            "snapshot": draw(t["best_frame"], [t["best"]]),
        })

    return {
        "frames": total,
        "fps": fps,
        "duration_sec": round(total / fps, 1) if fps else None,
        "potholes": confirmed,
        "summary": summarize([c["pothole"] for c in confirmed]),
    }


def _open_writer(out_path, fps, size) -> cv2.VideoWriter:
    # avc1 (H.264) открывается в браузерах; если кодека нет — mp4v для обычных плееров.
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
    raise RuntimeError("Не удалось открыть VideoWriter")
