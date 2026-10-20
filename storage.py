"""Где хранятся снимки и точки карты.

Локально — папка data/. На Vercel диск временный и у каждого экземпляра функции свой,
поэтому там используется Vercel Blob: снимок — images/<id>.jpg, точка — reports/<id>.json.
Blob включается, если в окружении есть BLOB_STORE_ID (OIDC на Vercel) или BLOB_READ_WRITE_TOKEN.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAX_REPORTS = 300


def use_blob() -> bool:
    return bool(os.environ.get("BLOB_STORE_ID") or os.environ.get("BLOB_READ_WRITE_TOKEN"))


class LocalStorage:
    url_prefix = "/files"

    def __init__(self):
        # На Vercel без Blob писать можно только в /tmp (и данные там не живут долго).
        base = Path("/tmp/potholes") if os.environ.get("VERCEL") else ROOT / "data"
        self.dir = base / "uploads"
        self.reports_file = base / "reports.json"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def save_image(self, name: str, data: bytes) -> str:
        (self.dir / name).write_bytes(data)
        return f"{self.url_prefix}/{name}"

    def add_report(self, report: dict) -> None:
        with self._lock:
            reports = self._read()
            reports.append(report)
            self.reports_file.write_text(json.dumps(reports, ensure_ascii=False, indent=2))

    async def list_reports(self) -> list[dict]:
        return self._read()[-MAX_REPORTS:]

    def _read(self) -> list[dict]:
        return json.loads(self.reports_file.read_text()) if self.reports_file.exists() else []


class BlobStorage:
    def save_image(self, name: str, data: bytes) -> str:
        from vercel.blob import sync as blob
        return blob.put_blob(f"images/{name}", data, access="public", content_type="image/jpeg").url

    def add_report(self, report: dict) -> None:
        from vercel.blob import sync as blob
        # created_at в начале имени: список отсортирован по времени без чтения файлов
        name = f"reports/{report['created_at'].replace(':', '')}_{report['id']}.json"
        blob.put_blob(name, json.dumps(report, ensure_ascii=False).encode(), access="public",
                      content_type="application/json")

    async def list_reports(self) -> list[dict]:
        from vercel import blob
        entries = [b async for b in blob.query_blobs(prefix="reports/")]
        entries = sorted(entries, key=lambda b: b.pathname)[-MAX_REPORTS:]
        sem = asyncio.Semaphore(16)

        async def fetch(entry):
            async with sem:
                try:
                    return json.loads((await blob.get_blob(entry.url, access="public")).body)
                except Exception:
                    return None

        return [r for r in await asyncio.gather(*map(fetch, entries)) if r]


storage: LocalStorage | BlobStorage = BlobStorage() if use_blob() else LocalStorage()
