"""Поиск ям из командной строки.

    python detect.py road.jpg              # одно фото
    python detect.py photos/               # папка с фото и видео
    python detect.py dashcam.mp4           # видео с регистратора / телефона

Результат в папке output/: размеченные фото/видео, report.json и potholes.geojson
(точки на карте для фото, где есть GPS в EXIF; открывается в geojson.io, QGIS, Google My Maps).
"""

import argparse
import json
from pathlib import Path

import cv2

from pothole_ai import (IMAGE_EXT, SEVERITY_RU, VIDEO_EXT, detect_image, detect_video, draw,
                        gps_from_exif, read_image, summarize)


def collect(paths: list[str]) -> list[Path]:
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXT | VIDEO_EXT)
        else:
            files.append(p)
    return files


def main():
    parser = argparse.ArgumentParser(description="Поиск ям на дорогах")
    parser.add_argument("inputs", nargs="+", help="фото, видео или папки")
    parser.add_argument("-o", "--out", default="output", help="папка для результатов")
    parser.add_argument("--conf", type=float, default=0.35, help="порог уверенности модели")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report, features = [], []

    for f in collect(args.inputs):
        ext = f.suffix.lower()
        if ext in IMAGE_EXT:
            frame, pil = read_image(f)
            potholes = detect_image(frame, conf=args.conf)
            cv2.imwrite(str(out / f"{f.stem}_detected.jpg"), draw(frame, potholes))
            gps = gps_from_exif(pil)
            summary = summarize(potholes)
            report.append({"file": str(f), "type": "image", "gps": gps, "summary": summary,
                           "potholes": [p.to_dict() for p in potholes]})
            if gps and potholes:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [gps[1], gps[0]]},
                    "properties": {"file": f.name, **summary},
                })
            print(f"{f.name}: ям {summary['count']}, опасность {summary['worst_ru']}"
                  f"{f', GPS {gps[0]}, {gps[1]}' if gps else ''}")

        elif ext in VIDEO_EXT:
            res = detect_video(f, out / f"{f.stem}_detected.mp4", conf=args.conf)
            snap_dir = out / f"{f.stem}_potholes"
            snap_dir.mkdir(exist_ok=True)
            items = []
            for i, c in enumerate(res["potholes"], 1):
                snap = snap_dir / f"{i:03d}_{c['time_sec']:.1f}s.jpg"
                cv2.imwrite(str(snap), c["snapshot"])
                items.append({**c["pothole"].to_dict(), "time_sec": c["time_sec"],
                              "frames_seen": c["frames_seen"], "snapshot": str(snap)})
            report.append({"file": str(f), "type": "video", "duration_sec": res["duration_sec"],
                           "summary": res["summary"], "potholes": items})
            print(f"{f.name}: уникальных ям {res['summary']['count']} за {res['duration_sec']} c")
            for it in items:
                print(f"   {it['time_sec']:>7.1f} c  опасность {SEVERITY_RU[it['severity']]:<8} "
                      f"уверенность {it['confidence']:.2f}")
        else:
            print(f"Пропускаю {f}: неизвестный формат")

    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out / "potholes.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2))
    print(f"\nОтчёт: {out / 'report.json'}")


if __name__ == "__main__":
    main()
