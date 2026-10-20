"""Обучение детектора ям на дорогах (YOLO).

Датасет: 1 243 фото ям с разметкой в формате YOLO (MIT),
https://github.com/jaygala24/pothole-detection

Запуск:
    python train.py                 # скачать датасет (если нет) и обучить
    python train.py --epochs 100    # дольше = точнее
"""

import argparse
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET_URL = "https://github.com/jaygala24/pothole-detection/releases/download/v1.0.0/Pothole.Dataset.IVCNZ.zip"
RAW_DIR = ROOT / "datasets" / "raw"
DATA_DIR = ROOT / "datasets" / "potholes"
WEIGHTS_OUT = ROOT / "models" / "pothole.pt"


def download_dataset() -> Path:
    src = RAW_DIR / "extracted" / "Pothole Dataset"
    if src.exists():
        return src
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "pothole.zip"
    if not zip_path.exists():
        print(f"Скачиваю датасет: {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR / "extracted")
    return src


def split_dataset(src: Path, val_ratio: float = 0.15, seed: int = 42) -> Path:
    """Раскладывает плоскую папку img-N.jpg/img-N.txt в структуру train/val для YOLO."""
    yaml_path = DATA_DIR / "data.yaml"
    if yaml_path.exists():
        return yaml_path

    images = sorted(p for p in src.glob("*.jpg") if p.with_suffix(".txt").exists())
    random.Random(seed).shuffle(images)
    n_val = int(len(images) * val_ratio)
    splits = {"val": images[:n_val], "train": images[n_val:]}

    for split, files in splits.items():
        (DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy2(img, DATA_DIR / "images" / split / img.name)
            shutil.copy2(img.with_suffix(".txt"), DATA_DIR / "labels" / split / img.with_suffix(".txt").name)
        print(f"{split}: {len(files)} изображений")

    yaml_path.write_text(
        f"path: {DATA_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: pothole\n"
    )
    return yaml_path


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26n.pt", help="базовая модель (yolo26n/s/m.pt)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    data_yaml = split_dataset(download_dataset())
    device = pick_device()
    print(f"Устройство: {device}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(ROOT / "runs"),
        name="pothole",
        exist_ok=True,
        patience=15,
        plots=True,
    )

    best = ROOT / "runs" / "pothole" / "weights" / "best.pt"
    WEIGHTS_OUT.parent.mkdir(exist_ok=True)
    shutil.copy2(best, WEIGHTS_OUT)
    print(f"\nГотово. Веса модели: {WEIGHTS_OUT}")

    metrics = YOLO(WEIGHTS_OUT).val(data=str(data_yaml), device=device, plots=False)
    print(f"mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}  "
          f"precision: {metrics.box.mp:.3f}  recall: {metrics.box.mr:.3f}")
    print(f"ONNX для приложения: {export_onnx(WEIGHTS_OUT, args.imgsz)}")


def export_onnx(weights: Path, imgsz: int = 640) -> Path:
    """Экспорт в ONNX: приложение (pothole_ai.py) работает через ONNX Runtime без PyTorch."""
    import onnx

    path = Path(YOLO(str(weights)).export(format="onnx", imgsz=imgsz, simplify=True))
    # Ultralytics пишет в метаданные абсолютный путь к датасету — убираем перед публикацией.
    model = onnx.load(path)
    for prop in model.metadata_props:
        if prop.key == "description":
            prop.value = "Pothole detector (YOLO26n), github.com/jaygala24/pothole-detection dataset"
    onnx.save(model, path)
    return path


if __name__ == "__main__":
    main()
