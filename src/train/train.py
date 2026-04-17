import argparse
import sys
from pathlib import Path

import yaml
from ultralytics import RTDETR, YOLO

# 프로젝트 루트를 path에 넣어 `python src/train/train.py`·Colab 등에서도 `src` import 가능
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train.losses import apply_wiou_patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train defect detection model.")
    parser.add_argument("--config", required=True, help="Path to training config yaml")
    return parser.parse_args()


def build_model(cfg: dict) -> YOLO | RTDETR:
    model_path = str(cfg["model"])
    model_type = str(cfg.get("model_type", "auto")).lower()

    if model_type == "auto":
        if "rtdetr" in model_path.lower() or "rt-detr" in model_path.lower():
            model_type = "rtdetr"
        else:
            model_type = "yolo"

    if model_type == "rtdetr":
        # Ultralytics에서는 rtdetr-r50.pt 대신 rtdetr-l.pt/x.pt 프리트레인을 사용한다.
        if model_path.lower() == "rtdetr-r50.pt":
            print("[train] model alias: rtdetr-r50.pt -> rtdetr-l.pt")
            model_path = "rtdetr-l.pt"
        return RTDETR(model_path)
    if model_type == "yolo":
        return YOLO(model_path)

    raise ValueError(f"Unsupported model_type: {model_type}. Use one of [auto, yolo, rtdetr].")


def build_train_kwargs(cfg: dict) -> dict:
    kwargs = {
        "data": cfg["data"],
        "imgsz": cfg.get("imgsz", 640),
        "epochs": cfg.get("epochs", 100),
        "batch": cfg.get("batch", 16),
        "device": cfg.get("device", 0),
        "workers": cfg.get("workers", 8),
        "project": cfg.get("project", "runs/train"),
        "name": cfg.get("name", "exp"),
    }

    if "mosaic" in cfg:
        kwargs["mosaic"] = cfg["mosaic"]
    if "mixup" in cfg:
        kwargs["mixup"] = cfg["mixup"]

    return kwargs


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    loss_name = apply_wiou_patch(bool(cfg.get("use_wiou", False)))
    print(f"[train] configured loss mode: {loss_name}")

    model = build_model(cfg)
    train_kwargs = build_train_kwargs(cfg)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
