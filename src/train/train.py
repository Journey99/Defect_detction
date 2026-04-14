import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train defect detection model.")
    parser.add_argument("--config", required=True, help="Path to training config yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = YOLO(cfg["model"])
    model.train(
        data=cfg["data"],
        imgsz=cfg.get("imgsz", 640),
        epochs=cfg.get("epochs", 100),
        batch=cfg.get("batch", 16),
        device=cfg.get("device", 0),
        workers=cfg.get("workers", 8),
        project=cfg.get("project", "runs/train"),
        name=cfg.get("name", "exp"),
    )


if __name__ == "__main__":
    main()
