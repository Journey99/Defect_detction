import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyTorch weights to ONNX.")
    parser.add_argument("--weights", required=True, help="Path to .pt weights")
    parser.add_argument("--output", default="runs/export/model.onnx", help="Output ONNX path")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    model.export(format="onnx", imgsz=args.imgsz)
    print(f"[export] ONNX export finished. Check artifacts near: {out.parent}")


if __name__ == "__main__":
    main()
