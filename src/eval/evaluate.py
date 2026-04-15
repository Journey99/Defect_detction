import argparse

from ultralytics import RTDETR, YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model.")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pt)")
    parser.add_argument("--data", default="data/data.yaml", help="Path to dataset yaml")
    parser.add_argument("--tta", action="store_true", help="Enable TTA for validation")
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "yolo", "rtdetr"],
        help="Model family for loading weights",
    )
    return parser.parse_args()


def build_model(weights: str, model_type: str) -> YOLO | RTDETR:
    selected = model_type.lower()
    if selected == "auto":
        selected = "rtdetr" if "rtdetr" in weights.lower() or "rt-detr" in weights.lower() else "yolo"

    if selected == "rtdetr":
        return RTDETR(weights)
    return YOLO(weights)


def main() -> None:
    args = parse_args()
    model = build_model(args.weights, args.model_type)
    metrics = model.val(data=args.data, augment=args.tta)
    print("[eval] mAP50:", metrics.box.map50)
    print("[eval] mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()
