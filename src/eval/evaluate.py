import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model.")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pt)")
    parser.add_argument("--data", default="data/data.yaml", help="Path to dataset yaml")
    parser.add_argument("--tta", action="store_true", help="Enable TTA for validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, augment=args.tta)
    print("[eval] mAP50:", metrics.box.map50)
    print("[eval] mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()
