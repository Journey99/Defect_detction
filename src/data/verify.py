import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify YOLO dataset integrity.")
    parser.add_argument("--dataset-yaml", required=True, help="Path to data.yaml")
    return parser.parse_args()

# 이미지 파일 수집
def collect_images(image_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

# 라벨 파일 검증
def validate_label_file(label_file: Path, num_classes: int, split: str) -> list[str]:
    errors: list[str] = []
    raw_text = label_file.read_text(encoding="utf-8").strip()
    if not raw_text:
        errors.append(f"[{split}] Empty label file: {label_file}")
        return errors

    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            errors.append(
                f"[{split}] Invalid column count ({len(parts)}) in {label_file} line {line_no}: {line}"
            )
            continue

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(
                f"[{split}] Non-numeric label values in {label_file} line {line_no}: {line}"
            )
            continue

        if class_id < 0 or class_id >= num_classes:
            errors.append(
                f"[{split}] Class id out of range in {label_file} line {line_no}: {class_id}"
            )

        coords = [x_center, y_center, width, height]
        if any(v < 0.0 or v > 1.0 for v in coords):
            errors.append(
                f"[{split}] Normalized coords out of [0,1] in {label_file} line {line_no}: {coords}"
            )
        if width <= 0.0 or height <= 0.0:
            errors.append(
                f"[{split}] Non-positive box size in {label_file} line {line_no}: w={width}, h={height}"
            )

    return errors


def validate_split(
    split: str,
    images_dir: Path,
    labels_dir: Path,
    num_classes: int,
) -> tuple[int, int, list[str]]:
    errors: list[str] = []
    images = collect_images(images_dir)
    labels = sorted([p for p in labels_dir.iterdir() if p.is_file() and p.suffix == ".txt"])

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)

    if missing_labels:
        preview = ", ".join(missing_labels[:10])
        errors.append(f"[{split}] Missing labels for images: {preview}")
    if orphan_labels:
        preview = ", ".join(orphan_labels[:10])
        errors.append(f"[{split}] Labels without images: {preview}")

    for label_file in labels:
        errors.extend(validate_label_file(label_file, num_classes, split))

    return len(images), len(labels), errors



def main() -> None:
    args = parse_args()
    data_yaml = Path(args.dataset_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    names = cfg.get("names")
    if not isinstance(names, dict) or not names:
        raise ValueError("`names` must be a non-empty dictionary in dataset yaml.")
    num_classes = len(names)

    dataset_root = Path(cfg["path"])
    required_dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    ]

    missing = [str(p) for p in required_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset directories:\n" + "\n".join(missing))

    print("[verify] Dataset directory check passed.")
    print(f"[verify] Number of classes: {num_classes}")

    train_images, train_labels, train_errors = validate_split(
        split="train",
        images_dir=dataset_root / "images" / "train",
        labels_dir=dataset_root / "labels" / "train",
        num_classes=num_classes,
    )
    val_images, val_labels, val_errors = validate_split(
        split="val",
        images_dir=dataset_root / "images" / "val",
        labels_dir=dataset_root / "labels" / "val",
        num_classes=num_classes,
    )

    all_errors = train_errors + val_errors
    print(f"[verify] train: {train_images} images, {train_labels} labels")
    print(f"[verify] val: {val_images} images, {val_labels} labels")

    if all_errors:
        preview = "\n".join(all_errors[:30])
        extra = len(all_errors) - 30
        if extra > 0:
            preview += f"\n... and {extra} more errors"
        raise ValueError("[verify] Dataset integrity check failed:\n" + preview)

    print("[verify] Dataset integrity check passed.")


if __name__ == "__main__":
    main()
