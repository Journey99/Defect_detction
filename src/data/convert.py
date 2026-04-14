import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw NEU-DET data to YOLO format.")
    parser.add_argument("--input", required=True, help="Path to raw dataset directory")
    parser.add_argument("--output", required=True, help="Path to output YOLO dataset directory")
    return parser.parse_args()


def ensure_output_dirs(out_dir: Path) -> None:
    for p in [
        out_dir / "images" / "train",
        out_dir / "images" / "val",
        out_dir / "labels" / "train",
        out_dir / "labels" / "val",
    ]:
        p.mkdir(parents=True, exist_ok=True)

# 이미지 인덱스 생성
def build_image_index(images_root: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    for image_path in images_root.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            image_index[image_path.name] = image_path
    return image_index

# VOC 바운딩 박스를 YOLO 바운딩 박스로 변환
def voc_to_yolo_bbox(xmin: float, ymin: float, xmax: float, ymax: float, width: float, height: float) -> tuple[float, float, float, float]:
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_w = (xmax - xmin) / width
    box_h = (ymax - ymin) / height
    return x_center, y_center, box_w, box_h

# 데이터 셋 변환
def convert_split(in_dir: Path, out_dir: Path, split_name: str, out_split: str) -> tuple[int, int]:
    annotations_dir = in_dir / split_name / "annotations"
    images_dir = in_dir / split_name / "images"
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_index = build_image_index(images_dir)
    converted = 0
    skipped = 0

    for xml_file in sorted(annotations_dir.glob("*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        file_name = root.findtext("filename")
        if not file_name:
            skipped += 1
            continue
        image_path = image_index.get(file_name)
        if image_path is None:
            skipped += 1
            continue

        size = root.find("size")
        if size is None:
            skipped += 1
            continue
        img_w = float(size.findtext("width", default="0"))
        img_h = float(size.findtext("height", default="0"))
        if img_w <= 0 or img_h <= 0:
            skipped += 1
            continue

        yolo_lines: list[str] = []
        for obj in root.findall("object"):
            class_name = obj.findtext("name", default="")
            if class_name not in CLASS_TO_ID:
                continue
            bbox = obj.find("bndbox")
            if bbox is None:
                continue

            xmin = float(bbox.findtext("xmin", default="0"))
            ymin = float(bbox.findtext("ymin", default="0"))
            xmax = float(bbox.findtext("xmax", default="0"))
            ymax = float(bbox.findtext("ymax", default="0"))
            if xmax <= xmin or ymax <= ymin:
                continue

            x_center, y_center, box_w, box_h = voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
            yolo_lines.append(
                f"{CLASS_TO_ID[class_name]} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
            )

        if not yolo_lines:
            skipped += 1
            continue

        dst_image = out_dir / "images" / out_split / image_path.name
        dst_label = out_dir / "labels" / out_split / f"{image_path.stem}.txt"
        shutil.copy2(image_path, dst_image)
        dst_label.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        converted += 1

    return converted, skipped


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input)
    out_dir = Path(args.output)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    ensure_output_dirs(out_dir)
    train_converted, train_skipped = convert_split(in_dir, out_dir, split_name="train", out_split="train")
    val_converted, val_skipped = convert_split(
        in_dir,
        out_dir,
        split_name="validation",
        out_split="val",
    )

    print("[convert] Conversion completed.")
    print(f"[convert] train: converted={train_converted}, skipped={train_skipped}")
    print(f"[convert] val: converted={val_converted}, skipped={val_skipped}")


if __name__ == "__main__":
    main()
