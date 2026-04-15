from pathlib import Path

import pytest
import yaml


DATA_YAML_PATH = Path("data/data.yaml")


def _load_data_yaml() -> dict:
    with DATA_YAML_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)
    return cfg


def test_data_yaml_exists() -> None:
    assert DATA_YAML_PATH.exists()


def test_data_yaml_schema_is_valid() -> None:
    cfg = _load_data_yaml()

    assert "path" in cfg
    assert "train" in cfg
    assert "val" in cfg
    assert "names" in cfg

    assert isinstance(cfg["path"], str) and cfg["path"].strip()
    assert isinstance(cfg["train"], str) and cfg["train"].strip()
    assert isinstance(cfg["val"], str) and cfg["val"].strip()
    assert isinstance(cfg["names"], dict) and cfg["names"]

    # YOLO 클래스 인덱스는 0부터 연속된 정수여야 한다.
    class_ids = sorted(int(k) for k in cfg["names"].keys())
    assert class_ids == list(range(len(class_ids)))
    assert all(isinstance(v, str) and v.strip() for v in cfg["names"].values())


def test_dataset_directory_layout_if_present() -> None:
    cfg = _load_data_yaml()
    dataset_root = Path(cfg["path"])

    if not dataset_root.exists():
        pytest.skip(f"Dataset root does not exist: {dataset_root}")

    required_dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    ]
    missing = [p for p in required_dirs if not p.exists()]
    assert not missing, f"Missing dataset directories: {missing}"
