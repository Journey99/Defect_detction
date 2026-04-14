from pathlib import Path


def test_data_yaml_exists() -> None:
    assert Path("data/data.yaml").exists()
