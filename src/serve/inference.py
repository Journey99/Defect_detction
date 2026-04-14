from pathlib import Path
from typing import Any

import onnxruntime as ort


class OnnxDefectDetector:
    def __init__(self, model_path: str) -> None:
        model = Path(model_path)
        if not model.exists():
            raise FileNotFoundError(f"ONNX model not found: {model}")
        self.session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        # Placeholder: add preprocessing/postprocessing for real predictions.
        return {"message": "inference placeholder", "input_size": len(image_bytes)}
