from src.serve.inference import OnnxDefectDetector


def test_inference_placeholder_signature() -> None:
    assert hasattr(OnnxDefectDetector, "predict")
