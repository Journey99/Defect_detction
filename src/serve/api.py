from fastapi import FastAPI, File, UploadFile

from src.serve.inference import OnnxDefectDetector

app = FastAPI(title="Industrial Defect Detection API")
detector: OnnxDefectDetector | None = None


@app.on_event("startup")
def startup() -> None:
    global detector
    # Model path can be replaced by env var or config later.
    # API can still run without model for endpoint sanity checks.
    try:
        detector = OnnxDefectDetector("runs/export/model.onnx")
    except FileNotFoundError:
        detector = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    image_bytes = await file.read()
    if detector is None:
        return {"error": "model not loaded", "hint": "run export first"}
    return detector.predict(image_bytes)
