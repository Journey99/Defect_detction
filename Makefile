PYTHON ?= python

.PHONY: setup convert verify train eval serve export

setup:
	$(PYTHON) -m pip install -r requirements.txt

convert:
	$(PYTHON) src/data/convert.py --input data/raw --output data/processed

verify:
	$(PYTHON) src/data/verify.py --dataset-yaml data/data.yaml

train:
	$(PYTHON) src/train/train.py --config configs/baseline.yaml

eval:
	$(PYTHON) src/eval/evaluate.py --weights runs/train/baseline/weights/best.pt

export:
	$(PYTHON) src/serve/export.py --weights runs/train/baseline/weights/best.pt --output runs/export/model.onnx

serve:
	uvicorn src.serve.api:app --host 0.0.0.0 --port 8000 --reload
