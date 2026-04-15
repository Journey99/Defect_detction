PYTHON ?= python

.PHONY: setup convert verify train eval serve export exp-yolo-ciou exp-yolo-wiou exp-yolo-wiou-tta exp-rtdetr

setup:
	$(PYTHON) -m pip install -r requirements.txt

convert:
	$(PYTHON) src/data/convert.py --input data/raw --output data/processed

verify:
	$(PYTHON) src/data/verify.py --dataset-yaml data/data.yaml

train:
	$(PYTHON) src/train/train.py --config configs/yolo_ciou.yaml

eval:
	$(PYTHON) src/eval/evaluate.py --weights runs/train/yolo_ciou/weights/best.pt --model-type yolo

export:
	$(PYTHON) src/serve/export.py --weights runs/train/yolo_ciou/weights/best.pt --output runs/export/model.onnx

serve:
	uvicorn src.serve.api:app --host 0.0.0.0 --port 8000 --reload

exp-yolo-ciou:
	$(PYTHON) src/train/train.py --config configs/yolo_ciou.yaml

exp-yolo-wiou:
	$(PYTHON) src/train/train.py --config configs/yolo_wiou.yaml

exp-yolo-wiou-tta:
	$(PYTHON) src/eval/evaluate.py --weights runs/train/yolo_wiou/weights/best.pt --model-type yolo --data data/data.yaml --tta

exp-rtdetr:
	$(PYTHON) src/train/train.py --config configs/rtdetr.yaml
