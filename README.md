# 🔩 Industrial Defect Detection
**NEU-DET 철강 표면 결함 탐지 | YOLOv11 · RT-DETR · WIoU · SAHI**

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| Task | Object Detection |
| Dataset | NEU-DET (철강 표면 결함 6클래스) |
| 주요 모델 | YOLOv11m, RT-DETR-R50 |
| 핵심 기법 | WIoU Loss, SAHI, TTA, Mosaic/MixUp Augmentation |
| 서빙 | FastAPI + ONNX Runtime |

### 탐지 클래스
`crazing` · `inclusion` · `patches` · `pitted_surface` · `rolled-in_scale` · `scratches`

---

## 📁 프로젝트 구조

```
defect-detection/
├── data/
│   ├── raw/                  # NEU-DET 원본 데이터
│   ├── processed/            # YOLO 포맷 변환 데이터
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   └── data.yaml
├── src/
│   ├── data/
│   │   ├── convert.py        # VOC XML → YOLO 포맷 변환
│   │   └── verify.py         # 데이터셋 무결성 검증
│   ├── train/
│   │   ├── train.py          # 학습 엔트리포인트
│   │   ├── losses.py         # WIoU Loss 구현
│   │   └── callbacks.py      # WandB 로깅 콜백
│   ├── eval/
│   │   ├── evaluate.py       # mAP 평가
│   │   └── tta.py            # Test Time Augmentation
│   └── serve/
│       ├── export.py         # ONNX 변환
│       ├── inference.py      # ONNX Runtime 추론 클래스
│       └── api.py            # FastAPI 앱
├── configs/
│   ├── baseline.yaml
│   ├── wiou.yaml
│   └── tta.yaml
├── requirements.txt
└── Makefile
```

---


**requirements.txt**
```
ultralytics==8.3.0
wandb==0.17.0
fastapi==0.111.0
uvicorn==0.30.0
onnxruntime==1.18.0
opencv-python==4.10.0.84
PyYAML==6.0.1
```

---

## 🚀 실행 방법

### Phase 1 — 베이스라인

```bash
# 데이터 변환 (로컬)
python src/data/convert.py --input data/raw --output data/processed

# 학습 (Kaggle / Colab)
python src/train/train.py --config configs/baseline.yaml

# 평가
python src/eval/evaluate.py --weights runs/train/baseline/weights/best.pt
```

### Phase 2 — WIoU + TTA

```bash
# WIoU loss 실험
python src/train/train.py --config configs/wiou.yaml

# TTA 평가
python src/eval/evaluate.py --weights runs/train/wiou/weights/best.pt --tta
```


```
또는 Makefile 사용:
```bash
make setup      # 환경 설치
make convert    # 데이터 변환
make train      # 베이스라인 학습
make eval       # 평가
make serve      # API 서버 실행
```

---

## 📊 실험 결과

| 모델 | Loss | mAP@0.5 | mAP@0.5:0.95 | FPS |
|------|------|---------|--------------|-----|
| YOLOv11m | CIoU | - | - | - |
| YOLOv11m | WIoU | - | - | - |
| YOLOv11m + TTA | WIoU | - | - | - |
| RT-DETR-R50 | - | - | - | - |



---

## 🔬 주요 기법 설명

### WIoU Loss (Wise-IoU, 2023)
어려운 샘플(outlier)에 집중하는 동적 가중치 부여 방식의 IoU Loss.
기존 CIoU 대비 소형 결함 박스 회귀 정밀도 향상.

### SAHI (Slicing Aided Hyper Inference)
고해상도 산업 이미지를 슬라이스로 분할하여 추론 후 결과를 병합.
미세 결함 탐지에 효과적.

### TTA (Test Time Augmentation)
추론 시 다중 스케일 + 좌우 반전 이미지로 앙상블.
mAP 향상 및 예측 안정성 증가.

---

## 📚 참고 문헌

- [YOLOv11](https://github.com/ultralytics/ultralytics)
- [RT-DETR: DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
- [Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism](https://arxiv.org/abs/2301.10051)
- [SAHI: Slicing Aided Hyper Inference](https://github.com/obss/sahi)
- [NEU Surface Defect Database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html)