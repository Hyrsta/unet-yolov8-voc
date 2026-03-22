# UNet + YOLOv8 Combined Inference on Pascal VOC

Semantic segmentation and object detection pipelines trained on Pascal VOC 2007, with a combined inference mode that chains both models on the same input image.

Built during an internship at **Neusoft Education Technology Group Co., Ltd.** (Jul 2024).

## Results

### Semantic Segmentation (UNet, VGG backbone)

| Metric | Score |
|--------|-------|
| mIoU | 74.68% |
| mPA | 83.03% |
| mPrecision | 87.43% |
| mRecall | 83.03% |

Evaluated on Pascal VOC 2007 (21 classes, 512x512 input).

### Object Detection (YOLOv8)

| Metric | Score |
|--------|-------|
| mAP | 89.50% |

Per-class AP results and confusion matrices are in [`results/`](results/).

## How It Works

`predict.py` runs both models sequentially on the same image:

1. **UNet** produces a pixel-level semantic segmentation mask (21 VOC classes)
2. **YOLOv8** draws bounding boxes with class labels on top of the segmented output

This gives you both "what is each pixel" and "where is each object" in a single pass.

## Project Structure

```
.
├── predict.py                  # Combined inference entry point
├── unet.py                     # UNet wrapper (inference, FPS bench, ONNX export)
├── yolo.py                     # YOLOv8 wrapper (inference, FPS bench, ONNX export)
│
├── unet_nets/                  # UNet architecture (VGG / ResNet50 backbone)
├── unet_utils/                 # UNet data loading, training utils, metrics
├── yolov8_nets/                # YOLOv8 architecture
├── yolov8_utils/               # YOLOv8 data loading, training utils, mAP eval
│
├── training/
│   ├── unet/
│   │   ├── train.py            # UNet training script
│   │   ├── get_miou.py         # mIoU evaluation
│   │   ├── voc_annotation.py   # VOC dataset annotation parser
│   │   └── summary.py          # Model summary (FLOPs, params)
│   └── yolov8/
│       ├── train.py            # YOLOv8 training script
│       ├── get_map.py          # mAP evaluation
│       ├── voc_annotation.py   # VOC dataset annotation parser
│       └── summary.py          # Model summary (FLOPs, params)
│
├── results/
│   ├── unet/
│   │   └── confusion_matrix.csv
│   └── yolov8/
│       ├── per_class_ap.txt    # AP per class + ground truth counts
│       └── mAP_summary.txt
│
├── img/                        # Sample input images
├── unet_model_data/            # UNet model config (weights not included)
├── yolov8_model_data/          # YOLOv8 class list + model config
├── requirements.txt
└── LICENSE
```

## Quick Start

### Install

```bash
git clone https://github.com/Hyrsta/unet-yolov8-voc.git
cd unet-yolov8-voc
pip install -r requirements.txt
```

### Download Weights

Model weights are not included in this repo (too large for Git). Place trained weights at:

- `unet_model_data/unet_custom_model.pth`
- `yolov8_model_data/yolov8_custom_model.pth`

### Run Combined Inference

```bash
python predict.py
# Enter image path when prompted, e.g.: img/street.jpg
```

### Batch Processing

Edit `predict.py` and set `mode = "dir_predict"`:

```bash
python predict.py
# Processes all .jpg files in img/ and saves results to img_out/
```

### ONNX Export

```python
from unet import Unet
from yolo import YOLO

# Export UNet
unet = Unet()
unet.convert_to_onnx(simplify=True, model_path="unet.onnx")

# Export YOLOv8
yolo = YOLO()
yolo.convert_to_onnx(simplify=True, model_path="yolov8.onnx")
```

## Training

Both models were trained on Pascal VOC 2007 with:

- **EMA** (Exponential Moving Average) for weight smoothing
- **Freeze-unfreeze strategy**: backbone frozen during initial epochs, then full fine-tuning
- **Hyperparameter tuning** across learning rate, batch size, and data augmentation
- **Cosine annealing** learning rate scheduler

### Train UNet

```bash
# 1. Prepare VOC dataset in VOCdevkit/VOC2007/
# 2. Generate annotation files
cd training/unet
python voc_annotation.py

# 3. Train
python train.py
```

### Train YOLOv8

```bash
# 1. Prepare VOC dataset in VOCdevkit/VOC2007/
# 2. Generate annotation files
cd training/yolov8
python voc_annotation.py

# 3. Train
python train.py
```

### Evaluate

```bash
# UNet: compute mIoU, mPA, per-class IoU
cd training/unet
python get_miou.py

# YOLOv8: compute mAP, per-class AP, precision-recall
cd training/yolov8
python get_map.py
```

## Tech Stack

Python, PyTorch, OpenCV, NumPy, Pillow, ONNX, TensorBoard

## Pascal VOC Classes (20 + background)

```
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep,
sofa, train, tvmonitor
```

## License

[MIT](LICENSE)
