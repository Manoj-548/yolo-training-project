import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, Bottleneck
import torch.nn as nn

# Load the model with safe globals to handle PyTorch 2.6 weights_only issue
torch.serialization.add_safe_globals([DetectionModel, nn.Sequential, Conv, nn.Conv2d, nn.BatchNorm2d, nn.SiLU, C2f, nn.ModuleList, Bottleneck])
model = YOLO('runs/detect/train_pipeline/weights/best.pt')

# Run prediction on test images with all relevant parameters set to true for optimal output
results = model.predict(
    source='images/test',
    save=True,
    save_txt=True,
    save_conf=True,
    save_crop=True,
    show_labels=True,
    show_conf=True,
    plots=True,
    conf=0.25,
    iou=0.7,
    max_det=300,
    half=False,
    dnn=False,
    device='cpu',
    verbose=True,
    imgsz=640,
    batch=16,
    workers=8,
    project='runs',
    name='predict',
    exist_ok=True
)

print("Prediction completed. Results saved in runs/predict directory.")
