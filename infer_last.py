import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, Bottleneck, SPPF
import torch.nn as nn

# Patch torch.load to use weights_only=False for trusted checkpoints
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# Add safe globals for PyTorch 2.6 weights loading
torch.serialization.add_safe_globals([DetectionModel, nn.Sequential, Conv, nn.Conv2d, nn.BatchNorm2d, nn.SiLU, C2f, nn.ModuleList, Bottleneck, SPPF, nn.MaxPool2d, nn.Upsample, nn.ReLU, nn.AdaptiveAvgPool2d, nn.Linear, nn.Dropout])

# Load model with best.pt
model_path = 'runs/detect/train_pipeline/weights/best.pt'
model = YOLO(model_path)

# Run inference on test image
results = model.predict(
    source='images/test/2.jpg',
    save=True,
    show_labels=True,
    show_conf=True,
    plots=True
)

print("Inference completed on images/test/2.jpg")
