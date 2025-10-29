import yaml
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

# Add safe globals for PyTorch 2.6 weights loading (additional)
torch.serialization.add_safe_globals([DetectionModel, nn.Sequential, Conv, nn.Conv2d, nn.BatchNorm2d, nn.SiLU, C2f, nn.ModuleList, Bottleneck, SPPF, nn.MaxPool2d, nn.Upsample, nn.ReLU, nn.AdaptiveAvgPool2d, nn.Linear, nn.Dropout])

# Load args from yaml
with open('../../../../Desktop/main/runs/train/args.yaml', 'r') as f:
    args = yaml.safe_load(f)

# Load model
model_path = '../../../../Desktop/main/' + args['model']
model = YOLO(model_path)

# Prepare train args, excluding task, mode, model
train_args = {k: v for k, v in args.items() if k not in ['task', 'mode', 'model']}
# Update data path to absolute
train_args['data'] = '../../../../Desktop/main/' + args['data']

# Run training
model.train(**train_args)

print("Training completed.")
