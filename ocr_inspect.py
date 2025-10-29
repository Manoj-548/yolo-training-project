import json
import cv2
import torch
import os
import glob
import numpy as np
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, Bottleneck, SPPF
import torch.nn as nn
import yaml
import argparse
try:
    import easyocr  # type: ignore
    ocr_available = True
except ImportError:
    print("EasyOCR not installed. Install with: pip install easyocr")
    ocr_available = False
    reader = None

# Load default.yaml for settings
with open('../../venv/Lib/site-packages/ultralytics/cfg/default.yaml', 'r') as f:
    default_config = yaml.safe_load(f)

# Patch torch.load to use weights_only=False for trusted checkpoints
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# Add safe globals for PyTorch 2.6 weights loading
torch.serialization.add_safe_globals([DetectionModel, nn.Sequential, Conv, nn.Conv2d, nn.BatchNorm2d, nn.SiLU, C2f, nn.ModuleList, Bottleneck, SPPF, nn.MaxPool2d, nn.Upsample, nn.ReLU, nn.AdaptiveAvgPool2d, nn.Linear, nn.Dropout])

# Hardcoded defaults
hardcoded_config = {
    'model_path': 'runs/train/weights/best.pt',
    'data': 'data.yaml',  # Path to data.yaml for OCR dataset
    'conf': 0.25,
    'iou': 0.7,
    'show_labels': True,
    'show_conf': True,
    'plots': True,
    'ocr_lang': 'en'  # OCR language for all keyboard characters
}

# Argument parser for flexibility
parser = argparse.ArgumentParser(description="Inspect images from dataset and extract features.")
parser.add_argument('--source', type=str, default='images/test/', help="Path to image, directory, or pattern (e.g., images/test/*.jpg)")
parser.add_argument('--output', type=str, default='dataset_features.json', help="Output JSON file for features")
parser.add_argument('--use_ocr', action='store_true', help="Use OCR to extract text from detected regions")
parser.add_argument('--ocr_lang', type=str, default='en', help="OCR language (default: en)")
parser.add_argument('--config', type=str, help="Path to custom config YAML file")
args = parser.parse_args()

# Load custom config if provided
config = hardcoded_config.copy()
if args.config and os.path.exists(args.config):
    with open(args.config, 'r') as f:
        custom_config = yaml.safe_load(f)
        config.update(custom_config)
elif os.path.exists('project_config.yaml'):
    # Load project config if available
    with open('project_config.yaml', 'r') as f:
        project_config = yaml.safe_load(f)
        config.update(project_config.get('model', {}))
        config.update(project_config.get('ocr', {}))
else:
    config.update({k: v for k, v in default_config.items() if k in config})

# Load model
model = YOLO(config['model_path'])

# Initialize OCR reader if available and requested
if ocr_available and args.use_ocr:
    reader = easyocr.Reader([args.ocr_lang])
else:
    reader = None

# Get list of images
if os.path.isdir(args.source):
    image_paths = glob.glob(os.path.join(args.source, '*.jpg')) + glob.glob(os.path.join(args.source, '*.png')) + glob.glob(os.path.join(args.source, '*.bmp'))
elif '*' in args.source:
    image_paths = glob.glob(args.source)
else:
    image_paths = [args.source]

print(f"Found {len(image_paths)} images to inspect.")

# Run inference on all images
all_features = []
for image_path in image_paths:
    print(f"Inspecting {image_path}...")
    results = model.predict(
        source=image_path,
        conf=config.get('conf', 0.25),
        iou=config.get('iou', 0.7),
        save=True,
        show_labels=config.get('show_labels', True),
        show_conf=config.get('show_conf', True),
        plots=config.get('plots', True)
    )

    # Extract features for this image
    for result in results:
        img_features = {
            'image_path': result.path,
            'detections': []
        }
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for i in range(len(boxes)):
                detection = {
                    'class': int(classes[i]),
                    'confidence': float(confs[i]),
                    'bbox': boxes[i].tolist()
                }

                # Extract OCR text if OCR is enabled
                if reader is not None:
                    # Crop the detected region
                    img = cv2.imread(result.path)
                    x1, y1, x2, y2 = map(int, boxes[i])
                    cropped = img[y1:y2, x1:x2]
                    if cropped.size > 0:
                        # Perform OCR on the cropped region
                        ocr_results = reader.readtext(cropped)
                        detected_text = ' '.join([text for (_, text, _) in ocr_results])
                        detection['ocr_text'] = detected_text
                    else:
                        detection['ocr_text'] = ''
                else:
                    detection['ocr_text'] = ''

                img_features['detections'].append(detection)
        all_features.append(img_features)

# Save all features to JSON
with open(args.output, 'w') as f:
    json.dump(all_features, f, indent=4)

print(f"Inspection completed. Features saved to {args.output}")
print(f"Total images processed: {len(all_features)}")
total_detections = sum(len(f['detections']) for f in all_features)
print(f"Total detections: {total_detections}")
