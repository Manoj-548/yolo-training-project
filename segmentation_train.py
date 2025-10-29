import torch
import cv2
import yaml
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, Bottleneck, SPPF
import torch.nn as nn
import os
import json

# Patch torch.load to use weights_only=False for trusted checkpoints
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# Add safe globals for PyTorch 2.6 weights loading
torch.serialization.add_safe_globals([DetectionModel, nn.Sequential, Conv, nn.Conv2d, nn.BatchNorm2d, nn.SiLU, C2f, nn.ModuleList, Bottleneck, SPPF, nn.MaxPool2d, nn.Upsample, nn.ReLU, nn.AdaptiveAvgPool2d, nn.Linear, nn.Dropout])

# Load default.yaml for settings
with open('../../venv/Lib/site-packages/ultralytics/cfg/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

def create_segmentation_dataset_from_features(features_file, output_dir="segmentation_data"):
    """Create segmentation dataset from OCR features for character detection."""
    import numpy as np
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    with open(features_file, 'r') as f:
        features = json.load(f)

    for i, img_feature in enumerate(features):
        detections = img_feature['detections']

        # Create a blank image (640x640) for segmentation
        img = np.zeros((640, 640, 3), dtype=np.uint8)

        # For segmentation, we need polygon masks instead of bounding boxes
        label_file = labels_dir / f"{i}.txt"
        with open(label_file, 'w') as lf:
            for det in detections:
                cls = det['class']
                bbox = det['bbox']
                # Convert bbox to polygon (simple rectangle for now)
                x1, y1, x2, y2 = bbox
                # Normalize coordinates
                x1, x2 = x1 / 640, x2 / 640
                y1, y2 = y1 / 640, y2 / 640
                # Create polygon points (rectangle)
                polygon = f"{x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}"
                lf.write(f"{cls} {polygon}\n")

        # Save blank image (in real scenario, load original image)
        cv2.imwrite(str(images_dir / f"{i}.jpg"), img)

    # Create data.yaml for segmentation
    data_yaml = {
        "train": str(images_dir),
        "val": str(images_dir),
        "nc": 62,  # Assuming 62 classes for OCR
        "names": [str(i) for i in range(62)],
        "task": "segment"  # Specify segmentation task
    }
    with open(output_dir / "data.yaml", 'w') as f:
        yaml.dump(data_yaml, f)

    print(f"✅ Segmentation dataset created in {output_dir}")
    return str(output_dir / "data.yaml")

def train_segmentation_model(data_yaml, weights="yolov8n-seg.pt"):
    """Train YOLOv8 segmentation model for character detection."""
    model = YOLO(weights)

    print("🚀 Training segmentation model for character detection...")

    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=8,  # Smaller batch for segmentation
        device="cpu",  # Force CPU since CUDA is not available
        project="runs",
        name="segment_train",
        exist_ok=True,
        resume=False,
        patience=5,
        task="segment",  # Explicitly set task to segmentation
        mosaic=0.0,  # Disable mosaic augmentation for small dataset
        mixup=0.0,  # Disable mixup augmentation for small dataset
        close_mosaic=0  # Disable close mosaic
    )

    best_weights = "runs/segment_train/weights/best.pt"
    print(f"✅ Segmentation training complete. Best weights: {best_weights}")
    return best_weights

def test_segmentation_on_image(model_path, image_path):
    """Test segmentation model on a specific image."""
    model = YOLO(model_path)

    print(f"🔍 Testing segmentation on {image_path}")

    results = model.predict(
        source=image_path,
        conf=config.get('conf', 0.25),
        iou=config.get('iou', 0.7),
        save=True,
        show_labels=True,
        show_conf=True,
        plots=True,
        task="segment"
    )

    print("✅ Segmentation test complete")
    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and test segmentation for character detection")
    parser.add_argument('--data', type=str, default='C:/Users/Acer/Desktop/main/synthetic_dataset_segmentation/data.yaml', help="Data YAML file")
    parser.add_argument('--image', type=str, help="Test image path")
    parser.add_argument('--weights', type=str, default='yolov8n-seg.pt', help="Base weights for segmentation")

    args = parser.parse_args()

    # Use the segmentation dataset created from annotations
    if os.path.exists(args.data):
        data_yaml = args.data
        print(f"Using data.yaml: {data_yaml}")
    else:
        print(f"Data file {args.data} not found. Please run create_segmentation_dataset.py first.")
        exit(1)

    # Train segmentation model
    best_weights = train_segmentation_model(data_yaml, args.weights)

    # Test on image if provided
    if args.image and os.path.exists(args.image):
        test_segmentation_on_image(best_weights, args.image)
