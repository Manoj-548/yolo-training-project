#!/usr/bin/env python3
"""
YOLOv8 CLI Tool for train, val, predict tasks.
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
import torch.serialization
from ultralytics.nn.tasks import DetectionModel

# Add safe globals for torch.load
torch.serialization.add_safe_globals([DetectionModel])

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 CLI Tool")
    parser.add_argument('--data', type=str, help='Path to data.yaml')
    parser.add_argument('--source', type=str, help='Path to image/video/folder')
    parser.add_argument('--weights', type=str, help='Path to weights (yolov8n.pt/best.pt/last.pt)')
    parser.add_argument('--skip-train', action='store_true', help='Skip training step')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (default=0.25)')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS (default=0.45)')
    parser.add_argument('--task', type=str, choices=['train', 'val', 'predict'], help='Task to perform: train, val, predict. For predict, --source examples: image (e.g., path/to/image.jpg), video (e.g., path/to/video.mp4), folder (e.g., path/to/folder/)')

    args = parser.parse_args()

    # Validate conf and iou
    if not (0 <= args.conf <= 1):
        print("Error: --conf must be between 0 and 1")
        sys.exit(1)
    if not (0 <= args.iou <= 1):
        print("Error: --iou must be between 0 and 1")
        sys.exit(1)

    # Load model
    if args.weights:
        if not Path(args.weights).exists():
            print(f"Error: Weights file {args.weights} not found")
            sys.exit(1)
        model = YOLO(args.weights)
    else:
        model = YOLO('yolov8n.pt')  # Default

    if args.task == 'train':
        if not args.data:
            print("Error: --data required for train task")
            sys.exit(1)
        if not Path(args.data).exists():
            print(f"Error: Data file {args.data} not found")
            sys.exit(1)
        if args.skip_train:
            print("Skipping training as requested")
        else:
            print("Starting training...")
            model.train(data=args.data, epochs=50, patience=50, imgsz=640, batch=16, device='auto', project='runs', name='train', exist_ok=True)
            print("Training completed")

    elif args.task == 'val':
        if not args.data:
            print("Error: --data required for val task")
            sys.exit(1)
        if not Path(args.data).exists():
            print(f"Error: Data file {args.data} not found")
            sys.exit(1)
        print("Starting validation...")
        results = model.val(data=args.data, conf=args.conf, iou=args.iou)
        print("Validation completed")
        print(results)

    elif args.task == 'predict':
        if not args.source:
            print("Error: --source required for predict task")
            sys.exit(1)
        if not Path(args.source).exists():
            print(f"Error: Source {args.source} not found")
            sys.exit(1)
        print(f"Starting prediction on {args.source}...")
        results = model.predict(source=args.source, conf=args.conf, iou=args.iou, save=True)
        print("Prediction completed")
        print(results)

    else:
        print("Error: --task must be specified")
        sys.exit(1)

if __name__ == "__main__":
    main()
