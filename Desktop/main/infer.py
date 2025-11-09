from ultralytics import YOLO
import sys
import yaml
import os
import torch
import torch.nn.functional as F
from ultralytics.utils.plotting import Annotator
import cv2
import requests
import tempfile
import json
import numpy as np
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

# Load class names from data.yaml
data_yaml_path = '../../../../Desktop/main/synthetic_dataset/data.yaml'
if os.path.exists(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg.get('names', [])
else:
    print(f"Warning: {data_yaml_path} not found. Using default class names.")
    class_names = [str(i) for i in range(95)]  # Default OCR classes

def get_image_path(source):
    """Get image path from various sources: local file, URL, or directory."""
    if source.startswith(('http://', 'https://')):
        # Handle URL
        try:
            response = requests.get(source, timeout=10)
            if response.status_code == 200:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                temp_file.write(response.content)
                temp_file.close()
                print(f"Downloaded image from URL: {source}")
                return temp_file.name
            else:
                print(f"Failed to download from URL: {source}")
                return None
        except Exception as e:
            print(f"Error downloading from URL {source}: {e}")
            return None
    elif os.path.isfile(source):
        return source
    else:
        print(f"Source not found or not supported: {source}")
        return None

def find_latest_weights():
    """Find the latest trained weights: prefer best.pt over last.pt, segmentation over detection."""
    possible_weights = [
        'runs/segment_train/weights/best.pt',
        'runs/segment_train/weights/last.pt',
        'runs/train/weights/best.pt',
        'runs/train/weights/last.pt',
        'synthetic_data/runs/train/weights/best.pt',
        'synthetic_data/runs/train/weights/last.pt',
        'last.pt',
        'best.pt'
    ]
    for weight in possible_weights:
        if os.path.exists(weight):
            print(f"Using weights: {weight}")
            return weight
    print("No weights found. Using default yolov8n.pt")
    return 'yolov8n.pt'

def group_characters_into_words(detected_chars, max_horizontal_gap=50):
    """Group detected characters into words based on horizontal proximity."""
    if not detected_chars:
        return []

    # Sort by y-coordinate (rows), then by x-coordinate (left to right)
    sorted_chars = sorted(detected_chars, key=lambda x: (x[1][1], x[1][0]))  # y, then x

    words = []
    current_word = [sorted_chars[0]]

    for char in sorted_chars[1:]:
        prev_char = current_word[-1]
        # Check if in same row (similar y) and close horizontally
        if abs(char[1][1] - prev_char[1][1]) < 30 and char[1][0] - prev_char[1][2] < max_horizontal_gap:
            current_word.append(char)
        else:
            # Start new word
            if current_word:
                words.append(current_word)
            current_word = [char]

    if current_word:
        words.append(current_word)

    return words

def interpret_word_meaning(word_text, word_bbox):
    """Interpret the meaning of a detected word based on full descriptions and visual context."""
    meanings = {
        'HT': 'High Temperature - Indicates elevated thermal readings on the device',
        'GE': 'General Electric - Manufacturer name, visible on equipment branding',
        'GC': 'Gas Chromatograph - Analytical instrument for separating chemical mixtures',
        'T1': 'Type 1 - Model designation for the specific device variant',
        'C': 'Controller - Central processing unit managing device operations',
        'SNAPSHOT': 'Image Capture - Digital photograph taken at a specific moment',
        'TEMPERATURE': 'Temperature - Measurement of heat level, displayed numerically or graphically',
        'SMOKE': 'Smoke - Visible particulate matter from combustion, appears as hazy gray clouds',
        'PRESSURE': 'Pressure - Force exerted per unit area, shown in PSI or bar units',
        'FLOW': 'Flow - Rate of fluid movement through pipes or channels',
        'LEVEL': 'Level - Height or quantity measurement in tanks or containers',
        'ALARM': 'Alarm - Warning signal for abnormal conditions, often with flashing indicators',
        'STATUS': 'Status - Current operational state of the system',
        '2025': 'Year 2025 - Calendar year when the image was captured',
        '07': 'July - Seventh month of the year',
        '26': '26th - Day of the month',
        '18': '6 PM - Time in 24-hour format',
        '23': '23 minutes - Minute component of timestamp',
        '20': '20 seconds - Second component of timestamp',
        '243': '243 milliseconds - Millisecond precision in timestamp',
        '53980021997': 'Serial Number - Unique identifier for the equipment unit'
    }

    # Try exact match first
    if word_text.upper() in meanings:
        return meanings[word_text.upper()]

    # Try partial matches or patterns
    if word_text.isdigit():
        if len(word_text) == 4 and word_text.startswith('20'):
            return f"Year {word_text} - Calendar year notation"
        elif len(word_text) == 2:
            return f"Month/Day {word_text} - Date component"
        elif len(word_text) > 8:
            return f"Serial Number {word_text} - Unique equipment identifier"

    # Contextual interpretations based on common industrial terms
    word_lower = word_text.lower()
    if 'temp' in word_lower:
        return f"Temperature Reading - Numerical value showing current heat measurement"
    elif 'press' in word_lower:
        return f"Pressure Reading - Value indicating force per unit area"
    elif 'flow' in word_lower:
        return f"Flow Rate - Measurement of fluid movement speed"
    elif 'level' in word_lower:
        return f"Level Indicator - Shows quantity or height in container"
    elif 'alarm' in word_lower:
        return f"Alarm Condition - Warning for system anomaly"
    elif 'status' in word_lower:
        return f"System Status - Current operational condition"

    # Default interpretation with visual context
    return f"Text: {word_text} - Detected text element in the image"

def run_inference(image_path, weights=None, annotations_path=None, conf=0.1, iou=0.5, save=True):
    """Run inference on an image with ground truth comparison, word grouping, and meaning interpretation."""
    if weights is None:
        weights = find_latest_weights()

    try:
        # Load the trained model weights for inference
        model = YOLO(weights)
        print("Model loaded successfully.")
        print(model.info())  # Print model stats
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Please check if the weights file exists and is compatible with the current ultralytics version.")
        return None

    # Load ground truth annotations from JSON
    if annotations_path is None:
        annotations_path = '../../../../Downloads/image_0_annotations.json'
    if os.path.exists(annotations_path):
        with open(annotations_path, 'r') as f:
            annotations_data = json.load(f)
        ground_truth = {ann['label']: ann['points'] for ann in annotations_data['annotations']}
        print("Ground truth annotations loaded.")
    else:
        ground_truth = {}
        print("No ground truth annotations found.")

    # Run inference
    results = model.predict(source=image_path, save=save, conf=conf, iou=iou)

    for result in results:
        print("Detected objects:")
        # Group boxes by coordinates to find highest confidence class per box
        boxes_by_coords = {}
        for box in result.boxes:
            coords = tuple(map(float, box.xyxy.tolist()[0]))
            if coords not in boxes_by_coords or box.conf.item() > boxes_by_coords[coords].conf.item():
                boxes_by_coords[coords] = box

        detected_chars = []
        for box in boxes_by_coords.values():
            cls = int(box.cls)
            conf_val = box.conf.item()
            class_name = class_names[cls] if cls < len(class_names) else str(cls)
            print(f"Class: {class_name} ({cls}), Confidence: {conf_val:.2f}, Bbox: {box.xyxy.tolist()}")
            detected_chars.append((class_name, box.xyxy.tolist()[0], conf_val))

        # Group characters into words
        words = group_characters_into_words(detected_chars)
        print(f"\nGrouped into {len(words)} words:")

        detected_words = []
        for word in words:
            word_text = ''.join([char[0] for char in word])
            # Calculate word bounding box
            word_bbox = [
                min([char[1][0] for char in word]),
                min([char[1][1] for char in word]),
                max([char[1][2] for char in word]),
                max([char[1][3] for char in word])
            ]
            avg_conf = sum([char[2] for char in word]) / len(word)
            meaning = interpret_word_meaning(word_text, word_bbox)
            print(f"Word: '{word_text}' at {word_bbox} with avg conf {avg_conf:.2f} - Meaning: {meaning}")
            detected_words.append((word_text, word_bbox, avg_conf, meaning))

        # Compare with ground truth
        print("\nGround Truth vs Detected:")
        for gt_label, gt_points in ground_truth.items():
            if gt_label in [d[0] for d in detected_chars]:
                detected = next((d for d in detected_chars if d[0] == gt_label), None)
                if detected:
                    print(f"✓ {gt_label}: Detected at {detected[1]} with conf {detected[2]:.2f}")
            else:
                print(f"✗ {gt_label}: Not detected")

        # Visualize the results on the image
        im0 = result.orig_img.copy()
        annotator = Annotator(im0)

        # Draw ground truth polygons
        for gt_label, gt_points in ground_truth.items():
            pts = np.array([[p['x'], p['y']] for p in gt_points], np.int32)
            cv2.polylines(im0, [pts], True, (0, 255, 0), 2)  # Green for ground truth
            cv2.putText(im0, gt_label, (int(pts[0][0]), int(pts[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw individual characters
        for box in boxes_by_coords.values():
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]
            cls = int(box.cls)
            conf_val = box.conf.item()
            class_name = class_names[cls] if cls < len(class_names) else str(cls)
            label = f"{class_name} {conf_val:.2f}"
            annotator.box_label(xyxy, label)

        # Draw word bounding boxes and meanings
        for word_text, word_bbox, avg_conf, meaning in detected_words:
            x1, y1, x2, y2 = map(int, word_bbox)
            cv2.rectangle(im0, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for words
            cv2.putText(im0, f"{word_text}: {meaning}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        annotated_img = annotator.result()

        # Show the image with annotations
        cv2.imshow("Inference Visualization (Green: Ground Truth, Blue: Detected, Red: Words)", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run YOLO inference on images")
    parser.add_argument("image_path", help="Path to input image or URL")
    parser.add_argument("--weights", help="Path to YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--annotations", help="Path to ground truth annotations JSON")
    parser.add_argument("--save", action="store_true", default=True, help="Save results")

    args = parser.parse_args()

    image_path = get_image_path(args.image_path)
    if image_path is None:
        sys.exit(1)

    run_inference(image_path, weights=args.weights, annotations_path=args.annotations, conf=args.conf, iou=args.iou, save=args.save)
