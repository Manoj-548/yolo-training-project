from PIL import Image
import json
import os

# Load the image to get size
image_path = r"C:\Users\Acer\Desktop\New folder (3)\HT-GE134GC-T1-C-Snapshot-20250626-121954-110-886770761944.BMP"
img = Image.open(image_path)
img_width, img_height = img.size
print(f"Image size: {img_width} x {img_height}")

# Load annotations
annotations_path = r'C:\Users\Acer\Downloads\image_0_annotations.json'
with open(annotations_path, 'r') as f:
    data = json.load(f)

annotations = data['annotations']

# Get unique labels
unique_labels = list(set(ann['label'] for ann in annotations if ann['label'] != 'Unlabeled'))
unique_labels.sort()
print(f"Unique labels: {unique_labels}")

# Create class mapping
class_mapping = {label: idx for idx, label in enumerate(unique_labels)}

# Create directories
os.makedirs('synthetic_dataset_segmentation/images/train', exist_ok=True)
os.makedirs('synthetic_dataset_segmentation/images/val', exist_ok=True)
os.makedirs('synthetic_dataset_segmentation/labels/train', exist_ok=True)
os.makedirs('synthetic_dataset_segmentation/labels/val', exist_ok=True)

# Create data.yaml
data_yaml_content = f"""
train: images/train
val: images/val

nc: {len(unique_labels)}
names: {unique_labels}
"""

with open('synthetic_dataset_segmentation/data.yaml', 'w') as f:
    f.write(data_yaml_content)

# Copy image to train and val
import shutil
shutil.copy(image_path, 'synthetic_dataset_segmentation/images/train/image_0.jpg')
shutil.copy(image_path, 'synthetic_dataset_segmentation/images/val/image_0.jpg')

# Create label file
label_file = 'synthetic_dataset_segmentation/labels/train/image_0.txt'
with open(label_file, 'w') as f:
    for ann in annotations:
        if ann['label'] == 'Unlabeled':
            continue
        label = ann['label']
        class_id = class_mapping[label]
        points = ann['points']
        # Normalize points
        normalized_points = []
        for p in points:
            x_norm = p['x'] / img_width
            y_norm = p['y'] / img_height
            normalized_points.extend([x_norm, y_norm])
        line = f"{class_id} " + " ".join(f"{p:.6f}" for p in normalized_points)
        f.write(line + '\n')

# Copy label to val
shutil.copy(label_file, 'synthetic_dataset_segmentation/labels/val/image_0.txt')

print("Segmentation dataset created.")
