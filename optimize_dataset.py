import os
import hashlib
from pathlib import Path
import shutil
import yaml
import cv2
import numpy as np
from collections import defaultdict
import json

class DatasetOptimizer:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.images_dir = self.base_path / "images"
        self.labels_dir = self.base_path / "labels"
        self.optimized_dir = self.base_path / "optimized_dataset"
        self.stats_file = self.base_path / "dataset_stats.json"
        
        # Create optimized directory
        self.optimized_dir.mkdir(exist_ok=True)
        (self.optimized_dir / "images").mkdir(exist_ok=True)
        (self.optimized_dir / "labels").mkdir(exist_ok=True)
        
    def get_image_hash(self, image_path):
        """Calculate perceptual hash of image"""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        # Resize and convert to grayscale
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Calculate hash
        return hashlib.md5(img.tobytes()).hexdigest()
    
    def analyze_dataset(self):
        """Analyze current dataset for duplicates and class distribution"""
        print("Analyzing dataset...")
        stats = {
            'total_images': 0,
            'duplicates': 0,
            'class_distribution': defaultdict(int),
            'image_hashes': defaultdict(list)
        }
        
        for split in ['train', 'val', 'test']:
            split_images = self.images_dir / split
            if not split_images.exists():
                continue
                
            for img_path in split_images.glob('*.*'):
                stats['total_images'] += 1
                img_hash = self.get_image_hash(img_path)
                if img_hash:
                    if len(stats['image_hashes'][img_hash]) > 0:
                        stats['duplicates'] += 1
                    stats['image_hashes'][img_hash].append(str(img_path))
                
                # Get class from filename
                class_name = img_path.stem.split('_')[0]
                stats['class_distribution'][class_name] += 1
        
        return stats
    
    def optimize_dataset(self, max_images_per_class=50):
        """Create optimized dataset with reduced duplicates and balanced classes"""
        stats = self.analyze_dataset()
        print(f"\nFound {stats['duplicates']} duplicate images")
        
        # Create optimized dataset structure
        for split in ['train', 'val', 'test']:
            (self.optimized_dir / "images" / split).mkdir(exist_ok=True)
            (self.optimized_dir / "labels" / split).mkdir(exist_ok=True)
        
        # Process each class
        processed_hashes = set()
        class_counts = defaultdict(int)
        
        for split in ['train', 'val', 'test']:
            split_images = self.images_dir / split
            if not split_images.exists():
                continue
            
            for img_path in split_images.glob('*.*'):
                class_name = img_path.stem.split('_')[0]
                
                # Skip if we have enough images for this class
                if class_counts[class_name] >= max_images_per_class:
                    continue
                
                img_hash = self.get_image_hash(img_path)
                if img_hash and img_hash not in processed_hashes:
                    # Copy image and label to optimized dataset
                    new_img_path = self.optimized_dir / "images" / split / img_path.name
                    label_path = self.labels_dir / split / f"{img_path.stem}.txt"
                    new_label_path = self.optimized_dir / "labels" / split / f"{img_path.stem}.txt"
                    
                    if label_path.exists():
                        shutil.copy2(img_path, new_img_path)
                        shutil.copy2(label_path, new_label_path)
                        processed_hashes.add(img_hash)
                        class_counts[class_name] += 1
        
        # Save optimization stats
        stats['optimized'] = {
            'images_per_class': max_images_per_class,
            'total_images': sum(class_counts.values()),
            'class_counts': dict(class_counts)
        }
        
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nOptimized dataset created with {sum(class_counts.values())} total images")
        return stats
    
    def cleanup_unused_files(self):
        """Remove unnecessary files to save space"""
        total_cleaned = 0
        
        # Clean predict directories
        predict_dirs = list(self.base_path.glob("runs/detect/predict*"))
        for path in predict_dirs:
            if path.is_dir():
                shutil.rmtree(path)
                total_cleaned += 1
        
        # Clean temporary files
        tmp_files = list(self.base_path.rglob("*.tmp"))
        for path in tmp_files:
            if path.is_file():
                path.unlink()
                total_cleaned += 1
        
        # Clean pycache
        pycache_dirs = list(self.base_path.rglob("__pycache__"))
        for path in pycache_dirs:
            if path.is_dir():
                shutil.rmtree(path)
                total_cleaned += 1
        
        print(f"\nCleaned up {total_cleaned} unnecessary files/directories")

if __name__ == "__main__":
    optimizer = DatasetOptimizer()
    stats = optimizer.optimize_dataset(max_images_per_class=50)  # Reduce to 50 images per class
    optimizer.cleanup_unused_files()
    
    # Print final stats
    print("\nDataset Optimization Summary:")
    print(f"Original total images: {stats['total_images']}")
    print(f"Duplicates found: {stats['duplicates']}")
    print(f"Optimized total images: {stats['optimized']['total_images']}")
    print("\nTop 10 classes by count:")
    sorted_classes = sorted(stats['class_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]
    for class_name, count in sorted_classes:
        print(f"{class_name}: {count} → {stats['optimized']['class_counts'].get(class_name, 0)}")