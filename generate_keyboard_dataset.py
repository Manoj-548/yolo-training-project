import os
import cv2
import numpy as np
from pathlib import Path
import yaml
import json
from PIL import Image, ImageDraw, ImageFont
import random

class KeyboardDatasetGenerator:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.fonts_dir = self.base_path / "fonts"
        self.output_dir = self.base_path / "images"
        self.labels_dir = self.base_path / "labels"
        
        # Create necessary directories
        self.fonts_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.labels_dir.mkdir(exist_ok=True)
        
        # Load class names
        with open('data.yaml', 'r') as f:
            self.data_config = yaml.safe_load(f)
            self.class_names = self.data_config['names']
            self.num_classes = len(self.class_names)
        
        # Create class to index mapping
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Font settings
        self.font_sizes = range(24, 73, 8)  # Different font sizes for variety
        self.backgrounds = ['white', 'light_gray', 'dark_gray']
        
    def create_key_image(self, key_text, font_path, font_size, background='white'):
        """Create an image of a keyboard key with the given text"""
        # Image size based on font size
        padding = int(font_size * 0.5)
        img_size = (font_size * 3, font_size * 3)
        
        # Create image with background
        bg_colors = {
            'white': (255, 255, 255),
            'light_gray': (240, 240, 240),
            'dark_gray': (200, 200, 200)
        }
        
        img = Image.new('RGB', img_size, bg_colors[background])
        draw = ImageDraw.Draw(img)
        
        # Load font
        try:
            font = ImageFont.truetype(str(font_path), font_size)
        except Exception:
            font = ImageFont.load_default()
        
        # Calculate text position to center it
        text_bbox = draw.textbbox((0, 0), key_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (img_size[0] - text_width) // 2
        y = (img_size[1] - text_height) // 2
        
        # Draw text
        draw.text((x, y), key_text, font=font, fill=(0, 0, 0))
        
        # Add key border
        border_width = max(1, int(font_size * 0.05))
        draw.rectangle([0, 0, img_size[0]-1, img_size[1]-1], 
                      outline=(100, 100, 100), width=border_width)
        
        return img
    
    def generate_dataset(self, num_images_per_class=100, split_ratio={'train': 0.7, 'val': 0.2, 'test': 0.1}):
        """Generate dataset with the specified number of images per class"""
        print(f"Generating dataset with {num_images_per_class} images per class...")
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / split).mkdir(exist_ok=True)
            (self.labels_dir / split).mkdir(exist_ok=True)
        
        # Download or check for fonts
        self._ensure_fonts_available()
        
        # Generate images for each class
        for class_name in self.class_names:
            print(f"Generating images for class: {class_name}")
            
            images_generated = 0
            while images_generated < num_images_per_class:
                # Randomly select parameters
                font_path = random.choice(list(self.fonts_dir.glob('*.ttf')))
                font_size = random.choice(self.font_sizes)
                background = random.choice(self.backgrounds)
                
                # Create image
                img = self.create_key_image(class_name, font_path, font_size, background)
                
                # Determine split based on ratios
                split = np.random.choice(
                    ['train', 'val', 'test'],
                    p=[split_ratio['train'], split_ratio['val'], split_ratio['test']]
                )
                
                # Save image with safe filename
                safe_name = class_name.replace('*', 'star').replace('/', 'slash').replace('\\', 'bslash')
                safe_name = ''.join(c if c.isalnum() or c in '-_' else f'sym{ord(c)}' for c in safe_name)
                img_path = self.output_dir / split / f"{safe_name}_{images_generated}.jpg"
                img.save(str(img_path))
                
                # Create and save label (YOLO format)
                label_path = self.labels_dir / split / f"{safe_name}_{images_generated}.txt"
                with open(label_path, 'w') as f:
                    # Format: class_idx x_center y_center width height
                    # Using full image for simplicity
                    f.write(f"{self.class_to_idx[class_name]} 0.5 0.5 0.8 0.8\n")
                
                images_generated += 1
                
            print(f"Generated {images_generated} images for {class_name}")
    
    def _ensure_fonts_available(self):
        """Ensure necessary fonts are available"""
        # List of open-source fonts to download if needed
        font_urls = [
            "https://github.com/google/fonts/raw/main/ofl/roboto/Roboto-Regular.ttf",
            "https://github.com/google/fonts/raw/main/ofl/opensans/OpenSans-Regular.ttf",
            # Add more fonts as needed
        ]
        
        if not list(self.fonts_dir.glob('*.ttf')):
            print("Downloading fonts...")
            import requests
            for url in font_urls:
                try:
                    response = requests.get(url)
                    font_name = url.split('/')[-1]
                    font_path = self.fonts_dir / font_name
                    with open(font_path, 'wb') as f:
                        f.write(response.content)
                except Exception as e:
                    print(f"Error downloading font {url}: {e}")
    
    def verify_dataset(self):
        """Verify the generated dataset"""
        stats = {'train': 0, 'val': 0, 'test': 0}
        class_counts = {name: {'train': 0, 'val': 0, 'test': 0} for name in self.class_names}
        
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            if not split_dir.exists():
                continue
                
            for img_path in split_dir.glob('*.jpg'):
                stats[split] += 1
                class_name = img_path.stem.split('_')[0]
                if class_name in class_counts:
                    class_counts[class_name][split] += 1
        
        print("\nDataset Statistics:")
        print(f"Total images: {sum(stats.values())}")
        print("\nSplit distribution:")
        for split, count in stats.items():
            print(f"{split}: {count} images")
        
        print("\nClass distribution (sample):")
        for class_name in list(self.class_names)[:10]:  # Show first 10 classes
            counts = class_counts[class_name]
            print(f"{class_name}: {sum(counts.values())} total "
                  f"(train: {counts['train']}, val: {counts['val']}, test: {counts['test']})")

if __name__ == "__main__":
    generator = KeyboardDatasetGenerator()
    generator.generate_dataset(num_images_per_class=100)
    generator.verify_dataset()