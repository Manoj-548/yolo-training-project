import os
import yaml
import json
import random
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from collections import defaultdict

class DatasetBalancer:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.config_file = self.base_path / "project_config.yaml"
        self.training_focus_file = self.base_path / "training_focus.json"
        
        # Load configurations
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)
            
        with open(self.training_focus_file, 'r') as f:
            self.training_focus = json.load(f)
        
        self.target_samples = self.training_focus['training_parameters']['target_samples_per_class']
        self.balanced_dir = self.base_path / "balanced_dataset"
        self.fonts_dir = self.base_path / "fonts"
        
    def setup_balanced_dataset(self):
        """Create directory structure for balanced dataset"""
        for split in ['train', 'val', 'test']:
            (self.balanced_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.balanced_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    def reduce_overrepresented_classes(self):
        """Reduce samples for overrepresented classes"""
        for split in ['train', 'val', 'test']:
            images_dir = self.base_path / "images" / split
            if not images_dir.exists():
                continue
                
            # Count samples per class
            class_samples = defaultdict(list)
            for img_path in images_dir.glob("*.*"):
                class_name = img_path.stem.split('_')[0]
                class_samples[class_name].append(img_path)
            
            # Balance each class
            for class_name, samples in class_samples.items():
                if len(samples) > self.target_samples:
                    # Randomly select target number of samples
                    selected_samples = random.sample(samples, self.target_samples)
                    
                    # Copy selected samples to balanced dataset
                    for sample in selected_samples:
                        new_img_path = self.balanced_dir / "images" / split / sample.name
                        label_path = self.base_path / "labels" / split / f"{sample.stem}.txt"
                        new_label_path = self.balanced_dir / "labels" / split / f"{sample.stem}.txt"
                        
                        shutil.copy2(sample, new_img_path)
                        if label_path.exists():
                            shutil.copy2(label_path, new_label_path)
                else:
                    # Copy all samples if class is underrepresented
                    for sample in samples:
                        new_img_path = self.balanced_dir / "images" / split / sample.name
                        label_path = self.base_path / "labels" / split / f"{sample.stem}.txt"
                        new_label_path = self.balanced_dir / "labels" / split / f"{sample.stem}.txt"
                        
                        shutil.copy2(sample, new_img_path)
                        if label_path.exists():
                            shutil.copy2(label_path, new_label_path)
    
    def generate_synthetic_samples(self):
        """Generate synthetic samples for missing classes"""
        # Use default font since we don't have custom fonts
        font = ImageFont.load_default()
        
        class_name_map = {
            '"': 'DQUOTE',
            "'": 'SQUOTE',
            '*': 'ASTERISK',
            '+': 'PLUS',
            ',': 'COMMA',
            '-': 'HYPHEN',
            '.': 'DOT',
            '/': 'FSLASH',
            ':': 'COLON',
            ';': 'SEMICOLON',
            '<': 'LT',
            '=': 'EQUALS',
            '>': 'GT',
            '?': 'QMARK',
            '[': 'LBRACKET',
            '\\': 'BSLASH',
            ']': 'RBRACKET',
            '^': 'CARET',
            '_': 'UNDERSCORE',
            '`': 'BACKTICK',
            '{': 'LCURLY',
            '|': 'PIPE',
            '}': 'RCURLY',
            '~': 'TILDE'
        }
        
        # Create reverse mapping for label creation
        reverse_map = {v: k for k, v in class_name_map.items()}
        
        # Generate samples for missing classes
        for class_name in self.training_focus['focus_classes']['new_classes']:
            samples_needed = self.target_samples
            
            for i in range(samples_needed):
                # Create image
                img = Image.new('RGB', (64, 64), color='white')
                draw = ImageDraw.Draw(img)
                
                # Position the character
                x = random.randint(16, 32)
                y = random.randint(16, 32)
                
                # Draw character
                draw.text((x, y), class_name, fill='black', font=font)
                
                # Apply random rotation
                angle = random.uniform(-15, 15)
                img = img.rotate(angle, expand=False, fillcolor='white')
                
                # Use mapped name for file system operations
                safe_name = class_name_map.get(class_name, class_name)
                img_name = f"{safe_name}_{i:03d}.png"
                label_name = f"{safe_name}_{i:03d}.txt"
                
                # Split between train/val/test (80/10/10)
                if i < samples_needed * 0.8:
                    split = 'train'
                elif i < samples_needed * 0.9:
                    split = 'val'
                else:
                    split = 'test'
                
                img_path = self.balanced_dir / "images" / split / img_name
                label_path = self.balanced_dir / "labels" / split / label_name
                
                # Ensure parent directories exist
                img_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                
                img.save(str(img_path))  # Convert path to string explicitly
                
                # Create YOLO format label
                # Center point is (0.5, 0.5), width and height are relative to image size
                with open(label_path, 'w') as f:
                    class_idx = self.config['dataset']['class_names'].index(class_name)
                    f.write(f"{class_idx} 0.5 0.5 0.5 0.5\n")
    
    def update_data_yaml(self):
        """Update data.yaml for the balanced dataset"""
        data_yaml = {
            'path': str(self.balanced_dir),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': self.config['dataset']['class_names']
        }
        
        with open(self.balanced_dir / "data.yaml", 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
    
    def create_early_stopping_config(self):
        """Create configuration for early stopping based on class performance"""
        early_stop_config = {
            'class_targets': {},
            'evaluation_frequency': 2,  # Evaluate every N epochs
            'patience': 5,  # Number of evaluations without improvement before stopping
            'min_epochs': 10,  # Minimum number of epochs to train
            'max_epochs': self.training_focus['training_parameters']['epochs'],
            'target_metrics': {
                'precision': 0.95,  # Target precision for well-represented classes
                'recall': 0.95,     # Target recall for well-represented classes
                'f1': 0.95          # Target F1 score for well-represented classes
            }
        }
        
        # Set different targets for different class categories
        for class_name in self.config['dataset']['class_names']:
            if class_name in [str(i) for i in range(10)]:  # Numbers
                early_stop_config['class_targets'][class_name] = {
                    'precision': 0.98,
                    'recall': 0.98,
                    'f1': 0.98
                }
            elif class_name in self.training_focus['focus_classes']['new_classes']:
                early_stop_config['class_targets'][class_name] = {
                    'precision': 0.90,
                    'recall': 0.90,
                    'f1': 0.90
                }
            else:
                early_stop_config['class_targets'][class_name] = {
                    'precision': 0.95,
                    'recall': 0.95,
                    'f1': 0.95
                }
        
        # Save early stopping configuration
        with open(self.balanced_dir / "early_stopping.json", 'w') as f:
            json.dump(early_stop_config, f, indent=2)
    
    def balance_dataset(self):
        """Main method to balance dataset and prepare for training"""
        print("Setting up balanced dataset directory...")
        self.setup_balanced_dataset()
        
        print("\nReducing overrepresented classes...")
        self.reduce_overrepresented_classes()
        
        print("\nGenerating synthetic samples for missing classes...")
        self.generate_synthetic_samples()
        
        print("\nUpdating data.yaml...")
        self.update_data_yaml()
        
        print("\nCreating early stopping configuration...")
        self.create_early_stopping_config()
        
        print("\nDataset balancing complete!")

if __name__ == "__main__":
    balancer = DatasetBalancer()
    balancer.balance_dataset()