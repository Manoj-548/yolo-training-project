import os
import yaml
import json
from pathlib import Path
from collections import defaultdict
import shutil

class DatasetClassFixer:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.config_file = self.base_path / "project_config.yaml"
        self.data_yaml = self.base_path / "data.yaml"
        self.stats_file = self.base_path / "dataset_stats.json"
        
        # Load configurations
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.valid_classes = set(self.config['dataset']['class_names'])
    
    def analyze_class_coverage(self, stats):
        """Analyze which classes are present and which are missing"""
        present_classes = set(stats['class_distribution'].keys())
        missing_classes = self.valid_classes - present_classes
        
        return {
            'present_classes': sorted(present_classes),
            'missing_classes': sorted(missing_classes),
            'coverage_percentage': (len(present_classes) / len(self.valid_classes)) * 100
        }
        
    def is_valid_class(self, class_name):
        """Check if a class name is valid (single character from our defined set)"""
        return class_name in self.valid_classes
    
    def split_multi_digit_filename(self, filename):
        """Split multi-digit filenames into individual characters"""
        base_name = filename.stem
        class_name = base_name.split('_')[0]
        
        # If it's a multi-digit number, split it
        if len(class_name) > 1 and all(c.isdigit() for c in class_name):
            return [c for c in class_name if self.is_valid_class(c)]
        
        # Return as single character if valid
        return [class_name] if self.is_valid_class(class_name) else []
    
    def fix_dataset(self):
        """Fix dataset by properly handling multi-digit numbers and ensuring valid classes"""
        stats = {
            'processed_files': 0,
            'split_files': 0,
            'invalid_files': 0,
            'class_distribution': defaultdict(int)
        }
        
        # Process each split (train/val/test)
        for split in ['train', 'val', 'test']:
            images_dir = self.base_path / "images" / split
            labels_dir = self.base_path / "labels" / split
            
            if not images_dir.exists():
                continue
            
            for img_path in images_dir.glob('*.*'):
                valid_classes = self.split_multi_digit_filename(img_path)
                
                if not valid_classes:
                    stats['invalid_files'] += 1
                    continue
                
                stats['processed_files'] += 1
                
                # If we need to split the file into multiple characters
                if len(valid_classes) > 1:
                    stats['split_files'] += 1
                    for idx, class_name in enumerate(valid_classes):
                        new_filename = f"{class_name}_{img_path.stem}_{idx}{img_path.suffix}"
                        new_img_path = images_dir / new_filename
                        shutil.copy2(img_path, new_img_path)
                        
                        # Copy and update corresponding label file
                        label_path = labels_dir / f"{img_path.stem}.txt"
                        if label_path.exists():
                            new_label_path = labels_dir / f"{new_filename.rsplit('.', 1)[0]}.txt"
                            shutil.copy2(label_path, new_label_path)
                        
                        stats['class_distribution'][class_name] += 1
                else:
                    # Single valid character
                    stats['class_distribution'][valid_classes[0]] += 1
        
        # Analyze class coverage
        coverage = self.analyze_class_coverage(stats)
        
        # Add missing classes to statistics with count 0
        for missing_class in coverage['missing_classes']:
            stats['class_distribution'][missing_class] = 0
        
        # Save updated statistics
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Update data.yaml with all classes (including missing ones)
        with open(self.data_yaml, 'r') as f:
            data_yaml = yaml.safe_load(f)
        
        data_yaml['names'] = sorted(list(self.valid_classes))  # Include all 92 classes
        
        with open(self.data_yaml, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        # Print detailed summary
        print(f"\nDataset Processing Summary:")
        print(f"Processed Files: {stats['processed_files']}")
        print(f"Split Files: {stats['split_files']}")
        print(f"Invalid Files: {stats['invalid_files']}")
        print(f"\nClass Coverage:")
        print(f"Total Expected Classes: {len(self.valid_classes)}")
        print(f"Classes Present: {len(coverage['present_classes'])}")
        print(f"Classes Missing: {len(coverage['missing_classes'])}")
        print(f"Coverage Percentage: {coverage['coverage_percentage']:.2f}%")
        
        print("\nMissing Classes:")
        for class_name in coverage['missing_classes']:
            print(f"  {class_name}")
        
        print("\nClass Distribution Summary:")
        for class_name, count in sorted(stats['class_distribution'].items()):
            print(f"{class_name}: {count}")

if __name__ == "__main__":
    fixer = DatasetClassFixer()
    fixer.fix_dataset()