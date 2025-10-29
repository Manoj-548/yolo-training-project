import os
import yaml
import shutil
from pathlib import Path
import time
import random

class LabelFixer:
    def __init__(self, config_path="project_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.valid_chars = set(self.config['dataset']['class_names'])
        # Add common number characters
        self.valid_chars.update(set('0123456789'))
        
    def is_valid_char(self, char):
        return char in self.valid_chars
        
    def safe_copy(self, src, dst, max_retries=3, base_delay=1):
        """Copy a file with retry mechanism"""
        for attempt in range(max_retries):
            try:
                # If destination exists, try to remove it first
                if Path(dst).exists():
                    try:
                        os.remove(dst)
                    except:
                        pass
                
                # Use low-level file operations instead of shutil
                with open(src, 'rb') as fsrc:
                    with open(dst, 'wb') as fdst:
                        while True:
                            chunk = fsrc.read(65536)  # 64KB chunks
                            if not chunk:
                                break
                            fdst.write(chunk)
                return True
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to copy {src} to {dst} after {max_retries} attempts: {str(e)}")
                    return False
                delay = (base_delay * (attempt + 1)) + random.uniform(0, 1)
                time.sleep(delay)
        return False

    def safe_remove(self, path, max_retries=3, base_delay=1):
        """Remove a file with retry mechanism"""
        path = Path(path)
        if not path.exists():
            return True
            
        for attempt in range(max_retries):
            try:
                path.unlink()
                return True
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to remove {path} after {max_retries} attempts: {str(e)}")
                    return False
                delay = (base_delay * (attempt + 1)) + random.uniform(0, 1)
                time.sleep(delay)
        return False

    def safe_copytree(self, src, dst, max_retries=3):
        """Create directory tree copy with retry mechanism"""
        src = Path(src)
        dst = Path(dst)
        
        if dst.exists():
            print(f"Backup directory {dst} already exists, skipping...")
            return True
            
        try:
            dst.mkdir(parents=True, exist_ok=True)
            for item in src.glob('*'):
                d = dst / item.name
                if item.is_dir():
                    self.safe_copytree(item, d)
                else:
                    self.safe_copy(item, d)
            return True
        except Exception as e:
            print(f"Error creating backup directory {dst}: {str(e)}")
            return False
        
    def fix_labels(self, base_path):
        """Fix label names to match the 92 character set"""
        for split in ['train', 'val', 'test']:
            img_dir = Path(base_path) / 'images' / split
            label_dir = Path(base_path) / 'labels' / split
            
            if not img_dir.exists() or not label_dir.exists():
                print(f"Skipping {split} - directory not found")
                continue
                
            print(f"\nProcessing {split} split...")
            
            # Create backup directories with retry mechanism
            backup_img = img_dir.parent / f"{split}_backup_imgs"
            backup_label = label_dir.parent / f"{split}_backup_labels"
            
            if not backup_img.exists():
                print(f"Creating backup of {split} images...")
                if not self.safe_copytree(img_dir, backup_img):
                    print(f"Failed to create backup for {split} images, skipping this split")
                    continue
                    
            if not backup_label.exists():
                print(f"Creating backup of {split} labels...")
                if not self.safe_copytree(label_dir, backup_label):
                    print(f"Failed to create backup for {split} labels, skipping this split")
                    continue
            
            # Process each image file
            file_count = 0
            success_count = 0
            
            for img_file in sorted(img_dir.glob('*.*')):  # Sort to process in consistent order
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    base_name = img_file.stem
                    
                    # Skip if the file is already in the correct format (char_number.jpg)
                    parts = base_name.split('_')
                    if len(parts) == 2 and len(parts[0]) == 1 and self.is_valid_char(parts[0]) and parts[1].isdigit():
                        continue
                    
                    # Handle single-character filenames without numbering
                    if len(base_name) == 1 and self.is_valid_char(base_name):
                        valid_chars = [base_name]
                    # Handle numeric filenames
                    elif base_name.isdigit():
                        valid_chars = [base_name[0]]  # Take first digit only
                    else:
                        # Handle multi-character filenames
                        chars = list(base_name)
                        valid_chars = [c for c in chars if self.is_valid_char(c)]
                    
                    if not valid_chars:
                        print(f"Warning: No valid characters in {img_file}")
                        continue
                    
                    file_count += 1
                    success = True
                    
                    # Create new files for each valid character
                    for i, char in enumerate(valid_chars):
                        new_img_name = f"{char}_{i}{img_file.suffix}"
                        new_label_name = f"{char}_{i}.txt"
                        
                        # Copy image with retry
                        if not self.safe_copy(img_file, img_dir / new_img_name):
                            success = False
                            continue
                            
                        # Copy label if exists
                        label_file = label_dir / f"{base_name}.txt"
                        if label_file.exists():
                            if not self.safe_copy(label_file, label_dir / new_label_name):
                                success = False
                                continue
                    
                    # Only remove original files if all copies succeeded
                    if success:
                        success_count += 1
                        self.safe_remove(img_file)
                        label_file = label_dir / f"{base_name}.txt"
                        if label_file.exists():
                            self.safe_remove(label_file)
            
            print(f"Processed {file_count} files in {split} split (Successfully converted: {success_count})")
                        
    def run(self):
        """Run the label fixing process"""
        print("Starting label fixing process with improved file handling...")
        self.fix_labels('.')
        print("Label fixing completed. Backup copies have been created.")
        print("Please verify the changes before proceeding with training.")

if __name__ == "__main__":
    fixer = LabelFixer()
    fixer.run()