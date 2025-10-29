#!/usr/bin/env python3
"""
Project Setup and Recovery Script
Ensures the OCR inspection project is properly configured and recovers missing data.
Designed for portability across different machines with Git cloning.
"""

import os
import sys
import yaml
import json
import shutil
import requests
from pathlib import Path
import subprocess
import argparse

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()

# Default URLs for downloading missing data (replace with your actual URLs)
DEFAULT_DATA_URLS = {
    'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
    'sample_dataset.zip': 'https://example.com/sample_ocr_dataset.zip',  # Replace with actual URL
    'project_config.yaml': None,  # Will be created locally
}

def log(message):
    """Simple logging function."""
    print(f"[SETUP] {message}")

def check_dependencies():
    """Check and install required Python packages."""
    required_packages = [
        'ultralytics',
        'opencv-python',
        'numpy',
        'torch',
        'torchvision',
        'easyocr',
        'pytesseract',
        'Pillow',
        'pyyaml'
    ]

    log("Checking Python dependencies...")
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        log(f"Installing missing packages: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
    else:
        log("All dependencies are installed.")

def download_file(url, dest_path):
    """Download a file from URL to destination path."""
    log(f"Downloading {url} to {dest_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def ensure_directory(path):
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)

def setup_project_config():
    """Create or update project configuration file."""
    config_path = PROJECT_ROOT / 'project_config.yaml'

    if not config_path.exists():
        log("Creating default project configuration...")
        default_config = {
            'project_name': 'OCR_Inspection_Project',
            'version': '1.0.0',
            'description': 'Flexible OCR inspection and training pipeline',
            'recovery': {
                'auto_download_weights': True,
                'generate_synthetic_data': True,
                'fallback_to_defaults': True
            },
            'data_sources': {
                'primary': 'local',
                'cloud_backup': 'onedrive',  # Can be 'onedrive', 'github', 'azure'
                'onedrive_path': 'Documents/OCR_Project_Backup'
            },
            'ocr': {
                'enabled': True,
                'engine': 'easyocr',
                'language': 'en',
                'fallback_engine': 'tesseract',
                'min_confidence': 0.5
            },
            'model': {
                'path': 'runs/train/weights/best.pt',
                'conf': 0.3,
                'iou': 0.5,
                'show_labels': True,
                'show_conf': True,
                'plots': False
            },
            'dataset': {
                'data_yaml': 'data.yaml',
                'synthetic_generation': True,
                'classes': 62,
                'class_names': [str(i) for i in range(62)]
            },
            'training': {
                'epochs': 50,
                'imgsz': 640,
                'batch': 16,
                'device': 'cpu',
                'patience': 5,
                'resume': True
            },
            'windows': {
                'new_window_for_non_ocr': False,
                'initiate_new_project': False
            },
            'advanced': {
                'torch_weights_only': False,
                'safe_globals': True,
                'logging': True,
                'system_stats': True
            }
        }

        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        log(f"Created {config_path}")
    else:
        log(f"Project config already exists at {config_path}")

def setup_weights():
    """Ensure model weights are available."""
    weights_dir = PROJECT_ROOT / 'runs' / 'train' / 'weights'
    ensure_directory(weights_dir)

    best_weights = weights_dir / 'best.pt'
    last_weights = weights_dir / 'last.pt'

    if not best_weights.exists() and not last_weights.exists():
        log("No trained weights found. Downloading default YOLOv8n weights...")
        default_weights = PROJECT_ROOT / 'yolov8n.pt'
        if not default_weights.exists():
            download_file(DEFAULT_DATA_URLS['yolov8n.pt'], default_weights)

        # Copy to expected locations
        shutil.copy(default_weights, best_weights)
        shutil.copy(default_weights, last_weights)
        log("Default weights set up.")
    else:
        log("Weights already exist.")

def setup_sample_data():
    """Set up sample dataset if none exists."""
    images_dir = PROJECT_ROOT / 'images'
    ensure_directory(images_dir)

    if not list(images_dir.glob('*.jpg')) and not list(images_dir.glob('*.png')):
        log("No sample images found. Creating synthetic sample data...")
        # Generate a simple synthetic image for testing
        import cv2
        import numpy as np

        sample_img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.putText(sample_img, "SAMPLE OCR TEXT", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.imwrite(str(images_dir / 'sample.jpg'), sample_img)
        log("Created sample image for testing.")
    else:
        log("Sample data already exists.")

def setup_data_yaml():
    """Create data.yaml if it doesn't exist."""
    data_yaml_path = PROJECT_ROOT / 'data.yaml'

    if not data_yaml_path.exists():
        log("Creating default data.yaml...")
        data_config = {
            'train': 'images/train',
            'val': 'images/val',
            'nc': 62,
            'names': [str(i) for i in range(62)]
        }

        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f)
        log(f"Created {data_yaml_path}")
    else:
        log("data.yaml already exists.")

def recover_from_cloud():
    """Attempt to recover data from cloud storage (OneDrive, etc.)."""
    log("Attempting cloud recovery...")

    # Load project config to get cloud settings
    config_path = PROJECT_ROOT / 'project_config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        cloud_backup = config.get('recovery', {}).get('cloud_backup', 'onedrive')
        onedrive_path = config.get('recovery', {}).get('onedrive_path', 'Documents/OCR_Project_Backup')

        if cloud_backup == 'onedrive':
            log(f"Attempting OneDrive recovery from {onedrive_path}")
            # Check for OneDrive path (typical Windows locations)
            possible_onedrive_paths = [
                Path.home() / 'OneDrive' / onedrive_path,
                Path.home() / 'OneDrive - Personal' / onedrive_path,
                Path.home() / 'OneDrive - Work' / onedrive_path,
                Path('C:/Users') / os.getlogin() / 'OneDrive' / onedrive_path,
            ]

            for onedrive_dir in possible_onedrive_paths:
                if onedrive_dir.exists():
                    log(f"Found OneDrive backup at {onedrive_dir}")
                    # Copy weights if they exist
                    backup_weights = onedrive_dir / 'weights'
                    if backup_weights.exists():
                        weights_dir = PROJECT_ROOT / 'runs' / 'train' / 'weights'
                        ensure_directory(weights_dir)
                        for weight_file in backup_weights.glob('*.pt'):
                            shutil.copy(weight_file, weights_dir / weight_file.name)
                            log(f"Recovered weight file: {weight_file.name}")

                    # Copy datasets if they exist
                    backup_images = onedrive_dir / 'images'
                    if backup_images.exists():
                        images_dir = PROJECT_ROOT / 'images'
                        if not images_dir.exists():
                            shutil.copytree(backup_images, images_dir)
                            log("Recovered images directory")

                    # Copy config files
                    for config_file in ['data.yaml', 'project_config.yaml']:
                        backup_config = onedrive_dir / config_file
                        if backup_config.exists():
                            shutil.copy(backup_config, PROJECT_ROOT / config_file)
                            log(f"Recovered {config_file}")

                    log("Cloud recovery completed successfully!")
                    return

            log("OneDrive backup not found in standard locations.")
        else:
            log(f"Cloud backup type '{cloud_backup}' not yet implemented.")
    else:
        log("No project config found for cloud recovery settings.")

def main():
    parser = argparse.ArgumentParser(description="Set up and recover OCR inspection project")
    parser.add_argument('--force', action='store_true', help="Force re-setup even if files exist")
    parser.add_argument('--cloud-recover', action='store_true', help="Attempt cloud recovery")
    args = parser.parse_args()

    log("Starting project setup and recovery...")

    # Check dependencies
    check_dependencies()

    # Set up configuration
    setup_project_config()

    # Set up weights
    setup_weights()

    # Set up sample data
    setup_sample_data()

    # Set up data.yaml
    setup_data_yaml()

    # Cloud recovery if requested
    if args.cloud_recover:
        recover_from_cloud()

    log("Project setup complete!")
    log("You can now run: python ocr_inspect.py --source images/test/ --use_ocr")

if __name__ == '__main__':
    main()
