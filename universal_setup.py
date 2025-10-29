#!/usr/bin/env python3
"""
Universal Project Setup and Recovery Script
Ensures any ML/AI project is properly configured and recovers missing data.
Designed for portability across different machines with Git cloning.
Supports OCR, object detection, segmentation, and other ML projects.
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

# Default URLs for downloading missing data
DEFAULT_DATA_URLS = {
    'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
    'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
    'sample_dataset.zip': 'https://example.com/sample_dataset.zip',
    'project_config.yaml': None,
}

def log(message):
    """Simple logging function."""
    print(f"[SETUP] {message}")

def detect_project_type():
    """Detect the type of project based on files present."""
    if (PROJECT_ROOT / 'ocr_inspect.py').exists():
        return 'ocr'
    elif (PROJECT_ROOT / 'segmentation_train.py').exists():
        return 'segmentation'
    elif (PROJECT_ROOT / 'train_numpy.py').exists():
        return 'numpy_training'
    elif (PROJECT_ROOT / 'data.yaml').exists() or (PROJECT_ROOT / 'runs').exists():
        return 'object_detection'
    else:
        return 'generic_ml'

def get_project_dependencies(project_type):
    """Get required packages based on project type."""
    base_packages = [
        'ultralytics',
        'opencv-python',
        'numpy',
        'torch',
        'torchvision',
        'Pillow',
        'pyyaml'
    ]

    type_specific_packages = {
        'ocr': ['easyocr', 'pytesseract'],
        'segmentation': ['segmentation-models-pytorch', 'albumentations'],
        'numpy_training': ['scikit-learn', 'pandas', 'matplotlib'],
        'object_detection': [],
        'generic_ml': ['scikit-learn', 'pandas']
    }

    return base_packages + type_specific_packages.get(project_type, [])

def check_dependencies(project_type):
    """Check and install required Python packages based on project type."""
    required_packages = get_project_dependencies(project_type)

    log(f"Checking dependencies for {project_type} project...")
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

def setup_project_config(project_type):
    """Create or update project configuration file based on project type."""
    config_path = PROJECT_ROOT / 'project_config.yaml'

    if not config_path.exists():
        log(f"Creating default configuration for {project_type} project...")

        # Base configuration
        default_config = {
            'project_name': f'{project_type.replace("_", " ").title()}_Project',
            'version': '1.0.0',
            'project_type': project_type,
            'description': f'Flexible {project_type.replace("_", " ")} pipeline',
            'recovery': {
                'auto_download_weights': True,
                'generate_synthetic_data': True,
                'fallback_to_defaults': True,
                'cloud_backup': 'onedrive',
                'onedrive_path': 'Documents/ML_Projects_Backup',
                'github_repo': 'your-username/ml-project'
            },
            'data_sources': {
                'primary': 'local',
                'cloud_backup': 'onedrive',
                'onedrive_path': 'Documents/ML_Projects_Backup',
                'backup_urls': DEFAULT_DATA_URLS
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
                'classes': 80,
                'class_names': [f'class_{i}' for i in range(80)]
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

        # Add project-specific configurations
        if project_type == 'ocr':
            default_config.update({
                'ocr': {
                    'enabled': True,
                    'engine': 'easyocr',
                    'language': 'en',
                    'fallback_engine': 'tesseract',
                    'min_confidence': 0.5
                },
                'dataset': {
                    'classes': 62,
                    'class_names': ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
                },
                'feature_extraction': {
                    'enabled': True,
                    'output_format': 'json',
                    'include_ocr_text': True,
                    'minimal_training_examples': True
                }
            })
        elif project_type == 'segmentation':
            default_config.update({
                'segmentation': {
                    'enabled': True,
                    'model_type': 'unet',
                    'encoder': 'resnet34',
                    'activation': 'sigmoid'
                }
            })
        elif project_type == 'numpy_training':
            default_config.update({
                'numpy_training': {
                    'enabled': True,
                    'framework': 'sklearn',
                    'cross_validation': True,
                    'feature_scaling': True
                }
            })

        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        log(f"Created {config_path}")
    else:
        log(f"Project config already exists at {config_path}")

def setup_weights(project_type):
    """Ensure model weights are available based on project type."""
    weights_dir = PROJECT_ROOT / 'runs' / 'train' / 'weights'
    ensure_directory(weights_dir)

    best_weights = weights_dir / 'best.pt'
    last_weights = weights_dir / 'last.pt'

    if not best_weights.exists() and not last_weights.exists():
        log("No trained weights found. Downloading default weights...")

        if project_type in ['ocr', 'object_detection']:
            default_weights_url = DEFAULT_DATA_URLS['yolov8n.pt']
            default_weights = PROJECT_ROOT / 'yolov8n.pt'
        else:
            default_weights_url = DEFAULT_DATA_URLS['yolov8s.pt']
            default_weights = PROJECT_ROOT / 'yolov8s.pt'

        if not default_weights.exists():
            download_file(default_weights_url, default_weights)

        shutil.copy(default_weights, best_weights)
        shutil.copy(default_weights, last_weights)
        log("Default weights set up.")
    else:
        log("Weights already exist.")

def setup_sample_data(project_type):
    """Set up sample dataset based on project type."""
    images_dir = PROJECT_ROOT / 'images'
    ensure_directory(images_dir)

    if not list(images_dir.glob('*.jpg')) and not list(images_dir.glob('*.png')):
        log(f"No sample images found. Creating synthetic sample data for {project_type}...")

        import cv2
        import numpy as np

        if project_type == 'ocr':
            sample_img = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.putText(sample_img, "SAMPLE OCR TEXT 123", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.imwrite(str(images_dir / 'sample_ocr.jpg'), sample_img)
        else:
            sample_img = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.rectangle(sample_img, (100, 100), (200, 200), (255, 0, 0), 3)
            cv2.rectangle(sample_img, (300, 300), (400, 400), (0, 255, 0), 3)
            cv2.imwrite(str(images_dir / 'sample_objects.jpg'), sample_img)

        log("Created sample image for testing.")
    else:
        log("Sample data already exists.")

def setup_data_yaml(project_type):
    """Create data.yaml based on project type."""
    data_yaml_path = PROJECT_ROOT / 'data.yaml'

    if not data_yaml_path.exists():
        log(f"Creating default data.yaml for {project_type}...")

        if project_type == 'ocr':
            data_config = {
                'train': 'images/train',
                'val': 'images/val',
                'nc': 62,
                'names': ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
            }
        else:
            data_config = {
                'train': 'images/train',
                'val': 'images/val',
                'nc': 80,
                'names': [f'class_{i}' for i in range(80)]
            }

        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f)
        log(f"Created {data_yaml_path}")
    else:
        log("data.yaml already exists.")

def recover_from_cloud(project_type):
    """Attempt to recover data from cloud storage based on project type."""
    log("Attempting cloud recovery...")

    config_path = PROJECT_ROOT / 'project_config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        cloud_backup = config.get('recovery', {}).get('cloud_backup', 'onedrive')
        onedrive_path = config.get('recovery', {}).get('onedrive_path', f'Documents/{project_type.title()}_Projects_Backup')

        if cloud_backup == 'onedrive':
            log(f"Attempting OneDrive recovery from {onedrive_path}")
            possible_onedrive_paths = [
                Path.home() / 'OneDrive' / onedrive_path,
                Path.home() / 'OneDrive - Personal' / onedrive_path,
                Path.home() / 'OneDrive - Work' / onedrive_path,
                Path('C:/Users') / os.getlogin() / 'OneDrive' / onedrive_path,
            ]

            for onedrive_dir in possible_onedrive_paths:
                if onedrive_dir.exists():
                    log(f"Found OneDrive backup at {onedrive_dir}")

                    # Recover project-specific data
                    if project_type == 'ocr':
                        ocr_files = ['dataset_features.json', 'inference_results.json']
                        for ocr_file in ocr_files:
                            backup_file = onedrive_dir / ocr_file
                            if backup_file.exists():
                                shutil.copy(backup_file, PROJECT_ROOT / ocr_file)
                                log(f"Recovered {ocr_file}")

                    # Recover common files
                    common_files = ['data.yaml', 'project_config.yaml']
                    for common_file in common_files:
                        backup_file = onedrive_dir / common_file
                        if backup_file.exists():
                            shutil.copy(backup_file, PROJECT_ROOT / common_file)
                            log(f"Recovered {common_file}")

                    # Recover weights
                    backup_weights = onedrive_dir / 'weights'
                    if backup_weights.exists():
                        weights_dir = PROJECT_ROOT / 'runs' / 'train' / 'weights'
                        ensure_directory(weights_dir)
                        for weight_file in backup_weights.glob('*.pt'):
                            shutil.copy(weight_file, weights_dir / weight_file.name)
                            log(f"Recovered weight file: {weight_file.name}")

                    # Recover datasets
                    backup_images = onedrive_dir / 'images'
                    if backup_images.exists():
                        images_dir = PROJECT_ROOT / 'images'
                        if not images_dir.exists():
                            shutil.copytree(backup_images, images_dir)
                            log("Recovered images directory")

                    log("Cloud recovery completed successfully!")
                    return

            log("OneDrive backup not found in standard locations.")
        else:
            log(f"Cloud backup type '{cloud_backup}' not yet implemented.")
    else:
        log("No project config found for cloud recovery settings.")

def main():
    parser = argparse.ArgumentParser(description="Set up and recover ML/AI project")
    parser.add_argument('--force', action='store_true', help="Force re-setup even if files exist")
    parser.add_argument('--cloud-recover', action='store_true', help="Attempt cloud recovery")
    parser.add_argument('--project-type', type=str, help="Override auto-detected project type")
    args = parser.parse_args()

    # Detect project type
    project_type = args.project_type or detect_project_type()
    log(f"Detected project type: {project_type}")

    log("Starting project setup and recovery...")

    # Check dependencies
    check_dependencies(project_type)

    # Set up configuration
    setup_project_config(project_type)

    # Set up weights
    setup_weights(project_type)

    # Set up sample data
    setup_sample_data(project_type)

    # Set up data.yaml
    setup_data_yaml(project_type)

    # Cloud recovery if requested
    if args.cloud_recover:
        recover_from_cloud(project_type)

    log("Project setup complete!")
    if project_type == 'ocr':
        log("You can now run: python ocr_inspect.py --source images/test/ --use_ocr")
    else:
        log("You can now run: python ../../main.py --data data.yaml --source images/test/")

if __name__ == '__main__':
    main()
