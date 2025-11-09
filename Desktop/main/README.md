# Custom PyTorch Object Detection Project

This project implements a custom object detection pipeline using PyTorch, featuring a ResNet-based detection model trained on synthetic datasets for OCR and industrial monitoring applications.

## Features

- **Custom PyTorch Dataset**: Efficient data loading from YOLO-format annotations
- **Custom Data Loader**: PyTorch DataLoader with augmentations and batching
- **Custom Detection Model**: ResNet50 backbone with YOLO-style detection heads
- **Training Pipeline**: Complete training loop with validation and metrics
- **Testing Framework**: Comprehensive evaluation with mAP, precision, recall
- **Inference Engine**: Real-time object detection on images
- **Standard Code Format**: PEP8 compliant, well-documented code

## Installation

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
4. Install dependencies: `pip install -r requirements.txt`

## Quick Start

```bash
# Activate virtual environment
venv\Scripts\activate

# Train the model
python scripts/train.py --config configs/config.yaml --data_dir data/synthetic_dataset

# Test the model
python scripts/test.py --config configs/config.yaml --data_dir data/synthetic_dataset --checkpoint checkpoints/best_model.pth

# Run inference
python scripts/infer.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth --image sample_image.jpg
```

## Dependencies

The project requires the following Python packages:

- torch>=1.9.0
- torchvision>=0.10.0
- numpy>=1.21.0
- pillow>=8.0.0
- opencv-python>=4.5.0
- matplotlib>=3.4.0
- tqdm>=4.62.0
- pyyaml>=5.4.0
- tensorboard>=2.7.0

## Project Structure
```text
├── data/                    # Dataset directory
│   ├── synthetic_dataset/   # Main dataset
│   │   ├── images/         # Train/val/test images
│   │   ├── labels/         # YOLO format annotations
│   │   └── data.yaml       # Dataset configuration
│   └── custom_dataset.py   # PyTorch Dataset class
├── models/                  # Model definitions
│   └── custom_model.py     # Detection model with loss
├── utils/                   # Utility functions
│   └── data_loader.py      # Data loading utilities
├── scripts/                 # Executable scripts
│   ├── train.py            # Training script
│   ├── test.py             # Testing script
│   └── infer.py            # Inference script
├── configs/                 # Configuration files
│   └── config.yaml         # Training hyperparameters
├── logs/                    # Training logs and TensorBoard
├── checkpoints/             # Model checkpoints
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Usage

### Training

1. Prepare your dataset in YOLO format
2. Configure training parameters in `configs/config.yaml`
3. Run training:

```bash
python scripts/train.py --config configs/config.yaml --data_dir data/synthetic_dataset
```

### Testing

Evaluate trained model on test set:

```bash
python scripts/test.py --config configs/config.yaml --data_dir data/synthetic_dataset --checkpoint checkpoints/best_model.pth --output_dir test_results
```

### Inference

Run detection on new images:

```bash
python scripts/infer.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth --image path/to/image.jpg --output_dir inference_results
```

## Configuration

Edit `configs/config.yaml` to adjust:

- Training parameters (epochs, batch size, learning rate)
- Model architecture (backbone, image size)
- Data loading (workers, augmentations)
- Loss weights and thresholds

## Metrics

The project tracks:

- **Training**: Loss components (objectness, localization, classification)
- **Validation**: mAP@0.5, precision, recall, F1-score
- **Testing**: Comprehensive evaluation metrics

## Dataset Format

Expected YOLO format:

- Images: `data/synthetic_dataset/images/train/`, `val/`, `test/`
- Labels: `data/synthetic_dataset/labels/train/`, `val/`, `test/`
- Config: `data/synthetic_dataset/data.yaml` with class names

## Model Architecture

- **Backbone**: ResNet50 (pretrained on ImageNet)
- **Detection Head**: YOLO-style with classification, regression, and objectness branches
- **Loss Function**: Custom YOLO loss with coordinate and confidence terms
- **Output**: Bounding boxes, class probabilities, confidence scores

## Contributing

1. Follow PEP8 coding standards
2. Add docstrings to all functions
3. Test changes thoroughly
4. Update documentation

## License

MIT License - see LICENSE file for details
