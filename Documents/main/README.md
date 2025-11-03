is# OCR Model Project

This project implements an OCR (Optical Character Recognition) model using Detectron2 for detecting digits (0-9) in images. It uses a Faster R-CNN model trained on a custom dataset from Roboflow.

## Prerequisites

- Python 3.x
- PyTorch
- Detectron2
- OpenCV
- Roboflow dataset (download from Roboflow and place in `./roboflow_data`)

## Installation

1. Install PyTorch (compatible version for Detectron2, e.g., 1.13):
   ```
   pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
   ```
   (Adjust CUDA version if needed, or use CPU-only: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`)

2. Install Detectron2:
   ```
   pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```
   (Ensure PyTorch is installed first, as above. For other installation options, see https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

3. Install OpenCV:
   ```
   pip install opencv-python
   ```

2. Download the dataset from Roboflow and place it in `./roboflow_data` with the following structure:
   ```
   roboflow_data/
   ├── train/
   │   ├── _annotations.coco.json
   │   └── images...
   └── valid/
       ├── _annotations.coco.json
       └── images...
   ```

## Usage

### Training the Model

Run the script in training mode (default):

```
python ocr.py --mode train --data_path ./roboflow_data
```

### Running Inference

Run inference on an image:

```
python ocr.py --mode infer --image_path path/to/your/image.jpg --weights output/model_final.pth
```

### Command Line Arguments

- `--mode`: Choose 'train' or 'infer' (default: train)
- `--data_path`: Path to the dataset directory (default: ./roboflow_data)
- `--image_path`: Path to the image for inference (required for infer mode)
- `--weights`: Path to the trained model weights (default: output/model_final.pth)

## Output

- Training: Saves model weights in `output/` directory
- Inference: Displays the image with detected digits highlighted

## Configuration

Adjust hyperparameters in the script:
- `NUM_CLASSES`: Number of classes (10 for digits)
- `MAX_ITER`: Number of training iterations
- `IMS_PER_BATCH`: Batch size
