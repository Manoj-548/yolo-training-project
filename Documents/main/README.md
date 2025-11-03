# OCR Model Project

This project implements an OCR (Optical Character Recognition) model using Detectron2 for detecting digits (0-9) in images. It uses a Faster R-CNN model trained on a custom dataset from Roboflow.

## Quick Start

### Setup Development Environment

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd main
   python setup.py
   ```

2. **Download dataset from Roboflow** and place it in `./roboflow_data` with the following structure:
   ```
   roboflow_data/
   ├── train/
   │   ├── _annotations.coco.json
   │   └── images...
   └── valid/
       ├── _annotations.coco.json
       └── images...
   ```

### IDE Setup

#### VS Code
- Open the project: `code .`
- VS Code will automatically detect the Python interpreter in `./venv/Scripts/python.exe`
- Use the integrated debugger with pre-configured launch configurations:
  - **OCR Training**: Trains the model with your dataset
  - **OCR Inference**: Runs inference on a test image
- Use built-in tasks (Ctrl+Shift+P → Tasks):
  - **Install Dependencies**: Installs all required packages
  - **Run Training**: Starts training process
  - **Run Linting**: Checks code quality

#### PyCharm
- Open this directory as a project
- PyCharm will automatically detect the virtual environment
- Pre-configured run configurations are available:
  - **OCR Training**: Trains the model
  - **OCR Inference**: Runs inference on test images
- Code inspection and formatting are pre-configured

## Prerequisites

- Python 3.x
- PyTorch
- Detectron2
- OpenCV
- Roboflow dataset

## Manual Installation

If you prefer manual setup:

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Unix-like
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch (compatible version for Detectron2):**
   ```bash
   pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
   ```

4. **Install Detectron2:**
   ```bash
   pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```

## Usage

### Training the Model

**VS Code:** Use the "OCR Training" launch configuration or run the task "Run Training"

**PyCharm:** Use the "OCR Training" run configuration

**Command Line:**
```bash
python ocr.py --mode train --data_path ./roboflow_data
```

### Running Inference

**VS Code:** Use the "OCR Inference" launch configuration (will prompt for image path)

**PyCharm:** Use the "OCR Inference" run configuration (modify image path as needed)

**Command Line:**
```bash
python ocr.py --mode infer --image_path path/to/your/image.jpg --weights output/model_final.pth
```

### Command Line Arguments

- `--mode`: Choose 'train' or 'infer' (default: train)
- `--data_path`: Path to the dataset directory (default: ./roboflow_data)
- `--image_path`: Path to the image for inference (required for infer mode)
- `--weights`: Path to the trained model weights (default: output/model_final.pth)

## Project Structure

```
main/
├── ocr.py                 # Main OCR script
├── requirements.txt        # Python dependencies
├── setup.py              # Setup script for development environment
├── README.md             # This file
├── .gitignore            # Git ignore file
├── .vscode/              # VS Code configuration
│   ├── settings.json     # VS Code settings
│   ├── launch.json       # Debug configurations
│   ├── tasks.json        # Build tasks
│   └── extensions.json   # Recommended extensions
├── .idea/                # PyCharm configuration
│   ├── misc.xml          # Project settings
│   ├── modules.xml       # Module configuration
│   ├── vcs.xml           # Version control
│   └── runConfigurations/ # Run configurations
├── venv/                 # Virtual environment (gitignored)
├── output/               # Model outputs (gitignored)
├── roboflow_data/        # Dataset (gitignored)
└── logs/                 # Log files (gitignored)
```

## Output

- **Training**: Saves model weights in `output/` directory
- **Inference**: Displays the image with detected digits highlighted

## Configuration

Adjust hyperparameters in `ocr.py`:
- `NUM_CLASSES`: Number of classes (10 for digits)
- `MAX_ITER`: Number of training iterations
- `IMS_PER_BATCH`: Batch size

## Development

### Code Quality

- **Linting**: Pylint is configured for both IDEs
- **Formatting**: Black formatter is configured (line length: 88)
- **Type Checking**: Basic type checking is enabled

### Git Workflow

1. Create feature branches: `git checkout -b feature-name`
2. Make changes and commit: `git add . && git commit -m "Description"`
3. Push and create pull request

### Testing

Run linting:
```bash
pylint ocr.py
```

## Troubleshooting

### Common Issues

1. **CUDA errors**: Ensure you have the correct CUDA version installed
2. **Detectron2 installation**: Make sure PyTorch is installed first
3. **Memory issues**: Reduce `IMS_PER_BATCH` in the configuration
4. **Dataset not found**: Ensure `roboflow_data` directory exists with correct structure

### Getting Help

- Check the logs in the `output/` directory
- Verify your dataset structure matches the expected format
- Ensure all dependencies are properly installed