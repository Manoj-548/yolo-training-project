# Universal ML/AI Project Setup

A robust, portable setup and recovery system for any ML/AI project. Supports OCR, object detection, segmentation, and other ML projects with automatic dependency management and cloud recovery.

## Features

- **Auto-Project Detection**: Automatically detects project type (OCR, object detection, segmentation, etc.)
- **Universal Dependencies**: Installs required packages based on detected project type
- **Cloud Recovery**: Automatic recovery from OneDrive/Microsoft account or Google Drive
- **Cross-Platform**: Works on any machine with Git cloning
- **Zero-Config Setup**: One-command setup for any new machine
- **Project-Specific Configs**: Tailored configurations for different ML tasks

## Supported Project Types

- **OCR Projects**: Text detection and recognition
- **Object Detection**: YOLO-based detection pipelines
- **Segmentation**: Image segmentation tasks
- **NumPy Training**: Traditional ML with scikit-learn
- **Generic ML**: Custom ML projects

## Quick Start

### First-Time Setup (on any machine)

1. **Clone any ML repository**:

   ```bash
   git clone https://github.com/your-username/any-ml-project.git
   cd any-ml-project
   ```

2. **Run the universal setup**:

   ```bash
   python universal_setup.py --cloud-recover
   ```

   This automatically:
   - Detects project type
   - Installs required dependencies
   - Downloads default model weights
   - Recovers data from OneDrive or Google Drive
   - Creates sample data

3. **Test the setup**:

   ```bash
   # For OCR projects
   python ocr_inspect.py --source images/test/ --use_ocr

   # For other projects
   python ../../main.py --data data.yaml --source images/test/
   ```

## Usage Examples

### OCR Projects

```bash
python universal_setup.py --cloud-recover
python ocr_inspect.py --source "image.jpg" --use_ocr --output results.json
```

### Object Detection

```bash
python universal_setup.py --cloud-recover
python ../../main.py --data data.yaml --source images/ --weights yolov8n.pt
```

### Segmentation

```bash
python universal_setup.py --cloud-recover
python segmentation_train.py --data data.yaml
```

### Custom Project Type

```bash
python universal_setup.py --project-type custom --cloud-recover
```

## Project Structure

```bash
any-ml-project/
├── universal_setup.py        # Universal setup script
├── project_config.yaml       # Auto-generated project config
├── data.yaml                # Dataset configuration
├── images/                  # Image datasets
├── runs/                   # Training outputs
├── [project-specific files] # OCR, segmentation, etc.
└── README.md               # This documentation
```

## Configuration

### Automatic Configuration (`project_config.yaml`)

The setup script creates project-specific configurations:

**OCR Projects:**

```yaml
project_type: ocr
ocr:
  enabled: true
  engine: easyocr
dataset:
  classes: 62
  class_names: ["0", "1", ..., "z"]
```

**Object Detection:**

```yaml
project_type: object_detection
model:
  path: runs/train/weights/best.pt
dataset:
  classes: 80
```

**Segmentation:**

```yaml
project_type: segmentation
segmentation:
  model_type: unet
  encoder: resnet34
```

## Cloud Recovery

### OneDrive Integration

- **Automatic Detection**: Scans standard OneDrive locations
- **Smart Recovery**: Recovers weights, datasets, configs
- **Project-Specific**: Different backup paths per project type

### OneDrive Configuration

```yaml
recovery:
  cloud_backup: "onedrive"
  onedrive_path: "Documents/ML_Projects_Backup"
```

### Google Drive Integration

- **Automatic Detection**: Scans standard Google Drive locations
- **Smart Recovery**: Recovers weights, datasets, configs
- **Project-Specific**: Different backup paths per project type

### Google Drive Configuration

```yaml
recovery:
  cloud_backup: "google_drive"
  google_drive_path: "My Drive/ML_Projects_Backup"
```

## Dependencies (Auto-Installed)

**Base Packages:**

- ultralytics, opencv-python, numpy, torch, torchvision, Pillow, pyyaml

**OCR Projects:**

- easyocr, pytesseract

**Segmentation:**

- segmentation-models-pytorch, albumentations

**NumPy Training:**

- scikit-learn, pandas, matplotlib

## Data Recovery Workflow

1. **Clone Repository**: `git clone <repo>`
2. **Run Setup**: `python universal_setup.py --cloud-recover`
3. **Auto-Detection**: Script identifies project type
4. **Dependency Install**: Missing packages installed
5. **Cloud Recovery**: OneDrive or Google Drive data restored
6. **Ready to Use**: Project fully functional

## Command Line Options

```bash
python universal_setup.py [options]

Options:
  --cloud-recover    Attempt cloud recovery from OneDrive
  --force           Force re-setup even if files exist
  --project-type    Override auto-detected project type
                     (ocr, object_detection, segmentation, numpy_training, generic_ml)
```

## Troubleshooting

### Common Issues

1. **Wrong Project Type**: Use `--project-type` to override
2. **Missing OneDrive**: Setup downloads default weights
3. **Dependency Errors**: Re-run setup script
4. **Permission Issues**: Ensure write access to project directory

### Logs

All operations logged with `[SETUP]` prefix for easy debugging.

## For Project Maintainers

### Adding New Project Types

1. Update `detect_project_type()` function
2. Add dependencies in `get_project_dependencies()`
3. Add config template in `setup_project_config()`
4. Update recovery logic in `recover_from_cloud()`

### Custom Recovery

Modify `recover_from_cloud()` to support additional cloud services (GitHub, Azure, etc.).

## Contributing

1. Fork and clone
2. Run `python universal_setup.py --cloud-recover`
3. Make changes
4. Test on different project types
5. Submit pull request

## License

MIT License - see LICENSE file for details.

## Support

- Run setup with `--help` for options
- Check logs for detailed error information
- Ensure OneDrive or Google Drive sync for cloud recovery
- Test with sample data first
