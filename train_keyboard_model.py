from ultralytics import YOLO
import yaml
import logging
from pathlib import Path
import torch
import shutil
import os

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def backup_weights(source_dir, backup_dir):
    """Backup existing weights before training"""
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(exist_ok=True)
    
    # Backup best and last weights if they exist
    for weight_file in ['best.pt', 'last.pt']:
        source_path = Path(source_dir) / weight_file
        if source_path.exists():
            shutil.copy2(source_path, backup_dir / f"{weight_file}.backup")

def train_model(logger):
    """Train the YOLO model with the new configuration"""
    try:
        # Load configurations
        with open('model_config.yaml', 'r') as f:
            model_config = yaml.safe_load(f)
        
        # Initialize model
        model = YOLO('yolov8n.pt')
        
        # Backup existing weights
        backup_weights('runs/train/weights', 'weights_backup')
        
        # Configure training parameters
        train_args = {
            'data': 'data.yaml',
            'epochs': model_config['training_config']['epochs'],
            'batch': model_config['training_config']['batch_size'],
            'imgsz': model_config['model_config']['input_size'],
            'patience': model_config['callbacks']['early_stopping']['patience'],
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'workers': os.cpu_count(),
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'auto',
            'verbose': True,
            'seed': 42,
            'save': True,
            'save_period': -1,  # Save on best validation only
        }
        
        # Start training
        logger.info("Starting model training...")
        results = model.train(**train_args)
        
        # Log training results
        logger.info("Training completed. Results:")
        logger.info(f"Best mAP50: {results.best_fitness:.4f}")
        logger.info(f"Final epoch: {results.epoch}")
        
        # Export model in different formats
        logger.info("Exporting model...")
        for export_format in model_config['export']['formats']:
            model.export(format=export_format, dynamic=model_config['export']['dynamic_batch'])
        
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return False

def validate_model(logger):
    """Validate the trained model"""
    try:
        # Load best weights
        model = YOLO('runs/train/weights/best.pt')
        
        # Run validation
        logger.info("Running validation...")
        results = model.val(
            data='data.yaml',
            split='val',
            conf=0.25,
            iou=0.45,
            max_det=300,
            device=0 if torch.cuda.is_available() else 'cpu'
        )
        
        # Log validation results
        logger.info("Validation Results:")
        logger.info(f"mAP50: {results.box.map50:.4f}")
        logger.info(f"mAP50-95: {results.box.map:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        return None

if __name__ == "__main__":
    logger = setup_logging()
    
    logger.info("Starting keyboard detection training pipeline...")
    
    if train_model(logger):
        logger.info("Training completed successfully")
        validate_model(logger)
    else:
        logger.error("Training failed")