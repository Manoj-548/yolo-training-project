from ultralytics import YOLO
import yaml
import logging
from pathlib import Path
import torch
import os
from datetime import datetime
import json
import psutil
import time

def setup_logging():
    log_file = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_system_resources():
    """Get current system resource usage"""
    resources = {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'gpu_memory_used': 0
    }
    
    # Only check GPU if available
    if torch.cuda.is_available():
        try:
            resources['gpu_memory_used'] = torch.cuda.memory_allocated()
        except:
            pass
            
    return resources

def background_train():
    logger = setup_logging()
    logger.info("Starting background training with optimized dataset")
    
    try:
        # Load existing training progress if available
        progress_file = Path("training_progress.json")
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        else:
            progress = {'completed_epochs': 0, 'best_map50': 0}
        
        # Configure model for CPU training
        model = YOLO('yolov8n.pt')  # Using nano model for CPU training
        
        # Training arguments
        train_args = {
            'data': 'data.yaml',
            'epochs': 150,
            'imgsz': 640,  # Reduced image size for CPU
            'batch': 8,    # Smaller batch size for CPU
            'patience': 20,
            'device': 'cpu',  # Force CPU training
            'workers': max(1, os.cpu_count() // 2),  # Use half of available CPU cores
            'exist_ok': True,
            'pretrained': True,
            'resume': True if progress['completed_epochs'] > 0 else False,
            'amp': True,  # Use mixed precision training
            'optimizer': 'auto',
            'project': 'background_training'
        }
        
        # Start training
        logger.info(f"Resuming from epoch {progress['completed_epochs']}" if progress['completed_epochs'] > 0 else "Starting new training")
        
        while True:
            # Check system resources
            resources = get_system_resources()
            if resources['cpu_percent'] > 90 or resources['memory_percent'] > 90:
                logger.info("System resources high, waiting...")
                time.sleep(300)  # Wait 5 minutes
                continue
            
            # Train for one epoch
            results = model.train(**train_args)
            
            # Update progress
            progress['completed_epochs'] = results.epoch
            if hasattr(results, 'maps'):
                current_map50 = results.maps.get('map50', 0)
                progress['best_map50'] = max(progress['best_map50'], current_map50)
            
            # Save progress
            with open(progress_file, 'w') as f:
                json.dump(progress, f)
            
            logger.info(f"Completed epoch {progress['completed_epochs']}, best mAP50: {progress['best_map50']:.4f}")
            
            # Check if training is complete
            if progress['completed_epochs'] >= train_args['epochs']:
                logger.info("Training completed!")
                break
            
            # Add a small delay to prevent system overload
            time.sleep(10)
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    background_train()