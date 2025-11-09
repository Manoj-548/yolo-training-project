import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
import logging
from datetime import datetime
import numpy as np

from ..models.custom_model import CustomDetectionModel, YOLOLoss
from ..utils.data_loader import create_data_loaders


def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def calculate_metrics(predictions, targets, iou_threshold=0.5):
    """
    Calculate mAP and other detection metrics.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        iou_threshold: IoU threshold for true positives

    Returns:
        dict: Dictionary containing metrics
    """
    # Simplified metrics calculation - in practice, you'd use libraries like pycocotools
    # This is a placeholder implementation
    return {
        'mAP': 0.0,  # Mean Average Precision
        'precision': 0.0,
        'recall': 0.0
    }


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, writer):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    total_obj_loss = 0.0
    total_noobj_loss = 0.0
    total_coord_loss = 0.0
    total_cls_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        images = batch['images'].to(device)
        targets = batch['targets']

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['total_loss']

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update running losses
        running_loss += loss.item()
        total_obj_loss += loss_dict['obj_loss'].item()
        total_noobj_loss += loss_dict['noobj_loss'].item()
        total_coord_loss += loss_dict['coord_loss'].item()
        total_cls_loss += loss_dict['cls_loss'].item()

        if batch_idx % 10 == 0:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # Calculate average losses
    avg_loss = running_loss / len(train_loader)
    avg_obj_loss = total_obj_loss / len(train_loader)
    avg_noobj_loss = total_noobj_loss / len(train_loader)
    avg_coord_loss = total_coord_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)

    # Log to TensorBoard
    writer.add_scalar('Loss/train/total', avg_loss, epoch)
    writer.add_scalar('Loss/train/obj', avg_obj_loss, epoch)
    writer.add_scalar('Loss/train/noobj', avg_noobj_loss, epoch)
    writer.add_scalar('Loss/train/coord', avg_coord_loss, epoch)
    writer.add_scalar('Loss/train/cls', avg_cls_loss, epoch)

    return avg_loss


def validate_epoch(model, val_loader, criterion, device, epoch, logger, writer):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['images'].to(device)
            targets = batch['targets']

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total_loss']

            running_loss += loss.item()

            # Store predictions and targets for metrics
            all_predictions.append(outputs)
            all_targets.extend(targets)

    avg_loss = running_loss / len(val_loader)

    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets)

    # Log to TensorBoard
    writer.add_scalar('Loss/val/total', avg_loss, epoch)
    writer.add_scalar('Metrics/val/mAP', metrics['mAP'], epoch)
    writer.add_scalar('Metrics/val/precision', metrics['precision'], epoch)
    writer.add_scalar('Metrics/val/recall', metrics['recall'], epoch)

    logger.info(f"Validation Loss: {avg_loss:.4f}, mAP: {metrics['mAP']:.4f}")

    return avg_loss, metrics


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, filename):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description='Train custom object detection model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    logger = setup_logging(config['log_dir'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create data loaders
    data_loaders = create_data_loaders(
        args.data_dir,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=config['img_size']
    )

    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    num_classes = data_loaders['datasets']['train'].get_num_classes()

    # Create model
    model = CustomDetectionModel(num_classes=num_classes)
    model.to(device)

    # Create loss function
    criterion = YOLOLoss(num_classes=num_classes)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step_size'], gamma=config['lr_gamma'])

    # Setup TensorBoard
    writer = SummaryWriter(log_dir=config['tensorboard_dir'])

    # Training loop
    best_loss = float('inf')
    for epoch in range(config['num_epochs']):
        logger.info(f"Starting epoch {epoch + 1}/{config['num_epochs']}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, writer)

        # Validate
        val_loss, metrics = validate_epoch(model, val_loader, criterion, device, epoch, logger, writer)

        # Update learning rate
        scheduler.step()

        # Save checkpoint
        checkpoint_filename = f"checkpoint_epoch_{epoch + 1}.pth"
        save_checkpoint(model, optimizer, epoch, val_loss, config['checkpoint_dir'], checkpoint_filename)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_checkpoint = save_checkpoint(model, optimizer, epoch, val_loss, config['checkpoint_dir'], 'best_model.pth')
            logger.info(f"New best model saved: {best_checkpoint}")

        logger.info(f"Epoch {epoch + 1} completed. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Apply optimization techniques after training
    logger.info("Applying model optimization techniques...")

    # Pruning
    if config.get('pruning_ratio', 0.0) > 0:
        logger.info(f"Applying pruning with ratio {config['pruning_ratio']}")
        example_input = torch.randn(1, 3, config['img_size'], config['img_size']).to(device)
        pruning_stats = model.prune_model(pruning_ratio=config['pruning_ratio'], example_input=example_input)
        logger.info(f"Pruning completed: Sparsity {pruning_stats['sparsity']:.3f}")

        # Fine-tune after pruning
        logger.info("Fine-tuning after pruning...")
        for epoch in range(10):  # Fine-tune for 10 epochs
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, writer)
            val_loss, _ = validate_epoch(model, val_loader, criterion, device, epoch, logger, writer)
            logger.info(f"Fine-tune Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")

        # Save pruned model
        pruned_checkpoint = save_checkpoint(model, optimizer, config['num_epochs'], val_loss, config['checkpoint_dir'], 'pruned_model.pth')
        logger.info(f"Pruned model saved: {pruned_checkpoint}")

    # Quantization
    if config.get('quantization_enabled', False):
        logger.info("Applying quantization...")
        # Prepare calibration data (use a subset of validation data)
        calibration_data = []
        for i, batch in enumerate(val_loader):
            if i >= 10:  # Use first 10 batches for calibration
                break
            calibration_data.append(batch['images'].to(device))

        quantized_model = model.quantize_model(calibration_data=calibration_data)

        # Save quantized model
        quantized_path = os.path.join(config['checkpoint_dir'], 'quantized_model.pth')
        torch.save(quantized_model.state_dict(), quantized_path)
        logger.info(f"Quantized model saved: {quantized_path}")

    writer.close()
    logger.info("Training and optimization completed!")


if __name__ == "__main__":
    main()
