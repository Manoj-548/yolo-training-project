import os
import torch
import yaml
import argparse
import logging
from datetime import datetime
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import json

from models.custom_model import CustomDetectionModel
from utils.data_loader import create_data_loaders


def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1, box2: Bounding boxes in format [x1, y1, x2, y2]

    Returns:
        float: IoU value
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def evaluate_predictions(predictions, targets, iou_threshold=0.5, conf_threshold=0.5):
    """
    Evaluate predictions against ground truth.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        iou_threshold: IoU threshold for matching
        conf_threshold: Confidence threshold for predictions

    Returns:
        dict: Evaluation metrics
    """
    all_pred_boxes = []
    all_true_boxes = []
    all_pred_classes = []
    all_true_classes = []

    for batch_idx, (pred, target) in enumerate(zip(predictions, targets)):
        # Process predictions
        cls_pred = pred['cls_pred']  # Shape: (batch, num_classes, grid_h, grid_w)
        reg_pred = pred['reg_pred']  # Shape: (batch, 4, grid_h, grid_w)
        obj_pred = pred['obj_pred']  # Shape: (batch, 1, grid_h, grid_w)

        batch_size, num_classes, grid_h, grid_w = cls_pred.shape

        for b in range(batch_size):
            # Get predictions above confidence threshold
            obj_scores = torch.sigmoid(obj_pred[b, 0])  # Shape: (grid_h, grid_w)
            cls_scores, cls_indices = torch.max(torch.sigmoid(cls_pred[b]), dim=0)  # Shape: (grid_h, grid_w)

            conf_scores = obj_scores * cls_scores

            # Filter by confidence
            mask = conf_scores > conf_threshold
            if mask.sum() == 0:
                continue

            # Convert grid predictions to bounding boxes
            for y in range(grid_h):
                for x in range(grid_w):
                    if mask[y, x]:
                        # Get bounding box coordinates
                        dx = reg_pred[b, 0, y, x].item()
                        dy = reg_pred[b, 1, y, x].item()
                        dw = reg_pred[b, 2, y, x].item()
                        dh = reg_pred[b, 3, y, x].item()

                        # Convert to absolute coordinates (assuming 640x640 input)
                        img_size = 640
                        cell_size = img_size / grid_h

                        x_center = (x + dx) * cell_size
                        y_center = (y + dy) * cell_size
                        w = torch.exp(torch.tensor(dw)) * cell_size
                        h = torch.exp(torch.tensor(dh)) * cell_size

                        x1 = (x_center - w / 2).item()
                        y1 = (y_center - h / 2).item()
                        x2 = (x_center + w / 2).item()
                        y2 = (y_center + h / 2).item()

                        all_pred_boxes.append([x1, y1, x2, y2])
                        all_pred_classes.append(cls_indices[y, x].item())

            # Process ground truth
            if len(target) > 0:
                for obj in target:
                    class_id, x_center, y_center, w, h = obj

                    # Convert normalized coordinates to absolute
                    x1 = (x_center - w / 2) * img_size
                    y1 = (y_center - h / 2) * img_size
                    x2 = (x_center + w / 2) * img_size
                    y2 = (y_center + h / 2) * img_size

                    all_true_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                    all_true_classes.append(int(class_id.item()))

    # Calculate mAP and other metrics
    if len(all_pred_boxes) == 0 or len(all_true_boxes) == 0:
        return {
            'mAP': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }

    # Simplified mAP calculation (in practice, use pycocotools or similar)
    # This is a placeholder - proper implementation would be more complex
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_classes, all_pred_classes[:len(all_true_classes)],
        average='weighted', zero_division=0
    )

    # Placeholder mAP calculation
    mAP = 0.0  # Would need proper implementation

    return {
        'mAP': mAP,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'num_predictions': len(all_pred_boxes),
        'num_ground_truth': len(all_true_boxes)
    }


def test_model(model, test_loader, device, logger):
    """Test the model on test dataset."""
    model.eval()
    all_predictions = []
    all_targets = []

    logger.info("Starting model evaluation...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch['images'].to(device)
            targets = batch['targets']

            # Forward pass
            outputs = model(images)

            all_predictions.append(outputs)
            all_targets.extend(targets)

            if batch_idx % 10 == 0:
                logger.info(f"Processed {batch_idx}/{len(test_loader)} batches")

    # Calculate metrics
    metrics = evaluate_predictions(all_predictions, all_targets)

    logger.info("Evaluation completed!")
    logger.info(f"mAP: {metrics['mAP']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"Predictions: {metrics['num_predictions']}")
    logger.info(f"Ground Truth: {metrics['num_ground_truth']}")

    return metrics


def save_results(metrics, output_dir, logger):
    """Save evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, 'test_results.json')

    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description='Test custom object detection model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Output directory for results')
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

    test_loader = data_loaders['test']
    num_classes = data_loaders['datasets']['test'].get_num_classes()

    # Create model
    model = CustomDetectionModel(num_classes=num_classes)
    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Test model
    metrics = test_model(model, test_loader, device, logger)

    # Save results
    save_results(metrics, args.output_dir, logger)

    logger.info("Testing completed!")


if __name__ == "__main__":
    main()
