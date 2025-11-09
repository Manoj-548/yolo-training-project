import os
import torch
import yaml
import argparse
import logging
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np

from models.custom_model import CustomDetectionModel


def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def preprocess_image(image_path, img_size=640):
    """
    Preprocess image for inference.

    Args:
        image_path (str): Path to input image
        img_size (int): Target image size

    Returns:
        torch.Tensor: Preprocessed image tensor
        PIL.Image: Original image for visualization
    """
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply transforms
    processed_image = transform(image)

    return processed_image.unsqueeze(0), image  # Add batch dimension


def postprocess_predictions(predictions, conf_threshold=0.5, img_size=640):
    """
    Post-process model predictions to get bounding boxes.

    Args:
        predictions (dict): Model predictions
        conf_threshold (float): Confidence threshold
        img_size (int): Original image size

    Returns:
        list: List of detected objects with boxes, classes, and confidences
    """
    cls_pred = predictions['cls_pred'][0]  # Remove batch dimension
    reg_pred = predictions['reg_pred'][0]
    obj_pred = predictions['obj_pred'][0]

    num_classes, grid_h, grid_w = cls_pred.shape

    detections = []

    # Get predictions above confidence threshold
    obj_scores = torch.sigmoid(obj_pred[0])  # Shape: (grid_h, grid_w)
    cls_scores, cls_indices = torch.max(torch.sigmoid(cls_pred), dim=0)  # Shape: (grid_h, grid_w)

    conf_scores = obj_scores * cls_scores

    # Filter by confidence
    mask = conf_scores > conf_threshold

    if mask.sum() == 0:
        return detections

    # Convert grid predictions to bounding boxes
    for y in range(grid_h):
        for x in range(grid_w):
            if mask[y, x]:
                # Get bounding box coordinates
                dx = reg_pred[0, y, x].item()
                dy = reg_pred[1, y, x].item()
                dw = reg_pred[2, y, x].item()
                dh = reg_pred[3, y, x].item()

                # Convert to absolute coordinates
                cell_size = img_size / grid_h

                x_center = (x + dx) * cell_size
                y_center = (y + dy) * cell_size
                w = torch.exp(torch.tensor(dw)) * cell_size
                h = torch.exp(torch.tensor(dh)) * cell_size

                x1 = max(0, (x_center - w / 2).item())
                y1 = max(0, (y_center - h / 2).item())
                x2 = min(img_size, (x_center + w / 2).item())
                y2 = min(img_size, (y_center + h / 2).item())

                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'class': cls_indices[y, x].item(),
                    'confidence': conf_scores[y, x].item()
                }

                detections.append(detection)

    return detections


def draw_detections(image, detections, class_names):
    """
    Draw bounding boxes and labels on the image.

    Args:
        image (PIL.Image): Input image
        detections (list): List of detections
        class_names (list): List of class names

    Returns:
        PIL.Image: Image with detections drawn
    """
    # Convert PIL to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for detection in detections:
        bbox = detection['bbox']
        class_id = detection['class']
        confidence = detection['confidence']

        x1, y1, x2, y2 = map(int, bbox)

        # Draw bounding box
        cv2.rectangle(opencv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add label
        label = f"{class_names[class_id]} {confidence:.2f}"
        cv2.putText(opencv_image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert back to PIL
    pil_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))

    return pil_image


def run_inference(model, image_path, device, config, logger):
    """
    Run inference on a single image.

    Args:
        model: Trained model
        image_path (str): Path to input image
        device: Device to run inference on
        config (dict): Configuration dictionary
        logger: Logger instance

    Returns:
        list: Detections
        PIL.Image: Image with detections drawn
    """
    logger.info(f"Running inference on {image_path}")

    # Preprocess image
    processed_image, original_image = preprocess_image(image_path, config['img_size'])
    processed_image = processed_image.to(device)

    # Run model
    model.eval()
    with torch.no_grad():
        predictions = model(processed_image)

    # Post-process predictions
    detections = postprocess_predictions(
        predictions,
        conf_threshold=config['conf_threshold'],
        img_size=config['img_size']
    )

    logger.info(f"Found {len(detections)} detections")

    return detections, original_image


def save_results(detections, output_image, output_dir, image_name, logger):
    """
    Save inference results.

    Args:
        detections (list): List of detections
        output_image (PIL.Image): Image with detections drawn
        output_dir (str): Output directory
        image_name (str): Name of input image
        logger: Logger instance
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save annotated image
    image_base = os.path.splitext(image_name)[0]
    image_path = os.path.join(output_dir, f"{image_base}_detections.jpg")
    output_image.save(image_path)

    # Save detections as JSON
    json_path = os.path.join(output_dir, f"{image_base}_detections.json")
    with open(json_path, 'w') as f:
        import json
        json.dump(detections, f, indent=4)

    logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run inference with custom object detection model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Output directory')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    logger = setup_logging(config['log_dir'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load class names from data config
    data_config_path = os.path.join(os.path.dirname(args.config), 'data', 'synthetic_dataset', 'data.yaml')
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    class_names = data_config['names']

    # Create model
    num_classes = len(class_names)
    model = CustomDetectionModel(num_classes=num_classes)
    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Handle quantized or pruned models that might be saved differently
        model.load_state_dict(checkpoint)
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Run inference
    detections, original_image = run_inference(model, args.image, device, config, logger)

    # Draw detections on image
    output_image = draw_detections(original_image, detections, class_names)

    # Save results
    image_name = os.path.basename(args.image)
    save_results(detections, output_image, args.output_dir, image_name, logger)

    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
