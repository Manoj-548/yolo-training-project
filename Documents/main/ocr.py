import os
import random
import cv2
import json
import argparse

import torch 
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

# --- Configuration ---
ROBOFLOW_DATA_PATH = "./roboflow_data" # Path where your data was downloaded
TRAIN_DATASET_NAME = "ocr_train"
VALID_DATASET_NAME = "ocr_valid"
CONFIG_FILE = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
NUM_CLASSES = 10 # 0-9 digits
MAX_ITER = 1500 # Adjust based on dataset size and validation loss
IMS_PER_BATCH = 4 # Adjust based on GPU memory

# --- Register Datasets ---
def register_roboflow_datasets():
    # Register the train dataset
    DatasetCatalog.register(
        TRAIN_DATASET_NAME, 
        lambda: json.load(open(os.path.join(ROBOFLOW_DATA_PATH, "train/_annotations.coco.json")))
    )
    MetadataCatalog.get(TRAIN_DATASET_NAME).set(
        thing_classes=[str(i) for i in range(NUM_CLASSES)],
        image_root=os.path.join(ROBOFLOW_DATA_PATH, "train")
    )
    
    # Register the valid dataset
    DatasetCatalog.register(
        VALID_DATASET_NAME, 
        lambda: json.load(open(os.path.join(ROBOFLOW_DATA_PATH, "valid/_annotations.coco.json")))
    )
    MetadataCatalog.get(VALID_DATASET_NAME).set(
        thing_classes=[str(i) for i in range(NUM_CLASSES)],
        image_root=os.path.join(ROBOFLOW_DATA_PATH, "valid")
    )
    print("Roboflow datasets registered successfully.")

# --- Training ---
def train_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
    
    # Update configurations for your custom dataset
    cfg.DATASETS.TRAIN = (TRAIN_DATASET_NAME,)
    cfg.DATASETS.TEST = (VALID_DATASET_NAME,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_FILE)
    cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = MAX_ITER
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

# --- Inference ---
def inference(image_path, model_weights_path="output/model_final.pth"):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
    cfg.MODEL.WEIGHTS = model_weights_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TEST = (VALID_DATASET_NAME,)
    
    predictor = DefaultPredictor(cfg)
    
    image = cv2.imread(image_path)
    outputs = predictor(image)
    
    # Visualize the results
    v = Visualizer(
        image[:, :, ::-1],
        MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
        scale=0.8,
        instance_mode=ColorMode.IMAGE_BW
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Inference", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OCR Model Training and Inference")
    parser.add_argument('--mode', choices=['train', 'infer'], default='train', help='Mode: train or infer')
    parser.add_argument('--data_path', type=str, default='./roboflow_data', help='Path to Roboflow dataset')
    parser.add_argument('--image_path', type=str, help='Path to image for inference (required if mode=infer)')
    parser.add_argument('--weights', type=str, default='output/model_final.pth', help='Path to model weights for inference')

    args = parser.parse_args()

    # Update data path
    ROBOFLOW_DATA_PATH = args.data_path

    # Check if data path exists
    if not os.path.exists(ROBOFLOW_DATA_PATH):
        print(f"Error: Dataset path '{ROBOFLOW_DATA_PATH}' does not exist. Please download the dataset from Roboflow.")
        exit(1)

    # Register datasets
    try:
        register_roboflow_datasets()
        print("Datasets registered successfully.")
    except Exception as e:
        print(f"Error registering datasets: {e}")
        exit(1)

    if args.mode == 'train':
        print("Starting training...")
        try:
            train_model()
            print("Training completed successfully.")
        except Exception as e:
            print(f"Error during training: {e}")
            exit(1)
    elif args.mode == 'infer':
        if not args.image_path:
            print("Error: --image_path is required for inference mode.")
            exit(1)
        if not os.path.exists(args.image_path):
            print(f"Error: Image path '{args.image_path}' does not exist.")
            exit(1)
        print(f"Running inference on {args.image_path}...")
        try:
            inference(args.image_path, args.weights)
            print("Inference completed.")
        except Exception as e:
            print(f"Error during inference: {e}")
            exit(1)
