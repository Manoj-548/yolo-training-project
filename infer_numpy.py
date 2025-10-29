import numpy as np
import cv2
from pathlib import Path
import pickle

# Load the trained model
script_dir = Path(__file__).parent
model_path = script_dir / 'model.pkl'
if model_path.exists():
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
else:
    print("Model file not found. Please train the model first.")
    exit(1)

# Inference on a test image
test_image_path = script_dir / "images/test/2.jpg"
img = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)
if img is not None:
    h, w = img.shape
    # Take a patch from the image (e.g., top-left)
    patch_size = (28, 28)
    patch = cv2.resize(img, patch_size)
    patch_flat = patch.flatten() / 255.0
    pred = model.forward(patch_flat.reshape(1, -1))
    pred_class = np.argmax(pred, axis=1)[0]
    print(f"Predicted class for test image {test_image_path.name}: {pred_class}")
else:
    print("Test image not found")
