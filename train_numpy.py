
import numpy as np
import os
import cv2
from pathlib import Path
import random
import time
from tqdm import tqdm
import logging
import torch
import torch.nn as nn

logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Dataset paths
script_dir = Path(__file__).parent
train_images = script_dir / "../../../../Desktop/main/synthetic_dataset/images/train"
train_labels = script_dir / "../../../../Desktop/main/synthetic_dataset/labels/train"
val_images = script_dir / "../../../../Desktop/main/synthetic_dataset/images/val"
val_labels = script_dir / "../../../../Desktop/main/synthetic_dataset/labels/val"

# Classes
num_classes = 43

# Image size for patches
patch_size = (28, 28)

# Load data
def load_data(images_dir, labels_dir):
    X = []
    y = []
    for img_file in os.listdir(images_dir):
        if img_file.endswith('.jpg'):
            img_path = images_dir / img_file
            label_path = labels_dir / img_file.replace('.jpg', '.txt')
            if label_path.exists():
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                h, w = img.shape
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls = int(parts[0])
                            x, y_, w_, h_ = map(float, parts[1:])
                            x1 = int((x - w_/2) * w)
                            y1 = int((y_ - h_/2) * h)
                            x2 = int((x + w_/2) * w)
                            y2 = int((y_ + h_/2) * h)
                            patch = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                            if patch.size == 0:
                                continue
                            patch = cv2.resize(patch, patch_size)
                            X.append(patch.flatten() / 255.0)
                            y.append(cls)
    X = np.array(X)
    y = np.array(y)
    return X, y

logging.info("Loading training data...")
X_train, y_train = load_data(train_images, train_labels)
logging.info(f"Training data: {X_train.shape}, labels: {y_train.shape}")

logging.info("Loading validation data...")
X_val, y_val = load_data(val_images, val_labels)
logging.info(f"Validation data: {X_val.shape}, labels: {y_val.shape}")

# One-hot encode labels
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

y_train_oh = one_hot(y_train, num_classes)
y_val_oh = one_hot(y_val, num_classes)

# Simple CNN using numpy
class SimpleCNN:
    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        self.num_classes = num_classes
        # Conv1: 1x28x28 -> 16x14x14
        self.conv1_w = np.random.randn(16, 1, 3, 3) * 0.1
        self.conv1_b = np.zeros(16)
        # Conv2: 16x14x14 -> 32x7x7
        self.conv2_w = np.random.randn(32, 16, 3, 3) * 0.1
        self.conv2_b = np.zeros(32)
        # FC: 32*7*7 -> 128
        self.fc1_w = np.random.randn(32*7*7, 128) * 0.1
        self.fc1_b = np.zeros(128)
        # FC: 128 -> num_classes
        self.fc2_w = np.random.randn(128, num_classes) * 0.1
        self.fc2_b = np.zeros(num_classes)

    def conv2d(self, x, w, b, stride=1, padding=0):
        batch, in_c, h, w_ = x.shape
        out_c, _, kh, kw = w.shape
        out_h = (h + 2*padding - kh) // stride + 1
        out_w = (w_ + 2*padding - kw) // stride + 1
        x_pad = np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)), 'constant')
        out = np.zeros((batch, out_c, out_h, out_w))
        for batch_idx in range(batch):
            for oc in range(out_c):
                for i in range(out_h):
                    for j in range(out_w):
                        out[batch_idx, oc, i, j] = np.sum(x_pad[batch_idx, :, i*stride:i*stride+kh, j*stride:j*stride+kw] * w[oc]) + b[oc]
        return out

    def maxpool2d(self, x, size=2, stride=2):
        batch, c, h, w = x.shape
        out_h = (h - size) // stride + 1
        out_w = (w - size) // stride + 1
        out = np.zeros((batch, c, out_h, out_w))
        for b in range(batch):
            for cc in range(c):
                for i in range(out_h):
                    for j in range(out_w):
                        out[b, cc, i, j] = np.max(x[b, cc, i*stride:i*stride+size, j*stride:j*stride+size])
        return out

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        # x: batch x 784
        batch = x.shape[0]
        x = x.reshape(batch, 1, 28, 28)
        x = self.conv2d(x, self.conv1_w, self.conv1_b, padding=1)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = self.conv2d(x, self.conv2_w, self.conv2_b, padding=1)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = x.reshape(batch, -1)
        x = x @ self.fc1_w + self.fc1_b
        x = self.relu(x)
        x = x @ self.fc2_w + self.fc2_b
        return x

    def backward(self, x, y, lr=0.01):
        batch = x.shape[0]
        # Forward
        x_in = x.reshape(batch, 1, 28, 28)
        conv1_out = self.conv2d(x_in, self.conv1_w, self.conv1_b, padding=1)
        relu1_out = self.relu(conv1_out)
        pool1_out = self.maxpool2d(relu1_out)
        conv2_out = self.conv2d(pool1_out, self.conv2_w, self.conv2_b, padding=1)
        relu2_out = self.relu(conv2_out)
        pool2_out = self.maxpool2d(relu2_out)
        flat = pool2_out.reshape(batch, -1)
        fc1_out = flat @ self.fc1_w + self.fc1_b
        relu3_out = self.relu(fc1_out)
        fc2_out = relu3_out @ self.fc2_w + self.fc2_b

        # Loss
        probs = np.exp(fc2_out - np.max(fc2_out, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        loss = -np.sum(y * np.log(probs + 1e-8)) / batch

        # Backward
        dloss = (probs - y) / batch
        dfc2_w = relu3_out.T @ dloss
        dfc2_b = np.sum(dloss, axis=0)
        drelu3 = dloss @ self.fc2_w.T
        dfc1 = drelu3 * (fc1_out > 0)
        dfc1_w = flat.T @ dfc1
        dfc1_b = np.sum(dfc1, axis=0)
        dflat = dfc1 @ self.fc1_w.T
        dpool2 = dflat.reshape(pool2_out.shape)
        # Maxpool backward (simple, not accurate)
        dconv2 = np.zeros_like(conv2_out)
        for b in range(batch):
            for c in range(32):
                for i in range(7):
                    for j in range(7):
                        window = relu2_out[b, c, i*2:(i+1)*2, j*2:(j+1)*2]
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        dconv2[b, c, i*2 + max_idx[0], j*2 + max_idx[1]] = dpool2[b, c, i, j]
        drelu2 = dconv2 * (conv2_out > 0)
        dconv2_w = np.zeros_like(self.conv2_w)
        dconv2_b = np.zeros(32)
        for b in range(batch):
            for oc in range(32):
                for ic in range(16):
                    for i in range(7):
                        for j in range(7):
                            dconv2_w[oc, ic] += np.sum(pool1_out[b, ic, i*1:(i+1)*3, j*1:(j+1)*3] * drelu2[b, oc, i, j])
                dconv2_b[oc] += np.sum(drelu2[b, oc])
        dpool1 = self.conv2d(drelu2, np.transpose(self.conv2_w, (1,0,2,3)), np.zeros(16), padding=1)
        dconv1 = np.zeros_like(conv1_out)
        for b in range(batch):
            for c in range(16):
                for i in range(14):
                    for j in range(14):
                        window = relu1_out[b, c, i*2:(i+1)*2, j*2:(j+1)*2]
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        dconv1[b, c, i*2 + max_idx[0], j*2 + max_idx[1]] = dpool1[b, c, i, j]
        drelu1 = dconv1 * (conv1_out > 0)
        dconv1_w = np.zeros_like(self.conv1_w)
        dconv1_b = np.zeros(16)
        for b in range(batch):
            for oc in range(16):
                for ic in range(1):
                    for i in range(14):
                        for j in range(14):
                            dconv1_w[oc, ic] += np.sum(x_in[b, ic, i*1:(i+1)*3, j*1:(j+1)*3] * drelu1[b, oc, i, j])
                dconv1_b[oc] += np.sum(drelu1[b, oc])

        # Update
        self.fc2_w -= lr * dfc2_w
        self.fc2_b -= lr * dfc2_b
        self.fc1_w -= lr * dfc1_w
        self.fc1_b -= lr * dfc1_b
        self.conv2_w -= lr * dconv2_w
        self.conv2_b -= lr * dconv2_b
        self.conv1_w -= lr * dconv1_w
        self.conv1_b -= lr * dconv1_b

        return loss

import pickle

model_path = script_dir / 'model.pkl'

if model_path.exists():
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Loaded saved model from model.pkl")
else:
    # Train
    model = SimpleCNN(784, num_classes)
    epochs = 5
    batch_size = 32
    lr = 0.001

    print("Starting training...")
    start_time = time.time()
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        epoch_start = time.time()
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train_oh = y_train_oh[indices]
        total_loss = 0
        num_batches = (X_train.shape[0] + batch_size - 1) // batch_size
        val_losses = []
        prev_val_loss = float('inf')
        for i in tqdm(range(0, X_train.shape[0], batch_size), desc=f"Epoch {epoch+1} Batches", leave=False, total=num_batches):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train_oh[i:i+batch_size]
            loss = model.backward(X_batch, y_batch, lr)
            total_loss += loss
            # Calculate validation loss for each batch for stability check
            val_preds_batch = model.forward(X_val)
            val_probs_batch = np.exp(val_preds_batch - np.max(val_preds_batch, axis=1, keepdims=True))
            val_probs_batch /= np.sum(val_probs_batch, axis=1, keepdims=True)
            val_loss_batch = -np.sum(y_val_oh * np.log(val_probs_batch + 1e-8)) / X_val.shape[0]
            val_losses.append(val_loss_batch)
        avg_loss = total_loss / num_batches
        avg_val_loss = np.mean(val_losses)
        epoch_end = time.time()
        tqdm.write(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}, Time: {epoch_end - epoch_start:.2f}s")
        
        # Early stopping condition based on validation loss improvement
        if avg_val_loss > prev_val_loss:
            print(f"Validation loss increased. Early stopping at epoch {epoch+1}.")
            break
        prev_val_loss = avg_val_loss
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")
    # Save the trained model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved as model.pkl")
    # Convert to PyTorch model and save as best.pt
    class SimpleCNNPyTorch(nn.Module):
        def __init__(self, num_classes=43):
            super(SimpleCNNPyTorch, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 32 * 7 * 7)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    pytorch_model = SimpleCNNPyTorch()
    pytorch_model.conv1.weight.data = torch.from_numpy(model.conv1_w).float()
    pytorch_model.conv1.bias.data = torch.from_numpy(model.conv1_b).float()
    pytorch_model.conv2.weight.data = torch.from_numpy(model.conv2_w).float()
    pytorch_model.conv2.bias.data = torch.from_numpy(model.conv2_b).float()
    pytorch_model.fc1.weight.data = torch.from_numpy(model.fc1_w).float()
    pytorch_model.fc1.bias.data = torch.from_numpy(model.fc1_b).float()
    pytorch_model.fc2.weight.data = torch.from_numpy(model.fc2_w).float()
    pytorch_model.fc2.bias.data = torch.from_numpy(model.fc2_b).float()
    torch.save(pytorch_model.state_dict(), '../../../../Desktop/main/best.pt')
    print("Model saved as ../../../../Desktop/main/best.pt")

# Evaluate
def evaluate(X, y):
    preds = model.forward(X)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(y, axis=1)
    acc = np.mean(pred_classes == true_classes)
    return acc

train_acc = evaluate(X_train, y_train_oh)
val_acc = evaluate(X_val, y_val_oh)
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# Inference on a test image
test_image_path = script_dir / "images/test/23.jpg"
img = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)
if img is not None:
    h, w = img.shape
    # Take a patch from the image (e.g., top-left)
    patch = cv2.resize(img, patch_size)
    patch_flat = patch.flatten() / 255.0
    pred = model.forward(patch_flat.reshape(1, -1))
    pred_class = np.argmax(pred, axis=1)[0]
    print(f"Predicted class for test image {test_image_path.name}: {pred_class}")
else:
    print("Test image not found")
