import torch
from pathlib import Path

# Load the weights
weights_path = Path("../../Desktop/main/runs/detect/train_pipeline/weights/best.pt")
checkpoint = torch.load(weights_path, weights_only=False)

# Print some info
print("Model keys:", list(checkpoint.keys()))
if 'model' in checkpoint:
    model = checkpoint['model']
    print("Model type:", type(model))
    if hasattr(model, 'yaml'):
        print("YAML:", model.yaml)
    if hasattr(model, 'nc'):
        print("Number of classes:", model.nc)
    if hasattr(model, 'names'):
        print("Class names:", model.names)
else:
    print("No 'model' key in checkpoint")

# Check if epoch is saved
if 'epoch' in checkpoint:
    print("Epoch:", checkpoint['epoch'])
if 'best_fitness' in checkpoint:
    print("Best fitness:", checkpoint['best_fitness'])
