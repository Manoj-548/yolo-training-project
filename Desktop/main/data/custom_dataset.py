import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import yaml


class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images and labels from YOLO format.
    """

    def __init__(self, data_dir, split='train', transform=None):
        """
        Initialize the dataset.

        Args:
            data_dir (str): Path to the data directory containing images and labels.
            split (str): Dataset split ('train', 'val', or 'test').
            transform: Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # Load data configuration
        config_path = os.path.join(data_dir, 'data.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Get class names
        self.class_names = self.config['names']
        self.num_classes = len(self.class_names)

        # Get image and label paths
        self.image_dir = os.path.join(data_dir, 'images', split)
        self.label_dir = os.path.join(data_dir, 'labels', split)

        # List all image files
        self.image_files = []
        if os.path.exists(self.image_dir):
            for file in os.listdir(self.image_dir):
                if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.image_files.append(file)

        self.image_files.sort()

    def __len__(self):
        """Return the total number of samples."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Sample containing image, targets, and metadata.
        """
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Load labels
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        targets.append([class_id, x_center, y_center, width, height])

        targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.empty(0, 5)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        sample = {
            'image': image,
            'targets': targets,
            'image_path': img_path,
            'image_name': img_name
        }

        return sample

    def get_class_names(self):
        """Return the list of class names."""
        return self.class_names

    def get_num_classes(self):
        """Return the number of classes."""
        return self.num_classes
