import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from ..data.custom_dataset import CustomDataset

def get_transforms(img_size=640):
    """
    Get data transforms for training and validation.

    Args:
        img_size (int): Size to resize images to.

    Returns:
        dict: Dictionary containing train and val transforms.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return {
        'train': train_transforms,
        'val': val_transforms,
        'test': val_transforms
    }


def create_data_loaders(data_dir, batch_size=16, num_workers=4, img_size=640):
    """
    Create data loaders for training, validation, and testing.

    Args:
        data_dir (str): Path to the data directory.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of workers for data loading.
        img_size (int): Size to resize images to.

    Returns:
        dict: Dictionary containing train, val, and test data loaders.
    """
    transforms_dict = get_transforms(img_size)

    # Create datasets
    train_dataset = CustomDataset(data_dir, split='train', transform=transforms_dict['train'])
    val_dataset = CustomDataset(data_dir, split='val', transform=transforms_dict['val'])
    test_dataset = CustomDataset(data_dir, split='test', transform=transforms_dict['val'])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'datasets': {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    }


def collate_fn(batch):
    """
    Custom collate function for batching samples.

    Args:
        batch: List of samples from the dataset.

    Returns:
        dict: Batched samples.
    """
    images = []
    targets = []
    image_paths = []
    image_names = []

    for sample in batch:
        images.append(sample['image'])
        targets.append(sample['targets'])
        image_paths.append(sample['image_path'])
        image_names.append(sample['image_name'])

    images = torch.stack(images, dim=0)

    return {
        'images': images,
        'targets': targets,
        'image_paths': image_paths,
        'image_names': image_names
    }
