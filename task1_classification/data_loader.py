"""
Data loading and preprocessing utilities for PneumoniaMNIST dataset.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from medmnist import INFO
from medmnist.dataset import PneumoniaMNIST


class PneumoniaDataset(Dataset):
    """Wrapper for PneumoniaMNIST dataset with custom transformations."""
    
    def __init__(self, split='train', transform=None, download=True):
        """
        Args:
            split: 'train', 'val', or 'test'
            transform: torchvision transforms to apply
            download: whether to download the dataset
        """
        self.dataset = PneumoniaMNIST(split=split, download=download)
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Convert to PIL Image if needed
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = torch.from_numpy(image).float() / 255.0
            if len(image.shape) == 2:
                image = image.unsqueeze(0)  # Add channel dimension
        
        # Convert label to binary (0 or 1)
        label = torch.tensor(label.item() if isinstance(label, np.ndarray) else label, dtype=torch.long)
        
        return image, label


def get_data_loaders(batch_size=64, num_workers=4, augment=True):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        batch_size: batch size for all loaders
        num_workers: number of worker processes
        augment: whether to apply data augmentation for training
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Define transformations
    if augment:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),  # Small rotation for medical images
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Small translation
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    # Validation and test transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets
    train_dataset = PneumoniaDataset(split='train', transform=train_transform)
    val_dataset = PneumoniaDataset(split='val', transform=eval_transform)
    test_dataset = PneumoniaDataset(split='test', transform=eval_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_dataset_info():
    """Get information about the PneumoniaMNIST dataset."""
    info = INFO['pneumoniamnist']
    return {
        'task': info['task'],
        'n_channels': info['n_channels'],
        'n_classes': len(info['label']),
        'labels': info['label'],
        'shape': (28, 28)
    }
