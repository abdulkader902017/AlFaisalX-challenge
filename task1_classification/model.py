"""
Model architectures for pneumonia classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PneumoniaCNN(nn.Module):
    """
    Custom CNN architecture for pneumonia classification.
    Designed for 28x28 grayscale images.
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(PneumoniaCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate feature map size after convolutions
        # Input: 28x28 -> after 4 poolings: 28/2^4 = 1.75 -> effectively 1
        self.feature_size = 256 * 1 * 1
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Conv block 1: 28x28 -> 14x14
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2: 14x14 -> 7x7
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3: 7x7 -> 3x3 (with ceil_mode would be better, but using standard)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Conv block 4: 3x3 -> 1x1
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class PneumoniaResNet(nn.Module):
    """
    ResNet-based model for pneumonia classification.
    Uses pretrained ResNet18 adapted for grayscale images.
    """
    
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.5):
        super(PneumoniaResNet, self).__init__()
        
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer to accept grayscale (1 channel instead of 3)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify final FC layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)


class PneumoniaEfficientNet(nn.Module):
    """
    EfficientNet-based model for pneumonia classification.
    Uses pretrained EfficientNet-B0 adapted for grayscale images.
    """
    
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.5):
        super(PneumoniaEfficientNet, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        
        # Modify first conv layer to accept grayscale
        self.efficientnet.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # Modify classifier
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        return self.efficientnet(x)


def get_model(model_name='cnn', num_classes=2, pretrained=True, dropout_rate=0.5):
    """
    Factory function to get the specified model.
    
    Args:
        model_name: 'cnn', 'resnet18', 'efficientnet_b0'
        num_classes: number of output classes
        pretrained: whether to use pretrained weights (for ResNet/EfficientNet)
        dropout_rate: dropout probability
        
    Returns:
        Model instance
    """
    if model_name == 'cnn':
        return PneumoniaCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_name == 'resnet18':
        return PneumoniaResNet(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    elif model_name == 'efficientnet_b0':
        return PneumoniaEfficientNet(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown model: {model_name}")
