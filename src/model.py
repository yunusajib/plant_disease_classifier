# src/model.py
import torch
import torch.nn as nn
from torchvision import models

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(PlantDiseaseModel, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Replace the final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)