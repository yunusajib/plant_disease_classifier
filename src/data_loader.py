# src/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import yaml

class PlantDiseaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Get all image paths and labels
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        # Get all classes (subdirectories)
        for idx, class_dir in enumerate(sorted(self.data_dir.glob('*/'))):
            self.class_to_idx[class_dir.name] = idx
            for img_path in class_dir.glob('*.[jp][pn][gf]'):  # jpg, png, jpeg
                self.images.append(img_path)
                self.labels.append(idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(phase):
    if phase == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                              [0.229, 0.224, 0.225])
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                              [0.229, 0.224, 0.225])
        ])

def get_dataloaders(config_path):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create datasets
    train_dataset = PlantDiseaseDataset(
        config['data']['train_path'],
        transform=get_transforms('train')
    )
    
    val_dataset = PlantDiseaseDataset(
        config['data']['val_path'],
        transform=get_transforms('val')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, train_dataset.class_to_idx