# src/utils.py
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

def setup_data_directories(base_path, output_dir, train_size=0.7, val_size=0.15):
    """
    Organizes data into train/val/test splits
    """
    # Update paths to use the color directory
    base_path = Path(base_path) / 'plantvillage dataset' / 'color'
    output_dir = Path(output_dir)
    
    print(f"Looking for data in: {base_path}")
    print(f"Will save processed data to: {output_dir}")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {split_dir}")

    # Get all class directories
    class_dirs = list(base_path.glob('*/'))
    print(f"Found {len(class_dirs)} class directories")
    
    if len(class_dirs) == 0:
        print("No class directories found! Directory contents:")
        print(list(base_path.glob('*')))
        return
    
    for class_dir in class_dirs:
        print(f"\nProcessing class: {class_dir.name}")
        
        # Get all images in the class
        images = list(class_dir.glob('*.[jp][pn][gf]'))
        print(f"Found {len(images)} images")
        
        if len(images) == 0:
            print("Warning: No images found in this class directory!")
            continue
            
        # Split into train, val, test
        train_imgs, temp_imgs = train_test_split(images, train_size=train_size, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, train_size=val_size/(1-train_size), random_state=42)
        
        print(f"Split sizes - Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
        
        # Create class directories in each split
        for split in ['train', 'val', 'test']:
            (output_dir / split / class_dir.name).mkdir(exist_ok=True)
        
        # Copy images to respective directories
        for img in train_imgs:
            shutil.copy2(img, output_dir / 'train' / class_dir.name / img.name)
        for img in val_imgs:
            shutil.copy2(img, output_dir / 'val' / class_dir.name / img.name)
        for img in test_imgs:
            shutil.copy2(img, output_dir / 'test' / class_dir.name / img.name)

    print("\nData processing completed!")