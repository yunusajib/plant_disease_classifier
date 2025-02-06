# train_model.py
import torch
from src.train import train_model
import os

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models/saved_models', exist_ok=True)
    
    print("Starting training...")
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    
    train_model('configs/config.yaml')

if __name__ == "__main__":
    main()