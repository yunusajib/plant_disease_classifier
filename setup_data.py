# setup_data.py
from src.utils import setup_data_directories
import os

if __name__ == "__main__":
    # Create a test script to verify the data structure
    data_path = 'data/raw'
    output_path = 'data/processed'
    
    # First, let's check what's in the color directory
    color_path = os.path.join(data_path, 'plantvillage dataset', 'color')
    print(f"Checking contents of color directory: {color_path}")
    if os.path.exists(color_path):
        print("Contents:", os.listdir(color_path))
    else:
        print("Color directory not found!")
        
    # Set up the data
    setup_data_directories(data_path, output_path)