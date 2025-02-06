# Check model file
import torch
checkpoint = torch.load('models/saved_models/best_model.pth', map_location='cpu')
print("Checkpoint keys:", checkpoint.keys())