import sys
import platform

print("System Information:")
print("Python version:", sys.version)
print("Python implementation:", platform.python_implementation())
print("Platform:", platform.platform())

try:
    import numpy as np
    print("\nNumPy version:", np.__version__)
    print("NumPy path:", np.__file__)
except Exception as e:
    print("\nNumPy import error:", e)

try:
    import torch
    print("\nPyTorch version:", torch.__version__)
    print("PyTorch path:", torch.__file__)
except Exception as e:
    print("\nPyTorch import error:", e)

try:
    import torchvision
    print("\nTorchVision version:", torchvision.__version__)
    print("TorchVision path:", torchvision.__file__)
except Exception as e:
    print("\nTorchVision import error:", e)