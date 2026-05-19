import sys
print("Python path:")
for path in sys.path:
    print(f"  {path}")

print("\nTesting torch import:")
try:
    import torch
    print(f"torch version: {torch.__version__}")
    print("torch imported successfully")
except Exception as e:
    print(f"Error importing torch: {e}")

print("\nTesting torch.nn import:")
try:
    import torch.nn as nn
    print("torch.nn imported successfully")
except Exception as e:
    print(f"Error importing torch.nn: {e}")

print("\nTesting torch.nn.functional import:")
try:
    import torch.nn.functional as F
    print("torch.nn.functional imported successfully")
except Exception as e:
    print(f"Error importing torch.nn.functional: {e}")