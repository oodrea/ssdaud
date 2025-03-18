import torch
from torchvision.models import efficientnet_b0

# Load the model
model = efficientnet_b0(weights="IMAGENET1K_V1")

# Filter only MBConv blocks and exclude sequential containers
print("Extracting MBConv blocks:\n")
for name, layer in model.named_modules():
    if "MBConv" in layer.__class__.__name__:  # Check if layer is explicitly an MBConv block
        print(f"{name} -> {layer}")
