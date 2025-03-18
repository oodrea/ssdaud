import torch
from torchvision.models import efficientnet_b0
from torchvision.models.feature_extraction import create_feature_extractor

# Load EfficientNet-B0 model
model = efficientnet_b0(weights="IMAGENET1K_V1")
model.eval()

# Define return nodes (extract all MBConv blocks)
return_nodes = {
    'features.3.0': 'MBConv1',
    'features.4.2': 'MBConv2',
    'features.5.0': 'MBConv3',
    'features.6.0': 'MBConv4',
    'features.7.0': 'MBConv5',
}

# Create feature extractor
feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

# Dummy input (batch_size=1, 3 color channels, 300x300 resolution for SSD)
dummy_input = torch.randn(1, 3, 300, 300)

# Forward pass through the feature extractor
output_features = feature_extractor(dummy_input)

# Print the resolutions of MBConv blocks
print("\nFeature Map Resolutions:")
for name, feature in output_features.items():
    print(f"{name}: {feature.shape}")  # Prints (batch, channels, height, width)
