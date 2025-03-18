import torch.nn as nn
import torchvision.models as models

class EfficientNetCustom(nn.Module):
    """Custom EfficientNet model wrapper"""

    def __init__(self, model_name="efficientnet_b0", num_classes=1000, batch_norm=True, pretrained=True):
        """
        Initialize the EfficientNet model.

        Arguments:
            model_name {str} -- EfficientNet variant (e.g., "efficientnet_b0", "efficientnet_b1", etc.)
            num_classes {int} -- Number of output classes (default: 1000 for ImageNet)
            batch_norm {bool} -- If True, keeps batch normalization layers (default: True)
            pretrained {bool} -- If True, loads pretrained weights (default: True)
        """
        super(EfficientNetCustom, self).__init__()

        # Load the model dynamically
        self.model = getattr(models, model_name)(weights="IMAGENET1K_V1" if pretrained else None)

        # Modify the classifier layer to match the desired number of classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        # If batch_norm is False, remove all batch norm layers
        if not batch_norm:
            self._remove_batch_norm(self.model)

    def forward(self, x):
        return self.model(x)

    def _remove_batch_norm(self, model):
        """Helper function to remove batch normalization layers from the model."""
        for module_name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                setattr(model, module_name, nn.Identity())  # Replace batch norm with identity

