import torch

import torch.nn as nn
from torchvision.models import efficientnet_b0, shufflenet_v2_x1_0, mobilenet_v2
from torchvision.models.feature_extraction import create_feature_extractor

class EfficientNetB0Backbone(nn.Module):
    def __init__(self):
        super(EfficientNetB0Backbone, self).__init__()
        # Load the pretrained EfficientNet-B0 model (ImageNet1K_V1)
        model = efficientnet_b0(weights="IMAGENET1K_V1")
        # Choose intermediate layers: for example, use 'features.3' and 'features.7'
        return_nodes = {'features.3': 'feat1', 'features.7': 'feat2'}
        self.backbone = create_feature_extractor(model, return_nodes=return_nodes)

    def forward(self, x):
        feats = self.backbone(x)
        # Return features as a list: first (higher resolution) then second (deeper feature)
        return [feats['feat1'], feats['feat2']]


class ShuffleNetV2Backbone(nn.Module):
    def __init__(self):
        super(ShuffleNetV2Backbone, self).__init__()
        # Load pretrained ShuffleNet-V2 x1.0
        model = shufflenet_v2_x1_0(weights="IMAGENET1K_V1")
        # For ShuffleNet, you might choose outputs after stage2 and stage4.
        return_nodes = {'stage2': 'feat1', 'stage4': 'feat2'}
        self.backbone = create_feature_extractor(model, return_nodes=return_nodes)

    def forward(self, x):
        feats = self.backbone(x)
        return [feats['feat1'], feats['feat2']]


class MobileNetV2Backbone(nn.Module):
    def __init__(self):
        super(MobileNetV2Backbone, self).__init__()
        # Load pretrained MobileNet-V2
        model = mobilenet_v2(weights="IMAGENET1K_V1")
        # For MobileNet-V2, select two feature layers from the features block
        return_nodes = {'features.4': 'feat1', 'features.7': 'feat2'}
        self.backbone = create_feature_extractor(model, return_nodes=return_nodes)

    def forward(self, x):
        feats = self.backbone(x)
        return [feats['feat1'], feats['feat2']]


def print_backbone_shapes(backbone, input_size=(1, 3, 300, 300)):
    x = torch.randn(input_size)
    features = backbone(x)
    for i, feat in enumerate(features):
        print(f"Feature {i} shape: {feat.shape}")

# Example usage for MobileNet:
backbone = MobileNetV2Backbone()
print_backbone_shapes(backbone)

backbone1 = ShuffleNetV2Backbone()
print_backbone_shapes(backbone1)

backbone2 = EfficientNetB0Backbone()
print_backbone_shapes(backbone2)