import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0
from torchvision.models.feature_extraction import create_feature_extractor

class SSDShuffleNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = shufflenet_v2_x1_0(weights="IMAGENET1K_V1" if pretrained else None)
        self.out_channels = [24, 116, 232, 464]  # Adjust based on extracted features

        self.feature_layers = {
            'layer1': self.backbone.conv1,          
            'layer2': self.backbone.stage2,         
            'layer3': self.backbone.stage3,         
            'layer4': self.backbone.stage4          
        }

    def forward(self, x):
        features = []
        for layer in self.feature_layers.values():
            x = layer(x)
            features.append(x)
        return features



