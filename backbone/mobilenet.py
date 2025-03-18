import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision.models.feature_extraction import create_feature_extractor

class SSDMobileNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        self.out_channels = [24, 32, 96, 320]  # Adjust based on extracted features

        self.feature_layers = {
            'layer1': self.backbone.features[:4],   
            'layer2': self.backbone.features[4:7],  
            'layer3': self.backbone.features[7:14], 
            'layer4': self.backbone.features[14:]   
        }

    def forward(self, x):
        features = []
        for layer in self.feature_layers.values():
            x = layer(x)
            features.append(x)
        return features


