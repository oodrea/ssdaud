import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from ultralytics import RTDETR
from torchvision.models import mobilenet_v2, shufflenet_v2_x1_0, efficientnet_b0

class CustomBackboneSSD(nn.Module):
    def __init__(self, backbone_type='mobilenet', **kwargs):
        super().__init__()
        
        self.backbone_type = backbone_type
        if backbone_type == 'mobilenet':
            self.backbone = mobilenet_v2(weights="IMAGENET1K_V1")
            self.out_channels = [24, 32, 96, 320]  
            
        elif backbone_type == 'shufflenet':
            self.backbone = shufflenet_v2_x1_0(weights="IMAGENET1K_V1")
            self.out_channels = [24, 116, 232, 464]  
            
        elif backbone_type == 'efficientnet':
            self.backbone = efficientnet_b0(weights="IMAGENET1K_V1")
            self.out_channels = [24, 40, 112, 320]  
            
        self.feature_layers = self._get_feature_layers()
        
    def _get_feature_layers(self):
        if self.backbone_type == 'mobilenet':
            return {
                'layer1': self.backbone.features[:4],   
                'layer2': self.backbone.features[4:7],  
                'layer3': self.backbone.features[7:14], 
                'layer4': self.backbone.features[14:]   
            }
        elif self.backbone_type == 'shufflenet':
            return {
                'layer1': self.backbone.conv1,          
                'layer2': self.backbone.stage2,         
                'layer3': self.backbone.stage3,         
                'layer4': self.backbone.stage4          
            }
        elif self.backbone_type == 'efficientnet':
            return {
                'layer1': nn.Sequential(self.backbone.features[:2]),
                'layer2': nn.Sequential(self.backbone.features[2:3]),
                'layer3': nn.Sequential(self.backbone.features[3:5]),
                'layer4': nn.Sequential(self.backbone.features[5:])
            }

    def forward(self, x):
        features = []
        for layer in self.feature_layers.values():
            x = layer(x)
            features.append(x)
        return features