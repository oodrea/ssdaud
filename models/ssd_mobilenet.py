import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from utils.init import xavier_init
from layers.detection import Detect

def build_dynamic_anchor_config(num_sources, scale_initial, scale_min, scale_max):
    """
    Build a list of dicts, one per feature map source,
    each containing a 'scale' and 'aspect_ratios' for that source.
    We'll do a simple scheme:
      - The first source uses scale_initial
      - The rest are linearly interpolated from scale_min to scale_max
    """

    # Example aspect ratios (adjust as you wish)
    aspect_ratios = [1.0, 2.0, 0.5]

    anchor_config = []

    # The first feature map uses scale_initial
    anchor_config.append({"scale": scale_initial, "aspect_ratios": aspect_ratios})

    # If there are multiple sources, define the remaining scales:
    # e.g. linearly from scale_min to scale_max
    remaining = num_sources - 1
    if remaining > 0:
        for i in range(remaining):
            # alpha goes from 0 to 1 across the remaining maps
            alpha = i / max(1, (remaining - 1))
            s = scale_min + alpha * (scale_max - scale_min)
            anchor_config.append({"scale": s, "aspect_ratios": aspect_ratios})

    return anchor_config

def generate_anchors_for_feature_map(fmap_shape, scale, aspect_ratios, image_size):
    """
    Generate anchors for one feature map.
    
    Args:
        fmap_shape: tuple (H, W) of the feature map.
        scale: a float (e.g., relative to the image size).
        aspect_ratios: list of aspect ratios (e.g. [1.0, 2.0, 0.5]).
        image_size: the size of the input image (assuming square input).
    
    Returns:
        anchors: Tensor of shape [H * W * num_ratios, 4] in [cx, cy, w, h] format,
                 with coordinates normalized between 0 and 1.
    """
    H, W = fmap_shape
    anchors = []
    # Loop over each cell in the feature map.
    for i in range(H):
        for j in range(W):
            # Compute the center in normalized coordinates.
            cx = (j + 0.5) / W
            cy = (i + 0.5) / H
            for ar in aspect_ratios:
                w = scale * (ar ** 0.5) / image_size
                h = scale / (ar ** 0.5) / image_size
                anchors.append([cx, cy, w, h])
    return torch.tensor(anchors, dtype=torch.float32)

def generate_dynamic_anchors(sources, anchor_config, image_size):
    """
    Generate dynamic anchors from a list of feature maps.
    
    Args:
        sources: list of feature maps (tensors) with shape [N, C, H, W].
        anchor_config: list of dicts (one per source) containing a 'scale' and 'aspect_ratios'
                       For example:
                         [
                           {"scale": 0.1, "aspect_ratios": [1.0, 2.0, 0.5]},
                           {"scale": 0.2, "aspect_ratios": [1.0, 2.0, 0.5]},
                           ...
                         ]
        image_size: size of the input image (e.g., 300).
    
    Returns:
        A tensor of anchors of shape [total_anchors, 4].
    """
    all_anchors = []
    for fmap, cfg in zip(sources, anchor_config):
        H, W = fmap.shape[2], fmap.shape[3]
        anchors = generate_anchors_for_feature_map((H, W), cfg["scale"], cfg["aspect_ratios"], image_size)
        all_anchors.append(anchors)
    return torch.cat(all_anchors, dim=0)

#############################################
# SSD MobileNet Implementation
#############################################

def get_out_channels_from_block(block):
    """
    Given a MobileNet v2 InvertedResidual block, return the number of output channels.
    We iterate in reverse over block.conv until we find a Conv2d.
    """
    for m in reversed(block.conv):
        if isinstance(m, nn.Conv2d):
            return m.out_channels
    raise ValueError("No Conv2d found in block.")

class SSDMobileNet(nn.Module):
    """
    SSD MobileNet V2 architecture with dynamic anchor generation.
    """
    def __init__(self, mode, base, extras, head, anchors, class_count, scale_initial, scale_min, scale_max):
        super(SSDMobileNet, self).__init__()

        self.mode = mode
        self.base = base
        self.extras = nn.ModuleList(extras)
        self.class_head = nn.ModuleList(head[0])
        self.loc_head = nn.ModuleList(head[1])
        self.anchors = anchors
        self.class_count = class_count

        # Store user CLI scale args:
        self.scale_initial = scale_initial
        self.scale_min = scale_min
        self.scale_max = scale_max

        if mode == 'test' or mode == 'val':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect.apply

    def forward(self, x):
        sources = []
        class_preds = []
        loc_preds = []

        # 1) Extract your 2 backbone feature maps
        for i, layer in enumerate(self.base.features):
            x = layer(x)
            if i == 7:
                sources.append(x)
            if i == 14:
                sources.append(x)
                break

        # 2) Apply extra layers
        for j, layer in enumerate(self.extras):
            x = F.relu(layer(x), inplace=True)
            sources.append(x)

        # 3) Dynamically build anchor config based on # of sources
        num_sources = len(sources)  # e.g. 6
        anchor_config = build_dynamic_anchor_config(num_sources,
                                                    self.scale_initial,  # store these in the model or pass them in
                                                    self.scale_min,
                                                    self.scale_max)
        # 4) Generate dynamic anchors
        dynamic_anchors = generate_dynamic_anchors(sources, anchor_config, image_size=300)
        print(f"Generated {dynamic_anchors.shape[0]} anchors dynamically")

        # 5) Predictions
        for i, (feat, conv) in enumerate(zip(sources, self.class_head)):
            pred = conv(feat).permute(0, 2, 3, 1).contiguous()
            class_preds.append(pred)

        for i, (feat, conv) in enumerate(zip(sources, self.loc_head)):
            pred = conv(feat).permute(0, 2, 3, 1).contiguous()
            loc_preds.append(pred)

        b = x.shape[0]
        class_preds = torch.cat([pred.view(b, -1, self.class_count) for pred in class_preds], 1)
        loc_preds = torch.cat([pred.view(b, -1, 4) for pred in loc_preds], 1)

        # If test or val, pass dynamic_anchors to detection
        if self.mode in ['test', 'val']:
            output = self.detect(self.class_count,
                                self.softmax(class_preds),
                                loc_preds,
                                dynamic_anchors)
        else:
            output = (class_preds, loc_preds)
        return output

    def init_weights(self, model_save_path=None, basenet=None, use_gpu=False):
        """
        Initializes the MobileNet v2 backbone (or custom EfficientNet) and SSD layers.
        Loads pretrained weights if available, then applies Xavier initialization.
        """
        if basenet and "vgg16_reducedfc.pth" not in basenet:
            weights_path = osp.join(model_save_path, basenet)
            print(f"Loading backbone weights from: {weights_path}")
            self.base.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
        else:
            print("Loading MobileNet V2 with ImageNet-1K pretrained weights.")
            self.base = mobilenet_v2(weights="IMAGENET1K_V1")

        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.to(self.device)

        self.extras.apply(xavier_init)
        self.class_head.apply(xavier_init)
        self.loc_head.apply(xavier_init)
        print(f"SSD MobileNet V2 backbone initialized successfully on {self.device}.")

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext in ['.pkl', '.pth']:
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry, only .pth and .pkl files are supported.')

#############################################
# Additional Helper Functions for Extra Layers and Multibox
#############################################

def get_extras(config, in_channels):
    layers = []
    print("Building extras layers (get_extras):")
    i = 0
    while i < len(config):
        if config[i] == 'S':
            print(f"Encountered 'S' marker at index {i}, skipping")
            i += 1
            continue
        if i + 1 < len(config) and config[i + 1] == 'S':
            if i + 2 >= len(config):
                raise ValueError("Config error: 'S' must be followed by a channel number")
            out_channels = config[i + 2]
            print(f"Creating extra layer with downsampling: {in_channels} -> {out_channels} (stride=2)")
            layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            layers.append(layer)
            in_channels = out_channels
            i += 3
        else:
            out_channels = config[i]
            print(f"Creating extra layer: {in_channels} -> {out_channels} (stride=1)")
            layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            layers.append(layer)
            in_channels = out_channels
            i += 1
    print("Finished building extras. Total extra layers:", len(layers))
    return layers

def build_extras(in_channels):
    """
    Build a set of extra layers for SSD.
    Here we define four extra layers that gradually downsample the feature map.
    """
    extras = nn.ModuleList([
        nn.Conv2d(in_channels, 256, kernel_size=2, stride=2, padding=1),   # e.g. from 19x19 to 10x10
        nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=1),             # 10x10 to 5x5
        nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=1),             # 5x5 to 3x3 (or similar)
        nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0),             # 3x3 to 1x1
    ])
    return extras

def multibox(config, base, extra_layers, class_count):
    class_layers = []
    loc_layers = []
    # For MobileNet V2, we use backbone features from indices 7 and 14.
    mobilenet_layers = [7, 14]
    for k, idx in enumerate(mobilenet_layers):
        # For MobileNet V2, we assume the block is an InvertedResidual;
        # we can get its output channels from the block’s last Conv2d.
        # Here we use a helper:
        in_channels = get_out_channels_from_block(base.features[idx])
        print(f"Multibox backbone head {k}: using in_channels = {in_channels}")
        class_layers.append(nn.Conv2d(in_channels, config[k] * class_count, kernel_size=3, padding=1))
        loc_layers.append(nn.Conv2d(in_channels, config[k] * 4, kernel_size=3, padding=1))
    # For the extra layers (we assume there are 4 extra layers):
    for k, layer in enumerate(extra_layers, start=2):
        print(f"Multibox extra head {k}: using extra layer with out_channels = {layer.out_channels}")
        class_layers.append(nn.Conv2d(layer.out_channels, config[k] * class_count, kernel_size=3, padding=1))
        loc_layers.append(nn.Conv2d(layer.out_channels, config[k] * 4, kernel_size=3, padding=1))
    return class_layers, loc_layers

def get_out_channels_from_block(block):
    """
    For a MobileNet V2 InvertedResidual block, return the number of output channels.
    We iterate in reverse over block.conv to find the last Conv2d.
    """
    for m in reversed(block.conv):
        if isinstance(m, nn.Conv2d):
            return m.out_channels
    raise ValueError("No Conv2d found in block!")

#############################################
# Configurations (example for SSD300)
#############################################

extras_config = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256]
}

mbox_config = {
    '300': [4, 6, 6, 6, 4, 4],
    '512': [4, 6, 6, 6, 6, 4, 4]
}

#############################################
# Build Model Function for SSD MobileNet V2
#############################################

def build_SSDMobileNet(mode, new_size, anchors, class_count, scale_initial, scale_min, scale_max):
    base = mobilenet_v2(weights="IMAGENET1K_V1")

    extras = build_extras(in_channels=160)

    head = multibox(config=mbox_config[str(new_size)],
                    base=base,
                    extra_layers=extras,
                    class_count=class_count)

    return SSDMobileNet(mode=mode,
                           base=base,
                           extras=extras,
                           head=head,
                           anchors=None,
                           class_count=class_count,
                           scale_initial=scale_initial,
                           scale_min=scale_min,
                           scale_max=scale_max)
