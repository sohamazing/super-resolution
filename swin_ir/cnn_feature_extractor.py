# swin_ir/cnn_feature_extractor.py
import torch
from torch import nn
from torch.nn import functional as F

class ConvResidualBlock(nn.Module):
    """
    A simple and effective residual block for CNNs.
    It maintains the same input and output dimensions.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
        # The shortcut connection ensures dimensions match for the residual addition.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.gelu(x + shortcut)

class CNNFeatureExtractor(nn.Module):
    """
    A lightweight CNN to extract features at multiple depths, but at the *same spatial resolution*.
    This is crucial for direct, one-to-one fusion with the transformer body.
    """
    def __init__(self, in_channels=3, embed_dim=180, num_layers=4):
        super().__init__()
        
        self.layers = nn.ModuleList()
        # The first layer maps the input image (e.g., 3 channels) to the main embedding dimension.
        self.layers.append(ConvResidualBlock(in_channels, embed_dim, stride=1))
        
        # All subsequent layers process features while maintaining the same channel and spatial dimensions.
        for _ in range(num_layers - 1):
            self.layers.append(ConvResidualBlock(embed_dim, embed_dim, stride=1))

    def forward(self, x):
        """Returns a list of feature maps, one from each stage of the CNN."""
        feature_maps = []
        # We pass the input through each layer sequentially...
        for layer in self.layers:
            x = layer(x)
            # ...and store the output of each layer for later fusion.
            feature_maps.append(x)
        return feature_maps

# downsampling during CNN?
# 
# class CNNFeatureExtractor(nn.Module):
#     """
#     A lightweight CNN to extract multi-scale features.
#     These features will be fused into the Swin Transformer layers.
#     """
#     def __init__(self, in_channels=3, base_dim=64):
#         super().__init__()
#         # --- Stage 1: Full Resolution Features ---
#         self.stage1 = nn.Sequential(
#             ConvResidualBlock(in_channels, base_dim, stride=1),
#             ConvResidualBlock(base_dim, base_dim, stride=1)
#         )
        
#         # --- Stage 2: Half Resolution Features ---
#         self.stage2 = nn.Sequential(
#             ConvResidualBlock(base_dim, base_dim * 2, stride=2),
#             ConvResidualBlock(base_dim * 2, base_dim * 2, stride=1)
#         )
        
#         # --- Stage 3: Quarter Resolution Features ---
#         self.stage3 = nn.Sequential(
#             ConvResidualBlock(base_dim * 2, base_dim * 4, stride=2),
#             ConvResidualBlock(base_dim * 4, base_dim * 4, stride=1)
#         )

#     def forward(self, x):
#         """Returns a list of feature maps at different scales."""
#         feat1 = self.stage1(x)
#         feat2 = self.stage2(feat1)
#         feat3 = self.stage3(feat2)
#         return [feat1, feat2, feat3]