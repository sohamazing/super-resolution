# swinir/fused_swinir_model.py
import torch
from torch import nn
from torch.nn import functional as F

from .swin_transformer import SwinTransformerBlock
from .cnn_feature_extractor import CNNFeatureExtractor

class ResidualSwinTransformerLayer(nn.Module):
    """
    A layer of Swin Transformer blocks, composed of sequential blocks
    with alternating regular (W-MSA) and shifted (SW-MSA) windows.
    """
    def __init__(self, dim, num_heads, window_size=8, num_blocks=6):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            # Every other block uses a shifted window
            shift_size = window_size // 2 if (i % 2 != 0) else 0
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size
                )
            )
            
        # A final convolutional layer for the residual connection
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        # The SwinTransformerBlock expects (B, H, W, C)
        # Assuming input x is (B, C, H, W), we permute it
        B, C, H, W = x.shape
        shortcut = x
        
        x = x.permute(0, 2, 3, 1) # (B, H, W, C)
        
        for block in self.blocks:
            x = block(x)
        
        # Permute back to (B, C, H, W) for the convolutional layer
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        
        # Add the main residual connection
        return shortcut + x

class FusionBlock(nn.Module):
    """
    A refined block to fuse features from the CNN and Transformer branches.
    """
    def __init__(self, dim):
        super().__init__()
        # A 1x1 convolution is used to refine the combined features, allowing the model
        # to learn the best way to integrate the local (CNN) and global (Transformer) information.
        self.conv = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, transformer_features, cnn_features):
        # The features are added element-wise, then passed through the refining convolution.
        return self.conv(transformer_features + cnn_features)

class FusedSwinTransformer(nn.Module):
    """
    The final, professionally implemented Fused-Scale SwinIR model for 4x Super-Resolution.
    """
    def __init__(self, in_channels=3, out_channels=3, embed_dim=180, num_heads=6, 
                 window_size=8, num_layers=4, num_blocks_per_layer=6, scale=4):
        super().__init__()
        self.scale = scale
        
        # --- 1. Shallow Feature Extraction (for Transformer path) ---
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)
        
        # --- 2. Parallel CNN Feature Extraction ---
        # This now correctly uses the refined CNN extractor.
        self.cnn_extractor = CNNFeatureExtractor(in_channels=in_channels, embed_dim=embed_dim, num_layers=num_layers)
        
        # --- 3. Deep Feature Extraction & Fusion ---
        self.transformer_body = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()
        
        for _ in range(num_layers):
            self.transformer_body.append(
                ResidualSwinTransformerLayer(
                    dim=embed_dim, num_heads=num_heads, window_size=window_size, num_blocks=num_blocks_per_layer
                )
            )
            # The FusionBlock is now simpler as all features have the same dimension.
            self.fusion_blocks.append(FusionBlock(dim=embed_dim))
            
        # This layer processes the features after the final fusion has occurred.
        self.conv_after_fusion = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            
        # --- 4. High-Resolution Reconstruction ---
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        # A bicubic upsampling of the input serves as a baseline for the final residual connection.
        # This helps stabilize training and allows the model to focus on learning the high-frequency details.
        shortcut = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        
        # The CNN and Transformer paths run in parallel.
        cnn_features = self.cnn_extractor(x)
        trans_features = self.conv_first(x)
        
        # The core loop: process with transformer, then fuse with corresponding CNN feature.
        for i in range(len(self.transformer_body)):
            trans_features = self.transformer_body[i](trans_features)
            trans_features = self.fusion_blocks[i](trans_features, cnn_features[i])
        
        fused_features = self.conv_after_fusion(trans_features)
        
        # The model's output is added to the bicubic shortcut to produce the final image.
        return self.upsample(fused_features) + shortcut