# # fused_gan/generator_hcast.py
import torch
from torch import nn
import torch.nn.functional as F
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent))

from swinir.swin_transformer import SwinTransformerBlock

class ConvBlock(nn.Module):
    """Standard conv block with optional normalization."""
    def __init__(self, in_channels, out_channels, use_norm=True, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=not use_norm),
            nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class HCASTGenerator(nn.Module):
    """
    Hierarchical CNN-Attention Super-Resolution Transformer (H-CAST) Generator.
    Optimized version with memory-efficient skip connections and better stability.
    """
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256],
                 embed_dim=180, num_heads=6, window_size=8, num_swin_blocks=6,
                 scale=4, dropout=0.1):
        super().__init__()
        self.scale = scale

        # --- 1. Initial Feature Extraction ---
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # --- 2. CNN Encoder Path (Downsampling) ---
        self.down_blocks = nn.ModuleList()
        in_c = features[0]
        for feature in features[1:]:
            self.down_blocks.append(
                nn.Sequential(
                    ConvBlock(in_c, feature, dropout=dropout),
                    nn.Conv2d(feature, feature, 4, 2, 1)  # Downsample
                )
            )
            in_c = feature

        # --- 3. Swin Transformer Bottleneck ---
        self.bottleneck_conv = nn.Conv2d(features[-1], embed_dim, 1, 1, 0)

        self.swin_body = nn.Sequential(
            *[SwinTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                drop=dropout  # Add dropout to Swin blocks
            ) for i in range(num_swin_blocks)]
        )

        self.bottleneck_conv_out = nn.Conv2d(embed_dim, features[-1], 3, 1, 1)

        # --- 4. CNN Decoder Path (Upsampling) ---
        self.up_blocks = nn.ModuleList()
        reversed_features = list(reversed(features))
        for i in range(len(reversed_features) - 1):
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(reversed_features[i] * 2, reversed_features[i+1], 4, 2, 1),
                    ConvBlock(reversed_features[i+1], reversed_features[i+1], dropout=dropout)
                )
            )

        # --- 5. Final Reconstruction Head ---
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(features[0] * 2, features[0] * (scale**2), 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features[0], out_channels, 3, 1, 1)
        )

    def forward(self, x):
        """
        Memory-optimized forward pass using pop() instead of list reversal.
        """
        # Store skip connections for U-Net architecture
        skip_connections = []

        # Encoder path
        x = self.initial_conv(x)
        skip_connections.append(x)

        for block in self.down_blocks:
            x = block(x)
            skip_connections.append(x)

        # --- Bottleneck: Swin Transformer ---
        x = self.bottleneck_conv(x)
        B, C, H, W = x.shape

        # Reshape for Swin: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        # Process through Swin blocks (input/output: B, H, W, C)
        x = self.swin_body(x)

        # Reshape back: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.bottleneck_conv_out(x)

        # --- Decoder path with memory-efficient skip connections ---
        # Instead of reversing the entire list, we pop from the end
        # This reduces memory by immediately freeing skip connections
        for block in self.up_blocks:
            skip_conn = skip_connections.pop()  # Get deepest skip first
            x = torch.cat([x, skip_conn], dim=1)
            x = block(x)

        # Final skip connection
        final_skip = skip_connections.pop()
        x = torch.cat([x, final_skip], dim=1)

        return self.reconstruction_head(x)

    def load_pretrained(self, pretrained_path):
        """Load pretrained weights with error handling."""
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.load_state_dict(state_dict, strict=True)
            print(f"Successfully loaded pretrained weights from {pretrained_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Continuing with random initialization...")


# ===== Memory-efficient version with gradient checkpointing =====
class HCASTGeneratorCheckpoint(HCASTGenerator):
    """
    Version with gradient checkpointing for even lower memory usage.
    Use this if you're hitting OOM errors.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Enable gradient checkpointing for Swin blocks
        for module in self.swin_body:
            if hasattr(module, 'set_grad_checkpointing'):
                module.set_grad_checkpointing(True)

    def forward(self, x):
        """Forward with gradient checkpointing on encoder/decoder."""
        from torch.utils.checkpoint import checkpoint

        skip_connections = []

        # Encoder (with checkpointing)
        x = checkpoint(self.initial_conv, x, use_reentrant=False)
        skip_connections.append(x)

        for block in self.down_blocks:
            x = checkpoint(block, x, use_reentrant=False)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck_conv(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)

        # Swin body (already uses checkpointing if enabled)
        x = self.swin_body(x)

        x = x.permute(0, 3, 1, 2)
        x = self.bottleneck_conv_out(x)

        # Decoder (with checkpointing)
        for block in self.up_blocks:
            skip_conn = skip_connections.pop()
            x = torch.cat([x, skip_conn], dim=1)
            x = checkpoint(block, x, use_reentrant=False)

        # Final
        final_skip = skip_connections.pop()
        x = torch.cat([x, final_skip], dim=1)

        return checkpoint(self.reconstruction_head, x, use_reentrant=False)