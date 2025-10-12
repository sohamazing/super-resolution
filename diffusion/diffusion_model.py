# diffusion/diffusion_model.py
"""
Conditional Diffusion Model for Image Super-Resolution.
Clean, modular implementation with proper normalization and residual connections.
"""
import torch
from torch import nn
import torch.nn.functional as F
import math

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embeddings for timestep encoding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: (batch_size,) tensor of timesteps
        Returns:
            (batch_size, dim) tensor of embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """
    Residual block with time embedding injection and GroupNorm.
    More stable than BatchNorm for small batch sizes.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        # First conv block
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        # Second conv block
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb):
        """
        Args:
            x: (B, in_channels, H, W)
            time_emb: (B, time_emb_dim)
        Returns:
            (B, out_channels, H, W)
        """
        h = self.block1(x)

        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]

        h = self.block2(h)

        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """
    Spatial self-attention block for capturing global context.
    Uses efficient implementation for images.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Normalize and compute Q, K, V
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        head_dim = C // self.num_heads
        q = q.reshape(B, self.num_heads, head_dim, H * W).transpose(-2, -1)
        k = k.reshape(B, self.num_heads, head_dim, H * W)
        v = v.reshape(B, self.num_heads, head_dim, H * W).transpose(-2, -1)

        # Scaled dot-product attention
        scale = head_dim ** -0.5
        attn = torch.softmax(q @ k * scale, dim=-1)
        h = attn @ v

        # Reshape back
        h = h.transpose(-2, -1).reshape(B, C, H, W)
        h = self.proj(h)

        return x + h


class DownBlock(nn.Module):
    """Downsampling block with residual connections."""
    def __init__(self, in_channels, out_channels, time_emb_dim, has_attn=False):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x, time_emb):
        x = self.res(x, time_emb)
        x = self.attn(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block with skip connections."""
    def __init__(self, in_channels, out_channels, time_emb_dim, has_attn=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_emb_dim)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x, skip, time_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, time_emb)
        x = self.attn(x)
        return x


class DiffusionUNet(nn.Module):
    """
    Conditional U-Net for super-resolution diffusion.

    Architecture:
    - Conditioning on LR image (concatenated)
    - Time embedding injection at each level
    - Attention at bottleneck for global context
    - Skip connections for preserving spatial information
    """
    def __init__(self, in_channels=6, out_channels=3, features=[64, 128, 256],
                 time_emb_dim=128, dropout=0.1):
        super().__init__()

        # Time embedding network
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, features[0], 3, padding=1)

        # Encoder (downsampling path)
        self.down1 = DownBlock(features[0], features[1], time_emb_dim, has_attn=False)
        self.down2 = DownBlock(features[1], features[2], time_emb_dim, has_attn=False)

        # Bottleneck with attention
        self.bottleneck = nn.ModuleList([
            ResidualBlock(features[2], features[2], time_emb_dim, dropout=dropout),
            AttentionBlock(features[2], num_heads=8),
            ResidualBlock(features[2], features[2], time_emb_dim, dropout=dropout)
        ])

        # Decoder (upsampling path)
        self.up1 = UpBlock(features[2], features[1], time_emb_dim, has_attn=False)
        self.up2 = UpBlock(features[1], features[0], time_emb_dim, has_attn=False)

        # Final output
        self.final = nn.Sequential(
            nn.GroupNorm(8, features[0]),
            nn.SiLU(),
            nn.Conv2d(features[0], out_channels, 3, padding=1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, t, lr_condition):
        """
        Args:
            x: (B, 3, H, W) - noisy SR image
            t: (B,) - timestep
            lr_condition: (B, 3, H, W) - upscaled LR image (bicubic)
        Returns:
            (B, 3, H, W) - predicted noise
        """
        # Get time embeddings
        time_emb = self.time_mlp(t)

        # Concatenate noisy image with LR condition
        x = torch.cat([x, lr_condition], dim=1)

        # Initial convolution
        x = self.init_conv(x)

        # Encoder
        x, skip1 = self.down1(x, time_emb)
        x, skip2 = self.down2(x, time_emb)

        # Bottleneck
        for layer in self.bottleneck:
            if isinstance(layer, ResidualBlock):
                x = layer(x, time_emb)
            else:
                x = layer(x)

        # Decoder
        x = self.up1(x, skip2, time_emb)
        x = self.up2(x, skip1, time_emb)

        # Final prediction
        return self.final(x)

    def get_num_params(self):
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Lightweight version for faster training
class DiffusionUNetLite(DiffusionUNet):
    """Smaller version for testing or resource-constrained environments."""
    def __init__(self, **kwargs):
        kwargs.setdefault('features', [32, 64, 128])
        kwargs.setdefault('time_emb_dim', 64)
        super().__init__(**kwargs)