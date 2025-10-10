# diffusion/diffusion_model.py
import torch
from torch import nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """Encodes the timestep 't' into a vector of sine and cosine values."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        # Create a set of frequencies
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # Apply frequencies to the time tensor
        embeddings = time[:, None] * embeddings[None, :]
        # Create sine and cosine pairs
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """A residual block with two convolutions and a time embedding."""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        # A 1x1 conv to match input/output channels for the residual connection
        self.identity_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        # Main convolutional path
        h = self.norm1(self.relu(self.conv1(x)))
        # Project time embedding and add to the feature map
        time_emb = self.relu(self.time_mlp(t)).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.norm2(self.relu(self.conv2(h)))
        # Add the original input to the output
        return h + self.identity_conv(x)

class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4, ff_hidden=512):
        super().__init__()
        self.channels = channels
        self.ln = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.ff_self = nn.Sequential(
            nn.Linear(channels, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # reshape for MHA: (b, h*w, c)
        x_reshaped = x.reshape(b, c, h * w).swapaxes(1, 2).contiguous()
        x_ln = self.ln(x_reshaped)
        attn_out, _ = self.mha(x_ln, x_ln, x_ln)
        attn_out = attn_out + x_reshaped
        out = self.ff_self(attn_out) + attn_out
        # reshape back to (b, c, h, w)
        return out.swapaxes(2, 1).contiguous().reshape(b, c, h, w)


class DiffusionUNet(nn.Module):
    """A professional U-Net with Residual blocks, Self-Attention, and Conditioning."""
    def __init__(self, in_channels=6, out_channels=3, time_emb_dim=32):
        super().__init__()
        
        # --- Time Embedding ---
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # --- Encoder (Downsampling Path) ---
        self.down1 = ResidualBlock(in_channels, 64, time_emb_dim)
        self.down2 = ResidualBlock(64, 128, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bot1 = ResidualBlock(128, 256, time_emb_dim)
        self.attention = SelfAttention(256)
        self.bot2 = ResidualBlock(256, 256, time_emb_dim)

        # --- Decoder (Upsampling Path) ---
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up1 = ResidualBlock(256 + 128, 128, time_emb_dim) # Input channels = upsampled + skip connection
        self.up2 = ResidualBlock(128 + 64, 64, time_emb_dim)  # Input channels = upsampled + skip connection
        
        # --- Final Output Layer ---
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, t, lr_image):
        # Concatenate the noisy image with the upscaled LR image condition
        x = torch.cat([x, lr_image], dim=1)
        
        # 1. Process the timestep
        t = self.time_mlp(t)
        
        # 2. Go down the encoder path
        x1 = self.down1(x, t)  # Save for skip connection
        p1 = self.pool(x1)
        x2 = self.down2(p1, t) # Save for skip connection
        p2 = self.pool(x2)
        
        # 3. Pass through the bottleneck
        x3 = self.bot1(p2, t)
        x3 = self.attention(x3)
        x3 = self.bot2(x3, t)
        
        # 4. Go up the decoder path
        u1 = self.up(x3)
        u1 = torch.cat([u1, x2], dim=1) # Apply skip connection
        u1 = self.up1(u1, t)
        
        u2 = self.up(u1)
        u2 = torch.cat([u2, x1], dim=1) # Apply skip connection
        u2 = self.up2(u2, t)
        
        # 5. Get the final prediction
        return self.out(u2)