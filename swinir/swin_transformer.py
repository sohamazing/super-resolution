# swinir/swin_transformer.py
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

def window_partition(x, window_size):
    """
    Partitions a feature map into non-overlapping windows.
    Args:
        x: Input tensor of shape (B, H, W, C).
        window_size (int): The size of the window.
    Returns:
        Tensor of shape (num_windows*B, window_size, window_size, C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Merges windows back into a feature map.
    Args:
        windows: Window tensor of shape (num_windows*B, window_size, window_size, C).
        window_size (int): The size of the window.
        H (int): Height of the original feature map.
        W (int): Width of the original feature map.
    Returns:
        Tensor of shape (B, H, W, C).
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """
    Windowed Multi-Head Self-Attention (W-MSA) with relative position bias.
    """
    def __init__(self, dim, window_size, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformer(nn.Module):
    """The main Swin Transformer Block."""
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(proj_drop) # Use proj_drop for MLP as well
        )
        
        # ✅ FIX: The attention mask creation depends on the input size,
        # so it must be calculated in the forward pass, not in __init__.
        self.attn_mask = None

    def create_mask(self, H, W, device):
        # Only create a mask if we are using a shifted window
        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1), device=device)
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            return attn_mask
        else:
            return None

    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x
        
        # ✅ FIX: Create the mask dynamically in the forward pass
        # This prevents the NameError from the original __init__
        if self.shift_size > 0:
            self.attn_mask = self.create_mask(H, W, x.device)

        x = self.norm1(x)

        # ✅ FIX: Correct logic for cyclic shift and windowing
        # The shift must happen on the 4D tensor *before* partitioning.
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
```eof

### 2. The Corrected Generator

This file now correctly imports `SwinTransformer` and passes the appropriate `attn_drop` and `proj_drop` arguments to it, resolving the `TypeError`.

```python:HCAST Generator (Corrected):fused_gan/generator_hcast.py
# fused_gan/generator_hcast.py
import torch
from torch import nn
import torch.nn.functional as F
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent))

# ✅ FIX: Correctly import the fixed SwinTransformer class
from swinir.swin_transformer import SwinTransformer

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
    """
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256],
                 embed_dim=180, num_heads=6, window_size=8, num_swin_blocks=6,
                 scale=4, dropout=0.1):
        super().__init__()
        self.scale = scale

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.down_blocks = nn.ModuleList()
        in_c = features[0]
        for feature in features[1:]:
            self.down_blocks.append(
                nn.Sequential(
                    ConvBlock(in_c, feature, dropout=dropout),
                    nn.Conv2d(feature, feature, 4, 2, 1)
                )
            )
            in_c = feature

        self.bottleneck_conv = nn.Conv2d(features[-1], embed_dim, 1, 1, 0)

        self.swin_body = nn.Sequential(
            *[SwinTransformer(
                dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                # ✅ FIX: Pass the dropout value to the correct parameters
                attn_drop=dropout,
                proj_drop=dropout
            ) for i in range(num_swin_blocks)]
        )

        self.bottleneck_conv_out = nn.Conv2d(embed_dim, features[-1], 3, 1, 1)

        self.up_blocks = nn.ModuleList()
        reversed_features = list(reversed(features))
        for i in range(len(reversed_features) - 1):
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(reversed_features[i] * 2, reversed_features[i+1], 4, 2, 1),
                    ConvBlock(reversed_features[i+1], reversed_features[i+1], dropout=dropout)
                )
            )

        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(features[0] * 2, features[0] * (scale**2), 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features[0], out_channels, 3, 1, 1)
        )

    def forward(self, x):
        skip_connections = []
        x = self.initial_conv(x)
        skip_connections.append(x)

        for block in self.down_blocks:
            x = block(x)
            skip_connections.append(x)

        x = self.bottleneck_conv(x)
        x = x.permute(0, 2, 3, 1) # Reshape for Swin: (B, C, H, W) -> (B, H, W, C)
        
        x = self.swin_body(x)

        x = x.permute(0, 3, 1, 2) # Reshape back: (B, H, W, C) -> (B, C, H, W)
        x = self.bottleneck_conv_out(x)

        for block in self.up_blocks:
            skip_conn = skip_connections.pop()
            x = torch.cat([x, skip_conn], dim=1)
            x = block(x)

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

# The gradient checkpointing version does not need to be changed as it inherits the fixes.
class HCASTGeneratorCheckpoint(HCASTGenerator):
    """
    Version with gradient checkpointing for even lower memory usage.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for module in self.swin_body:
            if hasattr(module, 'set_grad_checkpointing'):
                module.set_grad_checkpointing(True)

    def forward(self, x):
        from torch.utils.checkpoint import checkpoint
        skip_connections = []
        x = checkpoint(self.initial_conv, x, use_reentrant=False)
        skip_connections.append(x)
        for block in self.down_blocks:
            x = checkpoint(block, x, use_reentrant=False)
            skip_connections.append(x)
        x = self.bottleneck_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.swin_body(x)
        x = x.permute(0, 3, 1, 2)
        x = self.bottleneck_conv_out(x)
        for block in self.up_blocks:
            skip_conn = skip_connections.pop()
            x = torch.cat([x, skip_conn], dim=1)
            x = checkpoint(block, x, use_reentrant=False)
        final_skip = skip_connections.pop()
        x = torch.cat([x, final_skip], dim=1)
        return checkpoint(self.reconstruction_head, x, use_reentrant=False)
```eof