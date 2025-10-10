# esrgan/generator.py
import torch
from torch import nn
import torch.nn.functional as F

class ResidualDenseBlock(nn.Module):
    """Innermost block of the RRDB. Densely connected convolutional layers."""
    def __init__(self, in_channels=64, growth_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_channels, in_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """The main building block of the generator."""
    def __init__(self, in_channels=64):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(in_channels)
        self.rdb2 = ResidualDenseBlock(in_channels)
        self.rdb3 = ResidualDenseBlock(in_channels)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class UpsampleBlock(nn.Module):
    """A block for learnable upsampling with PixelShuffle."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(self.pixel_shuffle(self.conv(x)))

class GeneratorESRGAN(nn.Module):
    """The complete generator model for 4x upscaling with PixelShuffle."""
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=23):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_features) for _ in range(num_blocks)])
        self.conv_body = nn.Conv2d(num_features, num_features, 3, 1, 1)

        # Two upsampling blocks for 4x scale
        self.upsample = nn.Sequential(UpsampleBlock(num_features), UpsampleBlock(num_features))

        self.conv_hr = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        trunk = self.conv_body(self.body(feat))
        feat = feat + trunk

        feat = self.upsample(feat)
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out