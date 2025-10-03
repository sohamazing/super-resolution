# esrgan/discriminator.py
from torch import nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    """A VGG-style discriminator with spectral normalization."""
    def __init__(self, in_channels=3):
        super().__init__()

        def conv_block(in_feat, out_feat, stride=1):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_feat, out_feat, 3, stride=stride, padding=1)),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.features = nn.Sequential(
            # input: 3 x 256 x 256
            conv_block(in_channels, 64),
            conv_block(64, 64, stride=2), # 64 x 128 x 128
            conv_block(64, 128),
            conv_block(128, 128, stride=2), # 128 x 64 x 64
            conv_block(128, 256),
            conv_block(256, 256, stride=2), # 256 x 32 x 32
            conv_block(256, 512),
            conv_block(512, 512, stride=2), # 512 x 16 x 16
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            spectral_norm(nn.Conv2d(512, 1024, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(1024, 1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)