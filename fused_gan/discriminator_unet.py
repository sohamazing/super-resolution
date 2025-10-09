# fused_gan/discriminator_unet.py
import torch
from torch import nn

class ConvBlock(nn.Module):
    """A standard convolutional block for the U-Net discriminator."""
    def __init__(self, in_channels, out_channels, use_instance_norm=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels) if use_instance_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DiscriminatorUNet(nn.Module):
    """
    A U-Net based discriminator that provides a spatial "realism map" as output.
    This gives the generator patch-based feedback to improve local details.
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        # --- Encoder (Downsampling Path) ---
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.down_blocks = nn.ModuleList()
        in_c = features[0]
        for feature in features[1:]:
            self.down_blocks.append(ConvBlock(in_c, feature))
            in_c = feature

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.ReLU(), # Changed to ReLU in the bottleneck
        )

        # --- Decoder (Upsampling Path) ---
        self.up_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for feature in reversed(features):
            # Upsampling via transposed convolution
            self.up_blocks.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=4, stride=2, padding=1, bias=False)
            )
            # Convolutional layers to process the upsampled features
            self.up_convs.append(
                nn.Sequential(
                    nn.InstanceNorm2d(feature),
                    nn.ReLU(),
                    nn.Conv2d(feature, feature, kernel_size=3, stride=1, padding=1)
                )
            )

        # --- Final Output Layer ---
        # This final convolution maps the features to a single-channel realism map.
        self.final_conv = nn.Conv2d(features[0] * 2, 1, kernel_size=4, stride=2, padding=1, padding_mode="reflect")

    def forward(self, x):
        # The output of each downsampling block is saved for skip connections
        skip_connections = []

        x = self.initial(x)
        skip_connections.append(x)

        for block in self.down_blocks:
            x = block(x)
            skip_connections.append(x)

        x = self.bottleneck(x)

        # The skip connections are fed into the decoder in reverse order
        for i, (up_block, up_conv) in enumerate(zip(self.up_blocks, self.up_convs)):
            skip_conn = skip_connections[-(i + 1)]
            x = up_block(x)
            # Concatenate the skip connection features with the upsampled features
            x = torch.cat([x, skip_conn], dim=1)
            x = up_conv(x)

        # No sigmoid activation here. This is handled by the BCEWithLogitsLoss
        # for better numerical stability during training.
        return self.final_conv(x)