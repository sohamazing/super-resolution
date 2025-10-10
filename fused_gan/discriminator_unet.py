# fused_gan/discriminator_unet.py
import torch
from torch import nn

class ConvBlock(nn.Module):
    """A standard convolutional block for the U-Net discriminator."""
    def __init__(self, in_channels, out_channels, use_instance_norm=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, 
                     bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels) if use_instance_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DiscriminatorUNet(nn.Module):
    """
    U-Net based discriminator with corrected skip connection logic.
    Provides spatial "realism map" for patch-based feedback.
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        # --- Encoder (Downsampling Path) ---
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, 
                     padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.down_blocks = nn.ModuleList()
        in_c = features[0]
        for feature in features[1:]:
            self.down_blocks.append(ConvBlock(in_c, feature))
            in_c = feature

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1], kernel_size=4, stride=2, padding=1, 
                     padding_mode="reflect"),
            nn.ReLU(),
        )

        # --- Decoder (Upsampling Path) ---
        self.up_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        reversed_features = list(reversed(features))  # [512, 256, 128, 64]
        
        for i in range(len(reversed_features)):
            in_channels_concat = reversed_features[i] * 2  # Concat doubles channels
            out_channels = reversed_features[i]
            
            # Upsampling layer operates on concatenated features
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels_concat, out_channels, 
                                      kernel_size=4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
            # Additional processing after upsampling
            self.up_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

        # --- Final Output Layer ---
        self.final_conv = nn.Conv2d(features[0], 1, kernel_size=4, stride=2, padding=1, 
                                    padding_mode="reflect")

    def forward(self, x):
        # Store skip connections during encoding
        skip_connections = []
        
        x = self.initial(x)
        skip_connections.append(x)

        for block in self.down_blocks:
            x = block(x)
            skip_connections.append(x)

        x = self.bottleneck(x)

        for i, (up_block, up_conv) in enumerate(zip(self.up_blocks, self.up_convs)):
            # Get corresponding skip connection (reverse order)
            skip_idx = -(i + 1)
            skip_conn = skip_connections[skip_idx]
            
            # Concatenate before upsampling 
            x = torch.cat([x, skip_conn], dim=1)
            
            # Now upsample the concatenated features
            x = up_block(x)
            
            # Additional refinement
            x = up_conv(x)

        return self.final_conv(x)


# ============== VERIFICATION TEST ==============
def test_discriminator():
    """
    Test to verify the discriminator architecture handles the exact scenario 
    that caused the original error.
    """
    print("Testing DiscriminatorUNet Architecture...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    disc = DiscriminatorUNet().to(device)
    
    # Test with exact batch size from error (16) and typical HR image size
    batch_size = 16
    test_input = torch.randn(batch_size, 3, 128, 128).to(device)
    
    try:
        with torch.no_grad():
            output = disc(test_input)
        
        print(f"✓ SUCCESS!")
        print(f"  Input shape:  {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected output: [16, 1, H, W] (realism map)")
        
        # Verify output is a valid realism map
        assert output.shape[0] == batch_size
        assert output.shape[1] == 1  # Single channel realism map
        print("\n✓ All architecture checks passed!")
        return True
        
    except RuntimeError as e:
        print(f"✗ FAILED with error:")
        print(f"  {str(e)}")
        return False


if __name__ == "__main__":
    test_discriminator()