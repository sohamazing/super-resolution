# fused_gan/discriminator_unet.py
import torch
from torch import nn
import torch.nn.functional as F


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
    U-Net based discriminator with CORRECTED skip connection architecture.
    
    Architecture flow (for 128x128 input):
    Encoder:
      initial: 128 -> 64  [skip_0]
      down[0]: 64 -> 32   [skip_1]
      down[1]: 32 -> 16   [skip_2]
      down[2]: 16 -> 8    [skip_3]
      bottleneck: 8 -> 4
    
    Decoder (skip connections must match spatially):
      up[0]: concat(4, skip_3:8) -> up to 8 -> refine
      up[1]: concat(8, skip_2:16) -> up to 16 -> refine
      up[2]: concat(16, skip_1:32) -> up to 32 -> refine
      up[3]: concat(32, skip_0:64) -> up to 64 -> refine
      final: 64 -> 32 -> output
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
        # CRITICAL: Each decoder step must account for the output of the previous step
        self.up_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        reversed_features = list(reversed(features))  # [512, 256, 128, 64]
        
        # Build decoder blocks - each takes input from previous decoder's output
        for i in range(len(reversed_features)):
            # Input channels = output from previous decoder step (or bottleneck for first)
            # For first iteration: bottleneck outputs features[-1] = 512
            # For subsequent: previous up_conv outputs reversed_features[i-1]
            in_channels_up = reversed_features[i-1] if i > 0 else features[-1]
            out_channels = reversed_features[i]
            
            # Step 1: Upsample to match skip connection's spatial dimensions
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels_up, out_channels, 
                                      kernel_size=4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
            # Step 2: Process concatenated features (upsampled + skip connection)
            self.up_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

        # --- Final Output Layer ---
        self.final_conv = nn.Conv2d(features[0], 1, kernel_size=4, stride=2, padding=1, 
                                    padding_mode="reflect")

    def forward(self, x):
        """
        Forward pass with proper skip connection management.
        
        The key is: skip_connections are in encoder order [shallow -> deep]
        We need them in decoder order [deep -> shallow], so we reverse or pop.
        """
        # Store skip connections during encoding
        skip_connections = []
        
        x = self.initial(x)
        skip_connections.append(x)

        for block in self.down_blocks:
            x = block(x)
            skip_connections.append(x)

        x = self.bottleneck(x)

        # Reverse skip connections so we can iterate in decoder order (deep -> shallow)
        skip_connections = skip_connections[::-1]

        # Decoder: upsample, then concatenate with matching skip, then refine
        for i, (up_block, up_conv) in enumerate(zip(self.up_blocks, self.up_convs)):
            # Step 1: Upsample to match skip connection's spatial size
            x = up_block(x)
            
            # Step 2: Get matching skip connection
            skip_conn = skip_connections[i]
            
            # Step 3: Ensure spatial dimensions match (safety check for any rounding issues)
            if x.shape[2:] != skip_conn.shape[2:]:
                x = F.interpolate(x, size=skip_conn.shape[2:], mode='bilinear', align_corners=False)
            
            # Step 4: Concatenate along channel dimension
            x = torch.cat([x, skip_conn], dim=1)
            
            # Step 5: Process the concatenated features
            x = up_conv(x)

        return self.final_conv(x)


# ============== VERIFICATION TEST ==============
def test_discriminator():
    """
    Comprehensive test to verify the discriminator architecture.
    """
    print("="*60)
    print("Testing DiscriminatorUNet Architecture")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    disc = DiscriminatorUNet().to(device)
    
    # Test with multiple input sizes
    test_sizes = [128, 256]
    batch_size = 16
    
    for size in test_sizes:
        print(f"\n📊 Testing with input size: {size}x{size}")
        test_input = torch.randn(batch_size, 3, size, size).to(device)
        
        try:
            with torch.no_grad():
                output = disc(test_input)
            
            print(f"  ✓ Input shape:  {list(test_input.shape)}")
            print(f"  ✓ Output shape: {list(output.shape)}")
            print(f"  ✓ Expected: Single-channel realism map")
            
            # Verify output properties
            assert output.shape[0] == batch_size, f"Batch size mismatch"
            assert output.shape[1] == 1, f"Expected 1 channel, got {output.shape[1]}"
            assert len(output.shape) == 4, f"Expected 4D tensor"
            
            print(f"  ✓ All checks passed for {size}x{size}!")
            
        except RuntimeError as e:
            print(f"  ✗ FAILED with error:")
            print(f"     {str(e)}")
            return False
    
    print("\n" + "="*60)
    print("✓ ALL ARCHITECTURE TESTS PASSED!")
    print("="*60)
    return True


def debug_forward_pass():
    """
    Debug version that prints shapes at each step.
    """
    print("\n" + "="*60)
    print("DEBUG: Detailed Forward Pass Analysis")
    print("="*60)
    
    device = 'cpu'  # Use CPU for debugging
    disc = DiscriminatorUNet().to(device)
    
    x = torch.randn(2, 3, 128, 128).to(device)
    print(f"\nInput: {list(x.shape)}")
    
    # Manual forward pass with prints
    skip_connections = []
    
    # Encoder
    print("\n--- ENCODER ---")
    x = disc.initial(x)
    print(f"After initial: {list(x.shape)}")
    skip_connections.append(x)
    
    for i, block in enumerate(disc.down_blocks):
        x = block(x)
        print(f"After down_block[{i}]: {list(x.shape)}")
        skip_connections.append(x)
    
    x = disc.bottleneck(x)
    print(f"After bottleneck: {list(x.shape)}")
    
    # Decoder
    print("\n--- DECODER ---")
    skip_connections = skip_connections[::-1]
    
    for i, (up_block, up_conv) in enumerate(zip(disc.up_blocks, disc.up_convs)):
        print(f"\nDecoder step {i}:")
        print(f"  Before upsample: {list(x.shape)}")
        
        x = up_block(x)
        print(f"  After upsample: {list(x.shape)}")
        
        skip_conn = skip_connections[i]
        print(f"  Skip connection: {list(skip_conn.shape)}")
        
        if x.shape[2:] != skip_conn.shape[2:]:
            x = F.interpolate(x, size=skip_conn.shape[2:], mode='bilinear', align_corners=False)
            print(f"  After interpolate: {list(x.shape)}")
        
        x = torch.cat([x, skip_conn], dim=1)
        print(f"  After concat: {list(x.shape)}")
        
        x = up_conv(x)
        print(f"  After refinement: {list(x.shape)}")
    
    x = disc.final_conv(x)
    print(f"\nFinal output: {list(x.shape)}")
    print("="*60)


if __name__ == "__main__":
    # Run detailed debug first
    debug_forward_pass()
    
    # Then run full tests
    print("\n" * 2)
    test_discriminator()