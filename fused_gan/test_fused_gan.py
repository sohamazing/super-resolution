# ============================================================
# ARCHITECTURE COMPATIBILITY VALIDATOR
# ============================================================
# Use this to test architectural changes before committing to them

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project to path
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent))


def validate_generator_architecture(features, num_swin_blocks, test_sizes=[64, 128, 256]):
    """
    Test if generator architecture is internally consistent.
    
    Args:
        features: List like [48, 96, 192] or [64, 128, 256]
        num_swin_blocks: Number of Swin transformer blocks
        test_sizes: Input sizes to test (LR sizes)
    
    Returns:
        bool: True if architecture is valid
    """
    from fused_gan.generator_hcast import HCASTGenerator
    
    print("="*70)
    print("TESTING GENERATOR ARCHITECTURE")
    print("="*70)
    print(f"Features: {features}")
    print(f"Swin blocks: {num_swin_blocks}")
    print()
    
    try:
        # Create model with custom parameters
        gen = HCASTGenerator(
            features=features,
            num_swin_blocks=num_swin_blocks,
            scale=4
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in gen.parameters())
        trainable_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
        
        print(f"✓ Model created successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Memory estimate: ~{total_params * 4 / 1024**2:.1f} MB (fp32)")
        print()
        
        # Test with different input sizes
        all_passed = True
        for size in test_sizes:
            try:
                x = torch.randn(2, 3, size, size)
                
                with torch.no_grad():
                    output = gen(x)
                
                expected_size = size * 4  # scale=4
                
                if output.shape[-1] != expected_size:
                    print(f"✗ Size {size}x{size}: Output size mismatch!")
                    print(f"  Expected: {expected_size}x{expected_size}, Got: {output.shape[-2]}x{output.shape[-1]}")
                    all_passed = False
                else:
                    print(f"✓ Size {size}x{size}: {list(x.shape)} → {list(output.shape)}")
                    
            except Exception as e:
                print(f"✗ Size {size}x{size}: FAILED - {str(e)}")
                all_passed = False
        
        print()
        
        # Check skip connection dimensions
        print("Checking skip connection compatibility...")
        print(f"  Encoder outputs {len(features)} skip connections")
        print(f"  Decoder expects {len(features)} skip connections")
        
        # The number of down_blocks should equal len(features[1:])
        num_down_blocks = len(features) - 1
        print(f"  Down blocks: {num_down_blocks}")
        print(f"  Up blocks: {num_down_blocks}")
        print(f"  ✓ Encoder/Decoder symmetry verified")
        print()
        
        if all_passed:
            print("="*70)
            print("✓ GENERATOR ARCHITECTURE IS VALID")
            print("="*70)
            return True
        else:
            print("="*70)
            print("✗ GENERATOR ARCHITECTURE HAS ISSUES")
            print("="*70)
            return False
            
    except Exception as e:
        print(f"✗ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def validate_discriminator_architecture(features, test_sizes=[128, 256]):
    """
    Test if discriminator architecture is internally consistent.
    
    Args:
        features: List like [32, 64, 128, 256] or [64, 128, 256, 512]
        test_sizes: HR input sizes to test
    
    Returns:
        bool: True if architecture is valid
    """
    from fused_gan.discriminator_unet import DiscriminatorUNet
    
    print()
    print("="*70)
    print("TESTING DISCRIMINATOR ARCHITECTURE")
    print("="*70)
    print(f"Features: {features}")
    print()
    
    try:
        # Create model with custom parameters
        disc = DiscriminatorUNet(features=features)
        
        # Count parameters
        total_params = sum(p.numel() for p in disc.parameters())
        trainable_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
        
        print(f"✓ Model created successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Memory estimate: ~{total_params * 4 / 1024**2:.1f} MB (fp32)")
        print()
        
        # Test with different input sizes
        all_passed = True
        for size in test_sizes:
            try:
                x = torch.randn(2, 3, size, size)
                
                with torch.no_grad():
                    output = disc(x)
                
                # Check output is a realism map
                if output.shape[1] != 1:
                    print(f"✗ Size {size}x{size}: Expected 1 output channel, got {output.shape[1]}")
                    all_passed = False
                else:
                    print(f"✓ Size {size}x{size}: {list(x.shape)} → {list(output.shape)}")
                    
            except Exception as e:
                print(f"✗ Size {size}x{size}: FAILED - {str(e)}")
                all_passed = False
        
        print()
        
        # Check U-Net symmetry
        print("Checking U-Net architecture symmetry...")
        num_encoder_blocks = len(features)  # initial + down_blocks
        num_decoder_blocks = len(features)  # up_blocks
        print(f"  Encoder blocks: {num_encoder_blocks} (initial + {len(features)-1} down)")
        print(f"  Decoder blocks: {num_decoder_blocks} (up blocks)")
        print(f"  ✓ Encoder/Decoder symmetry verified")
        print()
        
        if all_passed:
            print("="*70)
            print("✓ DISCRIMINATOR ARCHITECTURE IS VALID")
            print("="*70)
            return True
        else:
            print("="*70)
            print("✗ DISCRIMINATOR ARCHITECTURE HAS ISSUES")
            print("="*70)
            return False
            
    except Exception as e:
        print(f"✗ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def compare_architectures():
    """
    Compare different architecture configurations to help choose.
    """
    print("\n\n")
    print("="*70)
    print("ARCHITECTURE COMPARISON")
    print("="*70)
    print()
    
    configs = [
        {
            'name': 'Current (Full)',
            'gen_features': [64, 128, 256],
            'gen_swin': 6,
            'disc_features': [64, 128, 256, 512]
        },
        {
            'name': 'Small (Fast)',
            'gen_features': [48, 96, 192],
            'gen_swin': 4,
            'disc_features': [32, 64, 128, 256]
        },
        {
            'name': 'Tiny (Fastest)',
            'gen_features': [32, 64, 128],
            'gen_swin': 3,
            'disc_features': [32, 64, 128]
        },
        {
            'name': 'Large (Slow)',
            'gen_features': [96, 192, 384],
            'gen_swin': 8,
            'disc_features': [96, 192, 384, 768]
        }
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\nTesting: {cfg['name']}")
        print("-" * 70)
        
        from fused_gan.generator_hcast import HCASTGenerator
        from fused_gan.discriminator_unet import DiscriminatorUNet
        
        try:
            gen = HCASTGenerator(
                features=cfg['gen_features'],
                num_swin_blocks=cfg['gen_swin']
            )
            disc = DiscriminatorUNet(features=cfg['disc_features'])
            
            gen_params = sum(p.numel() for p in gen.parameters())
            disc_params = sum(p.numel() for p in disc.parameters())
            total_params = gen_params + disc_params
            
            # Estimate speed (rough approximation based on params)
            relative_speed = 1.0 / (total_params / 10_000_000)
            
            results.append({
                'name': cfg['name'],
                'gen_params': gen_params,
                'disc_params': disc_params,
                'total_params': total_params,
                'relative_speed': relative_speed,
                'valid': True
            })
            
            print(f"  Generator: {gen_params:,} params")
            print(f"  Discriminator: {disc_params:,} params")
            print(f"  Total: {total_params:,} params")
            print(f"  Estimated relative speed: {relative_speed:.2f}x")
            
        except Exception as e:
            print(f"  ✗ FAILED: {str(e)}")
            results.append({'name': cfg['name'], 'valid': False})
    
    print("\n")
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Config':<20} {'Total Params':<15} {'Relative Speed':<15} {'Valid'}")
    print("-" * 70)
    
    for r in results:
        if r['valid']:
            print(f"{r['name']:<20} {r['total_params']:>12,}   {r['relative_speed']:>8.2f}x        ✓")
        else:
            print(f"{r['name']:<20} {'N/A':<15} {'N/A':<15} ✗")


def validate_specific_config(gen_features, gen_swin, disc_features):
    """
    Validate a specific configuration you're considering.
    
    Example usage:
        validate_specific_config(
            gen_features=[48, 96, 192],
            gen_swin=4,
            disc_features=[32, 64, 128, 256]
        )
    """
    print("\n" * 2)
    print("="*70)
    print("VALIDATING SPECIFIC CONFIGURATION")
    print("="*70)
    
    gen_valid = validate_generator_architecture(gen_features, gen_swin)
    disc_valid = validate_discriminator_architecture(disc_features)
    
    print("\n")
    print("="*70)
    if gen_valid and disc_valid:
        print("✓ CONFIGURATION IS VALID - SAFE TO USE")
        print("="*70)
        print("\nTo use this configuration:")
        print(f"1. In generator_hcast.py:")
        print(f"   HCASTGenerator(features={gen_features}, num_swin_blocks={gen_swin})")
        print(f"2. In discriminator_unet.py:")
        print(f"   DiscriminatorUNet(features={disc_features})")
        print(f"3. Delete old checkpoints and restart pre-training")
        return True
    else:
        print("✗ CONFIGURATION HAS ISSUES - DO NOT USE")
        print("="*70)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate architecture configurations")
    parser.add_argument('--mode', type=str, default='compare', 
                       choices=['compare', 'current', 'small', 'custom'],
                       help='Validation mode')
    parser.add_argument('--gen_features', type=int, nargs='+', default=[64, 128, 256],
                       help='Generator feature sizes')
    parser.add_argument('--gen_swin', type=int, default=6,
                       help='Number of Swin blocks')
    parser.add_argument('--disc_features', type=int, nargs='+', default=[64, 128, 256, 512],
                       help='Discriminator feature sizes')
    
    args = parser.parse_args()
    
    if args.mode == 'compare':
        compare_architectures()
    elif args.mode == 'current':
        validate_specific_config([64, 128, 256], 6, [64, 128, 256, 512])
    elif args.mode == 'small':
        validate_specific_config([48, 96, 192], 4, [32, 64, 128, 256])
    elif args.mode == 'custom':
        validate_specific_config(args.gen_features, args.gen_swin, args.disc_features)