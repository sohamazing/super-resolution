# utils/loss.py
import torch
from torch import nn
from torchvision.models import vgg19

from config import config

class VGGLoss(nn.Module):
    """
    Calculates the L1 loss between the feature maps of a pre-trained VGG19 network.
    This is used as the Perceptual Loss component.
    """
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights='IMAGENET1K_V1').features[:36].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(config.DEVICE) # Ensure VGG is moved to the selected device
        self.loss = nn.L1Loss()

    def forward(self, input_features, target_features):
        return self.loss(self.vgg(input_features), self.vgg(target_features))

class FusedGANLoss(nn.Module):
    """
    Manages the combined loss function for the Fused-GAN, including
    Adversarial, Pixel, and Perceptual losses.
    """
    def __init__(self, lambda_pixel=1.0, lambda_adv=0.005, lambda_percep=0.01):
        super().__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_adv = lambda_adv
        self.lambda_percep = lambda_percep
        
        # Numerically stable adversarial loss
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        # Robust pixel-wise loss
        self.pixel_loss = nn.L1Loss()
        # High-level feature loss
        self.perceptual_loss = VGGLoss()

    def calculate_d_loss(self, disc_real, disc_fake):
        """
        Calculates the discriminator loss.
        The goal is to train the discriminator to output 1 for real images and 0 for fake images.
        """
        # Loss for real images (compare discriminator output to a tensor of all 1s)
        loss_real = self.adversarial_loss(disc_real, torch.ones_like(disc_real))
        # Loss for fake images (compare discriminator output to a tensor of all 0s)
        loss_fake = self.adversarial_loss(disc_fake, torch.zeros_like(disc_fake))
        # The total discriminator loss is the average of the two
        return (loss_real + loss_fake) / 2

    def calculate_g_loss(self, disc_fake, fake_hr, real_hr):
        """
        Calculates the generator's combined loss.
        """
        # 1. Adversarial Loss: How well the generator is fooling the discriminator.
        # The generator wants the discriminator to output 1 for its fake images.
        adv_loss = self.adversarial_loss(disc_fake, torch.ones_like(disc_fake))
        # 2. Pixel Loss: How close the generated image is to the ground truth.
        pix_loss = self.pixel_loss(fake_hr, real_hr)
        # 3. Perceptual Loss: How similar the high-level features are.
        perc_loss = self.perceptual_loss(fake_hr, real_hr)
        # Calculate the final weighted loss for the generator
        total_loss = (
            self.lambda_pixel * pix_loss +
            self.lambda_adv * adv_loss +
            self.lambda_percep * perc_loss
        )
        
        # Return all components for detailed logging
        return total_loss, pix_loss, adv_loss, perc_loss