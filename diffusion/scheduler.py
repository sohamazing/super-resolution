# diffusion/scheduler.py
import torch
import torch.nn.functional as F

def linear_beta_schedule(timesteps):
    """Creates a linear noise schedule from beta_start to beta_end."""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

class Scheduler:
    def __init__(self, timesteps=1000, device='cpu'):
        self.timesteps = timesteps

        # Define the noise schedule (betas)
        self.betas = linear_beta_schedule(timesteps).to(device)

        # Pre-calculate the alpha values based on the betas
        self.alphas = (1. - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device) # Cumulative product of alphas
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)

        # Pre-calculated values for the forward process (add_noise)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(device)

        # Pre-calculated values for the reverse process (sampling)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(device)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod).to(device)


    def add_noise(self, x_start, t, noise=None):
        """
        Takes a clean image and a timestep 't' and returns a noisy version.
        This is the 'forward process' and uses a closed-form solution to jump to any 't'.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Get the pre-calculated values for the given timestep 't'
        sqrt_alphas_cumprod_t = self._get_index_from_list(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        # Apply the noise formula: noisy_image = sqrt(alpha_cumprod_t) * x_start + sqrt(1 - alpha_cumprod_t) * noise
        noisy_image = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_image

    def sample_previous_timestep(self, x_t, t, predicted_noise):
        """
        Takes the model's noise prediction and the current noisy image 'x_t'
        to calculate the slightly cleaner image at step 't-1'. This is the 'reverse process'.
        """
        # Get the pre-calculated values for the given timestep 't'
        betas_t = self._get_index_from_list(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = self._get_index_from_list(self.sqrt_recip_alphas, t, x_t.shape)

        # Calculate the mean of the distribution for the previous timestep using the model's prediction
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        # Get the variance and add a small amount of random noise (except for the last step)
        posterior_variance_t = self._get_index_from_list(self.posterior_variance, t, x_t.shape)
        if t[0].item() == 0:
            return model_mean
        else:
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def _get_index_from_list(self, values, t, x_shape):
        """Helper function now assumes 'values' is already on the correct device."""
        batch_size = t.shape[0]
        out = values.gather(-1, t) # No more .to(device) or .cpu() calls needed here, avoiding the MPS bug.
        # Reshape to match the image batch dimensions for broadcasting
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))