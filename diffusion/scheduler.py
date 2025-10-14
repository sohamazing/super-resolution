# diffusion/scheduler.py
"""
Diffusion noise schedulers with both DDPM and DDIM sampling support.
- DDPM: Original stochastic sampling (slower, higher quality)
- DDIM: Deterministic sampling (faster, configurable steps)
"""
import torch
import torch.nn.functional as F
import math

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in 'Improved Denoising Diffusion Probabilistic Models'.
    More stable than linear schedule for image generation.

    Args:
        timesteps: Number of diffusion steps
        s: Small offset to prevent singularities
    Returns:
        torch.Tensor: Beta values for each timestep
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Linear schedule for beta values (original DDPM).

    Args:
        timesteps: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value
    Returns:
        torch.Tensor: Beta values for each timestep
    """
    return torch.linspace(beta_start, beta_end, timesteps)


class DDPMScheduler:
    """
    DDPM (Denoising Diffusion Probabilistic Models) Scheduler.

    Original stochastic sampling - slower but potentially higher quality.
    Uses the full Markov chain with added noise at each step.
    """
    def __init__(self, timesteps=1000, schedule='cosine', device='cpu'):
        """
        Args:
            timesteps: Total number of diffusion steps
            schedule: 'cosine' or 'linear' beta schedule
            device: torch device (cpu, cuda, mps)
        """
        self.timesteps = timesteps
        self.device = device
        self.schedule_type = schedule

        # Define noise schedule
        if schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps).to(device)
        else:
            self.betas = linear_beta_schedule(timesteps).to(device)

        # Pre-compute useful values for efficiency
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)

        # For forward process (q)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)

        # For reverse process (p) - For predicting x_0
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod).to(device)

        # Posterior variance for sampling
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).to(device)

        # Clip for numerical stability
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

        # Coefficients for posterior mean calculation from clipped x_0
        self.posterior_mean_coef1 = (
            torch.sqrt(self.alphas_cumprod_prev) * self.betas / (1.0 - self.alphas_cumprod)
        ).to(device)
        self.posterior_mean_coef2 = (
            torch.sqrt(self.alphas) * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).to(device)


    def add_noise(self, x_start, t, noise=None):
        """
        Forward diffusion: Add noise to clean images.
        Uses closed-form solution: q(x_t | x_0) = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε
        
        Args:
            x_start: (B, C, H, W) - clean images
            t: (B,) - timestep for each image in batch
            noise: (B, C, H, W) - optional pre-generated noise
        Returns:
            (B, C, H, W) - noisy images at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def sample_prev_timestep(self, x_t, t, predicted_noise, clip_denoised=True):
        """
        DDPM reverse diffusion: A more stable version that predicts and clips x_0.
        """
        # 1. Predict x_0 from the model's noise prediction
        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        # Equation to predict x_0: x_0 = (x_t - sqrt(1 - ᾱ_t) * ε_θ) / sqrt(ᾱ_t)
        pred_x0 = sqrt_recip_alphas_cumprod_t * (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise)

        # 2. Clip the predicted x_0 for stability
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # 3. Compute the posterior mean using the clipped x_0
        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x_t.shape)

        model_mean = posterior_mean_coef1_t * pred_x0 + posterior_mean_coef2_t * x_t

        # 4. Add noise (DDPM is stochastic, except for t=0)
        if t[0].item() == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def _extract(self, values, t, shape):
        """
        Extract values at timestep t and reshape for broadcasting.
        """
        batch_size = t.shape[0]
        out = values.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    def __repr__(self):
        return (
            f"DDPMScheduler(\n"
            f"  timesteps={self.timesteps},\n"
            f"  schedule='{self.schedule_type}',\n"
            f"  device={self.device}\n"
            f")"
        )


class DDIMScheduler:
    """
    DDIM (Denoising Diffusion Implicit Models) Scheduler.

    Deterministic sampling - much faster with configurable steps.
    Can use fewer steps than training (e.g., 50 instead of 1000).
    """
    def __init__(self, timesteps=1000, schedule='cosine', device='cpu'):
        """
        Args:
            timesteps: Total number of training diffusion steps
            schedule: 'cosine' or 'linear' beta schedule
            device: torch device (cpu, cuda, mps)
        """
        self.timesteps = timesteps
        self.device = device
        self.schedule_type = schedule

        # Define noise schedule (same as DDPM)
        if schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps).to(device)
        else:
            self.betas = linear_beta_schedule(timesteps).to(device)

        # Pre-compute useful values
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

        # For forward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)

    def add_noise(self, x_start, t, noise=None):
        """
        Forward diffusion: Add noise to clean images.
        Same as DDPM.

        Args:
            x_start: (B, C, H, W) - clean images
            t: (B,) - timestep for each image in batch
            noise: (B, C, H, W) - optional pre-generated noise
        Returns:
            (B, C, H, W) - noisy images at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def sample_prev_timestep(self, x_t, t, t_prev, predicted_noise, eta=0.0, clip_denoised=True):
        """
        DDIM reverse diffusion: Remove noise deterministically (or semi-stochastic with eta > 0).

        Args:
            x_t: (B, C, H, W) - noisy image at timestep t
            t: (B,) - current timestep
            t_prev: (B,) - previous timestep (can skip steps)
            predicted_noise: (B, C, H, W) - model's noise prediction
            eta: Stochasticity parameter (0=deterministic, 1=DDPM-like)
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]
        Returns:
            (B, C, H, W) - less noisy image at timestep t_prev
        """
        # Get alpha values for current and previous timesteps
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_t.shape)

        # Handle t_prev = -1 case (final step)
        if t_prev[0].item() < 0:
            alpha_cumprod_t_prev = torch.ones_like(alpha_cumprod_t)
        else:
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, t_prev, x_t.shape)

        # Predict x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)

        # Clip for stability
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # Compute variance
        sigma_t = eta * torch.sqrt(
            (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) *
            (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
        )

        # Compute direction pointing to x_t
        pred_dir = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t**2) * predicted_noise

        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir

        # Add noise if eta > 0
        if eta > 0 and t[0].item() > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma_t * noise

        return x_prev

    def get_sampling_timesteps(self, num_inference_steps):
        """
        Get subset of timesteps for faster sampling.

        Args:
            num_inference_steps: Number of steps to use for inference
        Returns:
            List of timesteps to use
        """
        # Evenly spaced timesteps
        step = self.timesteps // num_inference_steps
        timesteps = list(range(0, self.timesteps, step))[:num_inference_steps]
        return list(reversed(timesteps))

    def _extract(self, values, t, shape):
        """Extract values at timestep t and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = values.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    def __repr__(self):
        return (
            f"DDIMScheduler(\n"
            f"  timesteps={self.timesteps},\n"
            f"  schedule='{self.schedule_type}',\n"
            f"  device={self.device}\n"
            f")"
        )


# Factory function for easy scheduler creation
def create_scheduler(scheduler_type='ddpm', timesteps=1000, schedule='cosine', device='cpu'):
    """
    Factory function to create appropriate scheduler.

    Args:
        scheduler_type: 'ddpm' or 'ddim'
        timesteps: Number of training timesteps
        schedule: 'cosine' or 'linear'
        device: torch device

    Returns:
        Scheduler instance
    """
    if scheduler_type.lower() == 'ddpm':
        return DDPMScheduler(timesteps=timesteps, schedule=schedule, device=device)
    elif scheduler_type.lower() == 'ddim':
        return DDIMScheduler(timesteps=timesteps, schedule=schedule, device=device)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Must be 'ddpm' or 'ddim'")
