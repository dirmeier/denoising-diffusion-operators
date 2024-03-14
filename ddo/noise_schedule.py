import numpy as np


def cosine_alpha_schedule(timesteps, s=0.008):
    """Cosine alpha schedule."""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = np.clip(alphas, a_min=0.0001, a_max=0.9999)
    return alphas


def linear_alpha_schedule(timesteps, beta_min=0.0001, beta_max=0.02):
    """Linear alpha schedule."""
    steps = timesteps + 1
    betas = np.linspace(beta_min, beta_max, steps)
    alphas = 1.0 - betas
    return alphas
