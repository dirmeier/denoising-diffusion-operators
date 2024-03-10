from jax import numpy as jnp


def cosine_alpha_schedule(timesteps, s=0.008):
    """Cosine alpha schedule."""
    steps = timesteps + 1
    x = jnp.linspace(0, steps, steps)
    alphas_cumprod = jnp.cos(((x / steps) + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = jnp.clip(alphas, a_min=0.001, a_max=0.9999)
    return alphas
