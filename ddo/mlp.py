import dataclasses

import jax
from flax import linen as nn
from jax import numpy as jnp


def _timestep_embedding(inputs, embedding_dim, max_positions=10000):
    assert len(inputs.shape) == 1
    half_dim = embedding_dim // 2
    emb = jnp.log(max_positions) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    emb = inputs[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    return emb


class MLP(nn.Module):
    output_dim = 2
    hidden_dims = [256, 256]
    embedding_dim = 256

    @nn.compact
    def __call__(self, inputs, times, is_training):
        dropout_rate = 0.1 if is_training else 0.0
        times = _timestep_embedding(times, self.embedding_dim)
        t_embedding = jax.nn.swish(
            nn.Dense(self.embedding_dim)(times)
        )
        h = nn.Dense(self.embedding_dim)(inputs)
        h += t_embedding

        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = jax.nn.swish(h)

        h = nn.LayerNorm(use_bias=True, use_scale=True)(h)
        h = nn.Dropout(dropout_rate)(h, deterministic = not is_training)
        h = nn.Dense(self.output_dim)(h)
        return h


