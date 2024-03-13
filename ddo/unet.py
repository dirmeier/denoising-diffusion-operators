from collections.abc import Sequence

import jax
from flax import linen as nn
from jax import numpy as jnp


def _timestep_embedding(timesteps, embedding_dim: int, dtype=jnp.float32):
    """Sinusoidal embedding.

    From https://github.com/google-research/vdm/blob/main/model_vdm.py#L298C1-L323C13
    Does not use timesteps * 1000 since we sample them discretely in (1, timesteps]
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = jnp.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
    emb = timesteps.astype(dtype)[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class UNetBlock(nn.Module):
    """UNet block for diffusion models.

    Does two convolutions and adds the time embedding in between. Also does
    group normalisation and dropout
    """

    n_out_channels: int
    n_groups: int = 8
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, times, is_training):
        time_embedding = nn.Dense(self.n_out_channels, use_bias=False)(
            nn.swish(times)
        )
        hidden = inputs

        # convolution with pre-layer norm
        hidden = nn.GroupNorm(self.n_groups)(hidden)
        hidden = nn.swish(hidden)
        hidden = nn.Conv(
            self.n_out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
        )(hidden)

        # time conditioning
        hidden = hidden + time_embedding[:, None, None, :]

        # convolution with pre-layer norm
        hidden = nn.GroupNorm(self.n_groups)(hidden)
        hidden = nn.swish(hidden)
        hidden = nn.Dropout(self.dropout_rate)(
            hidden, deterministic=not is_training
        )
        hidden = nn.Conv(
            self.n_out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
        )(hidden)

        if inputs.shape[-1] != self.n_out_channels:
            inputs = nn.Conv(
                self.n_out_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
            )(inputs)
        return hidden + inputs


class UNet(nn.Module):
    """UNet with time embeddings.

    Can be used as score network for a diffusion model
    """

    n_blocks: int
    n_channels: int
    dim_embedding: int
    channel_multipliers: Sequence[int]
    n_groups: int = 8

    def time_embedding(self, times):
        times = _timestep_embedding(times, self.dim_embedding)
        times = nn.Sequential(
            [
                nn.Dense(self.dim_embedding * 2),
                nn.swish,
                nn.Dense(self.dim_embedding * 2),
            ]
        )(times)
        return times

    @nn.compact
    def __call__(self, inputs, times, is_training, **kwargs):
        # time embedding using sinusoidal embedding
        times = self.time_embedding(times)

        B, H, W, C = inputs.shape
        hidden = inputs
        # lift data
        hidden = nn.Conv(
            self.n_channels, kernel_size=(3, 3), strides=(1, 1), padding="SAME"
        )(hidden)

        hs = []
        # left block of UNet
        for i, channel_mult in enumerate(self.channel_multipliers):
            n_outchannels = channel_mult * self.n_channels
            for _ in range(self.n_blocks):
                hidden = UNetBlock(n_outchannels)(hidden, times, is_training)
            hs.append(hidden)
            hidden = nn.max_pool(hidden, window_shape=(2, 2), strides=(2, 2))

        # middle block of UNet
        for _ in range(self.n_blocks):
            hidden = UNetBlock(n_out_channels=hidden.shape[-1])(
                hidden, times, is_training
            )
        hs.append(hidden)

        hidden = hs.pop()
        # right block of UNet
        for i, channel_mult in enumerate(reversed(self.channel_multipliers)):
            n_outchannels = channel_mult * self.n_channels
            hidden = nn.ConvTranspose(
                n_outchannels, kernel_size=(2, 2), strides=(2, 2)
            )(hidden)
            for bl in range(self.n_blocks):
                hidden = (
                    jnp.concatenate([hidden, hs.pop()], axis=-1)
                    if bl == 0
                    else hidden
                )
                hidden = UNetBlock(n_out_channels=n_outchannels)(
                    hidden, times, is_training
                )

        hidden = nn.GroupNorm(self.n_groups)(hidden)
        hidden = nn.swish(hidden)
        outputs = nn.Conv(C, kernel_size=(1, 1))(hidden)
        return outputs
