from collections.abc import Iterable, Sequence
from typing import Union

import jax
from flax import linen as nn
from flax.linen import initializers
from jax import numpy as jnp


def _timestep_embedding(timesteps, embedding_dim: int, dtype=jnp.float32):
    """Sinusoidal embedding.

    From https://github.com/google-research/vdm/blob/main/model_vdm.py#L298C1-L323C13
    """
    assert len(timesteps.shape) == 1
    timesteps *= 1000.0

    half_dim = embedding_dim // 2
    emb = jnp.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
    emb = timesteps.astype(dtype)[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class SpectralConvolution(nn.Module):
    """The Fourier part of an operator layer.

    Adopted from
    - https://github.com/neuraloperator/neuraloperator/blob/main
    /neuralop/layers/spectral_convolution.py#L553
    - https://github.com/neuraloperator/GANO/blob/main/GANO_GRF.ipynb
    """

    n_out_channels: int
    n_modes: Union[int, tuple[int]]
    n_groups: int

    @staticmethod
    def mult(lhs, rhs):
        return jnp.einsum("bcxy,coxy->boxy", lhs, rhs)

    @nn.compact
    def __call__(self, inputs, height, width):
        n_modes = (
            self.n_modes
            if isinstance(self.n_modes, tuple)
            else (self.n_modes, self.n_modes)
        )
        inputs = inputs.transpose(0, 3, 1, 2)
        B, C, H, W = inputs.shape

        lweights = self.param(
            "lhs",
            initializers.glorot_normal(dtype=jnp.complex64),
            (C, self.n_out_channels, *n_modes),
            jnp.complex64,
        )
        rweights = self.param(
            "rhs",
            initializers.glorot_normal(dtype=jnp.complex64),
            (C, self.n_out_channels, *n_modes),
        )

        hidden = inputs
        hidden_ft = jnp.fft.rfft2(hidden, norm="forward")

        # TODO(simon): check if this is correct
        out_ft = jnp.zeros(
            (inputs.shape[0], self.n_out_channels, height, width // 2 + 1),
            dtype=jnp.complex64,
        )
        out_ft.at[:, :, : n_modes[0], : n_modes[1]].set(
            self.mult(hidden_ft[:, :, : n_modes[0], : n_modes[1]], lweights)
        )
        out_ft.at[:, :, -n_modes[0] :, : n_modes[1]].set(
            self.mult(hidden_ft[:, :, -n_modes[0] :, : n_modes[1]], rweights)
        )

        outputs = jnp.fft.irfft2(out_ft, s=(height, width), norm="forward")
        outputs = outputs.transpose(0, 2, 3, 1)
        return outputs


class PointwiseForward(nn.Module):
    """Point-wise forward operation.

    Not clear if in the paper this is the same, as any point-wise function
    space transform will do. A 1x1 convolution fulfills this requirement.

    The resizing is needed, because some DiffusionFNO layers change
    dimensionality.
    """

    n_out_channels: int

    @nn.compact
    def __call__(self, inputs, height, width):
        hidden = nn.Conv(self.n_out_channels, (1, 1))(inputs)
        B, _, _, C = hidden.shape
        output = jax.image.resize(
            hidden, (B, height, width, C), method="bilinear"
        )
        return output


class DiffusionFNO(nn.Module):
    """Fourier neural operator layer."""

    n_out_channels: int
    n_modes: int
    n_groups: int = 0

    @nn.compact
    def __call__(self, inputs, height, width):
        hidden = inputs
        hidden_c = SpectralConvolution(
            self.n_out_channels, self.n_modes, self.n_groups
        )(hidden, height, width)
        hidden_p = PointwiseForward(self.n_out_channels)(hidden, height, width)
        outputs = nn.gelu(hidden_c + hidden_p)
        return outputs


class DiffusionUNOBlock(nn.Module):
    """UNet block for diffusion models.

    Does two convolutions and adds the time embedding in between.
    """

    n_out_channels: int
    n_modes: int
    n_groups: int = 0

    @nn.compact
    def __call__(self, inputs, times, height, width):
        time_embedding = nn.gelu(nn.Dense(self.n_out_channels)(times))
        hidden = inputs

        hidden = DiffusionFNO(self.n_out_channels, self.n_modes, self.n_groups)(
            hidden, height, width
        )
        hidden = nn.gelu(hidden)
        hidden = hidden + time_embedding[:, None, None, :]
        hidden = DiffusionFNO(self.n_out_channels, self.n_modes, self.n_groups)(
            hidden, height, width
        )

        return hidden


class UNO(nn.Module):
    """UNO network with time embeddings.

    Can be used as score network for a diffusion model that uses a UNet
    with neural operator layers.
    """

    n_channels: int
    channel_multipliers: Sequence[int]
    n_modes: Sequence[int]

    @staticmethod
    def time_embedding(times):
        times = _timestep_embedding(times, 64)
        times = nn.Sequential([nn.Dense(64), nn.gelu, nn.Dense(64)])(times)
        return times

    @nn.compact
    def __call__(self, inputs, times, **kwargs):
        # time embedding using sinusoidal embedding
        times = self.time_embedding(times)

        # shape of input (note that the channel/feature/filter dimension is last in Flax)
        B, H, W, C = inputs.shape
        hidden = inputs
        # increase number of channels (in paper called lifting)
        hidden = nn.Conv(self.n_channels, kernel_size=(1, 1))(hidden)

        # left part of u-net
        hs = []
        for i, (channel_mult, n_modes) in enumerate(
            zip(self.channel_multipliers, self.n_modes)
        ):
            n_outchannels = channel_mult * self.n_channels
            # do convolutions (via spectral convolution layers)
            # increase C dimension
            block = DiffusionUNOBlock(
                n_out_channels=n_outchannels, n_modes=n_modes
            )
            hidden = block(hidden, times, hidden.shape[1], hidden.shape[2])
            hs.append(hidden)
            # do pooling (via spectral convolutions too)
            # reduce H and W dimensions
            block = DiffusionUNOBlock(
                n_out_channels=n_outchannels, n_modes=n_modes // 2
            )
            hidden = block(
                hidden, times, hidden.shape[1] // 2, hidden.shape[2] // 2
            )

        # mid part of u-net
        # do convolutions (via spectral convolution layers)
        # leaves all dimensions intact
        # in original paper increases C dimension
        block = DiffusionUNOBlock(
            n_out_channels=hidden.shape[-1], n_modes=self.n_modes[-1]
        )
        hidden = block(hidden, times, hidden.shape[1], hidden.shape[2])
        hs.append(hidden)

        # right part of u-net
        hidden = hs.pop()
        for i, (channel_mult, n_modes) in enumerate(
            zip(reversed(self.channel_multipliers), reversed(self.n_modes))
        ):
            n_outchannels = channel_mult * 64
            # do convolutions (via spectral convolution layers)
            # increase H and W dimensions
            block = DiffusionUNOBlock(
                n_out_channels=n_outchannels, n_modes=n_modes
            )
            hidden = block(
                hidden, times, hidden.shape[1] * 2, hidden.shape[2] * 2
            )
            # do convolutions (via spectral convolution layers)
            # reduce C dimensions and does residual computations
            block = DiffusionUNOBlock(
                n_out_channels=n_outchannels, n_modes=n_modes
            )
            hidden = block(
                jnp.concatenate([hidden, hs.pop()], axis=-1),
                times,
                hidden.shape[1],
                hidden.shape[2],
            )

        # reduce number of channels
        outputs = nn.Conv(C, kernel_size=(1, 1))(hidden)
        return outputs
