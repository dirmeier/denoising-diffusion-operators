from collections.abc import Sequence

import jax
from flax import linen as nn
from flax.linen import initializers
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


class SpectralConvolution(nn.Module):
    """The Fourier part of an operator layer.

    Adopted from
    - https://github.com/neuraloperator/neuraloperator/blob/main
    /neuralop/layers/spectral_convolution.py#L553
    - https://github.com/neuraloperator/GANO/blob/main/GANO_GRF.ipynb
    """

    n_out_channels: int
    n_modes: int | tuple[int]

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
            jnp.complex64,
        )

        hidden = inputs
        hidden_ft = jnp.fft.rfft2(hidden, norm="forward")

        out_ft = jnp.zeros(
            (inputs.shape[0], self.n_out_channels, height, width // 2 + 1),
            dtype=jnp.complex64,
        )
        out_ft = out_ft.at[:, :, : n_modes[0], : n_modes[1]].set(
            self.mult(hidden_ft[:, :, : n_modes[0], : n_modes[1]], lweights)
        )
        out_ft = out_ft.at[:, :, -n_modes[0] :, : n_modes[1]].set(
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
        hidden = nn.Conv(
            self.n_out_channels, (1, 1), strides=(1, 1), padding="SAME"
        )(inputs)
        B, _, _, C = hidden.shape
        output = jax.image.resize(
            hidden, (B, height, width, C), method="bilinear"
        )
        return output


class FourierNeuralOperator(nn.Module):
    """Fourier neural operator layer."""

    n_out_channels: int
    n_modes: int

    @nn.compact
    def __call__(self, inputs, height, width):
        hidden = inputs
        hidden_c = SpectralConvolution(self.n_out_channels, self.n_modes)(
            hidden, height, width
        )
        hidden_p = PointwiseForward(self.n_out_channels)(hidden, height, width)
        outputs = hidden_c + hidden_p
        return outputs


FNO = FourierNeuralOperator


class UNOResidualBlock(nn.Module):
    """UNet block for diffusion operator models.

    Does two convolutions and adds the time embedding in between. Also does
    group normalisation and dropout
    """

    n_out_channels: int
    n_modes: int
    dropout_rate: float
    down_or_up_sample: bool = False

    @nn.compact
    def __call__(self, inputs, times, height, width, is_training):
        time_embedding = nn.Dense(self.n_out_channels, use_bias=False)(
            nn.swish(times)
        )
        hidden = inputs

        # convolution with pre-layer norm
        hidden = nn.BatchNorm()(hidden, use_running_average=not is_training)
        hidden = nn.swish(hidden)
        hidden = FNO(self.n_out_channels, self.n_modes)(hidden, height, width)

        # time conditioning
        hidden = hidden + time_embedding[:, None, None, :]

        # convolution with pre-layer norm
        hidden = nn.BatchNorm()(hidden, use_running_average=not is_training)
        hidden = nn.swish(hidden)
        hidden = nn.Dropout(self.dropout_rate)(
            hidden, deterministic=not is_training
        )
        hidden = FNO(self.n_out_channels, self.n_modes)(hidden, height, width)

        if (
            inputs.shape[-1] != self.n_out_channels
            and not self.down_or_up_sample
        ):
            residual = nn.Conv(
                self.n_out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
            )(inputs)
        else:
            residual = inputs
        if not self.down_or_up_sample:
            hidden = hidden + residual
        return hidden / 1.414213


class UNO(nn.Module):
    """UNO network with time embeddings.

    Can be used as score network for a diffusion model that uses a UNet
    with neural operator layers.
    """

    n_blocks: int
    n_channels: int
    n_modes: int
    channel_multipliers: Sequence[int]
    dropout_rate: float
    do_pooling_via_fno: bool

    def time_embedding(self, times):
        times = _timestep_embedding(times, self.n_channels * 2)
        times = nn.Sequential(
            [
                nn.Dense(self.n_channels * 8),
                nn.swish,
                nn.Dense(self.n_channels * 8),
                nn.swish,
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
            self.n_channels, kernel_size=(1, 1), strides=(1, 1), padding="SAME"
        )(hidden)

        hs = []
        # left block of UNet
        for channel_mult, n_modes in zip(
            self.channel_multipliers[:-1], self.n_modes[:-1]
        ):
            n_outchannels = channel_mult * self.n_channels
            for _ in range(self.n_blocks):
                block = UNOResidualBlock(
                    n_outchannels, n_modes, self.dropout_rate
                )
                hidden = block(
                    hidden, times, hidden.shape[1], hidden.shape[2], is_training
                )
                hs.append(hidden)
            if not self.do_pooling_via_fno:
                hidden = nn.avg_pool(
                    hidden, window_shape=(2, 2), strides=(2, 2)
                )
            else:
                # do pooling (via spectral convolutions too)
                block = UNOResidualBlock(
                    n_outchannels,
                    n_modes // 2,
                    self.dropout_rate,
                    down_or_up_sample=True,
                )
                hidden = block(
                    hidden,
                    times,
                    hidden.shape[1] // 2,
                    hidden.shape[2] // 2,
                    is_training,
                )

        # middle block of UNet
        for _ in range(self.n_blocks):
            block = UNOResidualBlock(
                hidden.shape[-1], self.n_modes[-1], self.dropout_rate
            )
            hidden = block(
                hidden, times, hidden.shape[1], hidden.shape[2], is_training
            )

        for channel_mult, n_modes in zip(
            reversed(self.channel_multipliers[:-1]), reversed(self.n_modes[:-1])
        ):
            n_outchannels = channel_mult * self.n_channels
            if not self.do_pooling_via_fno:
                hidden = jax.image.resize(
                    hidden,
                    (
                        B,
                        hidden.shape[1] * 2,
                        hidden.shape[2] * 2,
                        hidden.shape[3],
                    ),
                    method="bilinear",
                )
            else:
                block = UNOResidualBlock(
                    n_outchannels,
                    n_modes // 2,
                    self.dropout_rate,
                    down_or_up_sample=True,
                )
                hidden = block(
                    hidden,
                    times,
                    hidden.shape[1] * 2,
                    hidden.shape[2] * 2,
                    is_training,
                )
            for _ in range(self.n_blocks):
                block = UNOResidualBlock(
                    n_outchannels, n_modes, self.dropout_rate
                )
                hidden = jnp.concatenate([hidden, hs.pop()], axis=-1)
                hidden = block(
                    hidden, times, hidden.shape[1], hidden.shape[2], is_training
                )

        hidden = nn.BatchNorm()(hidden, use_running_average=not is_training)
        hidden = nn.swish(hidden)
        outputs = nn.Conv(
            C,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            kernel_init=nn.initializers.zeros,
        )(hidden)
        return outputs
