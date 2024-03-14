import jax
import numpy as np
from flax import linen as nn
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class DenoisingDiffusionOperator(nn.Module):
    score_model: nn.Module
    alpha_schedule: Array

    def setup(self):
        self.n_diffusions = len(self.alpha_schedule)
        self._alphas = jnp.asarray(self.alpha_schedule)
        self._betas = 1.0 - self._alphas
        self._alphas_bar = jnp.cumprod(self._alphas)
        self._sqrt_alphas_bar = jnp.sqrt(self._alphas_bar)
        self._sqrt_1m_alphas_bar = jnp.sqrt(1.0 - self._alphas_bar)

    def __call__(self, method="loss", **kwargs):
        return getattr(self, method)(**kwargs)

    def loss(self, y0, is_training):
        rng_key = self.make_rng("sample")
        time_key, rng_key = jr.split(rng_key)
        times = jr.randint(
            key=time_key,
            minval=0,
            maxval=self.n_diffusions,
            shape=(y0.shape[0],),
        )

        noise_key, rng_key = jr.split(rng_key)
        noise = jax.random.normal(noise_key, y0.shape)
        yt = self.q_pred_reparam(y0, times, noise)
        eps = self.score_model(yt, times, is_training=is_training)
        loss = jnp.sum(jnp.square(noise - eps), axis=range(1, yt.ndim))

        return loss

    def q_pred_mean(self, y0, t):
        shape = (-1,) + tuple(np.ones(y0.ndim - 1, dtype=np.int32).tolist())
        return self._sqrt_alphas_bar[t].reshape(shape) * y0

    def q_pred_reparam(self, y0, t, noise):
        shape = (-1,) + tuple(np.ones(y0.ndim - 1, dtype=np.int32).tolist())
        mean = self.q_pred_mean(y0, t)
        scale = self._sqrt_1m_alphas_bar[t].reshape(shape) * noise
        return mean + scale

    def sample(self, sample_shape=(32, 32, 32, 1), n=100, **kwargs):
        timesteps = np.arange(0, self.n_diffusions, self.n_diffusions // n)
        yt = jr.normal(self.make_rng("sample"), sample_shape)
        for t in reversed(np.arange(1, n)):
            tprev, tcurr = timesteps[(t - 1) : (t + 1)]
            yt = self._denoise(yt, tcurr, tprev)
        return yt

    def _denoise(self, yt, t, tprev):
        eps = self.score_model(yt, jnp.full(yt.shape[0], t), is_training=False)
        lhs = (yt - eps * self._sqrt_1m_alphas_bar[t]) / self._sqrt_alphas_bar[
            t
        ]
        lhs = self._sqrt_alphas_bar[tprev] * lhs
        rhs = self._sqrt_1m_alphas_bar[tprev] * eps
        # we use the implicit version that uses sigma_t = 0.0
        ytm1 = lhs + rhs
        return ytm1
