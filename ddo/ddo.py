import jax
from flax import linen as nn
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class DenoisingDiffusionOperator(nn.Module):
    score_model: nn.Module
    alpha_schedule: Array
    n_langevin_steps: int

    def setup(self):
        self.n_diffusions = len(self.alpha_schedule)
        self._alphas = self.alpha_schedule
        self._betas = jnp.asarray(1.0 - self._alphas)
        self._alphas_bar = jnp.cumprod(self._alphas)
        self._sqrt_alphas_bar = jnp.sqrt(self._alphas_bar)
        self._sqrt_1m_alphas_bar = jnp.sqrt(1.0 - self._alphas_bar)

    def __call__(self, method="loss", **kwargs):
        return getattr(self, method)(**kwargs)

    def loss(self, y_0, is_training):
        rng_key = self.make_rng("sample")
        time_key, rng_key = jr.split(rng_key)
        times = jr.choice(
            key=time_key,
            a=self.n_diffusions,
            shape=(y_0.shape[0],),
        )

        noise_key, rng_key = jr.split(rng_key)
        noise = jax.random.normal(noise_key, y_0.shape)
        y_t = self.q_pred_reparam(y_0, times, noise)
        eps = self.score_model(y_t, times, is_training=is_training)
        loss = jnp.sum(jnp.square(noise - eps), axis=[1, 2, 3])

        return loss

    def q_pred_mean(self, y_0, t):
        return self._sqrt_alphas_bar[t].reshape(-1, 1, 1, 1) * y_0

    def q_pred_reparam(self, y_0, t, noise):
        mean = self.q_pred_mean(y_0, t)
        scale = self._sqrt_1m_alphas_bar[t].reshape(-1, 1, 1, 1) * noise
        return mean + scale

    def sample(self, sample_shape=(32, 32, 32, 1)):
        init_key, rng_key = jr.split(self.make_rng("sample"))

        yt = jr.normal(init_key, sample_shape)
        for t in reversed(range(self.n_diffusions)):
            z = jr.normal(jr.fold_in(rng_key, t), sample_shape)
            eps = self.score_model(
                yt, jnp.full(yt.shape[0], t), is_training=False
            )
            yn = self._betas[t] / self._sqrt_1m_alphas_bar[t] * eps
            yn = yt - yn
            yn = yn / jnp.sqrt(self._alphas[t])
            yt = yn + jnp.sqrt(self._betas[t]) * z

        return yt
