import pathlib
import pickle
import warnings

from flax.training.train_state import TrainState

from util import get_minst_sdf_data, get_nine_gaussians

warnings.filterwarnings("ignore")

from dataclasses import dataclass

import distrax
import haiku as hk
import numpy as np
import jax
import optax
import pandas as pd
from jax import lax, nn, jit, grad
from jax import numpy as jnp
from jax import random
from jax import scipy as jsp
from scipy import integrate

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={"figure.figsize": (6, 3)})
sns.set_style("ticks", {"font.family": "serif", "font.serif": "Merriweather"})

from dataclasses import dataclass


def get_embedding(inputs, embedding_dim, max_positions=10000):
    assert len(inputs.shape) == 1
    half_dim = embedding_dim // 2
    emb = jnp.log(max_positions) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    emb = inputs[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    return emb

@dataclass
class ScoreModel(hk.Module):
    output_dim = 2
    hidden_dims = [256, 256]
    embedding_dim = 256

    def __call__(self, z, t, is_training):
        dropout_rate = 0.1 if is_training else 0.0
        t_embedding = hk.nets.MLP(
            [self.embedding_dim] * 2, activation=nn.swish
        )(get_embedding(t, self.embedding_dim))

        h = hk.Linear(self.embedding_dim)(z)
        h += t_embedding

        for dim in self.hidden_dims:
            h = hk.Linear(dim)(h)
            h = jax.nn.swish(h)

        h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)
        h = hk.Linear(self.output_dim)(h)
        return h





prng_seq = hk.PRNGSequence(1)


train_iter, _ = get_nine_gaussians(10_000)

import jax.random as jr


def train_epoch(rng_key, state, train_iter):
    @jax.jit
    def step_fn(step_key, batch):
        def loss_fn(params):
            loss = state.apply_fn(
                params=params,
                rng=step_key,
                method="loss",
                y=batch,
                is_training=True,
            )
            loss = jnp.mean(loss) / np.prod(batch.shape[1:])
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return loss, new_state

    epoch_loss, n_samples = 0.0, 0
    for batch in train_iter:
        batch_key, rng_key = jr.split(rng_key)
        batch_loss, state = step_fn(batch_key, batch)
        epoch_loss += batch_loss
        n_samples += batch.shape[0]
    epoch_loss /= n_samples
    return float(epoch_loss), state


def optim(state, train_iter, n_epochs=1000):
    train_key = jr.PRNGKey(2)
    losses = []
    for epoch in range(n_epochs):
        print(epoch)
        train_key, val_key = jr.split(jr.fold_in(train_key, epoch))
        train_loss, state = train_epoch(train_key, state, train_iter)
        losses.append(train_loss)

    return state.params, losses


class DDPM(hk.Module):
    def __init__(self, score_model, betas):
        super().__init__()
        self.score_model = score_model
        self.n_diffusions = len(betas)
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_bar = jnp.cumprod(self.alphas)
        self.sqrt_alphas_bar = jnp.sqrt(self.alphas_bar)
        self.sqrt_1m_alphas_bar = jnp.sqrt(1.0 - self.alphas_bar)
        self.sqrt_1m_alphas = jnp.sqrt(1.0 - self.alphas)

    def __call__(self, method="loss", **kwargs):
        return getattr(self, method)(**kwargs)

    def loss(self, y, is_training=True):
        t = random.randint(
            key=hk.next_rng_key(),
            minval=1,
            maxval=self.n_diffusions,
            shape=(y.shape[0],),
        ).reshape(-1, 1)
        noise = random.normal(hk.next_rng_key(), y.shape)
        perturbed_y = (
                self.sqrt_alphas_bar[t] * y +
                self.sqrt_1m_alphas_bar[t] * noise
        )
        eps = self.score_model(
            perturbed_y,
            t.reshape(-1),
            is_training,
        )
        loss = jnp.sum(jnp.square(noise - eps), axis=-1)
        return loss

    # def sample(self, sample_shape=(1,)):
    #     def _fn(i, x):
    #         t = self.n_diffusions - i
    #         z = random.normal(hk.next_rng_key(), x.shape)
    #         sc = self.score_model(
    #             x,
    #             jnp.full(x.shape[0], t),
    #             False,
    #         )
    #         xn = (1 - self.alphas[t]) / self.sqrt_1m_alphas_bar[t] * sc
    #         xn = x - xn
    #         xn = xn / jnp.sqrt(self.alphas[t])
    #         x = xn + self.betas[t] * z
    #         return x
    #
    #     z_T = random.normal(hk.next_rng_key(), sample_shape + (2,))
    #     z0 = hk.fori_loop(0, self.n_diffusions, _fn, z_T)
    #     return z0

    def sample(self, sample_shape=(1,), n=20):
        init_key, rng_key = jr.split(hk.next_rng_key())

        possible_timesteps = np.arange(0, self.n_diffusions, self.n_diffusions // n)
        print(possible_timesteps)
        yt = jr.normal(init_key, sample_shape + (2,))
        for t in reversed(np.arange(n)):
            if t == 0:
                break
            tprev = possible_timesteps[t - 1]
            tcurr = possible_timesteps[t]
            print(tprev, " ", tcurr)
            yt = self.denoise_ddim(jr.fold_in(rng_key, tcurr), yt, tcurr, tprev)

        return yt

    def denoise_ddim(self, rng_key, yt, t, tprev):
        noise_key, rng_key = jr.split(rng_key)
        noise = jr.normal(key=noise_key, shape=yt.shape)
        eps = self.score_model(
            yt, jnp.full(yt.shape[0], t), is_training=False
        )
        lhs = (yt - eps * jnp.sqrt(1.0 - self.alphas_bar[t])) / jnp.sqrt(self.alphas_bar[t])
        lhs = jnp.sqrt(self.alphas_bar[tprev]) * lhs
        rhs = jnp.sqrt(1.0 - self.alphas_bar[tprev]) * eps
        ytm1 = lhs + rhs
        return ytm1



def _ddpm(**kwargs):
    score_model = ScoreModel()
    model = DDPM(score_model, jnp.linspace(10e-4, 0.02, 100))
    return model(**kwargs)


model = hk.transform(_ddpm)
params = model.init(random.PRNGKey(0), y=jnp.ones((1000, 2)))



tx = optax.adamw(0.001)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)



if not pathlib.Path("./params.pkl").exists():
    params, losses = optim(state, train_iter)
    losses = jnp.asarray(losses)
else:
    with open('params.pkl', 'rb') as handle:
        params = pickle.load(handle)

def plot(losses, samples):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    # sns.lineplot(
    #     data=pd.DataFrame({"y": np.asarray(losses), "x": range(len(losses))}),
    #     y="y",
    #     x="x",
    #     color="black",
    #     ax=ax1
    # )
    sns.kdeplot(
        data=pd.DataFrame(np.asarray(samples), columns=["x", "y"]),
        x="x",
        y="y",
        fill=True,
        cmap="mako_r",
        ax=ax2
    )
    ax1.set(title="Loss profile", xlabel="", ylabel="Loss", xticks=[], xticklabels=[], yticks=[], yticklabels=[])
    ax2.set(title="Generated samples", xlabel="$y_0$", ylabel="$y_1$")
    plt.show()

samples = model.apply(
    params, random.PRNGKey(0), method="sample", sample_shape=(10000,)
)

plot(None, samples)