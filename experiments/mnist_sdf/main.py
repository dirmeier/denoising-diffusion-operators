import dataclasses
from collections.abc import Sequence

import jax
import numpy as np
from absl import logging, flags, app
from flax.training.early_stopping import EarlyStopping
from flax import linen as nn
from jax import numpy as jnp, random as jr, Array

from ddo import DenoisingDiffusionOperator, ScoreModelUNO, cosine_alpha_schedule
from util import get_train_state, get_checkpointer_fns, \
    get_minst_sdf_data, visualize_samples

FLAGS = flags.FLAGS
flags.DEFINE_enum("mode", None, ["train", "sample"], "")
flags.DEFINE_integer("epochs", 1000, "max number of epochs used for training")
flags.mark_flags_as_required(["mode"])


@dataclasses.dataclass
class DiffusionUNOConfig:
    n_channels: int = 64
    channel_multipliers: Sequence[int] = (1, 2)
    n_modes: Sequence[int] = (5, 5)


@dataclasses.dataclass
class DenoisingDiffusionOperatorConfig:
    score_model: nn.Module
    alpha_schedule: Array
    n_langevin_steps: int = 100


def get_model():
    score_model = ScoreModelUNO(**dataclasses.asdict(DiffusionUNOConfig()))
    ddo = DenoisingDiffusionOperator(
        **dataclasses.asdict(DenoisingDiffusionOperatorConfig(
            score_model=score_model,
            alpha_schedule=cosine_alpha_schedule(1000)
        )),
    )
    return ddo


def train_epoch(rng_key, state, train_iter):
    @jax.jit
    def step_fn(step_key, batch):
        def loss_fn(params):
            loss = state.apply_fn(
                variables=params,
                rngs={"sample": step_key},
                method="loss",
                y_0=batch,
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


def evaluate_model(rng_key, state, val_iter):
    @jax.jit
    def loss_fn(params, rng_key, batch):
        loss = state.apply_fn(
            variables=params,
            rngs={"sample": rng_key},
            method="loss",
            y_0=batch,
            is_training=False,
        )
        loss = jnp.mean(loss) / np.prod(batch.shape[1:])
        return loss

    loss, n_samples = 0.0, 0
    for batch in val_iter:
        val_key, rng_key = jr.split(rng_key)
        loss += loss_fn(state.params, val_key, batch)
        n_samples += batch.shape[0]
    return float(loss) / n_samples


def train(rng_key, train_iter, val_iter, n_epochs):
    state_key, rng_key = jr.split(rng_key)
    model = get_model()
    state = get_train_state(state_key, model, next(iter(train_iter)))
    ckpt_save_fn, ckpt_restore_fn = get_checkpointer_fns()

    train_key, rng_key = jr.split(rng_key)
    early_stop = EarlyStopping(patience=10, min_delta=1e-05)
    logging.info("training model")
    for epoch in range(n_epochs):
        train_key, val_key = jr.split(jr.fold_in(train_key, epoch))
        train_loss, state = train_epoch(train_key, state, train_iter)
        val_loss = evaluate_model(val_key, state, val_iter)

        logging.info(f"loss at epoch {epoch}: {train_loss}/{val_loss}")
        early_stop = early_stop.update(val_loss)
        if early_stop.should_stop:
            logging.info(f'met early stopping criterion, breaking at epoch {epoch}')
            break
        ckpt_save_fn(epoch, state, {"train_loss": train_loss, "val_loss": val_loss})

    return ckpt_restore_fn()


def sample(sample_key, state, size):
    model = get_model()
    logging.info(f"sampling model with size {size}")
    samples = model.apply(
        variables=state["params"],
        rngs={"sample": sample_key},
        method="sample",
        sample_shape=(32, size, size, 1)
    )
    return samples


def main(argv):
    del argv
    logging.set_verbosity(logging.INFO)
    rng_key = jr.PRNGKey(123)
    if FLAGS.mode == "train":
        train_iter, val_iter = get_minst_sdf_data(split=["train[:90%]", "train[90%:]"])
        train_key, rng_key = jr.split(rng_key)
        train(train_key, train_iter, val_iter, n_epochs=FLAGS.epochs)
    else:
        sample_key, rng_key = jr.split(rng_key)
        _, ckpt_restore_fn = get_checkpointer_fns()
        state = ckpt_restore_fn()
        for size in (32, 64, 128):
            samples = sample(sample_key, state, size)
            visualize_samples(samples, 2, 5)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
