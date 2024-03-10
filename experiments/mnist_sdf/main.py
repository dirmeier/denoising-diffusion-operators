import dataclasses
from collections.abc import Sequence

import jax
import numpy as np
import wandb
import hashlib

from absl import app, flags, logging
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt

from ddo import DenoisingDiffusionOperator, UNet, cosine_alpha_schedule, UNO
from ddo.mlp import MLP
from ddo.noise_schedule import linear_alpha_schedule
from util import (
    get_checkpointer_fns,
    get_minst_sdf_data,
    get_train_state,
    visualize_samples, get_nine_gaussians,
)

FLAGS = flags.FLAGS
flags.DEFINE_enum("mode", None, ["train", "sample"], "do training or sampling")
flags.DEFINE_enum("model", "unet", ["unet", "uno"], "score network type")
flags.DEFINE_integer("epochs", 1000, "number of epochs used for training")
flags.DEFINE_bool("usewand", False, "use wandb for logging")
flags.mark_flags_as_required(["mode"])


@dataclasses.dataclass
class ScoreModelConfig:
    n_blocks: int = 5
    dim_embedding: int = 128
    n_channels: int = 64
    channel_multipliers: Sequence[int] = (1, 2, 2)


@dataclasses.dataclass
class DDOConfig:
    alpha_schedule: np.array


def get_schedule():
    return linear_alpha_schedule(1000)


def get_model(model_name):
    if model_name == "uno":
        score_model = UNO(n_modes=(5, 5, 4), **dataclasses.asdict(ScoreModelConfig()))
    elif model_name == "unet":
        score_model = UNet(**dataclasses.asdict(ScoreModelConfig()))
    else:
        score_model = MLP()
    ddo = DenoisingDiffusionOperator(
        score_model=score_model,
        **dataclasses.asdict(DDOConfig(jnp.asarray(get_schedule())))
    )
    return ddo


def train_epoch(rng_key, state, train_iter):
    @jax.jit
    def step_fn(step_key, batch):
        def loss_fn(params):
            loss = state.apply_fn(
                variables=params,
                rngs={"sample": step_key, "dropout": step_key},
                method="loss",
                y0=batch,
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
            y0=batch,
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


def train(rng_key, train_iter, val_iter, model_name, n_epochs, usewand=False):
    state_key, rng_key = jr.split(rng_key)
    model = get_model(model_name)
    state = get_train_state(state_key, model, next(iter(train_iter)))
    ckpt_save_fn, ckpt_restore_fn, path_best_ckpt_fn = get_checkpointer_fns(model_name)

    
    train_loss = evaluate_model(jr.PRNGKey(0), state, train_iter)
    val_loss = evaluate_model(jr.PRNGKey(0), state, val_iter)
    if usewand:
        wandb.log({"epoch": -1, "loss": train_loss, "val_loss": val_loss})
    logging.info(f"pretraining loss: {train_loss}/{val_loss}")

    train_key, rng_key = jr.split(rng_key)
    logging.info("training model")
    for epoch in range(n_epochs):
        train_key, val_key = jr.split(jr.fold_in(train_key, epoch))
        train_loss, state = train_epoch(train_key, state, train_iter)
        val_loss = evaluate_model(val_key, state, val_iter)
        logging.info(f"loss at epoch {epoch}: {train_loss}/{val_loss}")
        if usewand:
            wandb.log({"epoch": epoch, "loss": train_loss, "val_loss": val_loss})
        ckpt_save_fn(
            epoch, state, {"train_loss": train_loss, "val_loss": val_loss}
        )
    ckpt = ckpt_restore_fn()
    if usewand:
        art = wandb.Artifact("ckpt", "ckpt")
        art.add_dir(path_best_ckpt_fn())
        wandb.log_artifact(art)
    return ckpt


def sample(sample_key, sample_method, state, size, model_name):
    model = get_model(model_name)
    logging.info(f"sampling model with size {size}")
    samples = model.apply(
        variables=state["params"],
        rngs={"sample": sample_key},
        method=sample_method,
        sample_shape=(128, size, size, 1),
    )
    return samples


def hash_value(config):
    h = hashlib.new('sha256')
    h.update(str(config).encode('utf-8'))
    return h.hexdigest()


def main(argv):
    del argv
    logging.set_verbosity(logging.INFO)
    if FLAGS.usewand:
        config = {
            "ddo_config": dataclasses.asdict(DDOConfig(get_schedule())), 
            "score_model_config": dataclasses.asdict(ScoreModelConfig())
        }
        wandb.init(project="ddo-uno", config=config)
        wandb.run.name = f"{FLAGS.model}-{hash_value(config)}"
    
    rng_key = jr.PRNGKey(23)
    if FLAGS.mode == "train":
        train_iter, val_iter = get_minst_sdf_data(
            split=["train[:90%]", "train[90%:]"],
        )
        train_key, rng_key = jr.split(rng_key)
        train(train_key, train_iter, val_iter, FLAGS.model, FLAGS.epochs, FLAGS.usewand)
        
    sample_key, rng_key = jr.split(rng_key)
    _, ckpt_restore_fn, _ = get_checkpointer_fns(FLAGS.model)
    state = ckpt_restore_fn()
    sizes = (32,) if FLAGS.model == "unet" else (32, 64, 128)

    for sample_method in ["sample", "sample_ddim"]:
        for size in sizes:
            samples = sample(sample_key, sample_method, state, size, FLAGS.model)
            visualize_samples(samples, sample_method, FLAGS.model, 2, 5, FLAGS.usewand)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
