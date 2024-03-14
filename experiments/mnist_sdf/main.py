import hashlib
import os

import jax
import numpy as np
import optax
import wandb
from absl import app, flags, logging
from flax.training.early_stopping import EarlyStopping
from jax import numpy as jnp
from jax import random as jr
from ml_collections import config_flags
from util import (
    get_checkpointer_fns,
    get_minst_sdf_data,
    get_train_state,
)

import ddo
from ddo import (
    UNO,
    DenoisingDiffusionOperator,
    UNet,
)

os.environ["WANDB__SERVICE_WAIT"] = "300"


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "model configuration")
flags.DEFINE_enum("mode", None, ["train", "sample"], "do training or sampling")
flags.DEFINE_enum("model", "unet", ["unet", "uno"], "score network type")
flags.DEFINE_enum(
    "dataset", "mnist_sdf", ["mnist_sdf", "mnist"], "data set to use"
)
flags.DEFINE_string("workdir", None, "work directory")
flags.DEFINE_bool("usewand", False, "use wandb for logging")
flags.mark_flags_as_required(["mode", "workdir", "config"])


def get_schedule(alpha_schedule, n_timesteps, **kwargs):
    return getattr(ddo, f"{alpha_schedule}_alpha_schedule")(n_timesteps)


def get_config():
    config = {
        "diffusion_model_config": FLAGS.config.model.diffusion_model.to_dict(),
        "score_model_config": FLAGS.config.model.score_model.to_dict(),
        "training_config": FLAGS.config.training.to_dict(),
    }
    return config


def get_model(model_name, config):
    if model_name == "uno":
        score_model = UNO(n_modes=(17, 9, 5, 3), **config["score_model_config"])
    elif model_name == "unet":
        score_model = UNet(**config["score_model_config"])
    else:
        raise ValueError("neither of 'uno'/'unet'")

    schedule = get_schedule(**config["diffusion_model_config"])
    ddo = DenoisingDiffusionOperator(
        score_model=score_model, alpha_schedule=schedule
    )
    return ddo


@jax.jit
def step_fn(step_key, state, batch):
    def loss_fn(params, rng):
        sample_key, dropout_key = jr.split(rng)
        ll, ups = state.apply_fn(
            variables={"params": params, "batch_stats": state.batch_stats},
            rngs={"sample": sample_key, "dropout": dropout_key},
            mutable=["batch_stats"],
            method="loss",
            y0=batch,
            is_training=True,
        )
        ll = jnp.mean(ll) / (32 * 32)
        return ll, ups

    (loss, new_updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, step_key
    )
    new_state = state.apply_gradients(grads=grads)
    new_ema_params = optax.incremental_update(
        new_state.params,
        new_state.ema_params,
        step_size=1.0 - FLAGS.config.training.ema_rate,
    )
    new_state = new_state.replace(
        ema_params=new_ema_params, batch_stats=new_updates["batch_stats"]
    )
    return loss, new_state


def train_epoch(rng_key, state, train_iter):
    epoch_loss, n_samples = 0.0, 0
    for batch in train_iter:
        batch_key, rng_key = jr.split(rng_key)
        batch_loss, state = step_fn(batch_key, state, batch)
        epoch_loss += batch_loss
        n_samples += batch.shape[0]
    epoch_loss /= n_samples
    return float(epoch_loss), state


def evaluate_model(rng_key, state, val_iter):
    @jax.jit
    def loss_fn(rng_key, batch):
        ll = state.apply_fn(
            variables={
                "params": state.params,
                "batch_stats": state.batch_stats,
            },
            rngs={"sample": rng_key},
            method="loss",
            y0=batch,
            is_training=False,
        )
        ll = jnp.mean(ll) / (32 * 32)
        return ll

    loss, n_samples = 0.0, 0
    for batch in val_iter:
        val_key, rng_key = jr.split(rng_key)
        loss += loss_fn(val_key, batch)
        n_samples += batch.shape[0]
    return float(loss) / n_samples


def train(rng_key, train_iter, val_iter, model_id, run=None):
    state_key, rng_key = jr.split(rng_key)

    model = get_model(FLAGS.model, get_config())
    state = get_train_state(
        state_key, model, next(iter(train_iter)), FLAGS.config.training
    )
    ckpt_save_fn, ckpt_restore_fn, _ = get_checkpointer_fns(
        os.path.join(FLAGS.workdir, "checkpoints", model_id)
    )

    logging.info("training model")
    early_stop = EarlyStopping(patience=20, min_delta=1e-06)
    epoch_key, rng_key = jr.split(rng_key)

    for epoch in range(1, FLAGS.config.training.n_epochs + 1):
        train_key, val_key, sample_key = jr.split(
            jr.fold_in(epoch_key, epoch), 3
        )
        train_loss, state = train_epoch(train_key, state, train_iter)
        val_loss = evaluate_model(val_key, state, val_iter)
        logging.info(f"loss at epoch {epoch}: {train_loss}/{val_loss}")
        ckpt_save_fn(
            epoch,
            {"state": state, "config": get_config()},
            {"train_loss": train_loss, "val_loss": val_loss},
        )
        early_stop.update(train_loss)
        if run is not None:
            wandb.log({"loss": train_loss, "val_loss": val_loss})
        if run is not None and epoch % 100 == 0:
            logging.info("sampling images")
            log_images(sample_key, state, epoch, run)
        if early_stop.should_stop:
            logging.info("early stopping criterion found. stopping training")
            break
    ckpt = ckpt_restore_fn()
    return ckpt


def log_images(sample_key, state, epoch, run):
    sizes = (32,) if FLAGS.model == "unet" else (32, 64, 128)
    params = state.ema_params
    batch_stats = state.batch_stats
    for size in sizes:
        samples = _sample(
            sample_key,
            {"params": params, "batch_stats": batch_stats},
            state.apply_fn,
            size,
        )
        samples = np.asarray(samples).transpose(0, 3, 1, 2)
        images = []
        for idx in range(samples.shape[0]):
            image = wandb.Image(
                samples[idx],
                caption=f"Dimensions: {samples.shape[2]}x{samples.shape[3]}",
            )
            images.append(image)
        wandb.log({f"ddim_epoch_{epoch}_size_{samples.shape[2]}": images})


def _sample(sample_key, variables, apply_fn, size):
    @jax.jit
    def _fn():
        return apply_fn(
            variables=variables,
            rngs={"sample": sample_key},
            method="sample",
            sample_shape=(64, size, size, 1),
        )

    return _fn()


def sample(sample_key, ckpt):
    state, config = ckpt["state"], ckpt["config"]
    params, batch_stats = state["ema_params"], state["batch_stats"]
    model = get_model(FLAGS.model, config)
    sizes = (32,) if FLAGS.model == "unet" else (32, 64, 128)
    for size in sizes:
        samples = _sample(
            sample_key,
            {"params": params, "batch_stats": batch_stats},
            model.apply,
            size,
        )
        visualize_samples(samples)


def visualize_samples(samples, n_rows=2, n_cols=5):
    import matplotlib.pyplot as plt

    pth = os.path.join(FLAGS.workdir, "figures")
    _, axes = plt.subplots(
        figsize=(2 * n_cols, 2 * n_rows), nrows=n_rows, ncols=n_cols
    )
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i, :, :, 0])
        ax.axis("off")

    plt.tight_layout()
    fl = f"/{FLAGS.dataset}-{FLAGS.model}-size_{samples.shape[1]}x{samples.shape[2]}.png"
    plt.savefig(pth + fl)


def hash_value(config):
    h = hashlib.new("sha256")
    h.update(str(config).encode("utf-8"))
    return h.hexdigest()


def main(argv):
    del argv
    logging.set_verbosity(logging.INFO)
    run = None
    config = get_config()
    config["dataset"] = FLAGS.dataset
    model_id = f"{FLAGS.model}-{hash_value(config)}"

    if FLAGS.usewand:
        run = wandb.init(
            project="ddo-uno",
            config=config,
            dir=os.path.join(FLAGS.workdir, "wandb"),
        )
        wandb.run.name = model_id

    ## train model
    rng_key = jr.PRNGKey(FLAGS.config.rng_key)
    if FLAGS.mode == "train":
        train_iter, val_iter = get_minst_sdf_data(
            dataset_name=FLAGS.dataset,
            split=["train", "test[:1000]"],
            outfolder=os.path.join(FLAGS.workdir, "data"),
            batch_size=FLAGS.config.training.batch_size,
        )
        train_key, rng_key = jr.split(rng_key)
        train(train_key, train_iter, val_iter, model_id=model_id, run=run)
    else:
        ## sample images and store
        sample_key, rng_key = jr.split(rng_key)
        _, ckpt_restore_fn, _ = get_checkpointer_fns(
            os.path.join(FLAGS.workdir, "checkpoints", model_id)
        )
        ckpt = ckpt_restore_fn()
        sample(sample_key, ckpt)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
