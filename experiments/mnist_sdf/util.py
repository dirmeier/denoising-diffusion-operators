import os
import pathlib

import jax
import numpy as np
import optax
import orbax.checkpoint
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt
from orbax.checkpoint.utils import get_save_directory
from scipy.ndimage import distance_transform_edt
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def _as_batched_numpy_iter(itr, batch_size):
    return tfds.as_numpy(itr.shuffle(2000).batch(batch_size).prefetch(500))


def get_nine_gaussians(num_samples):
    K = 9
    means = jnp.array([-2.0, 0.0, 2.0])
    means = jnp.array(jnp.meshgrid(means, means)).T.reshape(-1, 2)
    covs = jnp.tile((1 / 16 * jnp.eye(2)), [K, 1, 1])

    probs = tfd.Uniform().sample(seed=jr.PRNGKey(23), sample_shape=(K,))
    probs = probs / jnp.sum(probs)

    d = tfd.MixtureSameFamily(
        tfd.Categorical(probs=probs),
        tfd.MultivariateNormalFullCovariance(means, covs),
    )

    y = d.sample(seed=jr.PRNGKey(12345), sample_shape=(num_samples,))
    train_itr = tf.data.Dataset.from_tensor_slices(y[: int(num_samples * 0.9)])
    train_itr = _as_batched_numpy_iter(train_itr)
    val_itr = tf.data.Dataset.from_tensor_slices(y[int(num_samples * 0.9) :])
    val_itr = _as_batched_numpy_iter(val_itr)
    return train_itr, val_itr


def get_minst_sdf_data(
    num_samples=None, split="train", size=32, batch_size=128
):
    def distance_transform(data):
        data[data < 0.5] = 0.0
        data[data >= 0.5] = 1.0

        neg_distances = distance_transform_edt(data)
        sd_img = data - 1.0
        sd_img = sd_img.astype(np.uint8)
        signed_distances = distance_transform_edt(sd_img) - neg_distances
        signed_distances /= float(data.shape[1])
        return signed_distances

    def resize(data, size):
        data = jax.image.resize(
            data,
            (data.shape[0], size, size, data.shape[-1]),
            method="bicubic",
            antialias=True,
        )
        return np.copy(data)

    ds_builder = tfds.builder("mnist", try_gcs=True)
    ds_builder.download_and_prepare(download_dir="../../data")
    datasets = tfds.as_numpy(ds_builder.as_dataset(split=split, batch_size=-1))

    if isinstance(split, str):
        datasets = [datasets]

    itrs = []
    for dataset in datasets:
        ds = np.float32(dataset["image"] / 255.0)
        if num_samples is not None:
            ds = ds[:num_samples]
        ds = resize(ds, size)
        for i in range(ds.shape[0]):
            ds[i] = distance_transform(ds[i]).reshape(size, size, 1)
        itr = tf.data.Dataset.from_tensor_slices(ds)
        itr = _as_batched_numpy_iter(itr, batch_size)
        itrs.append(itr)
    return itrs


def get_train_state(rng_key, model, init_batch):
    params = model.init(
        {"params": rng_key, "sample": rng_key},
        method="loss",
        y0=init_batch,
        is_training=False,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(0.0002),
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def get_checkpointer_fns(model_name):
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=2,
        save_interval_steps=5,
        create=True,
        best_fn=lambda x: x["train_loss"],
        best_mode="min",
    )
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        os.path.join(
            pathlib.Path(os.path.abspath(__file__)).parent,
            "checkpoints",
            model_name,
        ),
        checkpointer,
        options,
    )

    def save_fn(epoch, ckpt, metrics):
        save_args = orbax_utils.save_args_from_target(ckpt)
        checkpoint_manager.save(
            epoch, ckpt, save_kwargs={"save_args": save_args}, metrics=metrics
        )

    def restore_fn():
        return checkpoint_manager.restore(checkpoint_manager.best_step())

    def path_best_ckpt_fn():
        return get_save_directory(
            checkpoint_manager.best_step(), checkpoint_manager.directory
        )

    return save_fn, restore_fn, path_best_ckpt_fn


def visualize_samples(
    samples, suffix, model_name, n_rows=2, n_cols=5, usewand=False
):
    path = pathlib.Path(__file__).resolve().parent / "figures"
    path.mkdir(exist_ok=True)

    _, axes = plt.subplots(
        figsize=(2 * n_cols, 2 * n_rows), nrows=n_rows, ncols=n_cols
    )
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i, :, :, 0])
        ax.axis("off")

    plt.tight_layout()
    fl = f"/mnist_sdf-{model_name}-{suffix}-size_{samples.shape[1]}x{samples.shape[2]}.png"
    plt.savefig(str(path) + fl)

    if usewand:
        samples = np.asarray(samples).transpose(0, 3, 1, 2)
        images = []
        for idx in range(samples.shape[0]):
            image = wandb.Image(
                samples[idx],
                caption=f"Dimensions: {samples.shape[2]}x{samples.shape[3]}",
            )
            images.append(image)
        wandb.log({f"synthetic_{suffix}_size_{samples.shape[2]}": images})
