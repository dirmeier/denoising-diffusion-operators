from typing import Any

import jax
import numpy as np
import optax
import orbax.checkpoint
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging
from flax import core, struct
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint.utils import get_save_directory
from scipy.ndimage import distance_transform_edt
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class EMATrainState(TrainState):
    ema_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    batch_stats: core.FrozenDict[str, Any] = struct.field(pytree_node=True)


def _as_batched_numpy_iter(itr, batch_size):
    return tfds.as_numpy(
        itr.shuffle(10_000)
        .batch(batch_size, drop_remainder=True)
        .prefetch(batch_size * 10)
    )


def get_minst_sdf_data(
    num_samples=None,
    dataset_name="mnist_sdf",
    outfolder="../../data",
    split="train",
    size=32,
    batch_size=128,
):
    assert dataset_name in ["mnist_sdf", "mnist"]

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

    ds_builder = tfds.builder("mnist", try_gcs=False, data_dir=outfolder)
    ds_builder.download_and_prepare(download_dir=outfolder)
    datasets = tfds.as_numpy(ds_builder.as_dataset(split=split, batch_size=-1))

    if isinstance(split, str):
        datasets = [datasets]

    itrs = []
    for dataset in datasets:
        ds = np.float32(dataset["image"] / 255.0)
        if num_samples is not None:
            ds = ds[:num_samples]
        ds = resize(ds, size)
        if dataset_name == "mnist_sdf":
            logging.info("using mnist_sdf data")
            for i in range(ds.shape[0]):
                ds[i] = distance_transform(ds[i]).reshape(size, size, 1)
        else:
            logging.info("using mnist data")
            ds = 2 * (ds - 0.5)  # 2 * (x - x_min) / (x_max - x_min) - 1
        logging.info(f"created ds of size: {ds.shape[0]}")
        itr = tf.data.Dataset.from_tensor_slices(ds)
        itr = _as_batched_numpy_iter(itr, batch_size)
        itrs.append(itr)
    return itrs


def get_train_state(rng_key, model, init_batch, config):
    variables = model.init(
        {"params": rng_key, "sample": rng_key},
        method="loss",
        y0=init_batch,
        is_training=False,
    )
    if config.optimizer.do_warmup and config.optimizer.do_decay:
        lr = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.optimizer.learning_rate,
            warmup_steps=config.optimizer.warmup_steps,
            decay_steps=config.optimizer.decay_steps,
            end_value=config.optimizer.end_learning_rate,
        )
    elif config.optimizer.do_warmup:
        lr = optax.linear_schedule(
            init_value=0.0,
            end_value=config.optimizer.learning_rate,
            transition_steps=config.optimizer.warmup_steps,
        )
    elif config.optimizer.do_decay:
        lr = optax.cosine_decay_schedule(
            init_value=config.optimizer.learning_rate,
            decay_steps=config.optimizer.decay_steps,
            alpha=config.optimizer.end_learning_rate
            / config.optimizer.learning_rate,
        )
    else:
        lr = config.optimizer.learning_rate

    tx = optax.adamw(lr, weight_decay=config.optimizer.weight_decay)
    if config.optimizer.do_gradient_clipping:
        tx = optax.chain(
            optax.clip_by_global_norm(config.optimizer.gradient_clipping), tx
        )

    return EMATrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        ema_params=variables["params"].copy(),
        batch_stats=variables["batch_stats"],
        tx=tx,
    )


def get_checkpointer_fns(outfolder):
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=5,
        save_interval_steps=20,
        create=True,
        best_fn=lambda x: x["train_loss"],
        best_mode="min",
    )
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        outfolder,
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
