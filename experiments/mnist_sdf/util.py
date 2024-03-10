import os
import pathlib

import jax
import numpy as np
import optax
import orbax.checkpoint
import tensorflow as tf
import tensorflow_datasets as tfds
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt


def get_minst_sdf_data(num_samples=None, split="train", size=32):
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

    def as_batched_numpy_iter(itr):
        return tfds.as_numpy(itr.shuffle(2000).batch(32).prefetch(100))

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
        itr = as_batched_numpy_iter(itr)
        itrs.append(itr)
    return itrs


def get_train_state(rng_key, model, init_batch):
    params = model.init(
        {"params": rng_key, "sample": rng_key},
        method="loss",
        y_0=init_batch,
        is_training=False
    )
    tx = optax.adam(0.0003)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def get_checkpointer_fns():
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=2,
        save_interval_steps=5,
        create=True,
        best_fn=lambda x: x['val_loss'],
        best_mode='min'
    )
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        os.path.join(pathlib.Path(os.path.abspath(__file__)).parent, "checkpoints"),
        checkpointer,
        options
    )

    def save_fn(epoch, ckpt, metrics):
        save_args = orbax_utils.save_args_from_target(ckpt)
        checkpoint_manager.save(
            epoch,
            ckpt,
            save_kwargs={"save_args": save_args},
            metrics=metrics
        )

    def restore_fn():
        return checkpoint_manager.restore(checkpoint_manager.best_step())

    return save_fn, restore_fn


def visualize_samples(samples, n_rows=2, n_cols=5):
    path = pathlib.Path(__file__).resolve().parent / "figures"
    path.mkdir(exist_ok=True)

    _, axes = plt.subplots(figsize=(2 * n_cols, 2 * n_rows), nrows=n_rows, ncols=n_cols)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i, :, :, 0])
        ax.axis('off')
    plt.suptitle(f"Synthetic images of dimension {samples.shape[1]}x{samples.shape[2]}")
    plt.tight_layout()
    plt.savefig(str(path) + f"/mnist_sdf-size_{samples.shape[1]}x{samples.shape[2]}.png")