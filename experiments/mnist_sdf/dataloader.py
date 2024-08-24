import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging
from scipy.ndimage import distance_transform_edt
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def _as_batched_numpy_iter(itr, batch_size):
    return tfds.as_numpy(
        itr.shuffle(10_000)
        .batch(batch_size, drop_remainder=True)
        .prefetch(batch_size * 10)
    )


def get_data_loaders(
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
