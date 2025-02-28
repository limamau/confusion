import array
import gzip
import os
import struct
import urllib.request

import jax.numpy as jnp


def load_mnist():
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data/mnist",
    )
    url = "https://storage.googleapis.com/cvdf-datasets/mnist"
    image_filename = "train-images-idx3-ubyte.gz"
    label_filename = "train-labels-idx1-ubyte.gz"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    # download images if not present
    image_path = os.path.join(data_dir, image_filename)
    if not os.path.exists(image_path):
        url = f"{url}/{image_filename}"
        urllib.request.urlretrieve(url, image_path)
        print(f"Downloaded {url} to {image_path}")

    # download labels if not present
    label_path = os.path.join(data_dir, label_filename)
    if not os.path.exists(label_path):
        url = f"{url}/{label_filename}"
        urllib.request.urlretrieve(url, label_path)
        print(f"Downloaded {url} to {label_path}")

    # load images
    with gzip.open(image_path, "rb") as fh:
        magic_number, num_images, rows, cols = struct.unpack(">IIII", fh.read(16))
        if magic_number != 2051:
            raise ValueError(f"Invalid magic number in image file: {magic_number}")
        shape = (num_images, 1, rows, cols)
        images = jnp.array(array.array("B", fh.read()), dtype=jnp.uint8).reshape(shape)

    # load labels
    with gzip.open(label_path, "rb") as fh:
        magic_number, num_labels = struct.unpack(">II", fh.read(8))
        if magic_number != 2049:
            raise ValueError(f"Invalid magic number in label file: {magic_number}")
        if num_images != num_labels:
            raise ValueError(
                f"Number of images ({num_images}) does not match number of labels ({num_labels})"
            )
        labels = jnp.array(array.array("B", fh.read()), dtype=jnp.uint8)

    return images, labels
