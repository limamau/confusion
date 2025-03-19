import os
from typing import Optional, Tuple

import jax.numpy as jnp
from jaxtyping import Array


def get_and_create_figs_dir(file_path: str, sub_dir: Optional[str] = None):
    script_dir = os.path.dirname(os.path.abspath(file_path))
    if sub_dir is None:
        figs_dir = os.path.join(script_dir, "figs")
    else:
        figs_dir = os.path.join(script_dir, "figs", sub_dir)
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
    return figs_dir


def denormalize(data: Array, mean: Array, std: Array) -> Array:
    return data * std + mean


def normalize(data: Array, mean=None, std=None) -> Tuple[Array, Array, Array]:
    if mean is None:
        mean = jnp.mean(data)

    if std is None:
        std = jnp.std(data)

    return (data - mean) / std, mean, std
