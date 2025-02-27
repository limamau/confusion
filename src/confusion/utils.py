from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Array


def denormalize(data: Array, mean: Array, std: Array) -> Array:
    return data * std + mean


def normalize(data: Array, mean=None, std=None) -> Tuple[Array, Array, Array]:
    if mean is None:
        mean = jnp.mean(data)

    if std is None:
        std = jnp.std(data)

    return (data - mean) / std, mean, std
