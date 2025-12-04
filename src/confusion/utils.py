import os
from typing import Iterator, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from confusion.models import AbstractModel


def dataloader(
    data: Array, conds: Optional[Array], batch_size: int, *, key: Key
) -> Iterator[Tuple[Array, Optional[Array]]]:
    dataset_size = data.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        key, subkey = jr.split(key, 2)
        perm = jr.permutation(subkey, indices)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            if conds is not None:
                yield data[batch_perm], conds[batch_perm]
            else:
                yield data[batch_perm], None
            start = end
            end = start + batch_size


def denormalize(
    data: Array,
    original_mean: Array,
    original_std: Array,
    imposed_mean: Union[Array, float] = 0.0,
    imposed_std: Union[Array, float] = 1.0,
) -> Array:
    data = (data - imposed_mean) / imposed_std
    data = data * original_std + original_mean
    return data


def get_and_create_figs_dir(file_path: str, sub_dir: Optional[str] = None):
    script_dir = os.path.dirname(os.path.abspath(file_path))
    if sub_dir is None:
        figs_dir = os.path.join(script_dir, "figs")
    else:
        figs_dir = os.path.join(script_dir, "figs", sub_dir)
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
    return figs_dir


def ignore_divergencies(samples, tol=1e2, print_divergencies=False):
    conv_idxs = jnp.abs(samples).max(axis=1) < tol
    num_divergencies = jnp.sum(~conv_idxs)
    print(f"Number of divergencies: {num_divergencies}")
    return samples[conv_idxs]


def normalize(
    data: Array,
    original_mean: Optional[Array] = None,
    original_std: Optional[Array] = None,
    imposed_mean: Union[Array, float] = 0.0,
    imposed_std: Union[Array, float] = 1.0,
    axis: Union[int, Sequence[int], None] = None,
) -> Tuple[Array, Array, Array]:
    if original_mean is None:
        original_mean = jnp.mean(data, axis=axis)

    if original_std is None:
        original_std = jnp.std(data, axis=axis)

    # prevent division by zero
    if jnp.any(original_std == 0):
        print("from normalization: 0 values in standard deviation - replacing with 1s.")
        original_std = jnp.where(original_std == 0, 1, original_std)

    data = (data - original_mean) / original_std
    data = data * imposed_std + imposed_mean

    return data, original_mean, original_std


def batch_avg_loss(
    model: AbstractModel,
    data: Array,
    times: Array,
    conds: Optional[Array],
    key: Key,
) -> Array:
    return jnp.mean(jax.vmap(model.loss)(data, times, conds, key))
